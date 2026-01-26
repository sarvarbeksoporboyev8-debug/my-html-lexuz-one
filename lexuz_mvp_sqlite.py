#!/usr/bin/env python3
r"""
LexUZ Ultra‑Cheap MVP (SQLite + local embeddings + optional ANN index + Telegram notifications + RAG Q&A)

What this script does
---------------------
1) Watches Lex.uz "calendar of published legal acts" pages to discover new documents.
2) Periodically re-checks older documents to detect updates (content hash change).
3) Ingests changed documents:
   - downloads HTML
   - extracts/cleans text
   - chunks it
   - embeds chunks (default: local SentenceTransformers multilingual E5)
   - stores everything in a local SQLite DB
   - updates a vector index on disk (FAISS or hnswlib if installed; otherwise falls back to NumPy brute force)
4) Sends Telegram notifications when new/updated docs are indexed.
5) Provides a CLI "ask" command: uses an LLM (OpenAI API) to normalize messy questions, retrieves relevant LexUZ chunks,
   then uses the LLM again to answer with citations (URLs).

Why SQLite (and not Postgres+pgvector)?
--------------------------------------
On Windows, pgvector is often the biggest setup blocker. SQLite has *zero* server setup.
You can still move to Postgres later; the ingestion/retrieval logic stays the same.

Install dependencies
--------------------
Minimum:
  pip install requests beautifulsoup4 lxml python-dotenv numpy

PDF text extraction (some newer LexUZ docs are PDFs):
  pip install pypdf

Scanned PDFs (no text layer) OCR (recommended):
  Uses Azure AI Document Intelligence (Read model) via REST (no extra pip deps).
  Requires setting AZURE_DI_ENDPOINT and AZURE_DI_KEY in .env.
  Enable with ENABLE_OCR=1.

Local embeddings (recommended, free):
  pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

ANN index (optional):
  # Try FAISS first (may or may not have wheels for your Python version)
  pip install faiss-cpu
  # or try hnswlib
  pip install hnswlib

OpenAI for question normalization + answering:
  (no extra pip pkg needed; this script uses raw HTTP via requests)

Telegram notifications (optional):
  (no extra pip pkg needed; raw HTTP via requests)

Quick start (Windows PowerShell)
-------------------------------
1) Create a folder and put this file there, e.g. C:\\lexuz\\lexuz_mvp_sqlite.py
2) Create a virtualenv:
      py -m venv venv
      .\\venv\\Scripts\\Activate.ps1
3) Install deps:
      pip install -U pip
      pip install requests beautifulsoup4 lxml python-dotenv numpy
      pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
      pip install faiss-cpu  # optional
      pip install hnswlib    # optional
4) Create a .env file in the same folder (example below)
5) Run:
      python .\\lexuz_mvp_sqlite.py doctor
      python .\\lexuz_mvp_sqlite.py run-once
      python .\\lexuz_mvp_sqlite.py ask "QQS stavkasi qancha? tushuntirib bering"

Example .env
------------
# Storage
DATA_DIR=./data
SQLITE_PATH=./data/lexuz.sqlite3

# LexUZ
LEX_BASE_URL=https://lex.uz
LEX_LANG=uz
LEX_CALENDAR_LANG_PARAM=4   # 4 often maps to O'ZB Latin in LexUZ URLs

# Watcher
POLL_DAYS=2
MAX_DOCS_PER_RUN=15
RECHECK_DAYS=21
RECHECK_PER_RUN=6
HTTP_RATE_LIMIT_SECONDS=0.7
HTTP_TIMEOUT_SECONDS=30

# Chunking
CHUNK_MAX_WORDS=350
CHUNK_OVERLAP_WORDS=60

# Retrieval
TOP_K=6
CANDIDATE_MULTIPLIER=6
MAX_CHUNKS_PER_DOC=3

# Embeddings (free local default)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=intfloat/multilingual-e5-small

# Vector index backend: auto|faiss|hnswlib|numpy
INDEX_BACKEND=auto

# OpenAI (for ask) — required for ask/rewrite/answer
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-5-nano
USE_OPENAI_REWRITE=1

# Telegram (optional)
TELEGRAM_BOT_TOKEN=123456:ABCDEF...
TELEGRAM_CHAT_ID=123456789

# OCR for scanned PDFs (Azure AI Document Intelligence)
# When LexUZ serves a PDF with no text layer (scan), set ENABLE_OCR=1 and configure Azure DI:
ENABLE_OCR=1
AZURE_DI_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com
AZURE_DI_KEY=...
AZURE_DI_API_VERSION=2024-11-30
AZURE_DI_MODEL=prebuilt-read
AZURE_DI_OUTPUT_FORMAT=text
AZURE_DI_OCR_HIGH_RES=1
AZURE_DI_POLL_TIMEOUT_SECONDS=180

Notes
-----
- This is an MVP. It focuses on being robust and cheap.
- For deletions: when documents update, old chunks/vectors are marked inactive in SQLite.
  FAISS/HNSW indices may still contain old vectors; the search results are filtered by "active=1".
  You can periodically run `rebuild-index` to compact.

License
-------
You can use/modify this file for your project.
"""

from __future__ import annotations

import base64
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import textwrap
from io import BytesIO
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urlencode, parse_qsl

import numpy as np
import requests
from bs4 import BeautifulSoup

# Optional deps
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Optional ANN backends
_FAISS = None
_HNSWLIB = None
try:  # pragma: no cover
    import faiss  # type: ignore

    _FAISS = faiss
except Exception:
    _FAISS = None

try:  # pragma: no cover
    import hnswlib  # type: ignore

    _HNSWLIB = hnswlib
except Exception:
    _HNSWLIB = None

# Optional local embeddings
_SENTENCE_TRANSFORMERS = None
try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer  # type: ignore

    _SENTENCE_TRANSFORMERS = SentenceTransformer
except Exception:
    _SENTENCE_TRANSFORMERS = None

# Optional PDF text extraction (for newer LexUZ docs served as PDFs)
_PYPDF = None
try:  # pragma: no cover
    from pypdf import PdfReader  # type: ignore

    _PYPDF = PdfReader
except Exception:
    _PYPDF = None




# Optional PDF text extraction engines (better than pypdf for some PDFs)
_PYMUPDF = None
_PDFMINER_EXTRACT_TEXT = None

try:  # pragma: no cover
    import fitz  # PyMuPDF
    _PYMUPDF = fitz
except Exception:
    _PYMUPDF = None

try:  # pragma: no cover
    from pdfminer.high_level import extract_text as _pdfminer_extract_text  # type: ignore
    _PDFMINER_EXTRACT_TEXT = _pdfminer_extract_text
except Exception:
    _PDFMINER_EXTRACT_TEXT = None

# OCR for scanned PDFs (no text layer)
# - Preferred (this project): Azure AI Document Intelligence (Read model) via REST.
#   Configure AZURE_DI_ENDPOINT + AZURE_DI_KEY and set ENABLE_OCR=1.
# - Optional fallback: local Tesseract OCR (pytesseract + system tesseract).
#   Kept for backward compatibility; disabled by default.
_PYTESSERACT = None
_PIL_IMAGE = None
try:  # pragma: no cover
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore

    _PYTESSERACT = pytesseract
    _PIL_IMAGE = Image
except Exception:
    _PYTESSERACT = None
    _PIL_IMAGE = None

# -----------------------------
# Utility helpers
# -----------------------------

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def iso(dt_obj: dt.datetime) -> str:
    return dt_obj.astimezone(dt.timezone.utc).isoformat(timespec="seconds")

def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sleep_rate_limited(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)

def is_windows() -> bool:
    return os.name == "nt"

def normalize_lex_base_url(url: str) -> str:
    """Normalize LexUZ base URL and prefer www.lex.uz (docs sometimes 404 on apex domain)."""
    url = (url or "").strip()
    if not url:
        url = "https://www.lex.uz"
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    p = urlparse(url)
    netloc = (p.netloc or "").lower()
    if netloc == "lex.uz":
        netloc = "www.lex.uz"
    return p._replace(netloc=netloc).geturl().rstrip("/")


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    # Storage
    data_dir: Path
    sqlite_path: Path

    # LexUZ
    lex_base_url: str
    lex_lang: str
    lex_calendar_lang_param: str

    # Watcher
    poll_days: int
    max_docs_per_run: int
    recheck_days: int
    recheck_per_run: int
    http_rate_limit_seconds: float
    http_timeout_seconds: int

    # Extraction quality / robustness
    min_text_words: int
    min_unique_words: int
    min_unique_ratio: float
    max_common_line_ratio: float
    max_pdf_url_tries: int

    # OCR (for scanned PDFs) - optional
    enable_ocr: bool
    ocr_lang: str
    ocr_max_pages: int
    ocr_dpi: int
    tesseract_cmd: str

    # Azure AI Document Intelligence (OCR for scanned PDFs / no text layer)
    azure_di_endpoint: str
    azure_di_key: str
    azure_di_api_version: str
    azure_di_model: str
    azure_di_output_format: str  # text|markdown
    azure_di_ocr_high_res: bool
    azure_di_poll_interval_seconds: float
    azure_di_poll_timeout_seconds: int

    # Scheduling
    error_retry_days: int

    # Chunking
    chunk_max_words: int
    chunk_overlap_words: int

    # Retrieval
    top_k: int
    candidate_multiplier: int
    max_chunks_per_doc: int

    # Embeddings
    embedding_provider: str  # local|openai
    embedding_model: str

    # Index backend
    index_backend: str  # auto|faiss|hnswlib|numpy

    # OpenAI (for ask)
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    use_openai_rewrite: bool

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # Bright Data proxy (optional)
    brightdata_enabled: bool
    brightdata_host: str
    brightdata_port: int
    brightdata_username: str
    brightdata_password: str

    # Misc
    user_agent: str

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw_html"

    @property
    def text_dir(self) -> Path:
        return self.data_dir / "clean_text"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "vector_index"

    @property
    def faiss_index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def hnsw_index_path(self) -> Path:
        return self.index_dir / "hnsw.index"

    @property
    def numpy_cache_path(self) -> Path:
        return self.index_dir / "numpy_cache.npz"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "run.lock"

    @staticmethod
    def from_env() -> "Config":
        # Load .env if available
        if load_dotenv is not None:
            # Load from current working directory
            load_dotenv(override=False)

        data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
        sqlite_path = Path(os.getenv("SQLITE_PATH", str(data_dir / "lexuz.sqlite3"))).resolve()

        lex_base_url = normalize_lex_base_url(os.getenv("LEX_BASE_URL", "https://www.lex.uz"))
        lex_lang = os.getenv("LEX_LANG", "uz").strip()
        lex_calendar_lang_param = os.getenv("LEX_CALENDAR_LANG_PARAM", "4").strip()

        poll_days = safe_int(os.getenv("POLL_DAYS", "2"), 2)
        max_docs_per_run = safe_int(os.getenv("MAX_DOCS_PER_RUN", "15"), 15)
        recheck_days = safe_int(os.getenv("RECHECK_DAYS", "21"), 21)
        recheck_per_run = safe_int(os.getenv("RECHECK_PER_RUN", "6"), 6)
        http_rate_limit_seconds = float(os.getenv("HTTP_RATE_LIMIT_SECONDS", "0.7"))
        http_timeout_seconds = safe_int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"), 30)

        # Extraction quality thresholds (tune if you see "text too short" or PDF-only docs)
        min_text_words = safe_int(os.getenv("MIN_TEXT_WORDS", "120"), 120)
        min_unique_words = safe_int(os.getenv("MIN_UNIQUE_WORDS", "60"), 60)
        min_unique_ratio = float(os.getenv("MIN_UNIQUE_RATIO", "0.15"))
        max_common_line_ratio = float(os.getenv("MAX_COMMON_LINE_RATIO", "0.25"))
        max_pdf_url_tries = safe_int(os.getenv("MAX_PDF_URL_TRIES", "4"), 4)

        # OCR (optional). Enable only if you install Tesseract and pytesseract.
        enable_ocr = os.getenv("ENABLE_OCR", "0").strip() not in ("0", "false", "False", "")
        ocr_lang = os.getenv("OCR_LANG", "uzb+rus+eng").strip()
        ocr_max_pages = safe_int(os.getenv("OCR_MAX_PAGES", "50"), 50)
        ocr_dpi = safe_int(os.getenv("OCR_DPI", "200"), 200)
        tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()

        # Azure AI Document Intelligence OCR (for scanned PDFs / no text layer)
        azure_di_endpoint = os.getenv("AZURE_DI_ENDPOINT", "").strip().rstrip("/")
        azure_di_key = os.getenv("AZURE_DI_KEY", "").strip()
        azure_di_api_version = os.getenv("AZURE_DI_API_VERSION", "2024-11-30").strip()
        azure_di_model = os.getenv("AZURE_DI_MODEL", "prebuilt-read").strip() or "prebuilt-read"
        azure_di_output_format = os.getenv("AZURE_DI_OUTPUT_FORMAT", "text").strip().lower()
        if azure_di_output_format not in ("text", "markdown"):
            azure_di_output_format = "text"
        azure_di_ocr_high_res = os.getenv("AZURE_DI_OCR_HIGH_RES", "0").strip().lower() not in ("0", "false", "no", "")
        azure_di_poll_interval_seconds = float(os.getenv("AZURE_DI_POLL_INTERVAL_SECONDS", "2.0"))
        azure_di_poll_timeout_seconds = safe_int(os.getenv("AZURE_DI_POLL_TIMEOUT_SECONDS", "180"), 180)

        # Error retry cadence (days)
        error_retry_days = safe_int(os.getenv("ERROR_RETRY_DAYS", "3"), 3)

        chunk_max_words = safe_int(os.getenv("CHUNK_MAX_WORDS", "350"), 350)
        chunk_overlap_words = safe_int(os.getenv("CHUNK_OVERLAP_WORDS", "60"), 60)

        top_k = safe_int(os.getenv("TOP_K", "6"), 6)
        candidate_multiplier = safe_int(os.getenv("CANDIDATE_MULTIPLIER", "6"), 6)
        max_chunks_per_doc = safe_int(os.getenv("MAX_CHUNKS_PER_DOC", "3"), 3)

        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small").strip()

        index_backend = os.getenv("INDEX_BACKEND", "auto").strip().lower()

        # LLM API config (OpenAI-compatible).
        # Supports either OPENAI_* (OpenAI) or OPENROUTER_* (OpenRouter) env vars.
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        openai_model_env = os.getenv("OPENAI_MODEL", "").strip()
        openai_base_env = os.getenv("OPENAI_BASE_URL", "").strip()

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openrouter_model = os.getenv("OPENROUTER_MODEL", "").strip()
        openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api").strip()

        if not openai_api_key and openrouter_api_key:
            openai_api_key = openrouter_api_key

        openai_model = openai_model_env or (openrouter_model if openrouter_model else "gpt-5-nano")
        openai_base_url = openai_base_env or (openrouter_base if openrouter_api_key else "https://api.openai.com")

        # Normalize base URL: strip trailing slashes and an optional trailing "/v1".
        openai_base_url = openai_base_url.strip().rstrip("/")
        if openai_base_url.endswith("/v1"):
            openai_base_url = openai_base_url[:-3]

        use_openai_rewrite = os.getenv("USE_OPENAI_REWRITE", "1").strip() not in ("0", "false", "False", "")

        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

        # Bright Data proxy configuration
        brightdata_enabled = os.getenv("BRIGHTDATA_ENABLED", "0").strip().lower() not in ("0", "false", "no", "")
        brightdata_host = os.getenv("BRIGHTDATA_HOST", "brd.superproxy.io").strip()
        brightdata_port = safe_int(os.getenv("BRIGHTDATA_PORT", "33335"), 33335)
        brightdata_username = os.getenv("BRIGHTDATA_USERNAME", "").strip()
        brightdata_password = os.getenv("BRIGHTDATA_PASSWORD", "").strip()

        user_agent = os.getenv(
            "USER_AGENT",
            "LexUZ-MVP-SQLite/1.0 (+https://example.invalid; contact: you@example.com)",
        ).strip()

        return Config(
            data_dir=data_dir,
            sqlite_path=sqlite_path,
            lex_base_url=lex_base_url,
            lex_lang=lex_lang,
            lex_calendar_lang_param=lex_calendar_lang_param,
            poll_days=poll_days,
            max_docs_per_run=max_docs_per_run,
            recheck_days=recheck_days,
            recheck_per_run=recheck_per_run,
            http_rate_limit_seconds=http_rate_limit_seconds,
            http_timeout_seconds=http_timeout_seconds,
            min_text_words=min_text_words,
            min_unique_words=min_unique_words,
            min_unique_ratio=min_unique_ratio,
            max_common_line_ratio=max_common_line_ratio,
            max_pdf_url_tries=max_pdf_url_tries,
            enable_ocr=enable_ocr,
            ocr_lang=ocr_lang,
            ocr_max_pages=ocr_max_pages,
            ocr_dpi=ocr_dpi,
            tesseract_cmd=tesseract_cmd,
            azure_di_endpoint=azure_di_endpoint,
            azure_di_key=azure_di_key,
            azure_di_api_version=azure_di_api_version,
            azure_di_model=azure_di_model,
            azure_di_output_format=azure_di_output_format,
            azure_di_ocr_high_res=azure_di_ocr_high_res,
            azure_di_poll_interval_seconds=azure_di_poll_interval_seconds,
            azure_di_poll_timeout_seconds=azure_di_poll_timeout_seconds,
            error_retry_days=error_retry_days,
            chunk_max_words=chunk_max_words,
            chunk_overlap_words=chunk_overlap_words,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            max_chunks_per_doc=max_chunks_per_doc,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            index_backend=index_backend,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            use_openai_rewrite=use_openai_rewrite,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            brightdata_enabled=brightdata_enabled,
            brightdata_host=brightdata_host,
            brightdata_port=brightdata_port,
            brightdata_username=brightdata_username,
            brightdata_password=brightdata_password,
            user_agent=user_agent,
        )


# -----------------------------
# Simple logger
# -----------------------------

def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def warn(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] WARNING: {msg}", file=sys.stderr, flush=True)

def err(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] ERROR: {msg}", file=sys.stderr, flush=True)


# -----------------------------
# Cross-platform single-process lock
# -----------------------------

class RunLock:
    """
    Cross-platform "best effort" lock using an exclusive lock file.
    - Prevents two cron/scheduled-task runs from overlapping.
    - If a lock is stale (older than ttl_seconds), it is removed.
    """
    def __init__(self, lock_path: Path, ttl_seconds: int = 60 * 60) -> None:
        self.lock_path = lock_path
        self.ttl_seconds = ttl_seconds
        self.acquired = False

    def acquire(self) -> bool:
        ensure_dir(self.lock_path.parent)
        if self.lock_path.exists():
            try:
                age = time.time() - self.lock_path.stat().st_mtime
                if age > self.ttl_seconds:
                    warn(f"Stale lock detected (age={int(age)}s). Removing {self.lock_path}.")
                    self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            # Exclusive create
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "started_at": iso(utc_now())}))
            self.acquired = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self.acquired:
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.acquired = False

    def __enter__(self) -> "RunLock":
        if not self.acquire():
            raise RuntimeError("Another run is already in progress (lock file exists).")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# -----------------------------
# SQLite storage
# -----------------------------

SCHEMA_VERSION = 1

class Store:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        ensure_dir(cfg.data_dir)
        ensure_dir(cfg.raw_dir)
        ensure_dir(cfg.text_dir)
        ensure_dir(cfg.index_dir)
        self.conn = sqlite3.connect(str(cfg.sqlite_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        # Speed/concurrency improvements
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            lang TEXT,
            source_date TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_fetched_at TEXT,
            last_changed_at TEXT,
            content_hash TEXT,
            etag TEXT,
            last_modified TEXT,
            status TEXT NOT NULL DEFAULT 'new',
            last_error TEXT,
            next_recheck_at TEXT,
            recheck_interval_days INTEGER DEFAULT 21,
            raw_html_path TEXT,
            text_path TEXT
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_next_recheck ON documents(next_recheck_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_last_seen ON documents(last_seen_at);")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            doc_id TEXT,
            reason TEXT,
            priority INTEGER DEFAULT 0,
            scheduled_at TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_queue_sched ON doc_queue(scheduled_at, priority);")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_active ON chunks(doc_id, active);")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_active ON vectors(active);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_chunk ON vectors(chunk_id);")

        # Schema version
        cur.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))

        # Repair (v3 bug): documents could be left with status='error' even after successful ingests.
        # If a document has active chunks and no last_error, it is almost certainly OK.
        cur.execute(
            """
            UPDATE documents
            SET status='ok'
            WHERE status='error'
              AND (last_error IS NULL OR last_error='')
              AND doc_id IN (SELECT DISTINCT doc_id FROM chunks WHERE active=1)
            """
        )

        self.conn.commit()

    # ---- meta helpers ----
    def meta_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def meta_set(self, key: str, value: str) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.conn.commit()

    # ---- queue helpers ----
    def enqueue_url(self, url: str, doc_id: Optional[str], reason: str, priority: int = 0, scheduled_at: Optional[str] = None) -> None:
        scheduled_at = scheduled_at or iso(utc_now())
        now = iso(utc_now())
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO doc_queue(url, doc_id, reason, priority, scheduled_at, attempts, last_error, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, 0, NULL, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                doc_id=COALESCE(excluded.doc_id, doc_queue.doc_id),
                reason=excluded.reason,
                priority=MAX(doc_queue.priority, excluded.priority),
                scheduled_at=MIN(doc_queue.scheduled_at, excluded.scheduled_at),
                updated_at=excluded.updated_at
            """,
            (url, doc_id, reason, priority, scheduled_at, now, now),
        )
        self.conn.commit()

    def dequeue_batch(self, limit: int) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        now = iso(utc_now())
        rows = cur.execute(
            """
            SELECT * FROM doc_queue
            WHERE scheduled_at <= ?
            ORDER BY priority DESC, scheduled_at ASC
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        return rows

    def queue_mark_attempt(self, queue_id: int, error: Optional[str] = None) -> None:
        cur = self.conn.cursor()
        now = iso(utc_now())
        cur.execute(
            """
            UPDATE doc_queue
            SET attempts = attempts + 1,
                last_error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (error, now, queue_id),
        )
        self.conn.commit()

    def queue_delete(self, queue_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM doc_queue WHERE id=?", (queue_id,))
        self.conn.commit()

    # ---- document helpers ----
    def upsert_document_placeholder(self, doc_id: str, url: str, title: Optional[str], lang: str, source_date: Optional[str]) -> None:
        now = iso(utc_now())
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO documents(doc_id, url, title, lang, source_date, first_seen_at, last_seen_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                url=excluded.url,
                title=COALESCE(excluded.title, documents.title),
                lang=COALESCE(excluded.lang, documents.lang),
                source_date=COALESCE(excluded.source_date, documents.source_date),
                last_seen_at=excluded.last_seen_at
            """,
            (doc_id, url, title, lang, source_date, now, now),
        )
        self.conn.commit()

    def get_document(self, doc_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        return cur.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()

    def list_recheck_due_docs(self, limit: int) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        now = iso(utc_now())
        rows = cur.execute(
            """
            SELECT * FROM documents
            WHERE next_recheck_at IS NOT NULL AND next_recheck_at <= ?
            ORDER BY next_recheck_at ASC
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        return rows


    def update_document_fetch(
        self,
        doc_id: str,
        *,
        title: Optional[str],
        content_hash: Optional[str],
        etag: Optional[str],
        last_modified: Optional[str],
        raw_html_path: Optional[str],
        text_path: Optional[str],
        changed: bool,
        error: Optional[str],
    ) -> None:
        """
        Record the result of a fetch/ingest attempt.

        Important:
        - status is set to 'ok' on success and 'error' on failure
        - last_error is cleared on success
        - next_recheck_at is scheduled sooner on errors (cfg.error_retry_days)
        """
        cur = self.conn.cursor()
        now = iso(utc_now())

        if error:
            status = "error"
            next_recheck = iso(utc_now() + dt.timedelta(days=max(1, int(self.cfg.error_retry_days))))
            last_error = str(error)[:2000]
        else:
            status = "ok"
            next_recheck = iso(utc_now() + dt.timedelta(days=int(self.cfg.recheck_days)))
            last_error = None

        cur.execute(
            """
            UPDATE documents SET
                title = COALESCE(?, title),
                last_fetched_at = ?,
                last_seen_at = ?,
                content_hash = COALESCE(?, content_hash),
                etag = COALESCE(?, etag),
                last_modified = COALESCE(?, last_modified),
                raw_html_path = COALESCE(?, raw_html_path),
                text_path = COALESCE(?, text_path),
                last_changed_at = CASE WHEN ? THEN ? ELSE last_changed_at END,
                status = ?,
                last_error = ?,
                next_recheck_at = ?
            WHERE doc_id = ?
            """,
            (
                title,
                now,
                now,
                content_hash,
                etag,
                last_modified,
                raw_html_path,
                text_path,
                1 if changed else 0,
                now,
                status,
                last_error,
                next_recheck,
                doc_id,
            ),
        )
        self.conn.commit()

    def deactivate_doc_chunks_and_vectors(self, doc_id: str) -> List[int]:
        """
        Mark all chunks and vectors of a document inactive.
        Returns vector_ids that were deactivated (useful to delete from hnswlib index if supported).
        """
        cur = self.conn.cursor()
        # get vector ids first
        vector_rows = cur.execute(
            """
            SELECT v.vector_id FROM vectors v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            WHERE c.doc_id = ? AND v.active = 1
            """,
            (doc_id,),
        ).fetchall()
        vector_ids = [int(r["vector_id"]) for r in vector_rows]

        cur.execute("UPDATE chunks SET active=0 WHERE doc_id=? AND active=1", (doc_id,))
        cur.execute(
            """
            UPDATE vectors SET active=0
            WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)
            """,
            (doc_id,),
        )
        self.conn.commit()
        return vector_ids

    # ---- chunks & vectors ----
    def insert_chunk(self, doc_id: str, chunk_index: int, text: str, text_hash: str) -> int:
        cur = self.conn.cursor()
        now = iso(utc_now())
        cur.execute(
            """
            INSERT INTO chunks(doc_id, chunk_index, text, text_hash, active, created_at)
            VALUES(?, ?, ?, ?, 1, ?)
            """,
            (doc_id, chunk_index, text, text_hash, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_vector(self, chunk_id: int, embedding_f16: np.ndarray) -> int:
        assert embedding_f16.dtype == np.float16
        dim = int(embedding_f16.shape[0])
        blob = embedding_f16.tobytes()
        cur = self.conn.cursor()
        now = iso(utc_now())
        cur.execute(
            """
            INSERT INTO vectors(chunk_id, dim, embedding, active, created_at)
            VALUES(?, ?, ?, 1, ?)
            """,
            (chunk_id, dim, sqlite3.Binary(blob), now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_chunk_by_vector_id(self, vector_id: int) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        row = cur.execute(
            """
            SELECT v.vector_id, v.active AS v_active, c.chunk_id, c.active AS c_active, c.text, c.doc_id, c.chunk_index,
                   d.url, d.title
            FROM vectors v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE v.vector_id = ?
            """,
            (vector_id,),
        ).fetchone()
        return row

    def iter_active_vectors(self) -> Iterable[Tuple[int, int, bytes]]:
        """
        Yields (vector_id, dim, embedding_blob) for active vectors.
        """
        cur = self.conn.cursor()
        for row in cur.execute("SELECT vector_id, dim, embedding FROM vectors WHERE active=1 ORDER BY vector_id ASC"):
            yield int(row["vector_id"]), int(row["dim"]), bytes(row["embedding"])

    def count_active_vectors(self) -> int:
        cur = self.conn.cursor()
        row = cur.execute("SELECT COUNT(*) AS n FROM vectors WHERE active=1").fetchone()
        return int(row["n"]) if row else 0

    def get_vector_dim(self) -> Optional[int]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT dim FROM vectors WHERE active=1 LIMIT 1").fetchone()
        if row:
            return int(row["dim"])
        return None


# -----------------------------
# HTTP client
# -----------------------------

# -----------------------------
# HTTP client (polite + robots-aware)
# -----------------------------

def parse_robots_crawl_delay(robots_txt: str) -> Optional[float]:
    """Parse Crawl-delay from robots.txt (User-agent: *)."""
    current_is_star = False
    for raw in robots_txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip()
            current_is_star = (ua == "*")
            continue
        if current_is_star and low.startswith("crawl-delay:"):
            val = line.split(":", 1)[1].strip()
            try:
                return float(val)
            except Exception:
                return None
    return None

def fetch_robots_crawl_delay(session: requests.Session, base_url: str, timeout_seconds: int) -> Optional[float]:
    """Fetch robots.txt and return Crawl-delay for User-agent: * if present."""
    try:
        robots_url = base_url.rstrip("/") + "/robots.txt"
        resp = session.get(robots_url, timeout=timeout_seconds)
        if resp.status_code != 200:
            return None
        txt = resp.text or ""
        return parse_robots_crawl_delay(txt)
    except Exception:
        return None


class HttpClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sess = requests.Session()

        # Be explicit about what we accept; LexUZ sometimes serves PDFs.
        self.sess.headers.update(
            {
                "User-Agent": cfg.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
                "Accept-Language": "uz,en;q=0.8,ru;q=0.7",
                "Connection": "keep-alive",
            }
        )

        # Configure Bright Data proxy if enabled
        self.proxy_enabled = cfg.brightdata_enabled
        self.proxies: Optional[Dict[str, str]] = None
        if self.proxy_enabled and cfg.brightdata_username and cfg.brightdata_password:
            proxy_url = (
                f"http://{cfg.brightdata_username}:{cfg.brightdata_password}"
                f"@{cfg.brightdata_host}:{cfg.brightdata_port}"
            )
            self.proxies = {"http": proxy_url, "https": proxy_url}
            log(f"[HTTP] Bright Data proxy enabled: {cfg.brightdata_host}:{cfg.brightdata_port}")

        self.rate_limit_seconds = float(cfg.http_rate_limit_seconds)

        # Respect robots.txt crawl-delay only if NOT using proxy.
        # With proxy, we can crawl faster since requests come from different IPs.
        if not self.proxy_enabled:
            respect_robots = os.getenv("RESPECT_ROBOTS", "1").strip().lower() not in ("0", "false", "no", "")
            if respect_robots:
                cd = fetch_robots_crawl_delay(self.sess, cfg.lex_base_url, cfg.http_timeout_seconds)
                if cd is not None and cd > self.rate_limit_seconds:
                    log(
                        f"[HTTP] robots.txt Crawl-delay={cd:.0f}s; overriding HTTP_RATE_LIMIT_SECONDS "
                        f"{self.rate_limit_seconds:.1f}s -> {cd:.0f}s"
                    )
                    self.rate_limit_seconds = float(cd)
        else:
            log(f"[HTTP] Proxy enabled, ignoring robots.txt crawl-delay. Using {self.rate_limit_seconds:.2f}s delay.")

    def get(self, url: str, *, etag: Optional[str] = None, last_modified: Optional[str] = None) -> requests.Response:
        headers: Dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        # fixed delay (+ tiny jitter) between requests
        sleep_rate_limited(self.rate_limit_seconds + random.uniform(0.0, 0.25))

        # very small retry loop (cheap + polite)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.sess.get(
                    url,
                    headers=headers,
                    timeout=self.cfg.http_timeout_seconds,
                    proxies=self.proxies,
                )
                return resp
            except Exception as e:
                last_exc = e
                backoff = 1.0 + attempt * 1.5
                warn(f"HTTP error for {url}: {e}. retrying in {backoff:.1f}s")
                time.sleep(backoff)
        raise RuntimeError(f"Failed to GET {url}: {last_exc}")


# -----------------------------
# LexUZ discovery and parsing
# -----------------------------

DOC_ID_RE = re.compile(r"/docs/(-?\d+)")
DOC_LINK_RE = re.compile(r"^/(?:uz|ru|en)?/docs/(-?\d+)$|^/docs/(-?\d+)$")

def extract_doc_id_from_url(url: str) -> Optional[str]:
    m = DOC_ID_RE.search(url)
    if m:
        return m.group(1)
    return None

def canonicalize_lex_url(base: str, href: str) -> str:
    """Make href absolute and keep host consistent with the configured base (lex.uz vs www.lex.uz)."""
    if href.startswith("http://") or href.startswith("https://"):
        u = href
    else:
        u = urljoin(base + "/", href)

    try:
        b = urlparse(base)
        p = urlparse(u)
        # If both are LexUZ hosts, force the candidate URL onto the base host to avoid apex-domain 404s.
        if (p.netloc in ("lex.uz", "www.lex.uz")) and (b.netloc in ("lex.uz", "www.lex.uz")):
            if p.netloc != b.netloc and b.netloc:
                u = p._replace(netloc=b.netloc).geturl()
    except Exception:
        pass
    return u


def candidate_doc_urls(cfg: Config, url: str) -> List[str]:
    """Generate a small set of equivalent LexUZ doc URLs to work around 404s / domain quirks."""
    if not url:
        return []
    # Ensure absolute + normalized host
    url0 = canonicalize_lex_url(cfg.lex_base_url, url)
    p0 = urlparse(url0)
    base = p0._replace(fragment="").geturl()
    candidates: List[str] = [base]

    # Toggle www/non-www
    try:
        alt = p0._replace(netloc=_toggle_www(p0.netloc)).geturl()
        if alt not in candidates and urlparse(alt).netloc:
            candidates.append(alt)
    except Exception:
        pass

    # Try with/without language prefix
    try:
        path = p0.path or ""
        if path.startswith(f"/{cfg.lex_lang}/docs/"):
            no_lang = path[len(f"/{cfg.lex_lang}") :]
            u = p0._replace(path=no_lang).geturl()
            if u not in candidates:
                candidates.append(u)
        elif path.startswith("/docs/"):
            with_lang = f"/{cfg.lex_lang}" + path
            u = p0._replace(path=with_lang).geturl()
            if u not in candidates:
                candidates.append(u)
    except Exception:
        pass

    # Toggle doc-id sign variant (LexUZ sometimes uses negative IDs: /docs/-1234567)
    try:
        extra: List[str] = []
        for cu in list(candidates):
            pu = urlparse(cu)
            m = re.search(r"/docs/(-?\d+)", pu.path or "")
            if not m:
                continue
            did = m.group(1)
            alt_did = did[1:] if did.startswith("-") else "-" + did
            alt_path = (pu.path or "").replace("/docs/" + did, "/docs/" + alt_did, 1)
            alt_url = pu._replace(path=alt_path).geturl()
            if alt_url not in candidates and alt_url not in extra:
                extra.append(alt_url)
        candidates.extend(extra)
    except Exception:
        pass

    # Dedup
    out: List[str] = []
    seen: set = set()
    for u in candidates:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

def calendar_url_for_date(cfg: Config, date: dt.date) -> str:
    # LexUZ expects dd.mm.yyyy
    d = f"{date.day:02d}.{date.month:02d}.{date.year:04d}"
    # Example observed: https://lex.uz/uz/search/calendar?date=20.08.2025&lang=4
    return f"{cfg.lex_base_url}/{cfg.lex_lang}/search/calendar?date={d}&lang={cfg.lex_calendar_lang_param}"

def parse_calendar_for_docs(cfg: Config, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml" if "lxml" in sys.modules else "html.parser")
    urls: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        # Filter doc links
        if "/docs/" not in href:
            continue
        if DOC_LINK_RE.match(href):
            urls.append(canonicalize_lex_url(cfg.lex_base_url, href))
        else:
            # Some links might include query strings, anchors etc.
            # Try extracting /docs/<id> anywhere in href.
            doc_id = extract_doc_id_from_url(href)
            if doc_id is not None:
                # keep the original href, but canonicalize if relative
                urls.append(canonicalize_lex_url(cfg.lex_base_url, href))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        # normalize: remove fragment
        parsed = urlparse(u)
        norm = parsed._replace(fragment="").geturl()
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out

# Text cleaning patterns (heuristics)
DROP_LINE_PATTERNS = [
    r"^\s*Markaz haqida\s*$",
    r"^\s*Sayt yangiliklari\s*$",
    r"^\s*Saytda reklama\s*$",
    r"^\s*RSS\s*$",
    r"^\s*Yopish\s*$",
    r"^\s*Close\s*$",
    r"^\s*ONLINE TRANSLATE\s*$",
    r"^\s*Hujjatga taklif yuborish\s*$",
    r"^\s*Audioni tinglash\s*$",
    r"^\s*Hujjat elementidan havola olish\s*$",
    r"^\s*Telegram\s*$",
    r"^\s*Facebook\s*$",
    r"^\s*Twitter\s*$",
    r"^\s*Instagram\s*$",
]
DROP_LINE_RE = re.compile("|".join(DROP_LINE_PATTERNS), re.IGNORECASE)

def extract_best_text_container(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Pick the DOM subtree with the most plain text.
    Works reasonably for many news/legal sites without site-specific selectors.
    """
    # Prefer semantic containers
    for tag in ["main", "article"]:
        el = soup.find(tag)
        if el and len(el.get_text(strip=True)) > 400:
            return el

    # Otherwise choose the div/section with maximum text length
    candidates = soup.find_all(["div", "section"], limit=2000)
    best = soup.body or soup
    best_len = 0
    for c in candidates:
        # skip small
        text = c.get_text(" ", strip=True)
        if len(text) < 800:
            continue
        # skip nav-like areas
        classes = " ".join(c.get("class", [])).lower()
        cid = (c.get("id") or "").lower()
        if any(x in classes for x in ["nav", "menu", "footer", "header", "sidebar"]) or any(x in cid for x in ["nav", "menu", "footer", "header", "sidebar"]):
            continue
        if len(text) > best_len:
            best_len = len(text)
            best = c
    return best

def clean_lex_html_to_text(html: str) -> Tuple[Optional[str], str]:
    soup = BeautifulSoup(html, "lxml" if "lxml" in sys.modules else "html.parser")

    # Title
    title = None
    if soup.title and soup.title.string:
        title = normalize_ws(soup.title.string)

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove typical navigation/header/footer blocks when possible
    for sel in ["header", "footer", "nav", "aside"]:
        for t in soup.find_all(sel):
            t.decompose()

    container = extract_best_text_container(soup)
    raw_text = container.get_text(separator="\n")

    # Line-level cleaning
    lines = [normalize_ws(l) for l in raw_text.splitlines()]
    cleaned: List[str] = []
    prev = ""
    for line in lines:
        if not line:
            continue
        # Drop known UI lines
        if DROP_LINE_RE.match(line):
            continue
        # Drop "Image" artifacts commonly appearing in extracted text
        if line.lower() == "image":
            continue
        if "play.google.com" in line or "apps.apple.com" in line or "www.uz" in line:
            continue
        # Drop very short menu items
        if len(line) <= 2:
            continue
        # Drop duplicate adjacent lines
        if line == prev:
            continue
        prev = line
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = normalize_ws(text)
    return title, text


def looks_like_pdf_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "application/pdf" in ct:
        return True
    try:
        head = resp.content[:5]
        return head.startswith(b"%PDF")
    except Exception:
        return False


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF (best-effort).

    Tries engines in this order (whichever is available):
      1) PyMuPDF (fitz)        -> often best for complex PDFs
      2) pdfminer.six          -> sometimes recovers text pypdf misses
      3) pypdf                 -> lightweight fallback

    Notes:
    - Many LexUZ PDFs are image-only scans. In that case, *all* text extractors will return very little.
      For an ultra-cheap MVP, we do NOT OCR by default (OCR is slower + requires language packs).
      Instead, we store the PDF and mark the document for recheck (LexUZ often publishes an HTML text
      version later).
    """
    if not pdf_bytes:
        return ""

    text_parts: List[str] = []

    # 1) PyMuPDF
    if _PYMUPDF is not None:
        try:
            doc = _PYMUPDF.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                try:
                    t = page.get_text("text") or ""
                except Exception:
                    t = ""
                if t:
                    text_parts.append(t)
            if text_parts:
                return postprocess_pdf_text("\n".join(text_parts))
        except Exception:
            pass

    # 2) pdfminer.six
    if _PDFMINER_EXTRACT_TEXT is not None:
        try:
            t = _PDFMINER_EXTRACT_TEXT(BytesIO(pdf_bytes)) or ""
            if t.strip():
                return postprocess_pdf_text(t)
        except Exception:
            pass

    # 3) pypdf
    if _PYPDF is not None:
        try:
            reader = _PYPDF(BytesIO(pdf_bytes))
            for page in getattr(reader, "pages", []):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t:
                    text_parts.append(t)
            if text_parts:
                return postprocess_pdf_text("\n".join(text_parts))
        except Exception:
            pass

    raise RuntimeError(
        "This LexUZ document is served as a PDF, but no PDF text extractor is available.\n"
        "Install at least one of:\n"
        "  pip install pypdf\n"
        "  pip install pdfminer.six\n"
        "  pip install pymupdf\n"
    )



def postprocess_ocr_text(text: str) -> str:
    """
    Cleanup for OCR output:
    - remove form-feed page breaks
    - remove repeated headers/footers (best-effort)
    - collapse duplicate lines
    """
    if not text:
        return ""
    t = text.replace("\x0c", "\n")
    raw_lines = [normalize_ws(l) for l in t.splitlines()]
    lines = [l for l in raw_lines if l]
    if not lines:
        return ""

    counts = Counter(lines)
    kept_once: set = set()
    cleaned: List[str] = []
    for l in lines:
        low = l.lower()

        # Drop pure page numbers
        if re.fullmatch(r"\d{1,4}", l):
            continue

        # If a short line repeats many times, it's usually a header/footer.
        if counts[l] >= 6 and len(l) <= 220:
            if "qmmb" in low or "lex.uz" in low:
                if l in kept_once:
                    continue
                kept_once.add(l)

        cleaned.append(l)

    # Collapse adjacent duplicates
    out: List[str] = []
    prev = ""
    for l in cleaned:
        if l == prev:
            continue
        prev = l
        out.append(l)

    return normalize_ws("\n".join(out))


def ocr_pdf_bytes(cfg: "Config", pdf_bytes: bytes) -> str:
    """
    OCR a PDF using Tesseract (via pytesseract) + PyMuPDF rendering.

    This is the cheapest option for scanned PDFs:
    - costs $0 in API fees
    - uses your CPU
    - slower than text extraction

    Requirements:
      pip install pymupdf pytesseract pillow
      and install the Tesseract binary on your system.
    """
    if not pdf_bytes:
        return ""

    if _PYMUPDF is None:
        raise RuntimeError("OCR requires PyMuPDF. Install with: pip install pymupdf")
    if _PYTESSERACT is None or _PIL_IMAGE is None:
        raise RuntimeError("OCR requires pytesseract + pillow. Install with: pip install pytesseract pillow")

    # Configure tesseract binary path if provided
    if getattr(cfg, "tesseract_cmd", ""):
        try:
            _PYTESSERACT.pytesseract.tesseract_cmd = cfg.tesseract_cmd  # type: ignore
        except Exception:
            pass

    try:
        doc = _PYMUPDF.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"PyMuPDF could not open PDF: {e}")

    n_pages = int(getattr(doc, "page_count", 0) or len(doc))
    max_pages = int(cfg.ocr_max_pages) if int(cfg.ocr_max_pages) > 0 else n_pages
    max_pages = min(n_pages, max_pages)

    # Render at ~OCR_DPI (default 200) for a balance of quality/speed
    zoom = float(cfg.ocr_dpi) / 72.0
    try:
        mat = _PYMUPDF.Matrix(zoom, zoom)
        gray = getattr(_PYMUPDF, "csGRAY", None)
    except Exception:
        mat = None
        gray = None

    parts: List[str] = []
    for i in range(max_pages):
        try:
            page = doc.load_page(i)
            if mat is not None:
                if gray is not None:
                    pix = page.get_pixmap(matrix=mat, colorspace=gray, alpha=False)
                else:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
            else:
                pix = page.get_pixmap(alpha=False)

            img_bytes = pix.tobytes("png")
            img = _PIL_IMAGE.open(BytesIO(img_bytes))  # type: ignore
            try:
                img = img.convert("RGB")
            except Exception:
                pass

            # PSM 6: assume a uniform block of text (often OK for legal docs)
            txt = _PYTESSERACT.image_to_string(img, lang=(cfg.ocr_lang or "eng"), config="--psm 6")  # type: ignore
            if txt and txt.strip():
                parts.append(txt)
        except Exception:
            # Skip problematic pages rather than failing the whole doc
            continue

    return postprocess_ocr_text("\n\n".join(parts))


# -----------------------------
# Azure AI Document Intelligence OCR (for scanned PDFs / no text layer)
# -----------------------------

class AzureDocumentIntelligence:
    """
    Minimal REST client for Azure AI Document Intelligence (v4.0) to OCR scanned PDFs.

    Used only as a fallback when:
      - the document is served as a PDF, and
      - local PDF text extraction yields low-quality / near-empty text, and
      - ENABLE_OCR=1

    Configuration (in .env):
      AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
      AZURE_DI_KEY=...
      AZURE_DI_API_VERSION=2024-11-30
      AZURE_DI_MODEL=prebuilt-read
      AZURE_DI_OUTPUT_FORMAT=text   # or markdown
      AZURE_DI_OCR_HIGH_RES=1
      AZURE_DI_POLL_TIMEOUT_SECONDS=180
    """

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.endpoint = (cfg.azure_di_endpoint or "").strip().rstrip("/")
        self.key = (cfg.azure_di_key or "").strip()
        self.api_version = (cfg.azure_di_api_version or "2024-11-30").strip()
        self.model = (cfg.azure_di_model or "prebuilt-read").strip() or "prebuilt-read"
        self.output_format = (cfg.azure_di_output_format or "text").strip().lower()
        if self.output_format not in ("text", "markdown"):
            self.output_format = "text"
        self.high_res = bool(cfg.azure_di_ocr_high_res)
        self.poll_interval_seconds = float(getattr(cfg, "azure_di_poll_interval_seconds", 2.0) or 2.0)
        self.poll_timeout_seconds = int(getattr(cfg, "azure_di_poll_timeout_seconds", 180) or 180)
        self._sess = requests.Session()

    def enabled(self) -> bool:
        return bool(self.endpoint and self.key)

    def _headers(self) -> Dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }

    def ocr_pdf_bytes(self, pdf_bytes: bytes) -> str:
        if not self.enabled():
            raise RuntimeError("Azure Document Intelligence is not configured (AZURE_DI_ENDPOINT/AZURE_DI_KEY).")
        if not pdf_bytes:
            return ""

        # Analyze using the prebuilt Read model (or whatever AZURE_DI_MODEL is set to).
        url = f"{self.endpoint}/documentintelligence/documentModels/{self.model}:analyze"
        params: Dict[str, Any] = {
            "api-version": self.api_version,
            "_overload": "analyzeDocument",
            "outputContentFormat": self.output_format,
        }
        if self.high_res:
            # Helps OCR small fonts / low-res scans
            params["features"] = "ocrHighResolution"

        body = {"base64Source": base64.b64encode(pdf_bytes).decode("ascii")}

        try:
            r = self._sess.post(url, headers=self._headers(), params=params, json=body, timeout=60)
        except Exception as e:
            raise RuntimeError(f"Azure DI analyze request failed: {e}")

        # Expect 202 Accepted + Operation-Location
        if r.status_code not in (202, 201):
            raise RuntimeError(f"Azure DI analyze request failed: HTTP {r.status_code} {r.text[:600]}")

        op_url = r.headers.get("Operation-Location") or r.headers.get("operation-location")
        if not op_url:
            raise RuntimeError("Azure DI did not return Operation-Location header.")

        deadline = time.time() + max(10, self.poll_timeout_seconds)
        last_status: Optional[str] = None

        while time.time() < deadline:
            try:
                rr = self._sess.get(op_url, headers={"Ocp-Apim-Subscription-Key": self.key}, timeout=60)
            except Exception as e:
                raise RuntimeError(f"Azure DI poll failed: {e}")

            if rr.status_code >= 400:
                raise RuntimeError(f"Azure DI poll failed: HTTP {rr.status_code} {rr.text[:600]}")

            try:
                data = rr.json()
            except Exception:
                raise RuntimeError(f"Azure DI poll returned non-JSON: {rr.text[:600]}")

            status = str(data.get("status") or "").lower()
            if status:
                last_status = status

            if status == "succeeded":
                ar = data.get("analyzeResult") or {}
                content = ar.get("content") or ""
                # Strip selection mark tokens that sometimes appear in content
                content = content.replace(":unselected:", "").replace(":selected:", "")
                return postprocess_ocr_text(content)

            if status == "failed":
                eobj = data.get("error") or {}
                raise RuntimeError(f"Azure DI analyze failed: {json.dumps(eobj, ensure_ascii=False)[:1200]}")

            # running/notStarted
            ra = rr.headers.get("Retry-After")
            sleep_s: float
            if ra:
                try:
                    sleep_s = float(ra)
                except Exception:
                    sleep_s = max(1.0, float(self.poll_interval_seconds))
            else:
                sleep_s = max(1.0, float(self.poll_interval_seconds))
            time.sleep(sleep_s)

        raise RuntimeError(
            f"Azure DI OCR timed out after {self.poll_timeout_seconds}s (last_status={last_status or 'unknown'})."
        )


_QMMB_LINE_RE = re.compile(r"^(?:QMMB|ҚҲММБ)\s*:", re.IGNORECASE)

def postprocess_pdf_text(text: str) -> str:
    """
    Clean up common PDF extraction artifacts:
    - repeated headers/footers (e.g., QMMB lines repeated on every page)
    - page numbers
    - duplicate lines
    """
    if not text:
        return ""

    # Normalize line whitespace (keep line breaks for now)
    raw_lines = [normalize_ws(l) for l in text.splitlines()]
    lines = [l for l in raw_lines if l]

    if not lines:
        return ""

    counts = Counter(lines)
    kept_once: set = set()
    cleaned: List[str] = []
    for l in lines:
        low = l.lower()

        # Drop pure page numbers / "page x" artifacts
        if re.fullmatch(r"\d{1,4}", l):
            continue
        if re.fullmatch(r"(page|бет)\s*\d{1,4}(\s*/\s*\d{1,4})?", low):
            continue

        # If a short line repeats many times, it's usually a header/footer.
        if counts[l] >= 6 and len(l) <= 200:
            if _QMMB_LINE_RE.match(l) or "lex.uz" in low:
                if l in kept_once:
                    continue
                kept_once.add(l)

        cleaned.append(l)

    # Collapse adjacent duplicates
    out: List[str] = []
    prev = ""
    for l in cleaned:
        if l == prev:
            continue
        prev = l
        out.append(l)

    return normalize_ws("\n".join(out))


# -----------------------------
# Text quality heuristics
# -----------------------------

@dataclass
class TextQuality:
    n_chars: int
    n_words: int
    unique_words: int
    unique_ratio: float
    n_lines: int
    common_line_ratio: float
    looks_like_ui: bool
    score: float

_UI_MARKERS = [
    "universal qidiruv",
    "kengaytirilgan qidiruv",
    "izlashni tozalash",
    "online translate",
    "about system",
    "publications",
    "directories",
    "legal explanations",
    "main page",
    "useful resourses",
    "rss",
]
# A few doc-ish hints that usually appear in real act bodies (Uzbek/Russian/English)
_DOC_HINT_PATTERNS = [
    r"\bQMMB\b", r"ҚҲММБ", r"\bmodda\b", r"\barticle\b", r"\bband\b",
    r"O['’`]?zbekiston Respublikasi", r"Ўзбекистон Республикаси",
    r"\bqaror\b", r"\bfarmon\b", r"\bqonun\b", r"\bdecree\b", r"\bresolution\b",
]

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёЎўҚқҒғҲҳʻʼ’]+", re.UNICODE)

def analyze_text_quality(text: str) -> TextQuality:
    t = (text or "").strip()
    n_chars = len(t)

    words = [w.lower() for w in _WORD_RE.findall(t)]
    n_words = len(words)
    unique_words = len(set(words))
    unique_ratio = (unique_words / n_words) if n_words else 0.0

    lines = [normalize_ws(l) for l in (text or "").splitlines()]
    lines = [l for l in lines if l]
    n_lines = len(lines)

    common_line_ratio = 0.0
    if n_lines >= 5:
        c = Counter(lines)
        common_line_ratio = (max(c.values()) / n_lines) if c else 0.0

    low = t.lower()
    ui_hits = sum(1 for m in _UI_MARKERS if m in low)
    looks_like_ui = ui_hits >= 2

    # If it has strong doc hints, don't treat it as UI
    if any(re.search(p, t, flags=re.IGNORECASE) for p in _DOC_HINT_PATTERNS):
        looks_like_ui = False

    # Quality score: prioritize unique words; penalize UI-y pages + heavy repetition
    score = float(unique_words) + 0.05 * float(min(n_words, 5000))
    if looks_like_ui:
        score -= 1000.0
    score -= 200.0 * float(common_line_ratio)

    return TextQuality(
        n_chars=n_chars,
        n_words=n_words,
        unique_words=unique_words,
        unique_ratio=unique_ratio,
        n_lines=n_lines,
        common_line_ratio=common_line_ratio,
        looks_like_ui=looks_like_ui,
        score=score,
    )

def is_text_acceptable(cfg: Config, q: TextQuality) -> bool:
    if q.looks_like_ui:
        return False
    if q.n_words < int(cfg.min_text_words):
        return False
    if q.unique_words < int(cfg.min_unique_words):
        return False
    if q.unique_ratio < float(cfg.min_unique_ratio):
        return False
    if q.n_lines >= 20 and q.common_line_ratio > float(cfg.max_common_line_ratio):
        return False
    return True

def with_query_params(url: str, params: Dict[str, str]) -> str:
    """Add/replace query parameters on a URL (safe for existing query strings)."""
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    for k, v in params.items():
        q[str(k)] = str(v)
    return p._replace(query=urlencode(q, doseq=True)).geturl()



def _url_root(u: str) -> str:
    p = urlparse(u)
    scheme = p.scheme or "https"
    return f"{scheme}://{p.netloc}"


def _toggle_www(netloc: str) -> str:
    nl = (netloc or "").strip()
    if not nl:
        return nl
    if nl.startswith("www."):
        return nl[len("www.") :]
    return "www." + nl


def guess_pdf_urls(cfg: Config, doc_url: str, html: str) -> List[str]:
    """
    Guess where the PDF for a doc might live.

    LexUZ sometimes serves documents as:
      - normal HTML text (best case)
      - an embedded PDF (often via /pdffile/<id> or /<lang>/pdfs/<id>)
      - a temporarily PDF-only page while HTML is being prepared

    We try to find PDF URLs by:
      1) parsing HTML for href/src/data attributes that look PDF-ish
      2) regex scanning raw HTML for /pdffile/<n> and /pdfs/<n>
      3) deriving from the numeric doc id as a fallback
    """
    roots: List[str] = []
    for candidate in [cfg.lex_base_url, doc_url]:
        try:
            r = _url_root(candidate)
            if r and r not in roots:
                roots.append(r)
            # add www/non-www variant
            p = urlparse(r)
            alt = p._replace(netloc=_toggle_www(p.netloc)).geturl()
            if alt and alt not in roots and urlparse(alt).netloc:
                roots.append(alt)
        except Exception:
            continue

    urls: List[str] = []

    def add(u: str) -> None:
        u = (u or "").strip()
        if not u:
            return
        urls.append(u)

    # 1) Parse HTML for explicit links/embeds
    try:
        soup = BeautifulSoup(html, "lxml" if "lxml" in sys.modules else "html.parser")
        attrs = []
        for tag, attr in [("a", "href"), ("iframe", "src"), ("embed", "src"), ("object", "data"), ("source", "src")]:
            for el in soup.find_all(tag):
                v = (el.get(attr) or "").strip()
                if v:
                    attrs.append(v)

        for v in attrs:
            low = v.lower()
            if "/pdffile/" in low or "/pdfs/" in low or low.endswith(".pdf"):
                # canonicalize relative URLs against a LexUZ root
                for root in roots:
                    add(canonicalize_lex_url(root, v))
                    break
    except Exception:
        pass

    # 2) Regex scan for common LexUZ PDF paths
    try:
        for m in re.finditer(r"(\/(?:[a-z]{2}\/)?(?:pdffile|pdfs)\/\d+)", html, flags=re.IGNORECASE):
            path = m.group(1)
            for root in roots:
                add(canonicalize_lex_url(root, path))
                break
    except Exception:
        pass

    # 3) Derive from doc id
    doc_id_str = extract_doc_id_from_url(doc_url)
    if doc_id_str:
        try:
            n = int(doc_id_str)
            dids = [str(abs(n)), str(n)]
            dids = [d for i, d in enumerate(dids) if d not in dids[:i]]

            for root in roots:
                for did in dids:
                    add(f"{root}/pdffile/{did}")
                    add(f"{root}/{cfg.lex_lang}/pdffile/{did}")
                    add(f"{root}/pdfs/{did}")
                    add(f"{root}/{cfg.lex_lang}/pdfs/{did}")
        except Exception:
            pass

    # Dedup while preserving order
    seen: set = set()
    out: List[str] = []
    for u in urls:
        try:
            # strip fragments
            pu = urlparse(u)
            u = pu._replace(fragment="").geturl()
        except Exception:
            pass
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# -----------------------------
# Chunking
# -----------------------------

def chunk_text_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    """
    Simple sliding window chunking by words.
    Works well for legal text and avoids complicated tokenizers.
    """
    text = normalize_ws(text)
    words = re.split(r"\s+", text)
    words = [w for w in words if w]
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    n = len(words)
    if overlap_words >= max_words:
        overlap_words = max(0, max_words // 4)

    while start < n:
        end = min(start + max_words, n)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_words)
    return chunks


# -----------------------------
# Embeddings
# -----------------------------

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

class Embedder:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.provider = cfg.embedding_provider.lower()
        self.model_name = cfg.embedding_model
        self._model = None

    def _ensure_local_model(self) -> None:
        if self._model is not None:
            return
        if _SENTENCE_TRANSFORMERS is None:
            raise RuntimeError(
                "Local embeddings requested but sentence-transformers is not installed.\n"
                "Install with: pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cpu"
            )
        log(f"[EMBED] Loading local embedding model: {self.model_name} (CPU)")
        self._model = _SENTENCE_TRANSFORMERS(self.model_name, device="cpu")

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self.provider == "local":
            self._ensure_local_model()
            # E5-style prefix improves retrieval
            prefixed = [f"passage: {t}" for t in texts]
            vecs = self._model.encode(prefixed, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype=np.float32)
            return vecs

        if self.provider == "openai":
            # Optional: implement OpenAI embedding endpoint (paid).
            # For ultra-cheap MVP, prefer local.
            raise RuntimeError("OpenAI embeddings provider not implemented in this MVP. Use local embeddings.")
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {self.provider}")

    def embed_query(self, text: str) -> np.ndarray:
        if self.provider == "local":
            self._ensure_local_model()
            q = f"query: {text}"
            vec = self._model.encode([q], show_progress_bar=False, normalize_embeddings=True)
            vec = np.asarray(vec, dtype=np.float32)[0]
            return vec
        if self.provider == "openai":
            raise RuntimeError("OpenAI embeddings provider not implemented in this MVP. Use local embeddings.")
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {self.provider}")


# -----------------------------
# Vector index (FAISS / hnswlib / numpy fallback)
# -----------------------------

class VectorIndex:
    """
    Persistent vector search over chunk embeddings.

    Backends:
      - faiss (if installed)    : fast ANN / exact depending on index type
      - hnswlib (if installed)  : fast ANN with native ID support
      - numpy fallback          : brute force dot product (slow but zero extra deps)

    Embeddings are cosine-normalized, so inner product == cosine similarity.
    """

    def __init__(self, cfg: Config, store: Store) -> None:
        self.cfg = cfg
        self.store = store
        self.backend = self._select_backend(cfg.index_backend)
        self.dim: Optional[int] = None

        self._faiss_index = None
        self._hnsw_index = None
        self._numpy_ids: Optional[np.ndarray] = None
        self._numpy_mat: Optional[np.ndarray] = None

        ensure_dir(cfg.index_dir)

        self._load_or_build()

    def _select_backend(self, desired: str) -> str:
        desired = desired.lower()
        if desired in ("faiss", "hnswlib", "numpy"):
            return desired
        # auto
        if _FAISS is not None:
            return "faiss"
        if _HNSWLIB is not None:
            return "hnswlib"
        return "numpy"

    def _read_embedding_blob(self, blob: bytes, dim: int) -> np.ndarray:
        vec = np.frombuffer(blob, dtype=np.float16, count=dim).astype(np.float32)
        # They were normalized before storing; keep as-is
        return vec

    def _load_or_build(self) -> None:
        dim = self.store.get_vector_dim()
        if dim is None:
            # No vectors yet: we can't build index until first embeddings exist.
            self.dim = None
            self._init_empty_index(dim_guess=None)
            return

        self.dim = dim

        # If meta says backend differs, rebuild
        meta_backend = self.store.meta_get("index_backend", None)
        meta_dim = self.store.meta_get("index_dim", None)
        if meta_backend and meta_backend != self.backend:
            warn(f"Index backend changed from {meta_backend} to {self.backend}. Will rebuild.")
            self.rebuild()

        if meta_dim and safe_int(meta_dim, dim) != dim:
            warn(f"Index dim changed from {meta_dim} to {dim}. Will rebuild.")
            self.rebuild()

        # Try load from disk
        if self.backend == "faiss" and self.cfg.faiss_index_path.exists() and _FAISS is not None:
            try:
                self._faiss_index = _FAISS.read_index(str(self.cfg.faiss_index_path))
                self.store.meta_set("index_backend", self.backend)
                self.store.meta_set("index_dim", str(dim))
                return
            except Exception as e:
                warn(f"Failed to load FAISS index; rebuilding. Error: {e}")
                self.rebuild()
                return

        if self.backend == "hnswlib" and self.cfg.hnsw_index_path.exists() and _HNSWLIB is not None:
            try:
                idx = _HNSWLIB.Index(space="cosine", dim=dim)
                idx.load_index(str(self.cfg.hnsw_index_path))
                self._hnsw_index = idx
                self.store.meta_set("index_backend", self.backend)
                self.store.meta_set("index_dim", str(dim))
                return
            except Exception as e:
                warn(f"Failed to load hnswlib index; rebuilding. Error: {e}")
                self.rebuild()
                return

        if self.backend == "numpy" and self.cfg.numpy_cache_path.exists():
            try:
                data = np.load(str(self.cfg.numpy_cache_path))
                self._numpy_ids = data["ids"].astype(np.int64)
                self._numpy_mat = data["mat"].astype(np.float32)
                self.store.meta_set("index_backend", self.backend)
                self.store.meta_set("index_dim", str(dim))
                return
            except Exception as e:
                warn(f"Failed to load numpy cache; rebuilding. Error: {e}")
                self.rebuild()
                return

        # Otherwise build from DB
        self.rebuild()

    def _init_empty_index(self, dim_guess: Optional[int]) -> None:
        # Initialize empty structures (used before any vectors exist)
        if self.backend == "faiss" and _FAISS is not None and dim_guess is not None:
            base = _FAISS.IndexHNSWFlat(dim_guess, 32, _FAISS.METRIC_INNER_PRODUCT)
            self._faiss_index = _FAISS.IndexIDMap2(base)
        elif self.backend == "hnswlib" and _HNSWLIB is not None and dim_guess is not None:
            idx = _HNSWLIB.Index(space="cosine", dim=dim_guess)
            idx.init_index(max_elements=1000, ef_construction=200, M=16)
            idx.set_ef(50)
            self._hnsw_index = idx
        else:
            self._numpy_ids = np.zeros((0,), dtype=np.int64)
            self._numpy_mat = np.zeros((0, dim_guess or 0), dtype=np.float32)

    def rebuild(self) -> None:
        dim = self.store.get_vector_dim()
        if dim is None:
            self.dim = None
            self._init_empty_index(dim_guess=None)
            self._persist()
            return
        self.dim = dim
        log(f"[INDEX] Rebuilding index backend={self.backend} dim={dim} ...")
        ids: List[int] = []
        mat: List[np.ndarray] = []
        for vector_id, vdim, blob in self.store.iter_active_vectors():
            if vdim != dim:
                continue
            ids.append(vector_id)
            mat.append(self._read_embedding_blob(blob, dim))
        if len(ids) == 0:
            self._init_empty_index(dim_guess=dim)
            self._persist()
            self.store.meta_set("index_backend", self.backend)
            self.store.meta_set("index_dim", str(dim))
            return

        X = np.stack(mat, axis=0).astype(np.float32)
        ids_arr = np.asarray(ids, dtype=np.int64)

        if self.backend == "faiss" and _FAISS is not None:
            base = _FAISS.IndexHNSWFlat(dim, 32, _FAISS.METRIC_INNER_PRODUCT)
            index = _FAISS.IndexIDMap2(base)
            index.add_with_ids(X, ids_arr)
            self._faiss_index = index
            self._hnsw_index = None
            self._numpy_ids = None
            self._numpy_mat = None

        elif self.backend == "hnswlib" and _HNSWLIB is not None:
            idx = _HNSWLIB.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=max(1000, int(len(ids_arr) * 1.2)), ef_construction=200, M=16)
            idx.add_items(X, ids_arr)
            idx.set_ef(50)
            self._hnsw_index = idx
            self._faiss_index = None
            self._numpy_ids = None
            self._numpy_mat = None

        else:
            # numpy
            self._numpy_ids = ids_arr
            self._numpy_mat = X
            self._faiss_index = None
            self._hnsw_index = None

        self._persist()
        self.store.meta_set("index_backend", self.backend)
        self.store.meta_set("index_dim", str(dim))
        self.store.meta_set("index_count", str(len(ids_arr)))

    def _persist(self) -> None:
        if self.dim is None:
            return
        if self.backend == "faiss" and self._faiss_index is not None and _FAISS is not None:
            _FAISS.write_index(self._faiss_index, str(self.cfg.faiss_index_path))
        elif self.backend == "hnswlib" and self._hnsw_index is not None:
            self._hnsw_index.save_index(str(self.cfg.hnsw_index_path))
        else:
            if self._numpy_ids is None or self._numpy_mat is None:
                return
            np.savez_compressed(str(self.cfg.numpy_cache_path), ids=self._numpy_ids, mat=self._numpy_mat)
        self.store.meta_set("index_last_saved_at", iso(utc_now()))

    def add_vectors(self, vector_ids: List[int], embeddings: np.ndarray) -> None:
        """
        Add new vectors to the index.
        embeddings: (n, dim) float32 normalized
        """
        if embeddings.size == 0 or not vector_ids:
            return
        if self.dim is None:
            self.dim = embeddings.shape[1]
            self.store.meta_set("index_dim", str(self.dim))
            # initialize now
            self._init_empty_index(dim_guess=self.dim)
            self.rebuild()
            return

        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {embeddings.shape[1]} expected {self.dim}")

        ids_arr = np.asarray(vector_ids, dtype=np.int64)
        X = embeddings.astype(np.float32)

        if self.backend == "faiss" and _FAISS is not None:
            if self._faiss_index is None:
                self.rebuild()
            assert self._faiss_index is not None
            self._faiss_index.add_with_ids(X, ids_arr)
            self._persist()

        elif self.backend == "hnswlib" and _HNSWLIB is not None:
            if self._hnsw_index is None:
                self.rebuild()
            assert self._hnsw_index is not None
            # Ensure capacity
            try:
                current = self._hnsw_index.get_current_count()
                max_el = self._hnsw_index.get_max_elements()
                needed = current + X.shape[0]
                if needed > max_el:
                    self._hnsw_index.resize_index(int(needed * 1.3))
            except Exception:
                pass
            self._hnsw_index.add_items(X, ids_arr)
            self._persist()

        else:
            # numpy: append in memory and persist
            if self._numpy_ids is None or self._numpy_mat is None:
                self.rebuild()
            assert self._numpy_ids is not None and self._numpy_mat is not None
            self._numpy_ids = np.concatenate([self._numpy_ids, ids_arr], axis=0)
            self._numpy_mat = np.vstack([self._numpy_mat, X])
            self._persist()

    def mark_deleted(self, vector_ids: List[int]) -> None:
        """
        Optional: delete vectors from index (only for hnswlib reliably).
        For FAISS, we keep inactive vectors and filter at query time (cheap MVP).
        """
        if not vector_ids:
            return
        if self.backend == "hnswlib" and self._hnsw_index is not None:
            for vid in vector_ids:
                try:
                    self._hnsw_index.mark_deleted(int(vid))
                except Exception:
                    pass
            self._persist()

    def search(self, query_vec: np.ndarray, k: int, candidate_multiplier: int) -> List[Tuple[int, float]]:
        """
        Returns list of (vector_id, score) with cosine similarity scores.
        Filters inactive vectors by checking SQLite.
        """
        if self.dim is None:
            return []
        if query_vec.shape[0] != self.dim:
            raise ValueError(f"Query dim mismatch: got {query_vec.shape[0]} expected {self.dim}")

        want = max(k, 1)
        cand_k = min(max(want * candidate_multiplier, want), 200)

        results: List[Tuple[int, float]] = []

        if self.backend == "faiss" and _FAISS is not None:
            if self._faiss_index is None:
                self.rebuild()
            assert self._faiss_index is not None
            D, I = self._faiss_index.search(query_vec.reshape(1, -1).astype(np.float32), cand_k)
            ids = I[0].tolist()
            scores = D[0].tolist()
            for vid, sc in zip(ids, scores):
                if vid == -1:
                    continue
                row = self.store.get_chunk_by_vector_id(int(vid))
                if not row:
                    continue
                if int(row["v_active"]) != 1 or int(row["c_active"]) != 1:
                    continue
                results.append((int(vid), float(sc)))
                if len(results) >= want:
                    break

        elif self.backend == "hnswlib" and _HNSWLIB is not None:
            if self._hnsw_index is None:
                self.rebuild()
            assert self._hnsw_index is not None
            ids, dists = self._hnsw_index.knn_query(query_vec.astype(np.float32), k=cand_k)
            ids = ids[0].tolist()
            dists = dists[0].tolist()
            for vid, dist in zip(ids, dists):
                # For cosine space in hnswlib, distance is (1 - cosine)
                sc = 1.0 - float(dist)
                row = self.store.get_chunk_by_vector_id(int(vid))
                if not row:
                    continue
                if int(row["v_active"]) != 1 or int(row["c_active"]) != 1:
                    continue
                results.append((int(vid), sc))
                if len(results) >= want:
                    break

        else:
            # numpy brute-force
            if self._numpy_ids is None or self._numpy_mat is None:
                self.rebuild()
            assert self._numpy_ids is not None and self._numpy_mat is not None
            if self._numpy_mat.shape[0] == 0:
                return []
            scores = self._numpy_mat @ query_vec.astype(np.float32)
            # Get top cand_k indices
            if scores.shape[0] <= cand_k:
                top_idx = np.argsort(-scores)
            else:
                top_idx = np.argpartition(-scores, cand_k)[:cand_k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            for i in top_idx:
                vid = int(self._numpy_ids[i])
                sc = float(scores[i])
                row = self.store.get_chunk_by_vector_id(vid)
                if not row:
                    continue
                if int(row["v_active"]) != 1 or int(row["c_active"]) != 1:
                    continue
                results.append((vid, sc))
                if len(results) >= want:
                    break

        return results


# -----------------------------
# Telegram notifications
# -----------------------------

class Telegram:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def enabled(self) -> bool:
        return bool(self.cfg.telegram_bot_token and self.cfg.telegram_chat_id)

    def send(self, text: str, *, disable_preview: bool = True) -> None:
        if not self.enabled():
            return
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.cfg.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code >= 300:
                warn(f"Telegram send failed: HTTP {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            warn(f"Telegram send error: {e}")


# -----------------------------
# OpenAI client (Responses API with fallback to Chat Completions)
# -----------------------------

class OpenAIClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.key = cfg.openai_api_key
        self.base = cfg.openai_base_url.rstrip("/")
        self.model = cfg.openai_model

    def enabled(self) -> bool:
        return bool(self.key)

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        # Optional OpenRouter identification headers (recommended by OpenRouter, but not required)
        site = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app = os.getenv("OPENROUTER_APP_NAME", "").strip()
        if site:
            h["HTTP-Referer"] = site
        if app:
            h["X-Title"] = app
        return h

    def _extract_output_text_from_responses(self, data: Dict[str, Any]) -> str:
        # The Python SDK exposes response.output_text; the raw HTTP response may include "output_text"
        if isinstance(data, dict):
            if "output_text" in data and isinstance(data["output_text"], str):
                return data["output_text"]
            # Otherwise, parse output array
            out = data.get("output")
            if isinstance(out, list):
                parts: List[str] = []
                for item in out:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                t = c.get("text")
                                if isinstance(t, str):
                                    parts.append(t)
                    # Some variants have item["text"]
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts).strip()
        return ""

    def generate_text(self, system: str, user: str, *, temperature: float = 0.2, max_output_tokens: int = 900) -> str:
        if not self.enabled():
            raise RuntimeError("OPENAI_API_KEY not set; cannot use ask/rewrite features.")

        base_lower = (self.base or "").lower()
        # Many OpenAI-compatible providers (e.g., OpenRouter) do not support the newer Responses API.
        # To save time and avoid an extra failing HTTP request, go straight to Chat Completions for OpenRouter.
        if "openrouter.ai" in base_lower:
            url2 = f"{self.base}/v1/chat/completions"
            payload2 = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_output_tokens,
            }
            r2 = requests.post(url2, headers=self._headers(), json=payload2, timeout=60)
            if r2.status_code >= 300:
                raise RuntimeError(f"LLM request failed: HTTP {r2.status_code} {r2.text[:400]}")
            data2 = r2.json()
            try:
                return data2["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(data2)[:2000]

        # Preferred: Responses API
        url = f"{self.base}/v1/responses"
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=60)
            if r.status_code < 300:
                data = r.json()
                text = self._extract_output_text_from_responses(data)
                if text:
                    return text
                # fallback: stringify json
                return json.dumps(data)[:2000]
            else:
                # fallback to chat completions
                warn(f"Responses API failed (HTTP {r.status_code}). Falling back to Chat Completions.")
        except Exception as e:
            warn(f"Responses API error: {e}. Falling back to Chat Completions.")

        # Fallback: Chat Completions
        url2 = f"{self.base}/v1/chat/completions"
        payload2 = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        r2 = requests.post(url2, headers=self._headers(), json=payload2, timeout=60)
        if r2.status_code >= 300:
            raise RuntimeError(f"OpenAI request failed: HTTP {r2.status_code} {r2.text[:400]}")
        data2 = r2.json()
        try:
            return data2["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data2)[:2000]


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from model output.
    Accepts:
      - pure JSON
      - JSON inside markdown fences
      - JSON preceded/followed by extra text
    """
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    # If already valid
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


# -----------------------------
# Ingestion pipeline
# -----------------------------

@dataclass
class IngestResult:
    doc_id: str
    url: str
    status: str  # "new"|"updated"|"unchanged"|"error"|"skipped"
    title: Optional[str] = None
    chunks_added: int = 0
    error: Optional[str] = None

class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.store = Store(cfg)
        self.http = HttpClient(cfg)
        self.embedder = Embedder(cfg)
        self.index = VectorIndex(cfg, self.store)
        self.telegram = Telegram(cfg)
        self.openai = OpenAIClient(cfg)
        self.azure_di = AzureDocumentIntelligence(cfg)

    def close(self) -> None:
        self.store.close()

    # --- discovery ---
    def discover_recent_docs(self, days: int) -> List[Tuple[str, str]]:
        """
        Returns list of (url, source_date_iso) discovered from calendar pages.
        """
        found: List[Tuple[str, str]] = []
        today = dt.date.today()
        for delta in range(days):
            d = today - dt.timedelta(days=delta)
            cal_url = calendar_url_for_date(self.cfg, d)
            try:
                resp = self.http.get(cal_url)
                if resp.status_code >= 300:
                    warn(f"[DISCOVER] Calendar fetch failed: HTTP {resp.status_code} {cal_url}")
                    continue
                urls = parse_calendar_for_docs(self.cfg, resp.text)
                for u in urls:
                    found.append((u, d.isoformat()))
                log(f"[DISCOVER] {d.isoformat()} found {len(urls)} doc link(s)")
            except Exception as e:
                warn(f"[DISCOVER] Error fetching {cal_url}: {e}")
        # Dedup
        seen = set()
        out: List[Tuple[str, str]] = []
        for u, sd in found:
            if u not in seen:
                seen.add(u)
                out.append((u, sd))
        return out

    def schedule_rechecks(self, limit: int) -> int:
        due = self.store.list_recheck_due_docs(limit=limit)
        n = 0
        for row in due:
            self.store.enqueue_url(row["url"], row["doc_id"], reason="recheck", priority=1)
            n += 1
        return n

    # --- ingestion ---

    def ingest_one(self, url: str, source_date: Optional[str] = None) -> IngestResult:
        """
        Fetch + extract + chunk + embed one LexUZ document.

        Key robustness features:
        - tries URL variants (www/non-www, with/without lang prefix)
        - if HTML has only metadata, tries a "print" view (often contains full text)
        - if still missing, tries to locate/download an embedded PDF
        - rejects obviously low-information / repetitive extraction (common with scanned PDFs)
        """
        # Normalize and try a couple of equivalent URLs (www/non-www, with/without /uz/)
        url_candidates = candidate_doc_urls(self.cfg, url)
        url_candidates = url_candidates or [url]

        doc_id = extract_doc_id_from_url(url) or sha256_text(url)[:16]
        existing = self.store.get_document(doc_id)

        # Ensure doc row exists early (so we can record fetch errors even on first-ever run).
        self.store.upsert_document_placeholder(
            doc_id, url_candidates[0], title=None, lang=self.cfg.lex_lang, source_date=source_date
        )

        # If we previously ingested this as a PDF, don't rely on HTML ETag/Last-Modified.
        use_conditional = True
        if existing:
            prev_raw = str(existing["raw_html_path"] or "").lower()
            if prev_raw.endswith(".pdf"):
                use_conditional = False

        etag = existing["etag"] if (existing and use_conditional) else None
        last_modified = existing["last_modified"] if (existing and use_conditional) else None

        resp: Optional[requests.Response] = None
        final_url = url_candidates[0]
        last_err: Optional[str] = None

        # Try candidates until one works
        for i, u in enumerate(url_candidates):
            try:
                # Only send conditional headers on the first canonical URL attempt.
                r = self.http.get(u, etag=etag if i == 0 else None, last_modified=last_modified if i == 0 else None)
            except Exception as e:
                last_err = str(e)
                continue

            # If HTML says not modified: for HTML docs we can return early;
            # for PDF-backed docs we still want to refresh by checking the PDF (HTML 304 isn't enough).
            if r.status_code == 304 and existing and (str(existing["raw_html_path"] or "").lower().endswith(".pdf")):
                try:
                    r = self.http.get(u)  # refetch without conditional headers
                except Exception as e:
                    last_err = str(e)
                    continue

            if r.status_code == 304:
                # Success: unchanged
                self.store.upsert_document_placeholder(doc_id, u, title=None, lang=self.cfg.lex_lang, source_date=source_date)
                self.store.update_document_fetch(
                    doc_id,
                    title=None,
                    content_hash=None,
                    etag=r.headers.get("ETag"),
                    last_modified=r.headers.get("Last-Modified"),
                    raw_html_path=None,
                    text_path=None,
                    changed=False,
                    error=None,
                )
                return IngestResult(doc_id=doc_id, url=u, status="unchanged")

            if r.status_code < 400:
                resp = r
                final_url = u
                break

            last_err = f"HTTP {r.status_code}"

        if resp is None:
            e = last_err or "Failed to fetch"
            self.store.update_document_fetch(
                doc_id,
                title=None,
                content_hash=None,
                etag=None,
                last_modified=None,
                raw_html_path=None,
                text_path=None,
                changed=False,
                error=e,
            )
            return IngestResult(doc_id=doc_id, url=url, status="error", error=e)

        # HTML fetched
        html = resp.text

        # Save raw HTML for debugging even if PDF fallback is used
        raw_html_path = self.cfg.raw_dir / f"{doc_id}.html"
        raw_html_path.write_text(html, encoding="utf-8", errors="ignore")

        best_title, best_text = clean_lex_html_to_text(html)
        best_raw_path = str(raw_html_path)
        best_stats = analyze_text_quality(best_text)
        best_source = "html"

        # Some LexUZ docs (especially very new ones) show only metadata in HTML.
        # Try print view first; it often contains the full text.
        if not is_text_acceptable(self.cfg, best_stats):
            print_url = None
            try:
                # Prefer numeric lact_id if doc_id is numeric; otherwise derive from URL
                if re.fullmatch(r"-?\d+", doc_id or ""):
                    lact = str(abs(int(doc_id)))
                    print_url = with_query_params(final_url, {"lact_id": lact, "version": "print"})
            except Exception:
                print_url = None

            if print_url:
                try:
                    r2 = self.http.get(print_url)
                    if r2.status_code < 400:
                        html2 = r2.text
                        raw_print_path = self.cfg.raw_dir / f"{doc_id}.print.html"
                        raw_print_path.write_text(html2, encoding="utf-8", errors="ignore")
                        t2_title, t2_text = clean_lex_html_to_text(html2)
                        t2_stats = analyze_text_quality(t2_text)
                        if t2_stats.score > best_stats.score:
                            best_title, best_text, best_stats = t2_title, t2_text, t2_stats
                            best_raw_path = str(raw_print_path)
                            best_source = "print"
                except Exception:
                    pass


        # Extra fallback: legacy "getpage.aspx" endpoint (sometimes has HTML when /docs/ is PDF-only)
        if not is_text_acceptable(self.cfg, best_stats):
            getpage_url = None
            try:
                if re.fullmatch(r"-?\d+", doc_id or ""):
                    lact = str(abs(int(doc_id)))
                    root = _url_root(final_url) or _url_root(self.cfg.lex_base_url) or self.cfg.lex_base_url
                    getpage_url = f"{root}/pages/getpage.aspx?lact_id={lact}"
            except Exception:
                getpage_url = None

            if getpage_url:
                try:
                    r3 = self.http.get(getpage_url)
                    if r3.status_code < 400:
                        html3 = r3.text
                        raw_gp_path = self.cfg.raw_dir / f"{doc_id}.getpage.html"
                        raw_gp_path.write_text(html3, encoding="utf-8", errors="ignore")
                        t3_title, t3_text = clean_lex_html_to_text(html3)
                        t3_stats = analyze_text_quality(t3_text)
                        if t3_stats.score > best_stats.score:
                            best_title, best_text, best_stats = t3_title, t3_text, t3_stats
                            best_raw_path = str(raw_gp_path)
                            best_source = "getpage"
                except Exception:
                    pass


        # If still low-information, try PDF fallback (unless skip_pdf is enabled)
        used_pdf = False
        pdf_error: Optional[str] = None
        skip_pdf = getattr(self, 'skip_pdf', False)
        if not is_text_acceptable(self.cfg, best_stats) and not skip_pdf:
            tries = 0
            ocr_tried = False
            for pdf_url in guess_pdf_urls(self.cfg, final_url, html):
                if tries >= int(self.cfg.max_pdf_url_tries):
                    break
                tries += 1
                try:
                    pdf_resp = self.http.get(pdf_url)
                    if pdf_resp.status_code >= 400:
                        pdf_error = f"{pdf_url} -> HTTP {pdf_resp.status_code}"
                        continue
                    if not looks_like_pdf_response(pdf_resp):
                        ct = pdf_resp.headers.get("Content-Type") or ""
                        pdf_error = f"{pdf_url} -> not a PDF (Content-Type={ct})"
                        continue

                    pdf_bytes = pdf_resp.content
                    pdf_path = self.cfg.raw_dir / f"{doc_id}.pdf"
                    pdf_path.write_bytes(pdf_bytes)

                    # Try fast local PDF text extraction first (free).
                    # If no extractor is installed (or extraction fails), we can still recover via OCR.
                    try:
                        pdf_text = extract_text_from_pdf_bytes(pdf_bytes)
                    except Exception as e:
                        pdf_text = ""
                        pdf_stats = analyze_text_quality(pdf_text)
                        pdf_error = f"{pdf_url} -> PDF text extraction failed: {e}"
                    else:
                        pdf_stats = analyze_text_quality(pdf_text)

                    # If text extraction is low-quality and OCR is enabled, try Azure AI Document Intelligence once.
                    if (not is_text_acceptable(self.cfg, pdf_stats)) and bool(self.cfg.enable_ocr) and (not ocr_tried):
                        ocr_tried = True
                        try:
                            if self.azure_di.enabled():
                                log(
                                    f"[AZURE OCR] {doc_id}: Azure AI Document Intelligence "
                                    f"(model={self.cfg.azure_di_model}, api={self.cfg.azure_di_api_version}, "
                                    f"format={self.cfg.azure_di_output_format}, high_res={int(self.cfg.azure_di_ocr_high_res)})"
                                )
                                ocr_text = self.azure_di.ocr_pdf_bytes(pdf_bytes)
                                source_tag = "pdf+azure"
                            else:
                                # Backward-compatible local OCR (requires pytesseract + system tesseract)
                                log(
                                    f"[OCR] {doc_id}: Azure DI not configured; falling back to local Tesseract OCR "
                                    f"(pages<=%s dpi=%s lang=%s)" % (self.cfg.ocr_max_pages, self.cfg.ocr_dpi, self.cfg.ocr_lang)
                                )
                                ocr_text = ocr_pdf_bytes(self.cfg, pdf_bytes)
                                source_tag = "pdf+ocr"

                            ocr_stats = analyze_text_quality(ocr_text)

                            # If OCR result is better than current best candidate, keep it (even if still not acceptable)
                            if ocr_stats.score > best_stats.score:
                                best_text = ocr_text
                                best_stats = ocr_stats
                                best_raw_path = str(pdf_path)
                                best_source = source_tag

                            if is_text_acceptable(self.cfg, ocr_stats):
                                best_text = ocr_text
                                best_stats = ocr_stats
                                best_raw_path = str(pdf_path)
                                used_pdf = True
                                best_source = source_tag
                                break
                            else:
                                pdf_error = (
                                    f"{pdf_url} -> {source_tag} low-quality extracted text "
                                    f"(words={ocr_stats.n_words}, unique={ocr_stats.unique_words}, "
                                    f"unique_ratio={ocr_stats.unique_ratio:.2f}, common_line_ratio={ocr_stats.common_line_ratio:.2f})"
                                )
                        except Exception as e:
                            pdf_error = f"{pdf_url} -> OCR failed: {e}"

                    if is_text_acceptable(self.cfg, pdf_stats):
                        best_text = pdf_text
                        best_stats = pdf_stats
                        best_raw_path = str(pdf_path)
                        used_pdf = True
                        best_source = "pdf"
                        break

                    # Low-quality extracted text (common with scanned PDFs)
                    pdf_error = (
                        f"{pdf_url} -> low-quality extracted text "
                        f"(words={pdf_stats.n_words}, unique={pdf_stats.unique_words}, "
                        f"unique_ratio={pdf_stats.unique_ratio:.2f}, common_line_ratio={pdf_stats.common_line_ratio:.2f})"
                    )
                except Exception as e:
                    pdf_error = f"{pdf_url} -> {e}"
                    continue

        # Save best-effort cleaned text (even if we later mark it as error) for debugging
        txt_path = self.cfg.text_dir / f"{doc_id}.txt"
        txt_path.write_text(best_text or "", encoding="utf-8", errors="ignore")

        # Still not acceptable? Record as error and retry later (LexUZ often adds HTML later).
        if not is_text_acceptable(self.cfg, best_stats):
            skip_pdf = getattr(self, 'skip_pdf', False)
            if skip_pdf:
                # In skip-pdf mode, mark as skipped (not error) so we don't retry
                e = "Skipped: PDF-only document (--skip-pdf mode)"
                self.store.upsert_document_placeholder(doc_id, final_url, best_title, self.cfg.lex_lang, source_date)
                self.store.update_document_fetch(
                    doc_id,
                    title=best_title,
                    content_hash=None,
                    etag=resp.headers.get("ETag"),
                    last_modified=resp.headers.get("Last-Modified"),
                    raw_html_path=best_raw_path,
                    text_path=str(txt_path),
                    changed=False,
                    error=e,
                )
                return IngestResult(doc_id=doc_id, url=final_url, status="skipped", title=best_title, error=e)
            else:
                e = (
                    "Could not extract full document text (likely PDF-only or scanned PDF). "
                    f"best_source={best_source}. "
                )
                if pdf_error:
                    e += f"Last PDF attempt: {pdf_error}"
                self.store.upsert_document_placeholder(doc_id, final_url, best_title, self.cfg.lex_lang, source_date)
                self.store.update_document_fetch(
                    doc_id,
                    title=best_title,
                    content_hash=None,
                    etag=resp.headers.get("ETag"),
                    last_modified=resp.headers.get("Last-Modified"),
                    raw_html_path=best_raw_path,
                    text_path=str(txt_path),
                    changed=False,
                    error=e,
                )
                return IngestResult(doc_id=doc_id, url=final_url, status="error", title=best_title, error=e)

        # Success path: compute content hash
        content_hash = sha256_text(re.sub(r"\s+", " ", best_text))

        is_new = existing is None
        changed = (existing is None) or (existing["content_hash"] != content_hash)

        # Ensure document row exists (store the canonical URL that worked)
        self.store.upsert_document_placeholder(doc_id, final_url, best_title, self.cfg.lex_lang, source_date)

        if not changed:
            self.store.update_document_fetch(
                doc_id,
                title=best_title,
                content_hash=content_hash,
                etag=resp.headers.get("ETag"),
                last_modified=resp.headers.get("Last-Modified"),
                raw_html_path=best_raw_path,
                text_path=str(txt_path),
                changed=False,
                error=None,
            )
            return IngestResult(doc_id=doc_id, url=final_url, status="unchanged", title=best_title)

        # Deactivate old vectors/chunks if update
        deactivated_vector_ids: List[int] = []
        if not is_new:
            deactivated_vector_ids = self.store.deactivate_doc_chunks_and_vectors(doc_id)
            # Try to delete from ANN index if backend supports it
            self.index.mark_deleted(deactivated_vector_ids)

        # Chunk and embed
        chunks = chunk_text_words(best_text, self.cfg.chunk_max_words, self.cfg.chunk_overlap_words)
        if not chunks:
            e = "No chunks produced."
            self.store.update_document_fetch(
                doc_id,
                title=best_title,
                content_hash=content_hash,
                etag=resp.headers.get("ETag"),
                last_modified=resp.headers.get("Last-Modified"),
                raw_html_path=best_raw_path,
                text_path=str(txt_path),
                changed=True,
                error=e,
            )
            return IngestResult(doc_id=doc_id, url=final_url, status="error", title=best_title, error=e)

        # Insert chunks
        chunk_ids: List[int] = []
        chunk_texts: List[str] = []
        for i, c in enumerate(chunks):
            th = sha256_text(c)
            cid = self.store.insert_chunk(doc_id, i, c, th)
            chunk_ids.append(cid)
            chunk_texts.append(c)

        # Embed
        vecs = self.embedder.embed_passages(chunk_texts)  # (n, dim) float32 normalized
        if vecs.ndim != 2 or vecs.shape[0] != len(chunk_ids):
            e = "Embedding failed: unexpected shape."
            self.store.update_document_fetch(
                doc_id,
                title=best_title,
                content_hash=content_hash,
                etag=resp.headers.get("ETag"),
                last_modified=resp.headers.get("Last-Modified"),
                raw_html_path=best_raw_path,
                text_path=str(txt_path),
                changed=True,
                error=e,
            )
            return IngestResult(doc_id=doc_id, url=final_url, status="error", title=best_title, error=e)

        # Store vectors and update index incrementally
        vector_ids: List[int] = []
        for i, cid in enumerate(chunk_ids):
            v = vecs[i].astype(np.float32)
            v_f16 = v.astype(np.float16)
            vid = self.store.insert_vector(cid, v_f16)
            vector_ids.append(vid)

        self.index.add_vectors(vector_ids, vecs.astype(np.float32))

        self.store.update_document_fetch(
            doc_id,
            title=best_title,
            content_hash=content_hash,
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
            raw_html_path=best_raw_path,
            text_path=str(txt_path),
            changed=True,
            error=None,
        )

        status = "new" if is_new else "updated"
        return IngestResult(doc_id=doc_id, url=final_url, status=status, title=best_title, chunks_added=len(chunks))

    def run_once(self) -> None:

        # 1) discover
        discovered = self.discover_recent_docs(days=self.cfg.poll_days)
        for url, source_date in discovered:
            doc_id = extract_doc_id_from_url(url)
            self.store.enqueue_url(url, doc_id, reason="calendar", priority=3)
            # store placeholder for tracking even if not ingested yet
            if doc_id:
                self.store.upsert_document_placeholder(doc_id, url, title=None, lang=self.cfg.lex_lang, source_date=source_date)

        # 2) schedule rechecks
        n_rechecks = self.schedule_rechecks(limit=self.cfg.recheck_per_run)
        if n_rechecks:
            log(f"[RECHECK] queued {n_rechecks} doc(s) for recheck")

        # 3) process queue
        batch = self.store.dequeue_batch(limit=self.cfg.max_docs_per_run)
        if not batch:
            log("[RUN] No queued documents to process.")
            return

        log(f"[RUN] Processing {len(batch)} document(s) ...")
        new_count = 0
        upd_count = 0
        unchanged_count = 0
        err_count = 0

        for row in batch:
            qid = int(row["id"])
            url = str(row["url"])
            reason = str(row["reason"] or "")
            doc_id = row["doc_id"]
            log(f"[RUN] ingest url={url} reason={reason}")
            self.store.queue_mark_attempt(qid, error=None)

            try:
                res = self.ingest_one(url)
            except Exception as e:
                res = IngestResult(doc_id=doc_id or "?", url=url, status="error", error=str(e))

            if res.status in ("new", "updated", "unchanged"):
                # done: remove from queue
                self.store.queue_delete(qid)
            else:
                # keep in queue but record error
                self.store.queue_mark_attempt(qid, error=res.error or res.status)

            if res.status == "new":
                new_count += 1
                self._notify_new(res)
            elif res.status == "updated":
                upd_count += 1
                self._notify_updated(res)
            elif res.status == "unchanged":
                unchanged_count += 1
            else:
                err_count += 1
                warn(f"[RUN] ingest failed for {url}: {res.error}")

        log(f"[RUN] done. new={new_count} updated={upd_count} unchanged={unchanged_count} error={err_count}")

    def _notify_new(self, res: IngestResult) -> None:
        if not self.telegram.enabled():
            return
        title = res.title or "(no title)"
        msg = f"🆕 LexUZ indexed NEW document\n\n{title}\n{res.url}\n\nchunks: {res.chunks_added}"
        self.telegram.send(msg)

    def _notify_updated(self, res: IngestResult) -> None:
        if not self.telegram.enabled():
            return
        title = res.title or "(no title)"
        msg = f"♻️ LexUZ UPDATED document re-indexed\n\n{title}\n{res.url}\n\nnew chunks: {res.chunks_added}"
        self.telegram.send(msg)


# -----------------------------
# RAG Q&A
# -----------------------------

@dataclass
class Retrieved:
    vector_id: int
    score: float
    doc_id: str
    url: str
    title: Optional[str]
    chunk_index: int
    text: str

def detect_language_simple(text: str) -> str:
    """
    Very rough heuristic: returns 'uz'|'ru'|'en' (best guess).
    """
    t = text.strip()
    # Cyrillic -> ru/uz-cyrl (we return ru)
    if re.search(r"[А-Яа-яЁёЎўҚқҒғҲҳ]", t):
        return "ru"
    # Uzbek latin markers
    if re.search(r"[ʻʼ’]|\bo['’]z\b|qonun|farmon|qaror|modda|soliq", t.lower()):
        return "uz"
    # Default english
    return "en"

def rewrite_question(openai: OpenAIClient, question: str) -> Dict[str, Any]:
    """
    Ask an LLM to normalize the question for better retrieval & answering.
    Returns dict with keys:
      - clean_question
      - search_query
      - answer_language
    """
    # Keep prompt short to save tokens
    system = (
        "You normalize user questions for searching and answering using Uzbekistan legal acts (LexUZ). "
        "Return ONLY valid JSON."
    )
    user = {
        "task": "Normalize the user question. Preserve meaning. Fix typos. "
                "If question mixes languages, choose the most appropriate for answering. "
                "Also produce a short search query.",
        "output_json_schema": {
            "clean_question": "string",
            "search_query": "string",
            "answer_language": "one of: uz, ru, en"
        },
        "user_question": question,
    }
    raw = openai.generate_text(system=system, user=json.dumps(user, ensure_ascii=False), temperature=0.1, max_output_tokens=300)
    obj = extract_json_object(raw)
    if not obj:
        # fallback heuristic
        lang = detect_language_simple(question)
        return {"clean_question": question.strip(), "search_query": question.strip(), "answer_language": lang}
    clean_q = str(obj.get("clean_question") or question).strip()
    search_q = str(obj.get("search_query") or clean_q).strip()
    al = str(obj.get("answer_language") or detect_language_simple(question)).strip().lower()
    if al not in ("uz", "ru", "en"):
        al = detect_language_simple(question)
    return {"clean_question": clean_q, "search_query": search_q, "answer_language": al}

def format_context_block(items: List[Retrieved], max_chars_each: int = 1200) -> str:
    blocks = []
    for i, it in enumerate(items, start=1):
        t = it.text
        if len(t) > max_chars_each:
            t = t[:max_chars_each].rsplit(" ", 1)[0] + " …"
        title = it.title or "(no title)"
        blocks.append(
            f"[SOURCE {i}]\n"
            f"Title: {title}\n"
            f"URL: {it.url}\n"
            f"Chunk: {it.chunk_index}\n"
            f"Text:\n{t}\n"
        )
    return "\n\n".join(blocks)

def answer_with_rag(openai: OpenAIClient, question: str, answer_language: str, context_items: List[Retrieved]) -> str:
    system = (
        "You are a careful legal information assistant. "
        "Answer using ONLY the provided sources from LexUZ. "
        "If the sources do not contain the answer, say you cannot find it in the provided texts and suggest what to search next. "
        "Do NOT invent citations. "
        "Write in the user's requested language."
    )

    context = format_context_block(context_items)
    user = {
        "answer_language": answer_language,
        "question": question,
        "instructions": [
            "Use the sources as evidence.",
            "When you mention a claim, cite it like [SOURCE 1], [SOURCE 2] etc.",
            "At the end, include a 'Sources' list with the URLs.",
            "Include a short disclaimer: 'This is not legal advice.'"
        ],
        "sources": context,
    }
    return openai.generate_text(system=system, user=json.dumps(user, ensure_ascii=False), temperature=0.2, max_output_tokens=900)

def rag_ask(cfg: Config, question: str) -> str:
    pipe = Pipeline(cfg)
    try:
        if not pipe.openai.enabled():
            raise RuntimeError("OPENAI_API_KEY is required for ask(). Set it in .env.")

        # Rewrite (optional)
        if cfg.use_openai_rewrite:
            rq = rewrite_question(pipe.openai, question)
            clean_q = rq["clean_question"]
            search_q = rq["search_query"]
            answer_lang = rq["answer_language"]
        else:
            clean_q = question.strip()
            search_q = clean_q
            answer_lang = detect_language_simple(question)

        # Retrieve
        qvec = pipe.embedder.embed_query(search_q)
        hits = pipe.index.search(qvec, k=cfg.top_k, candidate_multiplier=cfg.candidate_multiplier)

        if not hits:
            return "No documents indexed yet (or no matches). Run `run-once` or `backfill` first."

        # Load chunk text + metadata and apply per-doc cap
        results: List[Retrieved] = []
        per_doc: Dict[str, int] = {}
        for vid, score in hits:
            row = pipe.store.get_chunk_by_vector_id(vid)
            if not row:
                continue
            doc_id = str(row["doc_id"])
            per_doc.setdefault(doc_id, 0)
            if per_doc[doc_id] >= cfg.max_chunks_per_doc:
                continue
            per_doc[doc_id] += 1
            results.append(
                Retrieved(
                    vector_id=int(vid),
                    score=float(score),
                    doc_id=doc_id,
                    url=str(row["url"]),
                    title=row["title"],
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                )
            )
            if len(results) >= cfg.top_k:
                break

        # Compose answer
        answer = answer_with_rag(pipe.openai, clean_q, answer_lang, results)
        return answer
    finally:
        pipe.close()


# -----------------------------
# Backfill / export helpers
# -----------------------------

def cmd_backfill(cfg: Config, start_date: str, end_date: str, process: bool = False, skip_pdf: bool = False) -> None:
    """
    Enqueue docs from calendar pages in a date range (inclusive).
    If process=True, also process all queued documents (full backfill).
    If skip_pdf=True, skip documents that require PDF download (HTML-only mode).
    """
    pipe = Pipeline(cfg)
    pipe.skip_pdf = skip_pdf
    if skip_pdf:
        log("[BACKFILL] Skip-PDF mode enabled: only HTML documents will be indexed")
    try:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)
        if end < start:
            start, end = end, start
        cur = start
        n_urls = 0
        while cur <= end:
            cal_url = calendar_url_for_date(cfg, cur)
            try:
                resp = pipe.http.get(cal_url)
                if resp.status_code >= 300:
                    warn(f"[BACKFILL] HTTP {resp.status_code} {cal_url}")
                    cur += dt.timedelta(days=1)
                    continue
                urls = parse_calendar_for_docs(cfg, resp.text)
                for u in urls:
                    doc_id = extract_doc_id_from_url(u)
                    pipe.store.enqueue_url(u, doc_id, reason="backfill", priority=2)
                    if doc_id:
                        pipe.store.upsert_document_placeholder(doc_id, u, title=None, lang=cfg.lex_lang, source_date=cur.isoformat())
                n_urls += len(urls)
                log(f"[BACKFILL] {cur.isoformat()} +{len(urls)} urls")
            except Exception as e:
                warn(f"[BACKFILL] {cur.isoformat()} error: {e}")
            cur += dt.timedelta(days=1)

        log(f"[BACKFILL] queued total ~{n_urls} urls")

        # If --process flag is set, process all queued documents
        if process:
            log(f"[BACKFILL] Starting to process all queued documents...")
            processed = 0
            skipped = 0
            errors = 0
            while True:
                # Get next batch from queue
                batch = pipe.store.dequeue_batch(limit=50)
                if not batch:
                    break
                
                for row in batch:
                    qid = int(row["id"])
                    url = str(row["url"])
                    doc_id = row["doc_id"]
                    
                    try:
                        res = pipe.ingest_one(url)
                        if res.status in ("new", "updated", "unchanged", "skipped"):
                            pipe.store.queue_delete(qid)
                            if res.status in ("new", "updated"):
                                processed += 1
                                if processed % 100 == 0:
                                    log(f"[BACKFILL] Processed {processed} documents...")
                            elif res.status == "skipped":
                                skipped += 1
                        else:
                            pipe.store.queue_mark_attempt(qid, error=res.error or res.status)
                            errors += 1
                    except Exception as e:
                        pipe.store.queue_mark_attempt(qid, error=str(e))
                        errors += 1
                        warn(f"[BACKFILL] Error processing {url}: {e}")
            
            log(f"[BACKFILL] Complete. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    finally:
        pipe.close()

def cmd_export_corpus(cfg: Config, out_dir: Path, since_days: int) -> None:
    """
    Export cleaned document texts to a folder for offline inspection / (optional) fine-tuning data creation.
    """
    pipe = Pipeline(cfg)
    try:
        ensure_dir(out_dir)
        cutoff = iso(utc_now() - dt.timedelta(days=since_days))
        cur = pipe.store.conn.cursor()
        rows = cur.execute(
            "SELECT doc_id, url, title, text_path, last_changed_at, last_fetched_at FROM documents WHERE last_fetched_at >= ? ORDER BY last_fetched_at DESC",
            (cutoff,),
        ).fetchall()
        manifest = []
        for r in rows:
            doc_id = r["doc_id"]
            tp = r["text_path"]
            if not tp or not Path(tp).exists():
                continue
            dst = out_dir / f"{doc_id}.txt"
            dst.write_text(Path(tp).read_text(encoding="utf-8", errors="ignore"), encoding="utf-8", errors="ignore")
            manifest.append(
                {
                    "doc_id": doc_id,
                    "url": r["url"],
                    "title": r["title"],
                    "last_changed_at": r["last_changed_at"],
                    "last_fetched_at": r["last_fetched_at"],
                    "file": str(dst.name),
                }
            )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[EXPORT] exported {len(manifest)} docs to {out_dir}")
    finally:
        pipe.close()


# -----------------------------
# CLI commands
# -----------------------------

def cmd_doctor(cfg: Config) -> int:
    """
    Check environment + print what backend will be used.
    """
    ok = True
    log("=== LexUZ MVP doctor ===")

    log(f"Python: {sys.version.split()[0]}  OS: {os.name}")
    log(f"Data dir: {cfg.data_dir}")
    log(f"SQLite: {cfg.sqlite_path}")

    # Check LexUZ calendar fetch
    pipe = Pipeline(cfg)
    try:
        test_url = calendar_url_for_date(cfg, dt.date.today())
        log(f"LexUZ calendar test URL: {test_url}")
        try:
            resp = pipe.http.get(test_url)
            log(f"Calendar HTTP status: {resp.status_code}")
            if resp.status_code >= 300:
                ok = False
                warn("Calendar fetch failed. You may be blocked or the URL format changed.")
        except Exception as e:
            ok = False
            err(f"Calendar fetch exception: {e}")

        # Embeddings
        if cfg.embedding_provider.lower() == "local":
            if _SENTENCE_TRANSFORMERS is None:
                ok = False
                err("Local embedding provider selected but sentence-transformers is missing.")
            else:
                log(f"Embeddings: local model='{cfg.embedding_model}' (will download on first run)")
        else:
            warn(f"Embeddings provider '{cfg.embedding_provider}' is not implemented (use local).")
            ok = False

        # Index backend
        log(f"Index backend selected: {pipe.index.backend}")
        if pipe.index.backend == "faiss" and _FAISS is None:
            ok = False
            err("FAISS selected but import failed.")
        if pipe.index.backend == "hnswlib" and _HNSWLIB is None:
            ok = False
            err("hnswlib selected but import failed.")
        if pipe.index.backend == "numpy":
            warn("Using numpy brute-force index (works but slower). Consider installing faiss-cpu or hnswlib.")

        # OpenAI
        if cfg.openai_api_key:
            log(f"OpenAI: enabled (model={cfg.openai_model}, base={cfg.openai_base_url})")
        else:
            warn("OpenAI: OPENAI_API_KEY missing. 'ask' command won't work until set.")

        # Telegram
        if cfg.telegram_bot_token and cfg.telegram_chat_id:
            log("Telegram: enabled")
        else:
            warn("Telegram: disabled (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable)")

        # OCR (scanned PDFs / no text layer)
        if bool(cfg.enable_ocr):
            if cfg.azure_di_endpoint and cfg.azure_di_key:
                log(
                    "OCR: enabled via Azure AI Document Intelligence "
                    f"(model={cfg.azure_di_model}, api={cfg.azure_di_api_version}, "
                    f"format={cfg.azure_di_output_format}, high_res={int(cfg.azure_di_ocr_high_res)})"
                )
                if not (cfg.azure_di_endpoint.startswith("http://") or cfg.azure_di_endpoint.startswith("https://")):
                    warn("AZURE_DI_ENDPOINT doesn't look like a URL (expected https://...).")
            else:
                warn(
                    "OCR: ENABLE_OCR=1 but AZURE_DI_ENDPOINT/AZURE_DI_KEY not set. "
                    "Will try local Tesseract OCR (optional)."
                )
                log(f"OCR: local fallback (lang={cfg.ocr_lang}, dpi={cfg.ocr_dpi}, max_pages={cfg.ocr_max_pages})")
                if _PYMUPDF is None:
                    warn("Local OCR enabled but PyMuPDF (pymupdf) is missing: pip install pymupdf")
                if _PYTESSERACT is None or _PIL_IMAGE is None:
                    warn("Local OCR enabled but pytesseract/pillow is missing: pip install pytesseract pillow")
                if cfg.tesseract_cmd:
                    log(f"OCR: using TESSERACT_CMD={cfg.tesseract_cmd}")
                else:
                    warn("OCR: TESSERACT_CMD not set. If tesseract.exe isn't on PATH, OCR will fail.")
        else:
            log("OCR: disabled (set ENABLE_OCR=1 to enable scanned-PDF OCR via Azure DI)")
            if cfg.azure_di_endpoint and cfg.azure_di_key:
                log("OCR: Azure DI is configured but ENABLE_OCR=0, so it will not be used.")

    finally:
        pipe.close()

    log("=== doctor done ===")
    return 0 if ok else 1

def cmd_telegram_test(cfg: Config) -> None:
    t = Telegram(cfg)
    if not t.enabled():
        print("Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return
    t.send("LexUZ MVP is online ✅")
    print("Sent Telegram test message.")

def cmd_run_once(cfg: Config) -> None:
    lock = RunLock(cfg.lock_path)
    try:
        with lock:
            pipe = Pipeline(cfg)
            try:
                pipe.run_once()
            finally:
                pipe.close()
    except RuntimeError as e:
        warn(str(e))

def cmd_rebuild_index(cfg: Config) -> None:
    pipe = Pipeline(cfg)
    try:
        pipe.index.rebuild()
        log("[INDEX] rebuild complete")
    finally:
        pipe.close()

def cmd_ask(cfg: Config, question: str) -> None:
    answer = rag_ask(cfg, question)
    print("\n" + answer.strip() + "\n")

def cmd_stats(cfg: Config) -> None:
    pipe = Pipeline(cfg)
    try:
        cur = pipe.store.conn.cursor()
        docs = cur.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = cur.execute("SELECT COUNT(*) AS n FROM chunks WHERE active=1").fetchone()["n"]
        vecs = cur.execute("SELECT COUNT(*) AS n FROM vectors WHERE active=1").fetchone()["n"]
        queued = cur.execute("SELECT COUNT(*) AS n FROM doc_queue").fetchone()["n"]
        print(json.dumps({
            "documents_total": docs,
            "chunks_active": chunks,
            "vectors_active": vecs,
            "queue_size": queued,
            "index_backend": pipe.index.backend,
            "index_dim": pipe.index.dim,
        }, indent=2))
    finally:
        pipe.close()


def cmd_list_docs(cfg: Config, status: Optional[str], limit: int) -> None:
    """List documents in the SQLite DB (without loading the embedding model)."""
    store = Store(cfg)
    try:
        cur = store.conn.cursor()
        if status:
            cur.execute(
                """
                SELECT doc_id, status, url, title, last_fetched_at, last_error
                FROM documents
                WHERE status = ?
                ORDER BY last_seen_at DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            cur.execute(
                """
                SELECT doc_id, status, url, title, last_fetched_at, last_error
                FROM documents
                ORDER BY last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cur.fetchall()
        print(f"Documents shown: {len(rows)} (limit={limit})")
        for (doc_id, st, url, title, last_fetched_at, last_error) in rows:
            t = (title or "").strip()
            if len(t) > 90:
                t = t[:87] + "..."
            e = (last_error or "").strip()
            if len(e) > 120:
                e = e[:117] + "..."
            print(f"- {doc_id}  status={st}  fetched={last_fetched_at or '-'}")
            print(f"  url:   {url}")
            if t:
                print(f"  title: {t}")
            if e:
                print(f"  err:   {e}")
    finally:
        store.close()


def cmd_show_doc(cfg: Config, doc_id: str, chunk_limit: int = 5, text_chars: int = 2500) -> None:
    """Show one document + a few chunks (without loading the embedding model)."""
    store = Store(cfg)
    try:
        doc = store.get_document(doc_id)
        if not doc:
            print(f"Document not found: {doc_id}")
            return

        # sqlite3.Row supports dict-style indexing (row['col']) but not .get().
        # Convert to a plain dict for safer access in this CLI command.
        if not isinstance(doc, dict):
            doc = dict(doc)

        print("=" * 100)
        print(f"doc_id: {doc['doc_id']}")
        print(f"url: {doc['url']}")
        print(f"title: {doc.get('title')}")
        print(f"status: {doc.get('status')}")
        print(f"last_error: {doc.get('last_error')}")
        print(f"first_seen_at: {doc.get('first_seen_at')}")
        print(f"last_seen_at: {doc.get('last_seen_at')}")
        print(f"last_fetched_at: {doc.get('last_fetched_at')}")
        print(f"last_changed_at: {doc.get('last_changed_at')}")
        print(f"raw_html_path: {doc.get('raw_html_path')}")
        print(f"text_path: {doc.get('text_path')}")
        print("=" * 100)

        # Show cleaned text excerpt if available
        tp = doc.get("text_path")
        if tp and os.path.exists(tp):
            try:
                content = Path(tp).read_text(encoding="utf-8", errors="ignore")
                excerpt = content[:text_chars]
                print("\n=== CLEAN TEXT (excerpt) ===")
                print(excerpt)
                if len(content) > text_chars:
                    print(f"\n... ({len(content)} chars total)")
            except Exception as e:
                print(f"\nCould not read text_path: {e}")

        # Show chunks
        cur = store.conn.cursor()
        cur.execute(
            """
            SELECT chunk_index, text
            FROM chunks
            WHERE doc_id = ? AND active = 1
            ORDER BY chunk_index
            LIMIT ?
            """,
            (doc_id, chunk_limit),
        )
        rows = cur.fetchall()
        print(f"\n=== CHUNKS (showing {len(rows)} / limit={chunk_limit}) ===")
        for idx, t in rows:
            print(f"\n--- chunk {idx} ---")
            print(t[:2000] + ("..." if len(t) > 2000 else ""))
    finally:
        store.close()


def cmd_force_recheck(cfg: Config, *, status: Optional[str], doc_id: Optional[str], limit: int, all_docs: bool) -> None:
    """
    Force documents to be rechecked soon by setting next_recheck_at = now.
    This is useful after upgrading the script (e.g., improved PDF/HTML extraction).
    """
    store = Store(cfg)
    try:
        cur = store.conn.cursor()
        now = iso(utc_now())

        if doc_id:
            cur.execute("UPDATE documents SET next_recheck_at = ? WHERE doc_id = ?", (now, doc_id))
            store.conn.commit()
            print(f"Forced recheck for doc_id={doc_id}")
            return

        if all_docs:
            cur.execute("UPDATE documents SET next_recheck_at = ?", (now,))
            store.conn.commit()
            print("Forced recheck for ALL documents")
            return

        # Default: only error docs (or user-specified status)
        st = status or "error"
        cur.execute(
            """
            UPDATE documents
            SET next_recheck_at = ?
            WHERE doc_id IN (
                SELECT doc_id FROM documents
                WHERE status = ?
                ORDER BY last_seen_at DESC
                LIMIT ?
            )
            """,
            (now, st, limit),
        )
        store.conn.commit()
        print(f"Forced recheck for up to {limit} documents where status='{st}'")
    finally:
        store.close()



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lexuz_mvp_sqlite.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="LexUZ ultra-cheap MVP: calendar watcher + ingestion + local embeddings + vector search + RAG Q&A",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check environment and configuration.")

    sub.add_parser("run-once", help="Run one discovery + ingestion cycle.")

    sub.add_parser("stats", help="Show quick DB stats.")

    sp = sub.add_parser("list-docs", help="List documents in the SQLite DB (no model load).")
    sp.add_argument("--status", default=None, help="Filter by status (e.g., ok, error, new).")
    sp.add_argument("--limit", type=int, default=50, help="Max rows to show.")

    sp = sub.add_parser("show-doc", help="Show one document + a few chunks (no model load).")
    sp.add_argument("doc_id", help="Document id, e.g. -7934915")
    sp.add_argument("--chunks", type=int, default=5, help="How many chunks to print.")
    sp.add_argument("--text-chars", type=int, default=2500, help="How many chars of cleaned text to print.")

    sp = sub.add_parser("force-recheck", help="Force documents to be rechecked soon by setting next_recheck_at=now.")
    sp.add_argument("--status", default=None, help="Which status to target (default: error).")
    sp.add_argument("--doc-id", default=None, help="Force one specific document id.")
    sp.add_argument("--limit", type=int, default=200, help="Max docs to update when using --status (default 200).")
    sp.add_argument("--all", action="store_true", help="Force ALL documents (use carefully).")

    sub.add_parser("telegram-test", help="Send a test Telegram message (if configured).")

    sp = sub.add_parser("ask", help="Ask a question using RAG over indexed LexUZ documents.")
    sp.add_argument("question", type=str, help="Your question (any language; will answer in detected language).")

    sp = sub.add_parser("backfill", help="Enqueue docs from LexUZ calendar between two ISO dates (inclusive).")
    sp.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    sp.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    sp.add_argument("--process", action="store_true", help="Also process all queued documents (full backfill)")
    sp.add_argument("--skip-pdf", action="store_true", help="Skip PDF downloads, only index HTML documents (faster, cheaper)")

    sub.add_parser("rebuild-index", help="Rebuild vector index from active vectors in SQLite.")

    sp = sub.add_parser("export-corpus", help="Export cleaned texts to a folder.")
    sp.add_argument("--out-dir", required=True, help="Output directory path")
    sp.add_argument("--since-days", type=int, default=31, help="Export docs fetched within last N days (default 31)")

    return p

def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    cfg = Config.from_env()

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cmd = args.cmd
    if cmd == "doctor":
        return cmd_doctor(cfg)
    if cmd == "telegram-test":
        cmd_telegram_test(cfg)
        return 0
    if cmd == "run-once":
        cmd_run_once(cfg)
        return 0
    if cmd == "rebuild-index":
        cmd_rebuild_index(cfg)
        return 0
    if cmd == "ask":
        cmd_ask(cfg, args.question)
        return 0
    if cmd == "backfill":
        cmd_backfill(cfg, args.start, args.end, process=args.process, skip_pdf=args.skip_pdf)
        return 0
    if cmd == "export-corpus":
        cmd_export_corpus(cfg, Path(args.out_dir), args.since_days)
        return 0
    if cmd == "stats":
        cmd_stats(cfg)
        return 0
    if cmd == "list-docs":
        cmd_list_docs(cfg, status=args.status, limit=args.limit)
        return 0
    if cmd == "show-doc":
        cmd_show_doc(cfg, args.doc_id, chunk_limit=args.chunks, text_chars=args.text_chars)
        return 0
    if cmd == "force-recheck":
        cmd_force_recheck(cfg, status=args.status, doc_id=args.doc_id, limit=args.limit, all_docs=args.all)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
