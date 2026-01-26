#!/usr/bin/env python3
"""local_fts_index.py

Build a *very fast* local full‑text search (FTS) index for a folder of Lex.uz HTML files.

Why this works (and why Google feels "instant")
------------------------------------------------
Google/Perplexity are fast because they *search an index*, not raw pages.
This script builds that same kind of structure locally: an inverted index (SQLite FTS5 + BM25 ranking).

Input
-----
  A folder with ~11,000 HTML pages you already downloaded from lex.uz.

Output
------
  A single SQLite database file containing an FTS5 index.

Usage
-----
  python local_fts_index.py --html-dir ./lex_html --db ./data/lexuz_fts.sqlite3
  python local_fts_index.py --html-dir ./lex_html --db ./data/lexuz_fts.sqlite3 --reset

Then search with:
  python local_fts_search.py --db ./data/lexuz_fts.sqlite3 "mehnat ta'tili muddati"

Notes
-----
* FTS5 is included in most Python builds; this script will fail fast if it's missing.
* This builds a *chunk-level* index (better for RAG). Each document is split into overlapping chunks.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from bs4 import BeautifulSoup


# -----------------------------
# Text normalization
# -----------------------------

_APOSTROPHES = "\u02bb\u02bc\u2018\u2019\u201b\u2032\u00b4\u0060"


def normalize_text(s: str) -> str:
    """Normalize text for *both* indexing and querying.

    Uzbek Latin often contains apostrophe-like characters (Oʻzbekiston, O‘zb, O'z).
    SQLite FTS tokenizers often split on apostrophes, which hurts matching.
    We normalize these to nothing so 'Oʻzbekiston' and 'Ozbekiston' match.
    """
    if not s:
        return ""
    s = s.replace("\u00a0", " ")

    # Normalize various apostrophe-like chars to a plain apostrophe, then remove it.
    s = re.sub(f"[{_APOSTROPHES}]", "'", s)
    s = s.replace("'", "")

    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_doc_id_from_path(p: Path) -> str:
    """Best-effort extraction of lex.uz doc id from filename."""
    m = re.search(r"-?\d+", p.stem)
    return m.group(0) if m else p.stem


def html_to_text(html: str) -> Tuple[str, str]:
    """Extract (title, text) from an HTML page."""
    soup = BeautifulSoup(html, "lxml")

    # Remove obvious boilerplate.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        # Prefer on-page title when present.
        title = h1.get_text(" ", strip=True)

    # Try likely main containers first, then fall back.
    main = soup.select_one(
        "article, main, #content, .content, .document, .doc, .document-content, .doc-content, .text"
    )
    if main:
        text = main.get_text("\n", strip=True)
    else:
        body = soup.body
        text = body.get_text("\n", strip=True) if body else soup.get_text("\n", strip=True)

    # Clean lines.
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)

    # Final normalization.
    return normalize_text(title), normalize_text(text)


def chunk_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


# -----------------------------
# SQLite FTS schema + indexing
# -----------------------------


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # A small metadata table is handy for stats and future migrations.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )

    # Chunk-level index for RAG.
    # Column order matters for bm25() and snippet().
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            doc_id      UNINDEXED,
            url         UNINDEXED,
            title,
            chunk_index UNINDEXED,
            text,
            tokenize = 'unicode61',
            prefix = '2 3 4'
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS meta")
    cur.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.commit()


def iter_html_files(root: Path) -> Iterator[Path]:
    exts = {".html", ".htm"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@dataclass
class IndexRow:
    doc_id: str
    url: str
    title: str
    chunk_index: int
    text: str


def build_rows_for_file(p: Path, url_template: Optional[str], max_words: int, overlap: int) -> List[IndexRow]:
    raw = p.read_bytes()
    # Try utf-8 first, then fall back.
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        html = raw.decode("utf-8", errors="ignore")

    doc_id = extract_doc_id_from_path(p)
    url = url_template.format(doc_id=doc_id) if url_template else p.as_posix()

    title, text = html_to_text(html)
    if not text:
        return []

    chunks = chunk_words(text, max_words=max_words, overlap_words=overlap)
    rows: List[IndexRow] = []
    for i, ch in enumerate(chunks):
        rows.append(IndexRow(doc_id=doc_id, url=url, title=title, chunk_index=i, text=ch))
    return rows


def index_folder(
    html_dir: Path,
    db_path: Path,
    url_template: Optional[str],
    max_words: int,
    overlap: int,
    batch_size: int,
    limit_files: Optional[int],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200k pages in memory

    ensure_schema(conn)
    cur = conn.cursor()

    files = list(iter_html_files(html_dir))
    if limit_files is not None:
        files = files[:limit_files]

    t0 = time.time()
    inserted = 0

    batch: List[IndexRow] = []
    for idx, p in enumerate(files, 1):
        try:
            rows = build_rows_for_file(p, url_template, max_words, overlap)
            batch.extend(rows)
        except Exception as e:
            print(f"[WARN] Failed parsing {p}: {e}")
            continue

        if len(batch) >= batch_size:
            cur.execute("BEGIN")
            cur.executemany(
                "INSERT INTO chunks_fts(doc_id, url, title, chunk_index, text) VALUES (?,?,?,?,?)",
                [(r.doc_id, r.url, r.title, r.chunk_index, r.text) for r in batch],
            )
            conn.commit()
            inserted += len(batch)
            batch.clear()

        if idx % 250 == 0:
            dt = time.time() - t0
            rate = inserted / max(dt, 1e-9)
            print(f"[{idx}/{len(files)}] chunks={inserted} ({rate:.0f} chunks/s)")

    if batch:
        cur.execute("BEGIN")
        cur.executemany(
            "INSERT INTO chunks_fts(doc_id, url, title, chunk_index, text) VALUES (?,?,?,?,?)",
            [(r.doc_id, r.url, r.title, r.chunk_index, r.text) for r in batch],
        )
        conn.commit()
        inserted += len(batch)

    # Optimize the FTS index for faster queries.
    try:
        cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        conn.commit()
    except Exception:
        pass

    # Basic stats.
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("indexed_chunks", str(inserted)))
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("indexed_files", str(len(files))))
    cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("created_at", time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    dt = time.time() - t0
    print("\nDone!")
    print(f"  HTML files: {len(files)}")
    print(f"  Chunks:     {inserted}")
    print(f"  Time:       {dt:.1f}s")
    if dt > 0:
        print(f"  Rate:       {inserted/dt:.0f} chunks/s")
    print(f"  DB:         {db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SQLite FTS5 index from a folder of lex.uz HTML pages")
    ap.add_argument("--html-dir", required=True, type=Path, help="Folder with downloaded HTML files (recursive)")
    ap.add_argument("--db", required=True, type=Path, help="Output SQLite DB path")
    ap.add_argument(
        "--url-template",
        default="https://lex.uz/ru/docs/{doc_id}",
        help="URL template if you want citations to point to lex.uz (default: ru/docs). Use {doc_id}.",
    )
    ap.add_argument("--chunk-words", type=int, default=350)
    ap.add_argument("--overlap-words", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="Delete existing tables first")
    args = ap.parse_args()

    if not args.html_dir.exists():
        print(f"HTML dir not found: {args.html_dir}", file=sys.stderr)
        raise SystemExit(2)

    conn = sqlite3.connect(str(args.db))
    try:
        # Fail fast if FTS5 is missing.
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS __fts5_test")
        conn.commit()
    finally:
        conn.close()

    if args.reset and args.db.exists():
        conn = sqlite3.connect(str(args.db))
        reset_db(conn)
        conn.close()

    index_folder(
        html_dir=args.html_dir,
        db_path=args.db,
        url_template=args.url_template or None,
        max_words=args.chunk_words,
        overlap=args.overlap_words,
        batch_size=args.batch_size,
        limit_files=args.limit_files,
    )


if __name__ == "__main__":
    main()
