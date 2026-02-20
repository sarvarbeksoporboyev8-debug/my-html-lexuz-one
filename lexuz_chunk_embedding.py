#!/usr/bin/env python3
"""
Lex.uz chunk-level embedding index: paragraphs or 100–150 word segments.

Use this so retrieval hits the exact paragraph (e.g. Reklama 32-modda) instead of
the whole document. Embed with sentence-transformers; retrieval by cosine similarity.
Optional later: fine-tune with BCE/contrastive on (question, correct_chunk) pairs.

Usage:
  # Build index from 50–60 docs (list of {doc_id, url, title, text})
  python lexuz_chunk_embedding.py build --docs-json ./data/lexuz_docs.json --output ./data/lexuz_chunk_index.sqlite3

  # Or build from existing lexuz.sqlite3 (chunks table) – uses existing chunk text, re-embeds
  python lexuz_chunk_embedding.py build --db ./data/lexuz.sqlite3 --output ./data/lexuz_chunk_index.sqlite3 --max-words 150

  # Search
  python lexuz_chunk_embedding.py search --index ./data/lexuz_chunk_index.sqlite3 "maktab binolarida reklama taqiqlanadi" --top 5
"""

import argparse
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path
from typing import List, Optional

# Default model (same as search_ask.py for consistency)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MAX_WORDS = 150
DEFAULT_OVERLAP_WORDS = 20

# Modda-aware chunking for lex.uz laws (same idea as search_ask.py)
MODDA_RE = re.compile(r"(\b\d+\s*-\s*modda\b|\b\d+\s*modda\b)", re.IGNORECASE | re.UNICODE)


def split_by_modda(text: str) -> List[str]:
    """Split lex.uz full text by modda boundaries. Returns list of chunks (each 'X-modda' + text)."""
    if not (text or "").strip():
        return []
    parts = MODDA_RE.split(text.strip())
    if len(parts) <= 1:
        return []
    chunks = []
    i = 1
    while i < len(parts):
        header = (parts[i] or "").strip()
        i += 1
        body = []
        while i < len(parts) and not MODDA_RE.match((parts[i] or "").strip()):
            body.append(parts[i] or "")
            i += 1
        block = (header + " " + " ".join(body)).strip()
        if block:
            chunks.append(block)
    return chunks


def chunk_by_paragraph_or_words(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[str]:
    """
    Split text into chunks: by paragraph first; long paragraphs into 100–150 word
    segments with overlap so the exact relevant part (e.g. one modda) is one chunk.
    """
    if not (text or "").strip():
        return []
    text = text.strip()
    # Paragraphs: split on double newline or more
    raw_paras = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    for para in raw_paras:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        nw = len(words)
        if nw <= max_words:
            chunks.append(para)
            continue
        # Long paragraph -> sliding windows of max_words with overlap
        start = 0
        while start < nw:
            end = min(start + max_words, nw)
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk.strip())
            if end >= nw:
                break
            start = max(0, end - overlap_words)
    return chunks


def chunk_lex_doc(
    text: str,
    strategy: str = "modda",
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[tuple]:
    """
    Chunk doc text. strategy "modda" = split by modda first (for laws); "paragraph" = paragraph/words.
    Returns list of (chunk_text, modda_label). modda_label is e.g. "32-modda" or "".
    """
    if strategy == "modda":
        modda_chunks = split_by_modda(text)
        if len(modda_chunks) >= 2:
            out = []
            for block in modda_chunks:
                label = ""
                m = MODDA_RE.search(block)
                if m:
                    label = m.group(1).strip()
                out.append((block, label))
            return out
    chunks = chunk_by_paragraph_or_words(text, max_words, overlap_words)
    return [(c, "") for c in chunks]


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Lazy-load sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(model, texts: List[str], batch_size: int = 32) -> List[bytes]:
    """Embed each text; return list of BLOBs (float32 little-endian)."""
    import numpy as np
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 50)
    out: List[bytes] = []
    for v in vecs:
        out.append(v.astype(np.float32).tobytes())
    return out


def build_index_from_documents(
    documents: List[dict],
    output_path: Path,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_strategy: str = "modda",
) -> int:
    """
    documents = [{"doc_id", "url", "title", "text"}, ...]
    Chunk each doc: chunk_strategy "modda" = modda-aligned (best for laws), "paragraph" = paragraph/words.
    Returns number of chunks indexed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[tuple] = []  # (doc_id, url, title, chunk_text, modda_label)
    for doc in documents:
        doc_id = doc.get("doc_id") or doc.get("url", "")
        url = doc.get("url", "")
        title = doc.get("title", "")
        text = doc.get("text", "")
        for chunk_text, modda_label in chunk_lex_doc(text, strategy=chunk_strategy, max_words=max_words, overlap_words=overlap_words):
            all_chunks.append((doc_id, url, title, chunk_text, modda_label or ""))

    if not all_chunks:
        print("No chunks produced from documents.")
        return 0

    print(f"Chunked {len(documents)} docs into {len(all_chunks)} chunks. Embedding...")
    model = get_embedder(model_name)
    texts = [c[3] for c in all_chunks]
    blobs = embed_texts(model, texts)

    conn = sqlite3.connect(str(output_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("DROP TABLE IF EXISTS chunk_embeddings")
    conn.execute("""
        CREATE TABLE chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            modda_label TEXT,
            chunk_text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
    """)
    conn.execute("DELETE FROM chunk_embeddings")
    for i, (doc_id, url, title, chunk_text, modda_label) in enumerate(all_chunks):
        conn.execute(
            "INSERT INTO chunk_embeddings(doc_id, url, title, modda_label, chunk_text, embedding) VALUES (?,?,?,?,?,?)",
            (doc_id, url, title or "", modda_label or "", chunk_text, blobs[i]),
        )
    conn.commit()
    conn.close()
    print(f"Wrote {len(all_chunks)} chunks to {output_path}")
    return len(all_chunks)


def build_index_from_db(
    db_path: Path,
    output_path: Path,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_strategy: str = "modda",
) -> int:
    """
    Read documents + chunks from existing lexuz.sqlite3 (documents + chunks tables).
    Re-chunk by modda or paragraph/words; then embed and write.
    """
    db_path = Path(db_path)
    output_path = Path(output_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT d.doc_id, d.url, d.title,
               (SELECT GROUP_CONCAT(c.text, char(10)) FROM chunks c WHERE c.doc_id = d.doc_id AND c.active = 1 ORDER BY c.chunk_index) AS full_text
        FROM documents d
        WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id AND c.active = 1)
    """).fetchall()
    conn.close()
    documents = [
        {"doc_id": r["doc_id"], "url": r["url"] or "", "title": r["title"] or "", "text": r["full_text"] or ""}
        for r in rows
    ]
    if not documents:
        print("No documents with chunks found in DB.")
        return 0
    return build_index_from_documents(
        documents,
        output_path,
        max_words=max_words,
        overlap_words=overlap_words,
        model_name=model_name,
        chunk_strategy=chunk_strategy,
    )


def search_chunks(
    question: str,
    index_path: Path,
    top_k: int = 10,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> List[dict]:
    """
    Embed question, cosine similarity with all chunk embeddings, return top_k chunks
    with keys: chunk_text, url, title, doc_id, score.
    """
    import numpy as np
    index_path = Path(index_path)
    if not index_path.exists():
        return []
    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT chunk_id, doc_id, url, title, chunk_text, embedding FROM chunk_embeddings").fetchall()
    conn.close()
    if not rows:
        return []
    model = get_embedder(model_name)
    q_vec = model.encode(question, convert_to_numpy=True)
    q_vec = q_vec.astype(np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-9:
        return []
    scores = []
    for r in rows:
        blob = r["embedding"]
        d_vec = np.frombuffer(blob, dtype=np.float32)
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-9:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(q_vec, d_vec) / (q_norm * d_norm)))
    idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "chunk_text": rows[i]["chunk_text"],
            "url": rows[i]["url"],
            "title": rows[i]["title"],
            "doc_id": rows[i]["doc_id"],
            "score": float(scores[idx[j]]),
        }
        for j, i in enumerate(idx)
    ]


def main():
    ap = argparse.ArgumentParser(description="Lex.uz chunk embedding index: build or search")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # build from JSON
    build = sub.add_parser("build", help="Build index from documents")
    build.add_argument("--docs-json", type=Path, help="JSON file: list of {doc_id, url, title, text}")
    build.add_argument("--db", type=Path, help="Or: existing lexuz.sqlite3 with documents+chunks")
    build.add_argument("--output", type=Path, required=True, help="Output SQLite path (chunk_embeddings)")
    build.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="Max words per chunk (paragraph or window)")
    build.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS, help="Overlap for long paragraphs")
    build.add_argument("--chunk-strategy", type=str, default="modda", choices=("modda", "paragraph"), help="modda = split by X-modda (best for laws); paragraph = paragraph/words")
    build.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    # search
    search = sub.add_parser("search", help="Search index by question")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("query", nargs="+", help="Question text")
    search.add_argument("--top", type=int, default=5)
    search.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    args = ap.parse_args()

    if args.cmd == "build":
        chunk_strategy = getattr(args, "chunk_strategy", "modda")  # --chunk-strategy
        if args.docs_json and args.docs_json.exists():
            with open(args.docs_json, "r", encoding="utf-8") as f:
                documents = json.load(f)
            n = build_index_from_documents(
                documents,
                args.output,
                max_words=args.max_words,
                overlap_words=args.overlap_words,
                model_name=args.model,
                chunk_strategy=chunk_strategy,
            )
        elif args.db and args.db.exists():
            n = build_index_from_db(
                args.db,
                args.output,
                max_words=args.max_words,
                overlap_words=args.overlap_words,
                model_name=args.model,
                chunk_strategy=chunk_strategy,
            )
        else:
            print("Provide --docs-json or --db (existing lexuz.sqlite3).")
            sys.exit(1)
        print(f"Indexed {n} chunks.")
        return

    if args.cmd == "search":
        q = " ".join(args.query)
        hits = search_chunks(q, args.index, top_k=args.top, model_name=args.model)
        for i, h in enumerate(hits, 1):
            print("\n" + "=" * 80)
            print(f"#{i}  score={h['score']:.4f}  {h['title'][:60]}")
            print(h["url"])
            print(h["chunk_text"][:500] + ("..." if len(h["chunk_text"]) > 500 else ""))


if __name__ == "__main__":
    main()
