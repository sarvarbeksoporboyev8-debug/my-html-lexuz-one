#!/usr/bin/env python3
"""
Build FTS5 index from existing chunks table in lexuz.sqlite3

This uses the chunks you already have from backfill - no need to re-download HTML files.

Usage:
    python build_fts_from_chunks.py --source ./data/lexuz.sqlite3 --output ./data/lexuz_fts.sqlite3
    
Or add FTS to the same database:
    python build_fts_from_chunks.py --source ./data/lexuz.sqlite3 --in-place
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path


_APOSTROPHES = "\u02bb\u02bc\u2018\u2019\u201b\u2032\u00b4\u0060"


def normalize_text(s: str) -> str:
    """Normalize Uzbek text for better FTS matching."""
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(f"[{_APOSTROPHES}]", "'", s)
    s = s.replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_fts_index(source_db: Path, output_db: Path, in_place: bool = False) -> None:
    """Build FTS5 index from chunks table."""
    
    if in_place:
        output_db = source_db
        print(f"Adding FTS index to: {source_db}")
    else:
        print(f"Source DB: {source_db}")
        print(f"Output DB: {output_db}")
    
    # Connect to source
    src_conn = sqlite3.connect(str(source_db))
    src_conn.row_factory = sqlite3.Row
    
    # Check chunks exist
    count = src_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Found {count} chunks in source database")
    
    if count == 0:
        print("No chunks to index!")
        return
    
    # Connect to output (same or different)
    if in_place:
        out_conn = src_conn
    else:
        output_db.parent.mkdir(parents=True, exist_ok=True)
        out_conn = sqlite3.connect(str(output_db))
    
    out_conn.execute("PRAGMA journal_mode=WAL;")
    out_conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Drop existing FTS table if exists
    out_conn.execute("DROP TABLE IF EXISTS chunks_fts")
    
    # Create FTS5 table
    out_conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            doc_id      UNINDEXED,
            url         UNINDEXED,
            title,
            chunk_index UNINDEXED,
            text,
            tokenize = 'unicode61',
            prefix = '2 3 4'
        );
    """)
    out_conn.commit()
    
    # Get chunks with document info
    print("Building FTS index...")
    t0 = time.time()
    
    cursor = src_conn.execute("""
        SELECT 
            c.chunk_id,
            c.doc_id,
            d.url,
            d.title,
            c.chunk_index,
            c.text
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        ORDER BY c.doc_id, c.chunk_index
    """)
    
    batch = []
    inserted = 0
    
    for row in cursor:
        doc_id = row["doc_id"]
        url = row["url"] or f"https://lex.uz/docs/{doc_id}"
        title = normalize_text(row["title"] or "")
        chunk_index = row["chunk_index"]
        text = normalize_text(row["text"] or "")
        
        if text:
            batch.append((doc_id, url, title, chunk_index, text))
        
        if len(batch) >= 2000:
            out_conn.executemany(
                "INSERT INTO chunks_fts(doc_id, url, title, chunk_index, text) VALUES (?,?,?,?,?)",
                batch
            )
            out_conn.commit()
            inserted += len(batch)
            print(f"  Indexed {inserted} chunks...")
            batch = []
    
    if batch:
        out_conn.executemany(
            "INSERT INTO chunks_fts(doc_id, url, title, chunk_index, text) VALUES (?,?,?,?,?)",
            batch
        )
        out_conn.commit()
        inserted += len(batch)
    
    # Optimize
    print("Optimizing FTS index...")
    out_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    out_conn.commit()
    
    dt = time.time() - t0
    print(f"\nDone!")
    print(f"  Chunks indexed: {inserted}")
    print(f"  Time: {dt:.1f}s")
    print(f"  Output: {output_db}")
    
    if not in_place:
        out_conn.close()
    src_conn.close()


def main():
    ap = argparse.ArgumentParser(description="Build FTS5 index from existing chunks table")
    ap.add_argument("--source", required=True, type=Path, help="Source SQLite DB with chunks table")
    ap.add_argument("--output", type=Path, help="Output SQLite DB for FTS index")
    ap.add_argument("--in-place", action="store_true", help="Add FTS table to source DB instead of creating new file")
    args = ap.parse_args()
    
    if not args.source.exists():
        print(f"Source DB not found: {args.source}")
        raise SystemExit(1)
    
    if not args.in_place and not args.output:
        print("Either --output or --in-place is required")
        raise SystemExit(1)
    
    build_fts_index(args.source, args.output or args.source, args.in_place)


if __name__ == "__main__":
    main()
