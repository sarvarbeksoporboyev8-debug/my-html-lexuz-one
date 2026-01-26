#!/usr/bin/env python3
"""local_fts_search.py

Query a SQLite FTS5 index built by `local_fts_index.py`.

Examples:
  python local_fts_search.py --db ./data/lexuz_fts.sqlite3 "QQS stavkasi"
  python local_fts_search.py --db ./data/lexuz_fts.sqlite3 "yillik ta'til muddati" --k 8
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_APOSTROPHES = "\u02bb\u02bc\u2018\u2019\u201b\u2032\u00b4\u0060"


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(f"[{_APOSTROPHES}]", "'", s)
    s = s.replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_match_query(user_query: str) -> str:
    """Turn a user query into a safer FTS MATCH string.

    Keep it simple - just normalize and return tokens without wildcards.
    FTS5 handles stemming/matching well on its own.
    """
    q = normalize_text(user_query)
    q = re.sub(r"[^0-9A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    tokens = [t for t in q.split(" ") if t]
    if not tokens:
        return ""

    # Just return tokens as-is, no wildcards
    return " ".join(tokens)


@dataclass
class Hit:
    doc_id: str
    url: str
    title: str
    chunk_index: int
    snippet: str
    rank: float
    text: Optional[str] = None


class LexUZFTSSearcher:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def search(self, query: str, k: int = 8, per_doc: int = 2, candidate_k: int = 60, with_text: bool = False) -> List[Hit]:
        match = build_match_query(query)
        if not match:
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Pull more candidates, then enforce per-doc diversity via a window function.
        # bm25(): smaller is better.
        sql = """
        WITH ranked AS (
            SELECT
                rowid,
                doc_id,
                url,
                title,
                chunk_index,
                snippet(chunks_fts, 4, '[', ']', '…', 18) AS snippet,
                bm25(chunks_fts, 0.0, 0.0, 6.0, 0.0, 1.0) AS rank,
                text
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ),
        dedup AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY rank) AS rn
            FROM ranked
        )
        SELECT doc_id, url, title, chunk_index, snippet, rank, text
        FROM dedup
        WHERE rn <= ?
        ORDER BY rank
        LIMIT ?;
        """

        rows = cur.execute(sql, (match, int(candidate_k), int(per_doc), int(k))).fetchall()
        conn.close()

        hits: List[Hit] = []
        for r in rows:
            hits.append(
                Hit(
                    doc_id=str(r["doc_id"]),
                    url=str(r["url"]),
                    title=str(r["title"]),
                    chunk_index=int(r["chunk_index"]),
                    snippet=str(r["snippet"]),
                    rank=float(r["rank"]),
                    text=str(r["text"]) if with_text else None,
                )
            )
        return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Search LexUZ SQLite FTS index")
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("query", nargs="+", help="Query text")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per-doc", type=int, default=2)
    args = ap.parse_args()

    q = " ".join(args.query)
    s = LexUZFTSSearcher(args.db)
    hits = s.search(q, k=args.k, per_doc=args.per_doc, with_text=False)

    if not hits:
        print("No matches.")
        return

    for i, h in enumerate(hits, 1):
        print("\n" + "=" * 90)
        print(f"#{i}  rank={h.rank:.4f}  doc_id={h.doc_id}  chunk={h.chunk_index}")
        print(h.title)
        print(h.url)
        print("\n" + h.snippet)


if __name__ == "__main__":
    main()
