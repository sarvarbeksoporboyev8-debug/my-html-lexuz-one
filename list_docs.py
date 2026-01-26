import sqlite3

conn = sqlite3.connect("data/lexuz.sqlite3")
cur = conn.cursor()

cur.execute("""
SELECT doc_id, url, status, last_error, first_seen_at, last_seen_at, last_fetched_at
FROM documents
ORDER BY last_seen_at DESC
""")

rows = cur.fetchall()
print(f"Total documents: {len(rows)}\n")
for r in rows:
    print(r)

conn.close()