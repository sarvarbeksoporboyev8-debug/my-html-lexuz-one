import sqlite3

conn = sqlite3.connect("data/lexuz.sqlite3")
cur = conn.cursor()

cur.execute("""
SELECT d.doc_id, d.url, COUNT(c.chunk_id) AS chunks
FROM documents d
LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.active = 1
GROUP BY d.doc_id, d.url
ORDER BY chunks DESC
""")

for row in cur.fetchall():
    print(row)

conn.close()
