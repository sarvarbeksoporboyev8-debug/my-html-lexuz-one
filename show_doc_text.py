import sqlite3

DOC_ID_PART = "7934915"  # put part of ID or leave as-is

conn = sqlite3.connect("data/lexuz.sqlite3")
cur = conn.cursor()

# Find matching docs
cur.execute("""
SELECT doc_id, url, title, status, last_error
FROM documents
WHERE doc_id LIKE ? OR url LIKE ?
ORDER BY last_seen_at DESC
""", (f"%{DOC_ID_PART}%", f"%{DOC_ID_PART}%"))

docs = cur.fetchall()
if not docs:
    print("No matching documents found.")
    raise SystemExit(0)

for doc_id, url, title, status, last_error in docs[:5]:
    print("\n" + "="*100)
    print("doc_id:", doc_id)
    print("url:", url)
    print("title:", title)
    print("status:", status)
    print("last_error:", last_error)

    # Print chunks
    cur.execute("""
    SELECT chunk_index, text
    FROM chunks
    WHERE doc_id = ? AND active = 1
    ORDER BY chunk_index ASC
    """, (doc_id,))

    chunks = cur.fetchall()
    print(f"\nChunks: {len(chunks)}")
    for idx, text in chunks:
        print(f"\n--- chunk {idx} ---\n{text}")

conn.close()
