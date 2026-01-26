import sqlite3

DB_PATH = "data/lexuz.sqlite3"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("\n=== TABLES ===")
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
for (name,) in cur.fetchall():
    print("-", name)

def print_schema(table: str):
    print(f"\n=== SCHEMA: {table} ===")
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    for cid, name, ctype, notnull, dflt, pk in rows:
        print(f"{cid:>2}  {name:<24} {ctype:<12}  notnull={notnull}  pk={pk}  default={dflt}")

for t in ["documents", "doc_queue", "chunks", "vectors"]:
    print_schema(t)

conn.close()
