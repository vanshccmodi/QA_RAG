import sqlite3

conn = sqlite3.connect("rag.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    document TEXT,
    chunk_id INTEGER,
    text TEXT,
    embedding TEXT
)
""")

conn.commit()
