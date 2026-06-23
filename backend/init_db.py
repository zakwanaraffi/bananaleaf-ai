import sqlite3
import os

# path database
db_path = "backend/database/inspection.db"

# 🔥 bikin folder kalau belum ada
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# connect
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_name TEXT,
    image_path TEXT,
    class_name TEXT,
    confidence REAL,
    solution TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print(" Database berhasil dibuat tanpa error!")