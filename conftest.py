"""Test configuration ensuring repository root is on ``sys.path``."""
from __future__ import annotations

import sys
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure a test-friendly database with a bookings table exists
DB_PATH = ROOT / "data" / "db.sqlite"
DB_PATH.parent.mkdir(exist_ok=True)
with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            amount INTEGER
        )
        """
    )
    cur = conn.execute("SELECT COUNT(*) FROM bookings")
    if cur.fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO bookings (city, amount) VALUES (?, ?)",
            [("NYC", 100), ("London", 150), ("Tokyo", 200)],
        )
    conn.commit()
