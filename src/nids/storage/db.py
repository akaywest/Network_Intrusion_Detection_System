"""Connection helper for the NIDS log database (M5; plan §5.1, Chapter 3 §3.3.9).

foreign_keys and synchronous are per-connection PRAGMAs in SQLite, so every writer
connection goes through connect(). journal_mode=WAL is persistent and is set once by
schema.sql. The synchronous level is fixed for every run (plan §5.1) and recorded in
TEST_ENVIRONMENT.run_conditions by nids.storage.environment.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# PRAGMA synchronous level used for every run (plan §5.1). NORMAL is safe in WAL mode
# and is the value recorded in run_conditions. The PRAGMA statements below are literal
# strings: PRAGMA values cannot be bound as parameters, so they are not built at runtime.
SYNCHRONOUS = "NORMAL"


def connect(db_path: str | Path, *, autocommit: bool = False) -> sqlite3.Connection:
    """Open a writer connection with foreign keys ON and synchronous=NORMAL.

    autocommit=True sets isolation_level=None so the caller issues BEGIN/COMMIT
    explicitly (LatencyLogger, plan §5.3).
    """
    con = sqlite3.connect(str(db_path), isolation_level=None if autocommit else "")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA synchronous = NORMAL")
    return con
