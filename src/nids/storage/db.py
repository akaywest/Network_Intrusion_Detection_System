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


def ensure_additions(con: sqlite3.Connection) -> list[str]:
    """Add the Phase 5 MODEL_RUN columns to a database created before them (plan §7.1).

    New databases get them from schema.sql. The ALTER statements are literal strings
    (NFR6); a column already present is left alone. Returns the columns added.
    """
    have = {r[1] for r in con.execute("PRAGMA table_info(model_run)")}
    added = []
    if "rate_changes" not in have:
        con.execute("ALTER TABLE model_run ADD COLUMN rate_changes TEXT")
        added.append("rate_changes")
    if "end_reason" not in have:
        con.execute("ALTER TABLE model_run ADD COLUMN end_reason TEXT CHECK (end_reason IN "
                    "('complete','stopped','disconnected','cancelled','shutdown'))")
        added.append("end_reason")
    if added:
        con.commit()
    return added
