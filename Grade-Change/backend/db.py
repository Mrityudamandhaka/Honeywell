"""
SQLite connection and schema initialization for the feedback persistence layer.

Creates db/feedback.db and the feedback table on first run.
Every Accept/Reject operator decision writes here immediately — this is the
mechanism for evaluating suggestion quality over time.
"""

import os
import sqlite3


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "feedback.db")

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         INTEGER NOT NULL,
    suggestion_type  TEXT    NOT NULL,
    suggestion_detail TEXT   NOT NULL,
    source           TEXT    NOT NULL,
    decision         TEXT    NOT NULL CHECK(decision IN ('accepted','rejected')),
    timestamp        TEXT    NOT NULL
);
"""


def get_connection():
    """
    Return a sqlite3 connection to the feedback database, creating the db
    directory and table schema if they don't already exist.

    Returns
    -------
    sqlite3.Connection
    """
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")   # safer for concurrent web reads
    conn.execute(CREATE_FEEDBACK_TABLE)
    conn.commit()
    return conn
