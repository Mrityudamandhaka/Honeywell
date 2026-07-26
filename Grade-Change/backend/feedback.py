"""
Feedback logging and retrieval for operator Accept/Reject decisions.

Every suggestion the operator accepts or rejects is written to SQLite
immediately — not buffered in session state.  The log is the longitudinal
record used to measure and improve suggestion quality over time.
"""

from datetime import datetime, timezone
import pandas as pd
from backend.db import get_connection


def log_feedback(event_id, suggestion_type, suggestion_detail, source, decision):
    """
    Write a single operator decision to the feedback log.

    Parameters
    ----------
    event_id : int
    suggestion_type : str   e.g. 'setpoint_recommendation'
    suggestion_detail : str  human-readable text of what was suggested
    source : str             rationale/source tag from ai/recommend.py
    decision : str           'accepted' or 'rejected'
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO feedback
                (event_id, suggestion_type, suggestion_detail, source, decision, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, suggestion_type, suggestion_detail, source, decision, ts),
        )
        conn.commit()
    finally:
        conn.close()


def get_feedback_log():
    """
    Retrieve all feedback rows as a DataFrame.

    Returns
    -------
    pd.DataFrame  (empty DataFrame with schema if no rows yet)
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT id, event_id, suggestion_type, suggestion_detail, source, decision, timestamp "
            "FROM feedback ORDER BY timestamp DESC",
            conn,
        )
    finally:
        conn.close()
    return df


def acceptance_rate():
    """
    Compute the fraction of logged suggestions that were accepted.

    Returns
    -------
    float  between 0 and 1, or None if no feedback logged yet.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN decision='accepted' THEN 1 ELSE 0 END) as accepted "
            "FROM feedback"
        ).fetchone()
    finally:
        conn.close()

    total, accepted = row
    if not total:
        return None
    return accepted / total
