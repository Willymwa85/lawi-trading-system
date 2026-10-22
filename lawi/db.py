"""
LAWI Trading System - Database Layer
SQLite storage for predictions and accuracy tracking.
"""

import sqlite3
import logging
from datetime import datetime
from .config import DB_PATH

logger = logging.getLogger(__name__)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                bias        INTEGER NOT NULL,
                label       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                confluence  INTEGER NOT NULL,
                probabilities TEXT,
                sent_telegram INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS accuracy_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                prediction  TEXT NOT NULL,
                actual      TEXT,
                correct     INTEGER,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS training_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at  TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                accuracy    REAL,
                cv_accuracy REAL,
                n_samples   INTEGER,
                notes       TEXT
            );
        """)
    logger.info("Database initialised.")


def save_prediction(symbol: str, result: dict) -> int:
    """Save a prediction result. Returns the row ID."""
    import json
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (created_at, symbol, bias, label, confidence, confluence, probabilities, sent_telegram)
               VALUES (?,?,?,?,?,?,?,0)""",
            (
                now,
                symbol,
                result.get("bias", -1),
                result.get("label", "UNKNOWN"),
                result.get("confidence", 0.0),
                result.get("confluence", 0),
                json.dumps(result.get("probabilities", {})),
            )
        )
        return cur.lastrowid


def mark_sent(row_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE predictions SET sent_telegram=1 WHERE id=?", (row_id,))


def get_recent_predictions(symbol: str = None, limit: int = 30) -> list:
    with get_conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE symbol=? ORDER BY id DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_accuracy_stats(symbol: str = None) -> dict:
    """Return accuracy statistics from logged outcomes."""
    with get_conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM accuracy_log WHERE symbol=? AND correct IS NOT NULL",
                (symbol,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM accuracy_log WHERE correct IS NOT NULL"
            ).fetchall()

    if not rows:
        return {"total": 0, "correct": 0, "accuracy_pct": 0.0}

    total   = len(rows)
    correct = sum(1 for r in rows if r["correct"] == 1)
    return {
        "total":        total,
        "correct":      correct,
        "accuracy_pct": round(correct / total * 100, 2),
    }


def save_training_log(symbol: str, result: dict):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO training_log
               (trained_at, symbol, accuracy, cv_accuracy, n_samples, notes)
               VALUES (?,?,?,?,?,?)""",
            (
                now,
                symbol,
                result.get("accuracy"),
                result.get("cv_accuracy"),
                result.get("n_samples"),
                result.get("error", ""),
            )
        )
