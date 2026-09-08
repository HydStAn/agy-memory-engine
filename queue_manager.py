"""
Queue Manager for AGY Asynchronous Memory Pipeline
Handles thread-safe, fast SQLite turn queue in ~/.gemini/turn_queue.db
"""
import os
import sqlite3
import hashlib
from contextlib import contextmanager

from config import QUEUE_DB_PATH

_INITIALIZED_DBS = set()

def init_queue_db(db_path: str = QUEUE_DB_PATH):
    """Ensure the turn queue table exists with WAL mode and proper schema."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS turn_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE,
                source TEXT DEFAULT 'telegram',
                chat_id TEXT,
                user_prompt TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                extracted_summary TEXT,
                error TEXT,
                batch_id TEXT,
                processed_at TIMESTAMP
            );
        """)
        # Migrate existing table if missing batch_id or processed_at
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(turn_queue);")
        columns = [row[1] for row in cursor.fetchall()]
        if "batch_id" not in columns:
            conn.execute("ALTER TABLE turn_queue ADD COLUMN batch_id TEXT;")
        if "processed_at" not in columns:
            conn.execute("ALTER TABLE turn_queue ADD COLUMN processed_at TIMESTAMP;")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_queue_status ON turn_queue(status, id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_queue_batch ON turn_queue(batch_id);")
    _INITIALIZED_DBS.add(os.path.abspath(db_path))

def ensure_queue_db(db_path: str = QUEUE_DB_PATH):
    """Ensure queue DB schema is initialized once per db_path without running DDL on every call."""
    abs_path = os.path.abspath(db_path)
    if abs_path not in _INITIALIZED_DBS or not os.path.exists(abs_path):
        init_queue_db(db_path)

def reset_queue_db_guard(db_path: str = None):
    """Reset schema initialization guard (useful in test teardowns)."""
    if db_path:
        _INITIALIZED_DBS.discard(os.path.abspath(db_path))
    else:
        _INITIALIZED_DBS.clear()

def enqueue_turn(user_prompt: str, assistant_response: str, source: str = "telegram", chat_id: str = None, db_path: str = QUEUE_DB_PATH) -> bool:
    """Fast non-blocking insert of a turn into the queue (< 1ms)."""
    if not user_prompt or not user_prompt.strip():
        return False

    internal_markers = [
        "Multi-Layer Cognitive Memory Engine",
        "Du bist Stephans persönlicher autonomer KI-Assistent in Zürich für das Paket",
        "PROFIL Stephan:",
        "TPA BOT REPORT",
        "STATUS-SNAPSHOT [Paket:",
        "AGY Bot Integrity Watchdog"
    ]
    if any(m in user_prompt for m in internal_markers):
        return False

    ensure_queue_db(db_path)
    content_hash = hashlib.sha256(
        f"{source}:{chat_id}:{user_prompt.strip()}:{assistant_response.strip()}".encode("utf-8")
    ).hexdigest()
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            conn.execute("""
                INSERT INTO turn_queue (hash, source, chat_id, user_prompt, assistant_response, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(hash) DO NOTHING;
            """, (content_hash, source, str(chat_id) if chat_id is not None else None, user_prompt.strip(), assistant_response.strip()))
            conn.commit()
            return True
    except Exception:
        return False

def get_pending_stats(db_path: str = QUEUE_DB_PATH, retry_delay_seconds: int = 0) -> dict:
    """Return count and age in seconds of newest and oldest pending turn."""
    ensure_queue_db(db_path)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                count(*),
                COALESCE(strftime('%s', 'now') - strftime('%s', min(created_at)), 0),
                COALESCE(strftime('%s', 'now') - strftime('%s', max(created_at)), 0)
            FROM turn_queue
            WHERE status = 'pending'
              AND (error IS NULL OR processed_at IS NULL OR processed_at <= datetime('now', '-' || ? || ' seconds'))
        """, (retry_delay_seconds,))
        row = cursor.fetchone()
        return {
            "count": row[0] if row else 0,
            "oldest_age_seconds": row[1] if row and row[0] > 0 else 0,
            "newest_age_seconds": row[2] if row and row[0] > 0 else 0
        }

def get_pending_turns(limit: int = 25, db_path: str = QUEUE_DB_PATH, retry_delay_seconds: int = 0) -> list:
    """Fetch oldest pending turns for processing."""
    ensure_queue_db(db_path)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, source, chat_id, user_prompt, assistant_response, created_at
            FROM turn_queue
            WHERE status = 'pending'
              AND (error IS NULL OR processed_at IS NULL OR processed_at <= datetime('now', '-' || ? || ' seconds'))
            ORDER BY CASE WHEN error IS NULL THEN 0 ELSE 1 END, processed_at ASC, id ASC
            LIMIT ?
        """, (retry_delay_seconds, limit))
        rows = cursor.fetchall()
        return [{
            "id": r[0],
            "source": r[1],
            "chat_id": r[2],
            "user_prompt": r[3],
            "assistant_response": r[4],
            "created_at": r[5]
        } for r in rows]

def mark_turn_status(turn_ids: list, status: str, summary: str = None, error: str = None, batch_id: str = None, db_path: str = QUEUE_DB_PATH):
    """Update status of processed turns and assign batch_id and processed_at timestamp."""
    if not turn_ids:
        return
    ensure_queue_db(db_path)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        placeholders = ",".join("?" for _ in turn_ids)
        conn.execute(f"""
            UPDATE turn_queue
            SET status = ?, extracted_summary = ?, error = ?, batch_id = ?, processed_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
        """, [status, summary, error, batch_id] + list(turn_ids))
        conn.commit()

def prune_processed_turns(days: int = 7, db_path: str = QUEUE_DB_PATH):
    """Delete old processed / skipped items."""
    ensure_queue_db(db_path)
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.execute("""
                DELETE FROM turn_queue
                WHERE status IN ('processed', 'skipped')
                  AND datetime(created_at) < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.commit()
    except Exception:
        pass


def get_recent_turns(limit: int = 50, status: str = None, db_path: str = QUEUE_DB_PATH) -> list:
    """Fetch recent turns with optional status filter for dashboard inspection."""
    ensure_queue_db(db_path)
    with sqlite3.connect(db_path, timeout=5.0) as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute("""
                SELECT id, source, chat_id, user_prompt, assistant_response, datetime(created_at, 'localtime'), status, extracted_summary, error, batch_id, datetime(processed_at, 'localtime')
                FROM turn_queue
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT id, source, chat_id, user_prompt, assistant_response, datetime(created_at, 'localtime'), status, extracted_summary, error, batch_id, datetime(processed_at, 'localtime')
                FROM turn_queue
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()
        return [{
            "id": r[0],
            "source": r[1],
            "chat_id": r[2],
            "user_prompt": r[3],
            "assistant_response": r[4],
            "created_at": r[5],
            "status": r[6],
            "extracted_summary": r[7],
            "error": r[8],
            "batch_id": r[9],
            "processed_at": r[10]
        } for r in rows]

