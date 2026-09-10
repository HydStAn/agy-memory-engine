"""
Queue Manager for AGY Asynchronous Memory Pipeline
Handles thread-safe, fast SQLite turn queue in ~/.gemini/turn_queue.db
"""
import os
import sqlite3
import hashlib
import uuid
from pathlib import Path
from contextlib import contextmanager

from config import QUEUE_DB_PATH

_INITIALIZED_DBS = set()
_DB_IDENTITIES = {}


def _resolve_path(db_path: str | Path) -> Path:
    """Resolve database path expanding user directory cleanly."""
    return Path(db_path).expanduser().resolve()


@contextmanager
def _get_connection(db_path: str | Path, timeout: float = 5.0, isolation_level: str | None = None):
    """Provide a SQLite connection that is deterministically closed upon context exit."""
    resolved = _resolve_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved), timeout=timeout, isolation_level=isolation_level)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
        yield conn
    finally:
        conn.close()


def _get_queue_identity_conn(conn: sqlite3.Connection) -> str:
    """Retrieve or initialize persistent unique identity for this queue database."""
    row = conn.execute("SELECT value FROM queue_meta WHERE key='queue_id'").fetchone()
    if not row or not row[0]:
        raise RuntimeError('Queue identity is missing; reinitialize queue schema')
    return row[0]


def get_queue_identity(db_path: str = QUEUE_DB_PATH) -> str:
    """Return persistent queue identity from queue_meta table."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=5.0) as conn:
        return _get_queue_identity_conn(conn)


def serialize_hash_tuple(*parts: str | None) -> bytes:
    """Collision-safe serialization of tuple elements using length prefixing."""
    buf = []
    for p in parts:
        if p is None:
            buf.append(b"-1:")
        else:
            b = p.encode("utf-8")
            buf.append(f"{len(b)}:".encode("ascii") + b)
    return b"".join(buf)


def make_content_hash(
    source: str,
    chat_id: str | None,
    user_prompt: str,
    assistant_response: str,
    event_id: str | None = None,
) -> str:
    """Produce deterministic, collision-safe hash for queue entry deduplication."""
    chat_str = str(chat_id) if chat_id is not None else None
    if event_id is not None:
        payload = serialize_hash_tuple("event", source, chat_str, str(event_id))
    else:
        payload = serialize_hash_tuple("content", source, chat_str, user_prompt.strip(), assistant_response.strip())
    return hashlib.sha256(payload).hexdigest()


def compute_batch_id(queue_id: str, turns: list[dict]) -> str:
    """Deterministic stable batch_id based on queue identity + turn ids/hashes."""
    sorted_turns = sorted(turns, key=lambda t: t["id"])
    sig_elements = []
    for t in sorted_turns:
        t_id = str(t["id"])
        t_hash = str(t.get("hash") or "")
        sig_elements.append(f"{t_id}:{t_hash}")
    raw = f"{queue_id}:" + ",".join(sig_elements)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"batch_{digest}"


class BatchClaim(dict):
    """Dictionary-backed structure representing a claimed queue batch."""

    @property
    def batch_id(self) -> str:
        return self["batch_id"]

    @property
    def lease_token(self) -> str:
        return self["lease_token"]

    @property
    def source(self) -> str:
        return self["source"]

    @property
    def chat_id(self) -> str | None:
        return self["chat_id"]

    @property
    def turns(self) -> list[dict]:
        return self["turns"]


def init_queue_db(db_path: str = QUEUE_DB_PATH):
    """Ensure turn queue schema exists and is migrated under BEGIN IMMEDIATE."""
    resolved = _resolve_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection(resolved, timeout=10.0, isolation_level=None) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("BEGIN IMMEDIATE")
        try:
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
                    processed_at TIMESTAMP,
                    lease_token TEXT,
                    lease_expires_at TIMESTAMP,
                    attempt_count INTEGER DEFAULT 0,
                    event_id TEXT
                );
            """)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(turn_queue);")
            existing_columns = {row[1] for row in cursor.fetchall()}

            for col_name, col_def in [
                ("batch_id", "TEXT"),
                ("processed_at", "TIMESTAMP"),
                ("lease_token", "TEXT"),
                ("lease_expires_at", "TIMESTAMP"),
                ("attempt_count", "INTEGER DEFAULT 0"),
                ("event_id", "TEXT"),
            ]:
                if col_name not in existing_columns:
                    conn.execute(f"ALTER TABLE turn_queue ADD COLUMN {col_name} {col_def};")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_queue_status ON turn_queue(status, id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_queue_batch ON turn_queue(batch_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_queue_lease ON turn_queue(status, lease_expires_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turn_queue_event ON turn_queue(event_id);")

            conn.execute("CREATE TABLE IF NOT EXISTS queue_meta (key TEXT PRIMARY KEY, value TEXT);")
            conn.execute("INSERT OR IGNORE INTO queue_meta(key,value) VALUES ('queue_id',?)", (uuid.uuid4().hex,))

            conn.execute("PRAGMA user_version = 2;")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    _INITIALIZED_DBS.add(str(resolved))
    _DB_IDENTITIES[str(resolved)] = (resolved.stat().st_dev, resolved.stat().st_ino)


def ensure_queue_db(db_path: str = QUEUE_DB_PATH):
    """Ensure queue DB schema is initialized without running DDL on steady-state hot paths."""
    resolved = _resolve_path(db_path)
    abs_path = str(resolved)
    if abs_path in _INITIALIZED_DBS and resolved.exists() and _DB_IDENTITIES.get(abs_path) == (resolved.stat().st_dev, resolved.stat().st_ino):
        return

    if resolved.exists():
        try:
            with _get_connection(resolved, timeout=5.0) as conn:
                version = conn.execute("PRAGMA user_version;").fetchone()[0]
                if version > 2:
                    raise RuntimeError(f"Unsupported queue schema {version}")
                if version == 2:
                    _INITIALIZED_DBS.add(abs_path)
                    _DB_IDENTITIES[abs_path] = (resolved.stat().st_dev,resolved.stat().st_ino)
                    return
        except sqlite3.DatabaseError:
            raise

    init_queue_db(db_path)


def reset_queue_db_guard(db_path: str = None):
    """Reset schema initialization guard (useful in test teardowns)."""
    if db_path:
        _INITIALIZED_DBS.discard(str(_resolve_path(db_path)))
    else:
        _INITIALIZED_DBS.clear()


def enqueue_turn(
    user_prompt: str,
    assistant_response: str,
    source: str = "telegram",
    chat_id: str = None,
    db_path: str = QUEUE_DB_PATH,
    event_id: str = None,
    **kwargs
) -> bool:
    """Fast insert of a turn into the queue with collision-safe deduplication."""
    if "event_id" in kwargs and event_id is None:
        event_id = kwargs["event_id"]
    if "db_path" in kwargs:
        db_path = kwargs["db_path"]

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
    content_hash = make_content_hash(
        source=source,
        chat_id=chat_id,
        user_prompt=user_prompt,
        assistant_response=assistant_response,
        event_id=event_id
    )

    try:
        with _get_connection(db_path, timeout=5.0, isolation_level=None) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # BR05: Atomically update unbatched pending turns on repeated Stop events,
                # or record a continuation revision if earlier turn was already claimed/processed.
                if event_id is not None:
                    row = conn.execute(
                        "SELECT id, status, batch_id, assistant_response, user_prompt FROM turn_queue WHERE hash = ?",
                        (content_hash,)
                    ).fetchone()
                    if row:
                        ex_id, ex_status, ex_batch_id, ex_resp, ex_prompt = row
                        clean_user = user_prompt.strip()
                        clean_resp = assistant_response.strip()
                        # Exact duplicate - deduplicate cleanly
                        if ex_resp == clean_resp and ex_prompt == clean_user:
                            conn.execute("COMMIT")
                            return True
                        # Turn is still pending and unbatched - atomically update with expanded response
                        if ex_status == 'pending' and ex_batch_id is None:
                            conn.execute(
                                "UPDATE turn_queue SET user_prompt = ?, assistant_response = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?",
                                (clean_user, clean_resp, ex_id)
                            )
                            conn.execute("COMMIT")
                            return True
                        # Turn was already claimed or processed: record a continuation revision
                        rev = 2
                        while True:
                            cont_event_id = f"{event_id}:rev{rev}"
                            cont_hash = make_content_hash(
                                source=source,
                                chat_id=chat_id,
                                user_prompt=user_prompt,
                                assistant_response=assistant_response,
                                event_id=cont_event_id
                            )
                            cont_row = conn.execute(
                                "SELECT id, status, batch_id, assistant_response FROM turn_queue WHERE hash = ?",
                                (cont_hash,)
                            ).fetchone()
                            if not cont_row:
                                conn.execute("""
                                    INSERT INTO turn_queue (hash, source, chat_id, user_prompt, assistant_response, status, event_id)
                                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                                """, (
                                    cont_hash,
                                    source,
                                    str(chat_id) if chat_id is not None else None,
                                    clean_user,
                                    clean_resp,
                                    cont_event_id
                                ))
                                conn.execute("COMMIT")
                                return True
                            if cont_row[3] == clean_resp:
                                conn.execute("COMMIT")
                                return True
                            if cont_row[1] == 'pending' and cont_row[2] is None:
                                conn.execute(
                                    "UPDATE turn_queue SET assistant_response = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?",
                                    (clean_resp, cont_row[0])
                                )
                                conn.execute("COMMIT")
                                return True
                            rev += 1

                conn.execute("""
                    INSERT INTO turn_queue (hash, source, chat_id, user_prompt, assistant_response, status, event_id)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    ON CONFLICT(hash) DO NOTHING;
                """, (
                    content_hash,
                    source,
                    str(chat_id) if chat_id is not None else None,
                    user_prompt.strip(),
                    assistant_response.strip(),
                    str(event_id) if event_id is not None else None
                ))
                conn.execute("COMMIT")
                return True
            except Exception:
                conn.execute("ROLLBACK")
                raise
    except sqlite3.OperationalError as error:
        if getattr(error, "sqlite_errorcode", 0) & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            return False
        raise


def claim_batch(
    batch_size: int = 25,
    lease_duration_seconds: int = 300,
    retry_delay_seconds: int = 60,
    prefer_fresh: bool = False,
    exclude_batch_ids: list[str] | set[str] | None = None,
    db_path: str = QUEUE_DB_PATH
) -> BatchClaim | None:
    """Atomically claim a batch of turns partitioned strictly by (source, chat_id).

    Supports interleaving fresh turns and retry claims, and skipping specified batch IDs.
    """
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=10.0, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            queue_id = _get_queue_identity_conn(conn)
            cursor = conn.cursor()

            def _try_claim_retries():
                exclude_sql = ""
                params = [retry_delay_seconds]
                if exclude_batch_ids:
                    ph = ",".join("?" for _ in exclude_batch_ids)
                    exclude_sql = f"AND batch_id NOT IN ({ph})"
                    params.extend(list(exclude_batch_ids))

                cursor.execute(f"""
                    SELECT batch_id, source, chat_id
                    FROM turn_queue
                    WHERE batch_id IS NOT NULL AND (
                        (status = 'claimed' AND lease_expires_at IS NOT NULL AND datetime(lease_expires_at) <= datetime('now'))
                        OR (status = 'pending' AND (processed_at IS NULL OR datetime(processed_at) <= datetime('now', '-' || ? || ' seconds')))
                      )
                      {exclude_sql}
                    ORDER BY lease_expires_at ASC, processed_at ASC
                    LIMIT 1;
                """, tuple(params))
                expired_row = cursor.fetchone()
                if not expired_row:
                    return None
                exp_batch_id, exp_source, exp_chat_id = expired_row
                cursor.execute("""
                    SELECT id, hash, source, chat_id, user_prompt, assistant_response, created_at, batch_id, attempt_count
                    FROM turn_queue
                    WHERE batch_id = ?
                    ORDER BY id ASC;
                """, (exp_batch_id,))
                rows = cursor.fetchall()
                states = {row[0] for row in conn.execute('SELECT status FROM turn_queue WHERE batch_id=?', (exp_batch_id,))}
                if not states <= {'pending', 'claimed'}:
                    raise RuntimeError(f'Batch {exp_batch_id} has mixed completion state; reconcile before retry')
                if rows:
                    turns = [{
                        "id": r[0],
                        "hash": r[1],
                        "source": r[2],
                        "chat_id": r[3],
                        "user_prompt": r[4],
                        "assistant_response": r[5],
                        "created_at": r[6],
                        "batch_id": r[7],
                        "attempt_count": r[8] or 0
                    } for r in rows]
                    new_lease_token = uuid.uuid4().hex
                    cursor.execute("""
                        UPDATE turn_queue
                        SET status = 'claimed', lease_token = ?,
                            lease_expires_at = datetime('now', '+' || ? || ' seconds'),
                            attempt_count = COALESCE(attempt_count, 0) + 1
                        WHERE batch_id = ?;
                    """, (new_lease_token, lease_duration_seconds, exp_batch_id))
                    conn.execute("COMMIT")
                    return BatchClaim({
                        "batch_id": exp_batch_id,
                        "lease_token": new_lease_token,
                        "source": exp_source,
                        "chat_id": exp_chat_id,
                        "turns": turns
                    })
                return None

            def _try_claim_fresh():
                cursor.execute("""
                    SELECT source, chat_id
                    FROM turn_queue
                    WHERE status = 'pending' AND batch_id IS NULL
                      AND (error IS NULL OR processed_at IS NULL OR datetime(processed_at) <= datetime('now', '-' || ? || ' seconds'))
                    ORDER BY CASE WHEN error IS NULL THEN 0 ELSE 1 END, processed_at ASC, id ASC
                    LIMIT 1;
                """, (retry_delay_seconds,))
                target_row = cursor.fetchone()
                if not target_row:
                    return None

                target_source, target_chat_id = target_row

                cursor.execute("""
                    SELECT id, hash, source, chat_id, user_prompt, assistant_response, created_at, batch_id, attempt_count
                    FROM turn_queue
                    WHERE status = 'pending' AND batch_id IS NULL
                      AND source IS ?
                      AND chat_id IS ?
                      AND (error IS NULL OR processed_at IS NULL OR datetime(processed_at) <= datetime('now', '-' || ? || ' seconds'))
                    ORDER BY id ASC
                    LIMIT ?;
                """, (target_source, target_chat_id, retry_delay_seconds, batch_size))
                rows = cursor.fetchall()
                if not rows:
                    return None

                turns = [{
                    "id": r[0],
                    "hash": r[1],
                    "source": r[2],
                    "chat_id": r[3],
                    "user_prompt": r[4],
                    "assistant_response": r[5],
                    "created_at": r[6],
                    "batch_id": r[7],
                    "attempt_count": r[8] or 0
                } for r in rows]

                batch_id = compute_batch_id(queue_id, turns)
                new_lease_token = uuid.uuid4().hex
                turn_ids = [t["id"] for t in turns]
                placeholders = ",".join("?" for _ in turn_ids)

                cursor.execute(f"""
                    UPDATE turn_queue
                    SET status = 'claimed',
                        batch_id = ?,
                        lease_token = ?,
                        lease_expires_at = datetime('now', '+' || ? || ' seconds'),
                        attempt_count = COALESCE(attempt_count, 0) + 1
                    WHERE id IN ({placeholders})
                      AND status = 'pending'
                      AND batch_id IS NULL;
                """, [batch_id, new_lease_token, lease_duration_seconds] + turn_ids)

                if cursor.rowcount != len(turn_ids):
                    conn.execute("ROLLBACK")
                    return None

                for t in turns:
                    t["batch_id"] = batch_id
                    t["attempt_count"] = (t["attempt_count"] or 0) + 1

                conn.execute("COMMIT")
                return BatchClaim({
                    "batch_id": batch_id,
                    "lease_token": new_lease_token,
                    "source": target_source,
                    "chat_id": target_chat_id,
                    "turns": turns
                })

            if prefer_fresh:
                claim = _try_claim_fresh()
                if claim:
                    return claim
                claim = _try_claim_retries()
                if claim:
                    return claim
            else:
                claim = _try_claim_retries()
                if claim:
                    return claim
                claim = _try_claim_fresh()
                if claim:
                    return claim

            conn.execute("COMMIT")
            return None
        except Exception:
            conn.execute("ROLLBACK")
            raise


def acknowledge_batch(
    batch_id: str,
    lease_token: str,
    status: str = "processed",
    summary: str = None,
    db_path: str = QUEUE_DB_PATH
) -> bool:
    """Acknowledge a successfully processed batch, preventing stale lease owners from acknowledging."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=10.0, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE turn_queue
                SET status = ?,
                    extracted_summary = ?,
                    error = NULL,
                    processed_at = CURRENT_TIMESTAMP,
                    lease_token = NULL,
                    lease_expires_at = NULL
                WHERE batch_id = ?
                  AND lease_token = ?
                  AND status = 'claimed'
                  AND datetime(lease_expires_at) >= datetime('now');
            """, (status, summary, batch_id, lease_token))
            updated = cursor.rowcount > 0
            conn.execute("COMMIT")
            return updated
        except Exception:
            conn.execute("ROLLBACK")
            raise


def release_batch(
    batch_id: str,
    lease_token: str,
    error: str = None,
    db_path: str = QUEUE_DB_PATH
) -> bool:
    """Release a claimed batch on failure, returning turns to pending with backoff."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=10.0, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE turn_queue
                SET status = CASE WHEN COALESCE(attempt_count, 0) >= 10 THEN 'failed' ELSE 'pending' END,
                    error = ?,
                    processed_at = CURRENT_TIMESTAMP,
                    lease_token = NULL,
                    lease_expires_at = NULL
                WHERE batch_id = ?
                  AND lease_token = ?
                  AND status = 'claimed';
            """, (error, batch_id, lease_token))
            updated = cursor.rowcount > 0
            conn.execute("COMMIT")
            return updated
        except Exception:
            conn.execute("ROLLBACK")
            raise


def get_pending_stats(db_path: str = QUEUE_DB_PATH, retry_delay_seconds: int = 0) -> dict:
    """Return count and age in seconds of newest and oldest pending turn."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                count(*),
                COALESCE(strftime('%s', 'now') - strftime('%s', min(created_at)), 0),
                COALESCE(strftime('%s', 'now') - strftime('%s', max(created_at)), 0)
            FROM turn_queue
            WHERE (status = 'pending' OR (status = 'claimed' AND lease_expires_at IS NOT NULL AND datetime(lease_expires_at) <= datetime('now')))
              AND (error IS NULL OR processed_at IS NULL OR datetime(processed_at) <= datetime('now', '-' || ? || ' seconds'))
        """, (retry_delay_seconds,))
        row = cursor.fetchone()
        return {
            "count": row[0] if row else 0,
            "oldest_age_seconds": row[1] if row and row[0] > 0 else 0,
            "newest_age_seconds": row[2] if row and row[0] > 0 else 0
        }


def get_pending_turns(limit: int = 25, db_path: str = QUEUE_DB_PATH, retry_delay_seconds: int = 0) -> list:
    """Fetch oldest pending turns for processing or inspection."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=5.0) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, source, chat_id, user_prompt, assistant_response, created_at, batch_id, status, error, hash
            FROM turn_queue
            WHERE (status = 'pending' OR (status = 'claimed' AND lease_expires_at IS NOT NULL AND datetime(lease_expires_at) <= datetime('now')))
              AND (error IS NULL OR processed_at IS NULL OR datetime(processed_at) <= datetime('now', '-' || ? || ' seconds'))
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
            "created_at": r[5],
            "batch_id": r[6],
            "status": r[7],
            "error": r[8],
            "hash": r[9],
        } for r in rows]


def mark_turn_status(
    turn_ids: list,
    status: str,
    summary: str = None,
    error: str = None,
    batch_id: str = None,
    lease_token: str = None,
    db_path: str = QUEUE_DB_PATH
):
    """Update status of processed turns, clearing lease fields if completed."""
    if not turn_ids:
        return
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=5.0, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            placeholders = ",".join("?" for _ in turn_ids)
            sql = f"""
                UPDATE turn_queue
                SET status = ?, extracted_summary = ?, error = ?, batch_id = ?, processed_at = CURRENT_TIMESTAMP,
                    lease_token = CASE WHEN ? IN ('processed', 'skipped', 'pending') THEN NULL ELSE lease_token END,
                    lease_expires_at = CASE WHEN ? IN ('processed', 'skipped', 'pending') THEN NULL ELSE lease_expires_at END
                WHERE id IN ({placeholders})
            """
            params = [status, summary, error, batch_id, status, status] + list(turn_ids)
            if lease_token is not None:
                sql += " AND lease_token = ?"
                params.append(lease_token)
            else:
                sql += " AND status != 'claimed'"
            conn.execute(sql, params)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def prune_processed_turns(days: int = 7, db_path: str = QUEUE_DB_PATH) -> int:
    """Delete old processed / skipped items based on processed_at (fallback to created_at)."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=5.0, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM turn_queue
                WHERE status IN ('processed', 'skipped')
                  AND datetime(COALESCE(processed_at, created_at)) < datetime('now', '-' || ? || ' days')
            """, (days,))
            count = cursor.rowcount
            conn.execute("COMMIT")
            return count
        except Exception:
            conn.execute("ROLLBACK")
            raise


def get_recent_turns(limit: int = 50, status: str = None, db_path: str = QUEUE_DB_PATH) -> list:
    """Fetch recent turns with optional status filter for dashboard inspection."""
    ensure_queue_db(db_path)
    with _get_connection(db_path, timeout=5.0) as conn:
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
