"""
AGY Memory Engine - Self-Healing Vector Index Synchronization Module

Provides transactional outbox processing, revision and generation fencing,
lease-based worker concurrency, and freshness verification for sqlite-vec tables.
"""

import sys
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from config import DB_PATH, EMBEDDING_DIM, EMBEDDING_MODEL_NAME, VECTOR_SEARCH_ENABLED
from schema import (
    db_session,
    compute_model_fingerprint,
    get_db_generation,
    HAS_SQLITE_VEC,
)
from embedder import build_text_repr, embed_text

TABLE_TYPE_MAP = {
    "memories": "fact",
    "episodes": "episode",
    "learnings": "learning",
}

TYPE_TABLE_MAP = {
    "fact": "memories",
    "episode": "episodes",
    "learning": "learnings",
}


def get_active_model_fingerprint(conn=None) -> str:
    """Retrieve active model fingerprint from vector_index_config or calculate default."""
    if conn is not None:
        try:
            row = conn.execute("SELECT fingerprint FROM vector_index_config WHERE id = 1").fetchone()
            if row and row[0]:
                return row[0]
        except Exception:
            pass
    return compute_model_fingerprint(EMBEDDING_MODEL_NAME, EMBEDDING_DIM, "v1")


def build_canonical_text(conn, entity_type: str, entity_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Hydrate source record and build its canonical text representation along with sha256 hash.
    Returns (text_repr, text_hash) or (None, None) if entity does not exist.
    """
    cursor = conn.cursor()
    if entity_type == "memories":
        cursor.execute("SELECT category, fact, keywords FROM memories WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if not row:
            return None, None
        text = build_text_repr("fact", {"category": row[0], "fact": row[1], "keywords": row[2]})
    elif entity_type == "episodes":
        cursor.execute("SELECT topic, title, narrative, stance, keywords FROM episodes WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if not row:
            return None, None
        text = build_text_repr("episode", {
            "topic": row[0],
            "title": row[1],
            "narrative": row[2],
            "stance": row[3] or "neutral",
            "keywords": row[4],
        })
    elif entity_type == "learnings":
        cursor.execute("SELECT category, insight, context, keywords FROM learnings WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        if not row:
            return None, None
        text = build_text_repr("learning", {"category": row[0], "insight": row[1], "context": row[2], "keywords": row[3]})
    else:
        return None, None

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, text_hash


def claim_vector_jobs(conn, batch_size: int = 25, lease_seconds: int = 60) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Atomically claim due or expired vector index jobs under a short transaction.
    Returns (lease_token, claimed_jobs).
    """
    lease_token = uuid.uuid4().hex
    claimed_jobs = []

    if not conn.in_transaction:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass

    cursor = conn.cursor()
    cursor.execute("""
        SELECT entity_type, entity_id, op, desired_revision, attempts
        FROM vector_index_jobs
        WHERE (status = 'pending' AND datetime(next_attempt_at) <= datetime('now'))
           OR (status = 'claimed' AND datetime(lease_expires_at) < datetime('now'))
        ORDER BY next_attempt_at ASC
        LIMIT ?
    """, (batch_size,))
    rows = cursor.fetchall()

    if not rows:
        return lease_token, []

    for entity_type, entity_id, op, desired_revision, attempts in rows:
        cursor.execute("""
            UPDATE vector_index_jobs
            SET status = 'claimed',
                lease_token = ?,
                lease_expires_at = datetime('now', ?),
                attempts = attempts + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity_type = ? AND entity_id = ?
        """, (lease_token, f"+{lease_seconds} seconds", entity_type, entity_id))
        claimed_jobs.append({
            "entity_type": entity_type,
            "entity_id": entity_id,
            "op": op,
            "desired_revision": desired_revision,
            "attempts": attempts + 1,
        })

    conn.commit()
    return lease_token, claimed_jobs


def publish_vector_job(
    conn,
    lease_token: str,
    entity_type: str,
    entity_id: str,
    embedding: np.ndarray,
    text_hash: str,
    expected_revision: int,
    expected_gen: str,
    model_fp: str,
) -> bool:
    """
    Atomically publish computed embedding to vec virtual table and update vector_index_state
    under strict lease, revision, generation, and existence fencing.
    """
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return False

    if not conn.in_transaction:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass

    cursor = conn.cursor()

    # 1. Verify generation fencing
    cur_gen = get_db_generation(conn)
    if cur_gen != expected_gen:
        conn.rollback()
        return False

    # 2. Verify lease and desired revision
    cursor.execute("""
        SELECT desired_revision, status
        FROM vector_index_jobs
        WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
    """, (entity_type, entity_id, lease_token))
    job = cursor.fetchone()
    if not job or job[1] != "claimed" or job[0] != expected_revision:
        conn.rollback()
        return False

    # 3. Verify entity existence and current revision
    cursor.execute(f"SELECT 1 FROM {entity_type} WHERE id = ?", (entity_id,))
    if not cursor.fetchone():
        conn.rollback()
        return False

    cursor.execute("""
        SELECT revision FROM entity_revisions WHERE entity_type = ? AND entity_id = ?
    """, (entity_type, entity_id))
    rev_row = cursor.fetchone()
    if not rev_row or rev_row[0] != expected_revision:
        conn.rollback()
        return False

    # 4. Atomically update vec virtual table
    vec_table = f"vec_{entity_type}"
    try:
        cursor.execute(f"DELETE FROM {vec_table} WHERE id = ?", (entity_id,))
        cursor.execute(f"INSERT INTO {vec_table} (id, embedding) VALUES (?, ?)", (entity_id, embedding))
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to write into {vec_table} for {entity_id}: {e}\n")
        conn.rollback()
        return False

    # 5. Record verified index state
    cursor.execute("""
        INSERT OR REPLACE INTO vector_index_state
            (entity_type, entity_id, indexed_revision, generation, model_fingerprint, text_hash, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (entity_type, entity_id, expected_revision, expected_gen, model_fp, text_hash))

    # 6. Complete and delete job
    cursor.execute("""
        DELETE FROM vector_index_jobs
        WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
    """, (entity_type, entity_id, lease_token))

    conn.commit()
    return True


def publish_vector_reuse(
    conn,
    lease_token: str,
    entity_type: str,
    entity_id: str,
    text_hash: str,
    expected_revision: int,
    expected_gen: str,
    model_fp: str,
) -> bool:
    """
    Reuse existing vector when text representation hash is identical, updating state revision without re-embedding.
    """
    if not conn.in_transaction:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass

    cursor = conn.cursor()

    cur_gen = get_db_generation(conn)
    if cur_gen != expected_gen:
        conn.rollback()
        return False

    cursor.execute("""
        SELECT desired_revision, status
        FROM vector_index_jobs
        WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
    """, (entity_type, entity_id, lease_token))
    job = cursor.fetchone()
    if not job or job[1] != "claimed" or job[0] != expected_revision:
        conn.rollback()
        return False

    cursor.execute(f"SELECT 1 FROM {entity_type} WHERE id = ?", (entity_id,))
    if not cursor.fetchone():
        conn.rollback()
        return False

    cursor.execute("""
        SELECT revision FROM entity_revisions WHERE entity_type = ? AND entity_id = ?
    """, (entity_type, entity_id))
    rev_row = cursor.fetchone()
    if not rev_row or rev_row[0] != expected_revision:
        conn.rollback()
        return False

    cursor.execute("""
        UPDATE vector_index_state
        SET indexed_revision = ?,
            generation = ?,
            model_fingerprint = ?,
            indexed_at = CURRENT_TIMESTAMP
        WHERE entity_type = ? AND entity_id = ?
    """, (expected_revision, expected_gen, model_fp, entity_type, entity_id))

    cursor.execute("""
        DELETE FROM vector_index_jobs
        WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
    """, (entity_type, entity_id, lease_token))

    conn.commit()
    return True


def publish_vector_delete(
    conn,
    lease_token: str,
    entity_type: str,
    entity_id: str,
    expected_revision: int,
    expected_gen: str,
) -> bool:
    """
    Atomically delete vector and state records under lease and revision fencing.
    """
    if not conn.in_transaction:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass

    cursor = conn.cursor()

    cur_gen = get_db_generation(conn)
    if cur_gen != expected_gen:
        conn.rollback()
        return False

    cursor.execute("""
        SELECT desired_revision, status
        FROM vector_index_jobs
        WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
    """, (entity_type, entity_id, lease_token))
    job = cursor.fetchone()
    if not job or job[1] != "claimed" or job[0] != expected_revision:
        conn.rollback()
        return False

    if HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED:
        vec_table = f"vec_{entity_type}"
        try:
            cursor.execute(f"DELETE FROM {vec_table} WHERE id = ?", (entity_id,))
        except Exception:
            pass

    cursor.execute("DELETE FROM vector_index_state WHERE entity_type = ? AND entity_id = ?", (entity_type, entity_id))
    cursor.execute("DELETE FROM vector_index_jobs WHERE entity_type = ? AND entity_id = ? AND lease_token = ?", (entity_type, entity_id, lease_token))

    conn.commit()
    return True


def fail_vector_job(
    conn,
    lease_token: str,
    entity_type: str,
    entity_id: str,
    error_msg: str,
    max_attempts: int = 5,
    backoff_base: int = 5,
) -> None:
    """Record job failure, backing off exponentially or marking blocked if max attempts reached."""
    if not conn.in_transaction:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            pass

    cursor = conn.cursor()
    cursor.execute("""
        SELECT attempts
        FROM vector_index_jobs
        WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
    """, (entity_type, entity_id, lease_token))
    row = cursor.fetchone()
    if not row:
        return

    attempts = row[0]
    if attempts >= max_attempts:
        cursor.execute("""
            UPDATE vector_index_jobs
            SET status = 'blocked',
                last_error = ?,
                lease_token = NULL,
                lease_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
        """, (error_msg, entity_type, entity_id, lease_token))
    else:
        delay_seconds = backoff_base * (2 ** max(0, attempts - 1))
        cursor.execute("""
            UPDATE vector_index_jobs
            SET status = 'pending',
                next_attempt_at = datetime('now', ?),
                last_error = ?,
                lease_token = NULL,
                lease_expires_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity_type = ? AND entity_id = ? AND lease_token = ?
        """, (f"+{delay_seconds} seconds", error_msg, entity_type, entity_id, lease_token))

    conn.commit()


def drain_vector_jobs(db_path: str = None, batch_size: int = 25, max_batches: int = 10) -> int:
    """
    Worker drain loop: claims jobs, computes embeddings outside locks, and publishes with fencing.
    Returns the total count of successfully published or cleaned up vector jobs.
    """
    target_db = db_path or DB_PATH
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return 0

    processed_count = 0
    batches_run = 0

    while batches_run < max_batches:
        batches_run += 1
        with db_session(target_db) as conn:
            cur_gen = get_db_generation(conn)
            active_fp = get_active_model_fingerprint(conn)
            lease_token, jobs = claim_vector_jobs(conn, batch_size=batch_size, lease_seconds=60)

        if not jobs:
            break

        # Process each job: compute outside locks, then publish in short transaction
        for job in jobs:
            entity_type = job["entity_type"]
            entity_id = job["entity_id"]
            op = job["op"]
            desired_revision = job["desired_revision"]

            if op == "delete":
                with db_session(target_db) as conn:
                    ok = publish_vector_delete(conn, lease_token, entity_type, entity_id, desired_revision, cur_gen)
                    if ok:
                        processed_count += 1
                continue

            # Upsert operation: read source text representation
            with db_session(target_db) as conn:
                text, text_hash = build_canonical_text(conn, entity_type, entity_id)
                if text is None:
                    # Entity was deleted before worker ran
                    publish_vector_delete(conn, lease_token, entity_type, entity_id, desired_revision, cur_gen)
                    continue

                # Check if existing vector can be reused (same hash, same model, same gen)
                row = conn.execute("""
                    SELECT text_hash, model_fingerprint, generation
                    FROM vector_index_state
                    WHERE entity_type = ? AND entity_id = ?
                """, (entity_type, entity_id)).fetchone()
                can_reuse = (
                    row is not None
                    and row[0] == text_hash
                    and row[1] == active_fp
                    and row[2] == cur_gen
                )

            if can_reuse:
                with db_session(target_db) as conn:
                    ok = publish_vector_reuse(
                        conn, lease_token, entity_type, entity_id, text_hash, desired_revision, cur_gen, active_fp
                    )
                    if ok:
                        processed_count += 1
                continue

            # Compute embedding outside SQLite locks
            emb = embed_text(text)
            if emb is None:
                with db_session(target_db) as conn:
                    fail_vector_job(conn, lease_token, entity_type, entity_id, "Embedding generation returned None")
                continue

            # Publish under write lock with lease verification
            with db_session(target_db) as conn:
                ok = publish_vector_job(
                    conn,
                    lease_token=lease_token,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    embedding=emb,
                    text_hash=text_hash,
                    expected_revision=desired_revision,
                    expected_gen=cur_gen,
                    model_fp=active_fp,
                )
                if ok:
                    processed_count += 1
                else:
                    # Publish rejected by fencing; release or let retry
                    fail_vector_job(conn, lease_token, entity_type, entity_id, "Fencing mismatch during publish")

    return processed_count


def get_vector_sync_status(db_path: str = None) -> Dict[str, Any]:
    """Inspect and return operational vector index coverage and job statistics."""
    target_db = db_path or DB_PATH
    with db_session(target_db) as conn:
        cursor = conn.cursor()
        cur_gen = get_db_generation(conn)
        active_fp = get_active_model_fingerprint(conn)

        stats = {
            "generation": cur_gen,
            "active_fingerprint": active_fp,
            "vector_search_enabled": bool(HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED),
            "sources": {},
            "jobs": {"pending": 0, "claimed": 0, "blocked": 0, "total": 0},
            "total_source_records": 0,
            "total_eligible_vectors": 0,
            "total_missing_vectors": 0,
            "total_stale_vectors": 0,
            "degraded": False,
        }

        for table in ("memories", "episodes", "learnings"):
            cursor.execute(f"SELECT count(*) FROM {table}")
            src_count = cursor.fetchone()[0]
            stats["total_source_records"] += src_count

            # Eligible: source exists, indexed_revision matches current revision, gen matches, fp matches
            cursor.execute(f"""
                SELECT count(*)
                FROM vector_index_state s
                JOIN {table} m ON m.id = s.entity_id
                JOIN entity_revisions r ON r.entity_type = '{table}' AND r.entity_id = s.entity_id
                WHERE s.entity_type = '{table}'
                  AND s.indexed_revision = r.revision
                  AND s.generation = ?
                  AND s.model_fingerprint = ?
            """, (cur_gen, active_fp))
            eligible_count = cursor.fetchone()[0]
            stats["total_eligible_vectors"] += eligible_count

            # Missing: source exists, but not in vector_index_state
            cursor.execute(f"""
                SELECT count(*)
                FROM {table} m
                LEFT JOIN vector_index_state s ON s.entity_type = '{table}' AND s.entity_id = m.id
                WHERE s.entity_id IS NULL
            """)
            missing_count = cursor.fetchone()[0]
            stats["total_missing_vectors"] += missing_count

            # Stale: in state, but indexed_revision < current revision or gen/fp mismatch
            cursor.execute(f"""
                SELECT count(*)
                FROM vector_index_state s
                JOIN {table} m ON m.id = s.entity_id
                JOIN entity_revisions r ON r.entity_type = '{table}' AND r.entity_id = s.entity_id
                WHERE s.entity_type = '{table}'
                  AND (s.indexed_revision != r.revision OR s.generation != ? OR s.model_fingerprint != ?)
            """, (cur_gen, active_fp))
            stale_count = cursor.fetchone()[0]
            stats["total_stale_vectors"] += stale_count

            stats["sources"][table] = {
                "total": src_count,
                "eligible": eligible_count,
                "missing": missing_count,
                "stale": stale_count,
            }

        cursor.execute("SELECT status, count(*) FROM vector_index_jobs GROUP BY status")
        for st, cnt in cursor.fetchall():
            if st in stats["jobs"]:
                stats["jobs"][st] = cnt
            stats["jobs"]["total"] += cnt

        if stats["jobs"]["blocked"] > 0 or not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
            stats["degraded"] = True

    return stats


def reconcile_vector_index(db_path: str = None, apply: bool = False) -> Dict[str, Any]:
    """
    Reconcile source tables against vector_index_state and outbox jobs.
    If apply is True, enqueues missing or stale items for worker processing.
    """
    target_db = db_path or DB_PATH
    with db_session(target_db) as conn:
        cur_gen = get_db_generation(conn)
        active_fp = get_active_model_fingerprint(conn)
        cursor = conn.cursor()

        report = {
            "missing_enqueued": 0,
            "stale_enqueued": 0,
            "orphans_enqueued": 0,
            "applied": apply,
            "items": [],
        }

        for table in ("memories", "episodes", "learnings"):
            # Missing
            cursor.execute(f"""
                SELECT m.id, COALESCE(r.revision, 1)
                FROM {table} m
                LEFT JOIN entity_revisions r ON r.entity_type = '{table}' AND r.entity_id = m.id
                LEFT JOIN vector_index_state s ON s.entity_type = '{table}' AND s.entity_id = m.id
                WHERE s.entity_id IS NULL
            """)
            missing_rows = cursor.fetchall()
            for eid, rev in missing_rows:
                report["items"].append({"entity_type": table, "entity_id": eid, "reason": "missing", "revision": rev})
                report["missing_enqueued"] += 1
                if apply:
                    cursor.execute(f"""
                        INSERT INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at)
                        VALUES ('{table}', ?, 'upsert', ?, 'pending', 0, CURRENT_TIMESTAMP)
                        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                            op = 'upsert',
                            desired_revision = excluded.desired_revision,
                            status = 'pending',
                            lease_token = NULL,
                            lease_expires_at = NULL,
                            updated_at = CURRENT_TIMESTAMP
                    """, (eid, rev))

            # Stale
            cursor.execute(f"""
                SELECT m.id, r.revision
                FROM vector_index_state s
                JOIN {table} m ON m.id = s.entity_id
                JOIN entity_revisions r ON r.entity_type = '{table}' AND r.entity_id = s.entity_id
                WHERE s.entity_type = '{table}'
                  AND (s.indexed_revision != r.revision OR s.generation != ? OR s.model_fingerprint != ?)
            """, (cur_gen, active_fp))
            stale_rows = cursor.fetchall()
            for eid, rev in stale_rows:
                report["items"].append({"entity_type": table, "entity_id": eid, "reason": "stale", "revision": rev})
                report["stale_enqueued"] += 1
                if apply:
                    cursor.execute(f"""
                        INSERT INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at)
                        VALUES ('{table}', ?, 'upsert', ?, 'pending', 0, CURRENT_TIMESTAMP)
                        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                            op = 'upsert',
                            desired_revision = excluded.desired_revision,
                            status = 'pending',
                            lease_token = NULL,
                            lease_expires_at = NULL,
                            updated_at = CURRENT_TIMESTAMP
                    """, (eid, rev))

            # Orphans in state without source
            cursor.execute(f"""
                SELECT s.entity_id, s.indexed_revision
                FROM vector_index_state s
                LEFT JOIN {table} m ON m.id = s.entity_id
                WHERE s.entity_type = '{table}' AND m.id IS NULL
            """)
            orphan_rows = cursor.fetchall()
            for eid, rev in orphan_rows:
                report["items"].append({"entity_type": table, "entity_id": eid, "reason": "orphan", "revision": rev})
                report["orphans_enqueued"] += 1
                if apply:
                    cursor.execute(f"""
                        INSERT INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at)
                        VALUES ('{table}', ?, 'delete', ?, 'pending', 0, CURRENT_TIMESTAMP)
                        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                            op = 'delete',
                            desired_revision = excluded.desired_revision,
                            status = 'pending',
                            lease_token = NULL,
                            lease_expires_at = NULL,
                            updated_at = CURRENT_TIMESTAMP
                    """, (eid, rev))

        if apply:
            conn.commit()

    return report
