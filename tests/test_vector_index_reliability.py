"""
Unit and Integration Tests for AGY Memory Engine Vector Index Reliability.
Verifies all 13 core requirements of VECTOR_INDEX_RELIABILITY_PLAN.md:
transactional outbox, revision and generation fencing, lease recovery,
top-k prefiltering, and self-healing retries.
"""

import unittest
import tempfile
import os
import shutil
import sqlite3
import struct
import numpy as np
from unittest.mock import patch

from schema import (
    db_session,
    _init_schema,
    _upgrade_schema,
    get_db_generation,
    bump_db_generation,
    compute_model_fingerprint,
    HAS_SQLITE_VEC,
)
from config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME
from vector_index import (
    claim_vector_jobs,
    publish_vector_job,
    publish_vector_delete,
    publish_vector_reuse,
    fail_vector_job,
    drain_vector_jobs,
    get_vector_sync_status,
    reconcile_vector_index,
    get_active_model_fingerprint,
    build_canonical_text,
)
from agy_memory import upsert_fact, upsert_episode, upsert_learning


def dummy_embedding(val: float = 0.5) -> np.ndarray:
    arr = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    arr[0] = val
    return arr


class TestVectorIndexReliability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        # Initialize schema and upgrade
        with db_session(self.db_path) as conn:
            conn.commit()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_insert_memory_enqueues_and_worker_drain_indexes(self):
        """Inserting a memory enqueues outbox job and drain_vector_jobs publishes it."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.test.1", "dev", "Antigravity uses SQLite WAL mode", "sqlite,wal", connection=conn)
            conn.commit()

        status_before = get_vector_sync_status(self.db_path)
        self.assertEqual(status_before["sources"]["memories"]["total"], 1)
        self.assertEqual(status_before["sources"]["memories"]["eligible"], 0)
        self.assertEqual(status_before["sources"]["memories"]["missing"], 1)
        self.assertEqual(status_before["jobs"]["pending"], 1)

        with patch("vector_index.embed_text", return_value=dummy_embedding(0.9)):
            processed = drain_vector_jobs(self.db_path)
            self.assertEqual(processed, 1)

        status_after = get_vector_sync_status(self.db_path)
        self.assertEqual(status_after["sources"]["memories"]["eligible"], 1)
        self.assertEqual(status_after["sources"]["memories"]["missing"], 0)
        self.assertEqual(status_after["jobs"]["pending"], 0)

    def test_02_embedding_error_retries_and_self_heals(self):
        """Temporary embedding failure records backoff attempt and succeeds on next worker run."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.test.err", "general", "Temporary failure fact", connection=conn)
            conn.commit()

        # First run fails
        with patch("vector_index.embed_text", return_value=None):
            processed = drain_vector_jobs(self.db_path)
            self.assertEqual(processed, 0)

        with db_session(self.db_path) as conn:
            job = conn.execute("SELECT attempts, status, last_error FROM vector_index_jobs WHERE entity_id = 'fact.test.err'").fetchone()
            self.assertIsNotNone(job)
            self.assertEqual(job[0], 1)
            self.assertEqual(job[1], "pending")
            self.assertIn("Embedding generation returned None", job[2])

            # Reset next_attempt_at to now for immediate retry
            conn.execute("UPDATE vector_index_jobs SET next_attempt_at = datetime('now', '-1 minute')")
            conn.commit()

        # Second run succeeds
        with patch("vector_index.embed_text", return_value=dummy_embedding(0.8)):
            processed = drain_vector_jobs(self.db_path)
            self.assertEqual(processed, 1)

        status = get_vector_sync_status(self.db_path)
        self.assertEqual(status["sources"]["memories"]["eligible"], 1)

    def test_03_update_v1_to_v2_invalidates_v1_immediately_in_search(self):
        """Updating v1 to v2 immediately disqualifies v1 from semantic queries before worker runs."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.version", "dev", "Initial fact version 1", connection=conn)
            conn.commit()

        with patch("vector_index.embed_text", return_value=dummy_embedding(0.1)):
            drain_vector_jobs(self.db_path)

        status1 = get_vector_sync_status(self.db_path)
        self.assertEqual(status1["sources"]["memories"]["eligible"], 1)

        # Update to v2
        with db_session(self.db_path) as conn:
            upsert_fact("fact.version", "dev", "Updated fact version 2", connection=conn)
            conn.commit()

        # Prior to worker run, v1 vector must be flagged stale and ineligible
        status2 = get_vector_sync_status(self.db_path)
        self.assertEqual(status2["sources"]["memories"]["eligible"], 0)
        self.assertEqual(status2["sources"]["memories"]["stale"], 1)

        # Drain worker publishes v2
        with patch("vector_index.embed_text", return_value=dummy_embedding(0.2)):
            drain_vector_jobs(self.db_path)

        status3 = get_vector_sync_status(self.db_path)
        self.assertEqual(status3["sources"]["memories"]["eligible"], 1)
        self.assertEqual(status3["sources"]["memories"]["stale"], 0)

    def test_04_rapid_updates_fence_intermediate_worker_runs(self):
        """Worker computing v2 is rejected if v3 committed before worker publishes."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.race", "dev", "Version 1", connection=conn)
            conn.commit()

        with db_session(self.db_path) as conn:
            upsert_fact("fact.race", "dev", "Version 2", connection=conn)
            conn.commit()

        with db_session(self.db_path) as conn:
            cur_gen = get_db_generation(conn)
            active_fp = get_active_model_fingerprint(conn)
            lease_token, jobs = claim_vector_jobs(conn, batch_size=1)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["desired_revision"], 2)

        # In parallel, user updates to Version 3 before worker publishes Version 2
        with db_session(self.db_path) as conn:
            upsert_fact("fact.race", "dev", "Version 3", connection=conn)
            conn.commit()

        # Worker for Version 2 attempts to publish with expected_revision=2
        with db_session(self.db_path) as conn:
            ok = publish_vector_job(
                conn,
                lease_token=lease_token,
                entity_type="memories",
                entity_id="fact.race",
                embedding=dummy_embedding(0.5),
                text_hash="hash_v2",
                expected_revision=2,
                expected_gen=cur_gen,
                model_fp=active_fp,
            )
            # Must be rejected because current revision is now 3!
            self.assertFalse(ok)

        # Now worker runs again for Version 3
        with patch("vector_index.embed_text", return_value=dummy_embedding(0.6)):
            drain_vector_jobs(self.db_path)

        status = get_vector_sync_status(self.db_path)
        self.assertEqual(status["sources"]["memories"]["eligible"], 1)
        with db_session(self.db_path) as conn:
            rev = conn.execute("SELECT indexed_revision FROM vector_index_state WHERE entity_id = 'fact.race'").fetchone()[0]
            self.assertEqual(rev, 3)

    def test_05_rollback_write_transaction_reverts_job_and_revision(self):
        """Rollback of write transaction rolls back both entity and outbox job."""
        with db_session(self.db_path) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                upsert_fact("fact.rollback", "dev", "Will be rolled back", connection=conn)
                # Intentionally trigger rollback
                raise RuntimeError("Simulated transaction abort")
            except RuntimeError:
                conn.rollback()

        with db_session(self.db_path) as conn:
            mem = conn.execute("SELECT 1 FROM memories WHERE id = 'fact.rollback'").fetchone()
            rev = conn.execute("SELECT 1 FROM entity_revisions WHERE entity_id = 'fact.rollback'").fetchone()
            job = conn.execute("SELECT 1 FROM vector_index_jobs WHERE entity_id = 'fact.rollback'").fetchone()
            self.assertIsNone(mem)
            self.assertIsNone(rev)
            self.assertIsNone(job)

    def test_06_crash_after_claim_recovers_after_lease_expiry(self):
        """If worker crashes after claiming job, another worker can claim it once lease expires."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.crash", "general", "Crash recovery test", connection=conn)
            conn.commit()

        with db_session(self.db_path) as conn:
            lease1, jobs1 = claim_vector_jobs(conn, batch_size=1, lease_seconds=1)
            self.assertEqual(len(jobs1), 1)

            # Force lease expiration
            conn.execute("UPDATE vector_index_jobs SET lease_expires_at = datetime('now', '-5 seconds') WHERE entity_id = 'fact.crash'")
            conn.commit()

        with db_session(self.db_path) as conn:
            # Worker 2 claims it
            lease2, jobs2 = claim_vector_jobs(conn, batch_size=1, lease_seconds=60)
            self.assertEqual(len(jobs2), 1)
            self.assertNotEqual(lease1, lease2)

            cur_gen = get_db_generation(conn)
            active_fp = get_active_model_fingerprint(conn)

            # Stale worker 1 attempts to publish: must be rejected!
            ok1 = publish_vector_job(conn, lease1, "memories", "fact.crash", dummy_embedding(), "hash", 1, cur_gen, active_fp)
            self.assertFalse(ok1)

            # Worker 2 publishes successfully
            ok2 = publish_vector_job(conn, lease2, "memories", "fact.crash", dummy_embedding(), "hash", 1, cur_gen, active_fp)
            self.assertTrue(ok2)

    def test_07_entity_deletion_and_id_reuse(self):
        """Deleting an entity invalidates vectors immediately and id reuse bumps revision tombstone."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.del", "general", "To be deleted", connection=conn)
            conn.commit()

        with patch("vector_index.embed_text", return_value=dummy_embedding()):
            drain_vector_jobs(self.db_path)

        with db_session(self.db_path) as conn:
            # Delete fact
            conn.execute("DELETE FROM memories WHERE id = 'fact.del'")
            conn.commit()

        # Immediately ineligible in status
        status = get_vector_sync_status(self.db_path)
        self.assertEqual(status["total_source_records"], 0)
        self.assertEqual(status["total_eligible_vectors"], 0)

        # Worker cleans up
        drain_vector_jobs(self.db_path)
        status_after_cleanup = get_vector_sync_status(self.db_path)
        self.assertEqual(status_after_cleanup["total_missing_vectors"], 0)

        # Re-insert with same ID: revision must be bumped (tombstone: 1 -> delete: 2 -> new insert: 3)
        with db_session(self.db_path) as conn:
            upsert_fact("fact.del", "general", "Recreated fact", connection=conn)
            conn.commit()

        with db_session(self.db_path) as conn:
            rev = conn.execute("SELECT revision FROM entity_revisions WHERE entity_id = 'fact.del'").fetchone()[0]
            self.assertEqual(rev, 3)

        with patch("vector_index.embed_text", return_value=dummy_embedding()):
            drain_vector_jobs(self.db_path)

        with db_session(self.db_path) as conn:
            idx_rev = conn.execute("SELECT indexed_revision FROM vector_index_state WHERE entity_id = 'fact.del'").fetchone()[0]
            self.assertEqual(idx_rev, 3)

    def test_08_database_restore_generation_fencing(self):
        """Generation mismatch after restore rejects in-flight worker publications."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.restore", "infra", "Server IP 10.0.0.1", connection=conn)
            conn.commit()

        with db_session(self.db_path) as conn:
            cur_gen = get_db_generation(conn)
            active_fp = get_active_model_fingerprint(conn)
            lease_token, jobs = claim_vector_jobs(conn, batch_size=1)

            # Database restore occurs: generation bumped
            new_gen = bump_db_generation(conn)
            self.assertNotEqual(cur_gen, new_gen)

            # In-flight worker tries to publish under old generation
            ok = publish_vector_job(conn, lease_token, "memories", "fact.restore", dummy_embedding(), "hash", 1, cur_gen, active_fp)
            self.assertFalse(ok)

        # Reconcile detects stale generation and enqueues repair
        rep = reconcile_vector_index(self.db_path, apply=True)
        self.assertGreaterEqual(rep["missing_enqueued"] + rep["stale_enqueued"], 1)

        with patch("vector_index.embed_text", return_value=dummy_embedding()):
            drain_vector_jobs(self.db_path)

        status = get_vector_sync_status(self.db_path)
        self.assertEqual(status["sources"]["memories"]["eligible"], 1)

    @unittest.skipUnless(HAS_SQLITE_VEC, "sqlite-vec extension required for top-k prefilter test")
    def test_09_top_k_prefiltering_prevents_stale_result_truncation(self):
        """WHERE id IN eligible ensures closer stale vectors do NOT crowd out valid results in KNN."""
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute("CREATE VIRTUAL TABLE vec_test USING vec0(id text primary key, embedding float[4]);")
        conn.execute("CREATE TABLE eligible (id text primary key);")

        def pack_v(vals):
            return struct.pack(f"{len(vals)}f", *vals)

        # item1 is closest (dist 0.0) but stale (NOT in eligible)
        # item2 and item3 are farther but eligible
        conn.execute("INSERT INTO vec_test VALUES (?, ?)", ("item1", pack_v([1.0, 0.0, 0.0, 0.0])))
        conn.execute("INSERT INTO vec_test VALUES (?, ?)", ("item2", pack_v([0.8, 0.6, 0.0, 0.0])))
        conn.execute("INSERT INTO vec_test VALUES (?, ?)", ("item3", pack_v([0.5, 0.5, 0.5, 0.5])))

        conn.execute("INSERT INTO eligible VALUES ('item2'), ('item3')")

        query = pack_v([1.0, 0.0, 0.0, 0.0])

        # Without prefilter (old pattern): item1 occupies 1 of 2 top slots, leaving only 1 result after outer filter
        old_cur = conn.execute("""
            SELECT v.id FROM vec_test v
            JOIN eligible e ON e.id = v.id
            WHERE v.embedding MATCH ? AND k = 2
        """, (query,))
        old_results = [r[0] for r in old_cur.fetchall()]
        self.assertEqual(len(old_results), 1)
        self.assertEqual(old_results, ["item2"])

        # With prefilter (new pattern): both eligible items returned!
        new_cur = conn.execute("""
            SELECT id FROM vec_test
            WHERE embedding MATCH ? AND k = 2 AND id IN (SELECT id FROM eligible)
            ORDER BY distance ASC
        """, (query,))
        new_results = [r[0] for r in new_cur.fetchall()]
        self.assertEqual(len(new_results), 2)
        self.assertEqual(new_results, ["item2", "item3"])

    def test_10_missing_extension_does_not_crash_deletes(self):
        """Connections without sqlite-vec extension can delete memories without trigger OperationalError."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.plain_del", "general", "Plain delete test", connection=conn)
            conn.commit()

        # Open plain sqlite connection without loading sqlite-vec
        plain_conn = sqlite3.connect(self.db_path)
        try:
            plain_conn.execute("DELETE FROM memories WHERE id = 'fact.plain_del'")
            plain_conn.commit()
        except sqlite3.OperationalError as e:
            self.fail(f"Plain delete failed with OperationalError: {e}")
        finally:
            plain_conn.close()

        # Job must be enqueued with op='delete'
        with db_session(self.db_path) as conn:
            job = conn.execute("SELECT op, status FROM vector_index_jobs WHERE entity_id = 'fact.plain_del'").fetchone()
            self.assertIsNotNone(job)
            self.assertEqual(job[0], "delete")

    def test_11_content_hash_reuse_skips_embedding(self):
        """Updating record with unchanged canonical text reuses vector without re-embedding."""
        with db_session(self.db_path) as conn:
            upsert_fact("fact.reuse", "dev", "Identical content", "kw1", connection=conn)
            conn.commit()

        with patch("vector_index.embed_text", return_value=dummy_embedding(0.3)) as mock_embed:
            drain_vector_jobs(self.db_path)
            self.assertEqual(mock_embed.call_count, 1)

        # Update with exact same category, fact, keywords
        with db_session(self.db_path) as conn:
            upsert_fact("fact.reuse", "dev", "Identical content", "kw1", connection=conn)
            conn.commit()

        # Next drain must detect same hash and reuse vector without calling embed_text
        with patch("vector_index.embed_text", return_value=dummy_embedding(0.3)) as mock_embed:
            processed = drain_vector_jobs(self.db_path)
            self.assertEqual(processed, 1)
            mock_embed.assert_not_called()

        status = get_vector_sync_status(self.db_path)
        self.assertEqual(status["sources"]["memories"]["eligible"], 1)

    def test_12_episodes_and_learnings_lifecycle(self):
        """Full lifecycle verification for episodes and learnings layers."""
        with db_session(self.db_path) as conn:
            upsert_episode("ep.1", "dev", "Architecture Sprint", "Designed outbox pattern", connection=conn)
            upsert_learning("learn.1", "workflow", "Always use short transactions", "Concurrency testing", connection=conn)
            conn.commit()

        with patch("vector_index.embed_text", return_value=dummy_embedding(0.4)):
            processed = drain_vector_jobs(self.db_path)
            self.assertEqual(processed, 2)

        status = get_vector_sync_status(self.db_path)
        self.assertEqual(status["sources"]["episodes"]["eligible"], 1)
        self.assertEqual(status["sources"]["learnings"]["eligible"], 1)
        self.assertEqual(status["total_missing_vectors"], 0)


if __name__ == "__main__":
    unittest.main()
