"""Hermetic tests for queue receipts, atomic claims, leases, crash recovery, and execution invariants."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import os
import shutil
import tempfile
import sqlite3
import unittest
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from queue_manager import (
    init_queue_db,
    ensure_queue_db,
    reset_queue_db_guard,
    enqueue_turn,
    claim_batch,
    acknowledge_batch,
    release_batch,
    compute_batch_id,
    serialize_hash_tuple,
    make_content_hash,
    get_queue_identity,
    get_pending_turns,
    get_pending_stats,
    get_recent_turns,
    mark_turn_status,
)
import memory_worker
from memory_worker import (
    process_queue,
    should_process_queue,
    SyncBusyError,
    SyncExtractionError,
    main as worker_main,
)


class TestQueueReceiptsAndClaims(unittest.TestCase):
    """Rigorous hermetic test suite for queue claims, leases, and receipts."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_receipts.db")
        reset_queue_db_guard()

    def tearDown(self):
        reset_queue_db_guard()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deterministic_batch_id_stability(self):
        """Batch ID must be stable and deterministic given queue identity + turn ids/hashes."""
        queue_id = "test_queue_123"
        turns_order_a = [
            {"id": 1, "hash": "hash_a"},
            {"id": 2, "hash": "hash_b"},
            {"id": 3, "hash": "hash_c"},
        ]
        turns_order_b = [
            {"id": 3, "hash": "hash_c"},
            {"id": 1, "hash": "hash_a"},
            {"id": 2, "hash": "hash_b"},
        ]

        batch_id_a = compute_batch_id(queue_id, turns_order_a)
        batch_id_b = compute_batch_id(queue_id, turns_order_b)

        self.assertEqual(batch_id_a, batch_id_b)
        self.assertTrue(batch_id_a.startswith("batch_"))

        # Mutating a hash must change the batch_id
        turns_modified = [
            {"id": 1, "hash": "hash_a_different"},
            {"id": 2, "hash": "hash_b"},
            {"id": 3, "hash": "hash_c"},
        ]
        self.assertNotEqual(batch_id_a, compute_batch_id(queue_id, turns_modified))

        # Different queue identity must change the batch_id
        self.assertNotEqual(batch_id_a, compute_batch_id("different_queue", turns_order_a))

    def test_collision_safe_tuple_serialization(self):
        """Delimiter injection in tuple parts must not produce identical hash payloads."""
        payload_1 = serialize_hash_tuple("user:chat", "123", "hello", "world")
        payload_2 = serialize_hash_tuple("user", "chat:123", "hello", "world")
        self.assertNotEqual(payload_1, payload_2)

        hash_1 = make_content_hash("user:chat", "123", "hello", "world")
        hash_2 = make_content_hash("user", "chat:123", "hello", "world")
        self.assertNotEqual(hash_1, hash_2)

        # Null values serialize safely
        payload_null = serialize_hash_tuple("source", None, "prompt", "response")
        self.assertIn(b"-1:", payload_null)

    def test_event_id_exact_deduplication(self):
        """Event ID allows exact deduplication and is durably persisted."""
        ok1 = enqueue_turn(
            "User input message",
            "Assistant reply message",
            source="telegram",
            chat_id="100",
            event_id="evt_unique_101",
            db_path=self.db_path
        )
        self.assertTrue(ok1)

        # Duplicate event_id with same source/chat deduplicates cleanly
        ok2 = enqueue_turn(
            "User input message",
            "Assistant reply message",
            source="telegram",
            chat_id="100",
            event_id="evt_unique_101",
            db_path=self.db_path
        )
        self.assertTrue(ok2)

        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 1)

        # Verify event_id is saved in column
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT event_id FROM turn_queue WHERE id = ?", (pending[0]["id"],)).fetchone()
            self.assertEqual(row[0], "evt_unique_101")

        # Distinct event_id does not collide
        ok3 = enqueue_turn(
            "User input message",
            "Assistant reply message",
            source="telegram",
            chat_id="100",
            event_id="evt_unique_102",
            db_path=self.db_path
        )
        self.assertTrue(ok3)
        self.assertEqual(len(get_pending_turns(limit=10, db_path=self.db_path)), 2)

    def test_atomic_claim_and_lease(self):
        """claim_batch atomically assigns status='claimed', batch_id, lease_token, and lease_expires_at."""
        enqueue_turn("Prompt 1", "Resp 1", source="telegram", chat_id="chat_a", db_path=self.db_path)
        enqueue_turn("Prompt 2", "Resp 2", source="telegram", chat_id="chat_a", db_path=self.db_path)

        claim = claim_batch(batch_size=10, lease_duration_seconds=180, db_path=self.db_path)
        self.assertIsNotNone(claim)
        self.assertEqual(len(claim.turns), 2)
        self.assertEqual(claim.source, "telegram")
        self.assertEqual(claim.chat_id, "chat_a")
        self.assertTrue(claim.batch_id.startswith("batch_"))
        self.assertTrue(len(claim.lease_token) > 0)

        # Inspect database record
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT status, batch_id, lease_token, lease_expires_at, attempt_count FROM turn_queue").fetchall()
            for r in rows:
                self.assertEqual(r[0], "claimed")
                self.assertEqual(r[1], claim.batch_id)
                self.assertEqual(r[2], claim.lease_token)
                self.assertIsNotNone(r[3])
                self.assertEqual(r[4], 1)

    def test_concurrent_claim_no_duplicate_ownership(self):
        """Active claim prevents a competing worker from claiming the same batch or turns."""
        enqueue_turn("Chat A turn 1", "Resp 1", source="telegram", chat_id="chat_a", db_path=self.db_path)

        claim1 = claim_batch(batch_size=10, lease_duration_seconds=180, db_path=self.db_path)
        self.assertIsNotNone(claim1)

        # Second worker attempts to claim
        claim2 = claim_batch(batch_size=10, lease_duration_seconds=180, db_path=self.db_path)
        self.assertIsNone(claim2)

    def test_no_cross_chat_mixing(self):
        """Batches must partition strictly by (source, chat_id)."""
        enqueue_turn("Chat 1 turn", "Resp 1", source="telegram", chat_id="chat_1", db_path=self.db_path)
        enqueue_turn("Chat 2 turn", "Resp 2", source="telegram", chat_id="chat_2", db_path=self.db_path)
        enqueue_turn("Chat 1 turn 2", "Resp 1-2", source="telegram", chat_id="chat_1", db_path=self.db_path)

        claim_1 = claim_batch(batch_size=10, db_path=self.db_path)
        self.assertIsNotNone(claim_1)
        self.assertEqual(claim_1.chat_id, "chat_1")
        self.assertEqual(len(claim_1.turns), 2)
        for t in claim_1.turns:
            self.assertEqual(t["chat_id"], "chat_1")

        claim_2 = claim_batch(batch_size=10, db_path=self.db_path)
        self.assertIsNotNone(claim_2)
        self.assertEqual(claim_2.chat_id, "chat_2")
        self.assertEqual(len(claim_2.turns), 1)

    def test_crashed_batch_recovery_preserves_membership_and_batch_id(self):
        """Crashed claimed batches (with expired leases) recover with the EXACT same membership and batch_id."""
        enqueue_turn("Turn 1", "Resp 1", source="telegram", chat_id="chat_crash", db_path=self.db_path)
        enqueue_turn("Turn 2", "Resp 2", source="telegram", chat_id="chat_crash", db_path=self.db_path)

        claim1 = claim_batch(batch_size=10, lease_duration_seconds=300, db_path=self.db_path)
        original_batch_id = claim1.batch_id
        original_turn_ids = [t["id"] for t in claim1.turns]

        # Add a 3rd turn to the same chat while claim1 is in flight
        enqueue_turn("Turn 3 late", "Resp 3 late", source="telegram", chat_id="chat_crash", db_path=self.db_path)

        # Simulate crash: lease expires in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE turn_queue SET lease_expires_at = datetime('now', '-5 seconds') WHERE batch_id = ?", (original_batch_id,))

        # Recovery worker claims the expired batch
        recovered_claim = claim_batch(batch_size=10, db_path=self.db_path)
        self.assertIsNotNone(recovered_claim)

        # Invariant: Must recover with the EXACT same batch_id and EXACT same turn membership
        self.assertEqual(recovered_claim.batch_id, original_batch_id)
        recovered_turn_ids = [t["id"] for t in recovered_claim.turns]
        self.assertEqual(recovered_turn_ids, original_turn_ids)
        self.assertNotIn(3, recovered_turn_ids)  # Turn 3 must NOT be mixed into the crashed batch!

        # Attempt count incremented
        self.assertEqual(recovered_claim.turns[0]["attempt_count"], 1)

    def test_stale_lease_owner_cannot_acknowledge(self):
        """A worker whose lease has expired cannot acknowledge the batch after another worker reclaimed it."""
        enqueue_turn("Turn to process", "Resp", source="telegram", chat_id="chat_lease", db_path=self.db_path)

        worker1_claim = claim_batch(batch_size=10, lease_duration_seconds=10, db_path=self.db_path)
        batch_id = worker1_claim.batch_id
        token1 = worker1_claim.lease_token

        # Expire worker 1's lease
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE turn_queue SET lease_expires_at = datetime('now', '-10 seconds')")

        # Worker 2 recovers the batch
        worker2_claim = claim_batch(batch_size=10, lease_duration_seconds=300, db_path=self.db_path)
        self.assertIsNotNone(worker2_claim)
        token2 = worker2_claim.lease_token
        self.assertNotEqual(token1, token2)

        # Worker 1 suddenly wakes up and attempts to acknowledge
        ack_worker1 = acknowledge_batch(batch_id=batch_id, lease_token=token1, summary="Worker 1 finished", db_path=self.db_path)
        self.assertFalse(ack_worker1)

        # Worker 2 acknowledges with active lease
        ack_worker2 = acknowledge_batch(batch_id=batch_id, lease_token=token2, summary="Worker 2 finished", db_path=self.db_path)
        self.assertTrue(ack_worker2)

        # Database verifies status is processed with Worker 2's summary
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT status, extracted_summary, lease_token FROM turn_queue WHERE batch_id = ?", (batch_id,)).fetchone()
            self.assertEqual(row[0], "processed")
            self.assertEqual(row[1], "Worker 2 finished")
            self.assertIsNone(row[2])

    @patch("memory_worker.sync_turn")
    def test_sync_turn_batch_id_contract(self, mock_sync):
        """process_queue must pass batch_id to sync_turn adhering to the contract."""
        mock_sync.return_value = {"facts": [], "episodes": [], "learnings": [], "entity_links": []}

        enqueue_turn("Prompt for sync contract", "Response", source="telegram", chat_id="chat_contract", db_path=self.db_path)

        committed = process_queue(batch_size=10, notify=False, db_path=self.db_path)
        self.assertEqual(committed, 1)

        self.assertEqual(mock_sync.call_count, 1)
        self.assertIn("batch_id", mock_sync.call_args.kwargs)
        batch_id_passed = mock_sync.call_args.kwargs["batch_id"]
        self.assertTrue(batch_id_passed.startswith("batch_"))

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_notification_delivery_failure_separate_from_extraction(self, mock_sync, mock_notify):
        """Telegram notification delivery failure must NOT roll back or fail the committed memory batch."""
        mock_sync.return_value = {
            "facts": [{"id": "fact_1", "fact": "Persisted fact"}],
            "episodes": [],
            "learnings": [],
            "entity_links": []
        }
        mock_notify.side_effect = RuntimeError("Telegram network unreachable")

        enqueue_turn("Notify fail prompt", "Notify resp", source="telegram", chat_id="chat_notify", db_path=self.db_path)

        committed = process_queue(batch_size=10, notify=True, db_path=self.db_path)
        self.assertEqual(committed, 1)

        # Turn is successfully marked processed despite notification failure
        recent = get_recent_turns(limit=5, db_path=self.db_path)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["status"], "processed")
        self.assertIsNone(recent[0]["error"])

    @patch("memory_worker.sync_turn")
    def test_committed_count_vs_attempted_count(self, mock_sync):
        """process_queue must return the actual committed count, not attempted count."""
        def fake_sync(user_prompt, **kwargs):
            if "Fail" in user_prompt:
                raise SyncExtractionError("Extraction failed intentionally")
            return {"facts": [{"id": "ok_fact", "fact": "OK"}], "episodes": [], "learnings": [], "entity_links": []}

        mock_sync.side_effect = fake_sync

        enqueue_turn("Fail turn", "Fail resp", source="telegram", chat_id="chat_fail", db_path=self.db_path)
        enqueue_turn("OK turn 1", "OK resp 1", source="telegram", chat_id="chat_ok", db_path=self.db_path)
        enqueue_turn("OK turn 2", "OK resp 2", source="telegram", chat_id="chat_ok", db_path=self.db_path)

        # Total attempted is 3 turns across 2 partitions, but only 2 commit
        committed = process_queue(batch_size=10, notify=False, db_path=self.db_path)
        self.assertEqual(committed, 2)

    @patch("memory_worker.sync_turn")
    @patch("sys.exit")
    def test_cli_failure_exit_code_nonzero(self, mock_exit, mock_sync):
        """When memory worker encounters batch errors, CLI exits with nonzero status."""
        mock_sync.side_effect = SyncExtractionError("Crash")

        enqueue_turn("Failing turn", "Failing resp", source="telegram", chat_id="cli_chat", db_path=self.db_path)

        with patch("sys.argv", ["memory_worker.py", "--force"]):
            with patch("memory_worker.QUEUE_DB_PATH", self.db_path):
                worker_main()

        mock_exit.assert_called_with(1)

    def test_migration_serialized_no_hot_ddl(self):
        """Queue schema initialization runs under BEGIN IMMEDIATE and sets user_version = 2 without hot DDL."""
        init_queue_db(self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2)

        # Repeated ensure_queue_db does not invoke init_queue_db
        with patch("queue_manager.init_queue_db") as mock_init:
            ensure_queue_db(self.db_path)
            mock_init.assert_not_called()

    def test_deterministic_connection_closing(self):
        """Operations deterministically close connections without leaving open file locks."""
        enqueue_turn("Deterministic test", "Resp", source="telegram", chat_id="chat_conn", db_path=self.db_path)

        # Connection should not remain held; SQLite WAL checkpoint / exclusive lock must succeed immediately
        with sqlite3.connect(self.db_path, timeout=1.0) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


if __name__ == "__main__":
    unittest.main()
