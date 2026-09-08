
# Set temporary paths before importing modules that capture configuration defaults.
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

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from queue_manager import (
    init_queue_db,
    ensure_queue_db,
    reset_queue_db_guard,
    enqueue_turn,
    get_pending_turns,
    get_pending_stats,
    get_recent_turns,
    mark_turn_status,
    prune_processed_turns
)
from memory_worker import (
    process_queue,
    should_process_queue,
    SyncBusyError,
    SyncExtractionError
)


class TestQueueHardeningF04F16(unittest.TestCase):
    """Test F04 content hash identity and F16 DDL removal on hot path."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_queue_harden.db")
        reset_queue_db_guard()

    def tearDown(self):
        reset_queue_db_guard()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_content_hash_differentiates_chats(self):
        """Identical user prompt & response across different chats must have distinct hashes."""
        prompt = "Kannst du mir helfen?"
        response = "Ja gerne!"

        ok1 = enqueue_turn(prompt, response, source="telegram", chat_id="chat_111", db_path=self.db_path)
        ok2 = enqueue_turn(prompt, response, source="telegram", chat_id="chat_222", db_path=self.db_path)

        self.assertTrue(ok1)
        self.assertTrue(ok2)

        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 2)
        chat_ids = {t["chat_id"] for t in pending}
        self.assertEqual(chat_ids, {"chat_111", "chat_222"})

    def test_content_hash_differentiates_sources(self):
        """Identical turn across different sources must not collide."""
        prompt = "Status check"
        response = "Everything running smoothly."

        ok1 = enqueue_turn(prompt, response, source="telegram", chat_id="chat_1", db_path=self.db_path)
        ok2 = enqueue_turn(prompt, response, source="cli", chat_id="chat_1", db_path=self.db_path)

        self.assertTrue(ok1)
        self.assertTrue(ok2)

        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 2)
        sources = {t["source"] for t in pending}
        self.assertEqual(sources, {"telegram", "cli"})

    def test_content_hash_deduplicates_same_chat_and_source(self):
        """Identical turn for the same chat and source deduplicates cleanly."""
        prompt = "Gleiche Nachricht"
        response = "Gleiche Antwort"

        ok1 = enqueue_turn(prompt, response, source="telegram", chat_id="chat_999", db_path=self.db_path)
        ok2 = enqueue_turn(prompt, response, source="telegram", chat_id="chat_999", db_path=self.db_path)

        self.assertTrue(ok1)
        self.assertTrue(ok2)

        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 1)

    def test_ddl_not_run_on_hot_path(self):
        """Schema DDL should execute only once, not on repeated hot-path operations."""
        # First call initializes DB
        enqueue_turn("First prompt", "First resp", db_path=self.db_path)

        with patch("queue_manager.init_queue_db") as mock_init:
            enqueue_turn("Second prompt", "Second resp", db_path=self.db_path)
            get_pending_stats(db_path=self.db_path)
            get_pending_turns(db_path=self.db_path)
            mark_turn_status([1], status="processed", db_path=self.db_path)
            prune_processed_turns(db_path=self.db_path)
            get_recent_turns(db_path=self.db_path)

            mock_init.assert_not_called()


class TestWorkerPartitioningAndErrorHandling(unittest.TestCase):
    """Test F03 error classification, pending retention, and F04 chat partitioning."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_worker_harden.db")
        reset_queue_db_guard()

    def tearDown(self):
        reset_queue_db_guard()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_failed_head_does_not_starve_fresh_turns(self):
        enqueue_turn('Failing old prompt', 'response', source='hook', chat_id='bad', db_path=self.db_path)
        with patch('memory_worker.sync_turn', side_effect=SyncExtractionError('bad extraction')):
            process_queue(batch_size=1, notify=False, db_path=self.db_path)
        self.assertFalse(should_process_queue(db_path=self.db_path)[0])
        enqueue_turn('Healthy new prompt', 'response', source='hook', chat_id='good', db_path=self.db_path)
        with patch('memory_worker.sync_turn', return_value={key: [] for key in ('facts', 'episodes', 'learnings', 'entity_links')}) as sync:
            process_queue(batch_size=1, notify=False, db_path=self.db_path)
        self.assertIn('Healthy new prompt', sync.call_args.kwargs['user_prompt'])
        pending = get_pending_turns(db_path=self.db_path)
        self.assertEqual([row['chat_id'] for row in pending], ['bad'])

    def test_empty_unconfirmed_result_is_retained(self):
        enqueue_turn("Remember the current deployment host", "Host saved", db_path=self.db_path)
        with patch("memory_worker.sync_turn", return_value={}):
            process_queue(notify=False, db_path=self.db_path)
        rows = get_pending_turns(db_path=self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertIn('did not confirm', get_recent_turns(db_path=self.db_path)[0]['error'])

    def test_response_tail_participates_in_deduplication(self):
        for suffix in ('first', 'second'):
            enqueue_turn('Remember this conversation', 'same prefix ' * 100 + suffix, db_path=self.db_path)
        self.assertEqual(len(get_pending_turns(db_path=self.db_path)), 2)

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_partitioning_by_source_and_chat(self, mock_sync, mock_notify):
        """Batches must be partitioned strictly by (source, chat_id)."""
        mock_sync.return_value = {
            "facts": [{"id": "f1", "fact": "Fact 1"}],
            "episodes": [],
            "learnings": [],
            "entity_links": []
        }

        enqueue_turn("Chat A turn 1", "Resp A1", source="telegram", chat_id="100", db_path=self.db_path)
        enqueue_turn("Chat B turn 1", "Resp B1", source="telegram", chat_id="200", db_path=self.db_path)
        enqueue_turn("Chat A turn 2", "Resp A2", source="telegram", chat_id="100", db_path=self.db_path)

        processed_count = process_queue(batch_size=10, notify=True, db_path=self.db_path)
        self.assertEqual(processed_count, 3)

        # sync_turn should have been called twice (once for chat 100, once for chat 200)
        self.assertEqual(mock_sync.call_count, 2)

        # Call args inspection: verify Chat A dialogue does not contain Chat B dialogue
        first_call_prompt = mock_sync.call_args_list[0].kwargs["user_prompt"]
        second_call_prompt = mock_sync.call_args_list[1].kwargs["user_prompt"]

        self.assertIn("Chat A turn 1", first_call_prompt)
        self.assertIn("Chat A turn 2", first_call_prompt)
        self.assertNotIn("Chat B turn 1", first_call_prompt)

        self.assertIn("Chat B turn 1", second_call_prompt)
        self.assertNotIn("Chat A turn 1", second_call_prompt)

        # Notifications should be routed to each respective chat
        self.assertEqual(mock_notify.call_count, 2)
        notified_chats = {call.kwargs["chat_id"] for call in mock_notify.call_args_list}
        self.assertEqual(notified_chats, {"100", "200"})

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_sync_busy_error_retains_pending(self, mock_sync, mock_notify):
        """When sync_turn raises SyncBusyError, turns stay pending with error and timestamp."""
        mock_sync.side_effect = SyncBusyError("Database locked by another worker")

        enqueue_turn("Busy prompt", "Busy resp", source="telegram", chat_id="555", db_path=self.db_path)

        process_queue(batch_size=10, notify=True, db_path=self.db_path)

        # Must still be returned by get_pending_turns
        pending = get_pending_turns(db_path=self.db_path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["user_prompt"], "Busy prompt")

        # Inspect database record
        recent = get_recent_turns(limit=5, db_path=self.db_path)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["status"], "pending")
        self.assertIn("Database locked", recent[0]["error"])
        self.assertIsNotNone(recent[0]["processed_at"])

        # No telegram notification sent on error
        mock_notify.assert_not_called()

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_sync_extraction_error_retains_pending(self, mock_sync, mock_notify):
        """When sync_turn raises SyncExtractionError, turns stay pending with error."""
        mock_sync.side_effect = SyncExtractionError("Invalid model JSON output")

        enqueue_turn("Broken prompt", "Broken resp", source="telegram", chat_id="777", db_path=self.db_path)

        process_queue(batch_size=10, notify=True, db_path=self.db_path)

        pending = get_pending_turns(db_path=self.db_path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["user_prompt"], "Broken prompt")

        recent = get_recent_turns(limit=5, db_path=self.db_path)
        self.assertEqual(recent[0]["status"], "pending")
        self.assertIn("Invalid model JSON", recent[0]["error"])
        self.assertIsNotNone(recent[0]["processed_at"])
        mock_notify.assert_not_called()

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_partition_error_isolation(self, mock_sync, mock_notify):
        """One partition failure must not prevent another partition from succeeding."""
        def fake_sync(user_prompt, **kwargs):
            if "Chat Error" in user_prompt:
                raise SyncExtractionError("Model crashed on Chat Error")
            return {
                "facts": [{"id": "f.ok", "fact": "Success fact"}],
                "episodes": [],
                "learnings": [],
                "entity_links": []
            }

        mock_sync.side_effect = fake_sync

        enqueue_turn("Chat Error msg", "Resp Err", source="telegram", chat_id="fail_chat", db_path=self.db_path)
        enqueue_turn("Chat OK msg", "Resp OK", source="telegram", chat_id="ok_chat", db_path=self.db_path)

        process_queue(batch_size=10, notify=True, db_path=self.db_path)

        recent = get_recent_turns(limit=10, db_path=self.db_path)
        err_turn = next(t for t in recent if t["chat_id"] == "fail_chat")
        ok_turn = next(t for t in recent if t["chat_id"] == "ok_chat")

        self.assertEqual(err_turn["status"], "pending")
        self.assertIn("Model crashed", err_turn["error"])

        self.assertEqual(ok_turn["status"], "processed")
        self.assertIsNone(ok_turn["error"])
        self.assertIn("1 facts", ok_turn["extracted_summary"])

        # Notification sent ONLY for ok_chat
        self.assertEqual(mock_notify.call_count, 1)
        self.assertEqual(mock_notify.call_args[1]["chat_id"], "ok_chat")

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_telegram_notification_source_filter(self, mock_sync, mock_notify):
        """Telegram notification is only sent for source='telegram' and with valid chat_id."""
        mock_sync.return_value = {
            "facts": [{"id": "f.cli", "fact": "CLI fact"}],
            "episodes": [],
            "learnings": [],
            "entity_links": []
        }

        # CLI source should not trigger telegram notification
        enqueue_turn("CLI msg", "CLI resp", source="cli", chat_id="12345", db_path=self.db_path)
        process_queue(batch_size=10, notify=True, db_path=self.db_path)
        mock_notify.assert_not_called()

        # Telegram source without chat_id should not notify
        enqueue_turn("Telegram no chat", "Resp", source="telegram", chat_id=None, db_path=self.db_path)
        process_queue(batch_size=10, notify=True, db_path=self.db_path)
        mock_notify.assert_not_called()

        # Telegram source with chat_id should notify
        enqueue_turn("Telegram valid", "Resp", source="telegram", chat_id="8888", db_path=self.db_path)
        process_queue(batch_size=10, notify=True, db_path=self.db_path)
        mock_notify.assert_called_once()
        self.assertEqual(mock_notify.call_args[1]["chat_id"], "8888")

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_clean_extraction_zero_items_marked_processed(self, mock_sync, mock_notify):
        """A clean extraction with 0 items must be marked processed, not pending or failed."""
        mock_sync.return_value = {"facts": [], "episodes": [], "learnings": [], "entity_links": []}

        enqueue_turn("Smalltalk ohne Fakten", "Nichts zu merken", source="telegram", chat_id="333", db_path=self.db_path)

        process_queue(batch_size=10, notify=True, db_path=self.db_path)

        recent = get_recent_turns(limit=5, db_path=self.db_path)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["status"], "processed")
        self.assertEqual(recent[0]["extracted_summary"], "No persistent entities found")
        self.assertIsNone(recent[0]["error"])
        mock_notify.assert_not_called()

    @patch("memory_worker.send_telegram_notification")
    @patch("memory_worker.sync_turn")
    def test_retry_success_clears_error(self, mock_sync, mock_notify):
        """A turn that initially failed should clear its error upon successful retry."""
        mock_sync.side_effect = SyncBusyError("Lock busy")
        enqueue_turn("Retry prompt", "Retry resp", source="telegram", chat_id="444", db_path=self.db_path)

        # First run fails
        process_queue(batch_size=10, notify=True, db_path=self.db_path)
        recent = get_recent_turns(limit=5, db_path=self.db_path)
        self.assertEqual(recent[0]["status"], "pending")
        self.assertIn("Lock busy", recent[0]["error"])

        # Backoff suppresses an immediate retry without losing the pending turn.
        mock_sync.reset_mock()
        self.assertEqual(process_queue(batch_size=10, notify=False, db_path=self.db_path), 0)
        mock_sync.assert_not_called()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE turn_queue SET processed_at = datetime('now', '-61 seconds')")

        # Second eligible run succeeds.
        mock_sync.side_effect = None
        mock_sync.return_value = {
            "facts": [{"id": "f.retry", "fact": "Recovered fact"}],
            "episodes": [],
            "learnings": [],
            "entity_links": []
        }
        process_queue(batch_size=10, notify=True, db_path=self.db_path)

        recent_after = get_recent_turns(limit=5, db_path=self.db_path)
        self.assertEqual(recent_after[0]["status"], "processed")
        self.assertIsNone(recent_after[0]["error"])
        self.assertIn("1 facts", recent_after[0]["extracted_summary"])
        mock_notify.assert_called_once_with(mock_notify.call_args[0][0], chat_id="444")


if __name__ == "__main__":
    unittest.main()
