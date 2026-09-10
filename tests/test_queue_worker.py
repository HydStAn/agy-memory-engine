
# Set temporary paths before importing modules that capture configuration defaults.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import unittest
import unittest.mock
import os
import tempfile
import sqlite3
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from queue_manager import (
    init_queue_db,
    enqueue_turn,
    get_pending_turns,
    get_pending_stats,
    get_recent_turns,
    mark_turn_status,
    prune_processed_turns
)
from memory_worker import format_notification


class TestQueueManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_queue.db")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_and_enqueue(self):
        init_queue_db(self.db_path)
        ok = enqueue_turn("Hallo, wie gehts?", "Mir geht es gut!", source="telegram", chat_id="12345", db_path=self.db_path)
        self.assertTrue(ok)

        # Enqueue same turn should deduplicate and return True without duplicate insert
        ok_dup = enqueue_turn("Hallo, wie gehts?", "Mir geht es gut!", source="telegram", chat_id="12345", db_path=self.db_path)
        self.assertTrue(ok_dup)

        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["user_prompt"], "Hallo, wie gehts?")
        self.assertEqual(pending[0]["chat_id"], "12345")

    def test_pending_stats(self):
        stats_empty = get_pending_stats(db_path=self.db_path)
        self.assertEqual(stats_empty["count"], 0)

        enqueue_turn("Turn A", "Resp A", db_path=self.db_path)
        stats = get_pending_stats(db_path=self.db_path)
        self.assertEqual(stats["count"], 1)
        self.assertGreaterEqual(stats["newest_age_seconds"], 0)

    def test_mark_and_prune(self):
        enqueue_turn("Frage 1", "Antwort 1", db_path=self.db_path)
        enqueue_turn("Frage 2", "Antwort 2", db_path=self.db_path)
        
        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 2)
        
        turn1_id = pending[0]["id"]
        mark_turn_status([turn1_id], status="processed", summary="1 fact synced", batch_id="batch_20260902_090000", db_path=self.db_path)

        pending_after = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending_after), 1)
        self.assertEqual(pending_after[0]["user_prompt"], "Frage 2")

    def test_batch_tracking_in_recent_turns(self):
        enqueue_turn("Turn 1", "Antwort 1", db_path=self.db_path)
        enqueue_turn("Turn 2", "Antwort 2", db_path=self.db_path)
        enqueue_turn("Turn 3", "Antwort 3", db_path=self.db_path)

        pending = get_pending_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(pending), 3)

        turn_ids = [pending[0]["id"], pending[1]["id"]]
        batch_name = "batch_20260902_120000"
        mark_turn_status(turn_ids, status="processed", summary="2 facts extracted", batch_id=batch_name, db_path=self.db_path)

        recent = get_recent_turns(limit=10, db_path=self.db_path)
        self.assertEqual(len(recent), 3)

        # Pending turn
        turn3 = [t for t in recent if t["user_prompt"] == "Turn 3"][0]
        self.assertEqual(turn3["status"], "pending")
        self.assertIsNone(turn3["batch_id"])
        self.assertIsNone(turn3["processed_at"])

        # Batched turns
        batched = [t for t in recent if t["user_prompt"] in ("Turn 1", "Turn 2")]
        self.assertEqual(len(batched), 2)
        for t in batched:
            self.assertEqual(t["status"], "processed")
            self.assertEqual(t["batch_id"], batch_name)
            self.assertEqual(t["extracted_summary"], "2 facts extracted")
            self.assertIsNotNone(t["processed_at"])


class TestWorkerNotification(unittest.TestCase):
    def test_format_notification(self):
        changes = {
            "facts": [{"id": "user.test.dog", "category": "user", "fact": "Golden Retriever 4 Jahre", "is_update": False}],
            "episodes": [{"id": "travel.foehr_2026", "topic": "travel", "title": "Föhr Urlaub", "is_update": True}],
            "learnings": [{"id": "finance.investing", "category": "finance", "insight": "80/20 Allokation", "is_update": False}],
            "entity_links": [{"source": "infra.beelink", "target": "service.immich", "relation": "hosts"}]
        }
        msg = format_notification(changes)
        self.assertIn("Autonomous memory updated", msg)
        self.assertIn("Golden Retriever 4 Jahre", msg)
        self.assertIn("Föhr Urlaub", msg)
    @unittest.mock.patch("subprocess.run")
    def test_send_telegram_notification(self, mock_run):
        import memory_worker
        mock_run.return_value.returncode = 0

        mock_bin = unittest.mock.MagicMock()
        mock_bin.exists.return_value = True
        mock_bin.__str__.return_value = "/mock/bin/send_telegram.py"

        with unittest.mock.patch.object(memory_worker, "SEND_TELEGRAM_BIN", mock_bin):
            # When chat_id is None, should route to --reports
            memory_worker.send_telegram_notification("Test Message", chat_id=None)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertIn("--reports", cmd)
            self.assertIn("Test Message", cmd)

            # When chat_id is provided, should route to --chat-id
            mock_run.reset_mock()
            memory_worker.send_telegram_notification("Test Message 2", chat_id="12345")
            mock_run.assert_called_once()
            cmd2 = mock_run.call_args[0][0]
            self.assertIn("--chat-id", cmd2)
            self.assertIn("12345", cmd2)
            self.assertNotIn("--reports", cmd2)


if __name__ == "__main__":
    unittest.main()
