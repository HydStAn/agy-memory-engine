import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import unittest
import os
import tempfile
import json
import shutil
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from queue_manager import init_queue_db, enqueue_turn, claim_batch
from scripts.queue_cli import (
    cmd_status,
    cmd_claim,
    cmd_ack,
    cmd_skip,
    cmd_release,
    cmd_commit,
    cmd_prune,
    main,
)
import argparse


class TestQueueCLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_db = os.path.join(self.temp_dir, "test_queue.db")
        self.memory_db = os.path.join(self.temp_dir, "test_memory.db")
        init_queue_db(self.queue_db)
        os.environ["AGY_MEMORY_DB_PATH"] = self.memory_db

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_status_and_claim_flow(self):
        enqueue_turn("User: What is the server IP?", "The IP is 192.168.1.50", source="antigravity", chat_id="chat-1", db_path=self.queue_db)

        # Status check
        args = argparse.Namespace(db_path=self.queue_db, force=True)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_status(args)
            self.assertEqual(code, 0)
            res = json.loads(fake_out.getvalue())
            self.assertEqual(res["count"], 1)
            self.assertTrue(res["can_process"])

        # Claim check
        claim_args = argparse.Namespace(db_path=self.queue_db, batch_size=10, lease_seconds=120)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_claim(claim_args)
            self.assertEqual(code, 0)
            claim_res = json.loads(fake_out.getvalue())
            self.assertIsNotNone(claim_res)
            batch_id = claim_res["batch_id"]
            lease_token = claim_res["lease_token"]
            self.assertEqual(len(claim_res["turns"]), 1)

        # Commit check
        extraction_data = {
            "facts": [{"id": "infra.server.ip", "category": "infra", "fact": "Server IP is 192.168.1.50"}],
            "learnings": [{"id": "ops.ip_check", "category": "workflow", "insight": "Verify IP via status"}],
            "episodes": [],
            "entity_links": []
        }
        commit_args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=batch_id,
            lease_token=lease_token,
            data=json.dumps(extraction_data),
        )
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_commit(commit_args)
            self.assertEqual(code, 0)
            commit_res = json.loads(fake_out.getvalue())
            self.assertTrue(commit_res["acknowledged"])
            self.assertEqual(commit_res["committed"]["facts"], 1)
            self.assertEqual(commit_res["committed"]["learnings"], 1)

    def test_cli_skip_and_release(self):
        enqueue_turn("Hello!", "Hi there!", source="antigravity", chat_id="chat-2", db_path=self.queue_db)
        claim_args = argparse.Namespace(db_path=self.queue_db, batch_size=10, lease_seconds=120)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_claim(claim_args)
            claim_res = json.loads(fake_out.getvalue())
            batch_id = claim_res["batch_id"]
            lease_token = claim_res["lease_token"]

        # Release
        rel_args = argparse.Namespace(db_path=self.queue_db, batch_id=batch_id, lease_token=lease_token, error="retry later")
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_release(rel_args)
            self.assertEqual(code, 0)
            res = json.loads(fake_out.getvalue())
            self.assertTrue(res["released"])

        # Re-claim and test ack/skip
        claim_again = claim_batch(batch_size=10, lease_duration_seconds=120, retry_delay_seconds=0, db_path=self.queue_db)
        self.assertIsNotNone(claim_again)
        skip_args = argparse.Namespace(db_path=self.queue_db, batch_id=claim_again.batch_id, lease_token=claim_again.lease_token, summary="Trivial banter")
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_skip(skip_args)
            self.assertEqual(code, 0)
            res = json.loads(fake_out.getvalue())
            self.assertTrue(res["acknowledged"])
            self.assertEqual(res["status"], "skipped")

    def test_cli_prune(self):
        args = argparse.Namespace(db_path=self.queue_db, days=7)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_prune(args)
            self.assertEqual(code, 0)
            res = json.loads(fake_out.getvalue())
            self.assertTrue(res["pruned"])

