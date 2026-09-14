import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import unittest
import os
import tempfile
import json
import shutil
import sqlite3
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from queue_manager import init_queue_db, enqueue_turn, claim_batch
from schema import db_session
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
        os.environ["AGY_MEMORY_DB"] = self.memory_db
        os.environ["AGY_TURN_QUEUE_DB"] = self.queue_db
        with db_session(self.memory_db) as conn:
            pass

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
            data_file=None,
            memory_db=self.memory_db,
        )
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_commit(commit_args)
            self.assertEqual(code, 0)
            commit_res = json.loads(fake_out.getvalue())
            self.assertTrue(commit_res["acknowledged"])
            self.assertEqual(commit_res["committed"]["facts"], 1)
            self.assertEqual(commit_res["committed"]["learnings"], 1)

        # Verify in memory db
        with sqlite3.connect(self.memory_db) as conn:
            row = conn.execute("SELECT fact FROM memories WHERE id = 'infra.server.ip'").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "Server IP is 192.168.1.50")

    def test_cli_commit_lease_fencing(self):
        enqueue_turn("User: Secret?", "Yes", source="antigravity", chat_id="chat-fence", db_path=self.queue_db)
        claim = claim_batch(db_path=self.queue_db)
        self.assertIsNotNone(claim)

        # 1. Wrong lease token
        bad_args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=claim.batch_id,
            lease_token="invalid-token",
            data=json.dumps({"facts": [{"id": "bad.token", "category": "infra", "fact": "should not write"}]}),
            data_file=None,
            memory_db=self.memory_db,
        )
        with patch("sys.stderr", new=StringIO()):
            code = cmd_commit(bad_args)
            self.assertEqual(code, 1)

        # Verify nothing written to memory.db
        with sqlite3.connect(self.memory_db) as conn:
            row = conn.execute("SELECT 1 FROM memories WHERE id = 'bad.token'").fetchone()
            self.assertIsNone(row)

        # 2. Expired lease
        with sqlite3.connect(self.queue_db) as q_conn:
            q_conn.execute("UPDATE turn_queue SET lease_expires_at = datetime('now', '-10 seconds') WHERE batch_id = ?", (claim.batch_id,))

        exp_args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=claim.batch_id,
            lease_token=claim.lease_token,
            data=json.dumps({"facts": [{"id": "bad.expired", "category": "infra", "fact": "should not write"}]}),
            data_file=None,
            memory_db=self.memory_db,
        )
        with patch("sys.stderr", new=StringIO()):
            code = cmd_commit(exp_args)
            self.assertEqual(code, 1)

        with sqlite3.connect(self.memory_db) as conn:
            row = conn.execute("SELECT 1 FROM memories WHERE id = 'bad.expired'").fetchone()
            self.assertIsNone(row)

    def test_cli_commit_atomic_rollback(self):
        enqueue_turn("Turn 1", "Resp 1", source="antigravity", chat_id="chat-atomic", db_path=self.queue_db)
        claim = claim_batch(db_path=self.queue_db)
        self.assertIsNotNone(claim)

        # Payload has a valid fact followed by an invalid category
        payload = {
            "facts": [
                {"id": "valid.one", "category": "infra", "fact": "Valid infra info"},
                {"id": "invalid.two", "category": "nonexistent_cat_123", "fact": "Invalid"}
            ]
        }
        args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=claim.batch_id,
            lease_token=claim.lease_token,
            data=json.dumps(payload),
            data_file=None,
            memory_db=self.memory_db,
        )
        with patch("sys.stderr", new=StringIO()):
            code = cmd_commit(args)
            self.assertEqual(code, 1)

        # Neither should be written to memory.db
        with sqlite3.connect(self.memory_db) as conn:
            row = conn.execute("SELECT 1 FROM memories WHERE id = 'valid.one'").fetchone()
            self.assertIsNone(row)

    def test_cli_commit_malformed_item_rejected(self):
        enqueue_turn("Turn 1", "Resp 1", source="antigravity", chat_id="chat-malformed", db_path=self.queue_db)
        claim = claim_batch(db_path=self.queue_db)
        self.assertIsNotNone(claim)

        # Missing required 'fact' text
        payload = {"facts": [{"id": "missing.fact", "category": "infra"}]}
        args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=claim.batch_id,
            lease_token=claim.lease_token,
            data=json.dumps(payload),
            data_file=None,
            memory_db=self.memory_db,
        )
        with patch("sys.stderr", new=StringIO()):
            code = cmd_commit(args)
            self.assertEqual(code, 1)

        # Turn must NOT be marked skipped
        with sqlite3.connect(self.queue_db) as conn:
            status = conn.execute("SELECT status FROM turn_queue WHERE batch_id = ?", (claim.batch_id,)).fetchone()[0]
            self.assertEqual(status, "claimed")

    def test_cli_commit_replay_receipt(self):
        enqueue_turn("Turn 1", "Resp 1", source="antigravity", chat_id="chat-replay", db_path=self.queue_db)
        claim = claim_batch(db_path=self.queue_db)
        self.assertIsNotNone(claim)

        payload = {"facts": [{"id": "replay.fact", "category": "infra", "fact": "Replay data"}]}
        args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=claim.batch_id,
            lease_token=claim.lease_token,
            data=json.dumps(payload),
            data_file=None,
            memory_db=self.memory_db,
        )
        with patch("sys.stdout", new=StringIO()):
            code1 = cmd_commit(args)
            self.assertEqual(code1, 0)

        # Second commit for the same batch should return replay receipt
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code2 = cmd_commit(args)
            self.assertEqual(code2, 0)
            res = json.loads(fake_out.getvalue())
            self.assertTrue(res.get("replay"))
            self.assertTrue(res["acknowledged"])

    def test_cli_commit_data_file(self):
        enqueue_turn("Turn 1", "Resp 1", source="antigravity", chat_id="chat-file", db_path=self.queue_db)
        claim = claim_batch(db_path=self.queue_db)
        self.assertIsNotNone(claim)

        file_path = os.path.join(self.temp_dir, "payload.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"facts": [{"id": "from.file", "category": "infra", "fact": "Data from file"}]}, f)

        args = argparse.Namespace(
            db_path=self.queue_db,
            batch_id=claim.batch_id,
            lease_token=claim.lease_token,
            data="-",
            data_file=file_path,
            memory_db=self.memory_db,
        )
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_commit(args)
            self.assertEqual(code, 0)
            res = json.loads(fake_out.getvalue())
            self.assertTrue(res["acknowledged"])

        with sqlite3.connect(self.memory_db) as conn:
            row = conn.execute("SELECT fact FROM memories WHERE id = 'from.file'").fetchone()
            self.assertIsNotNone(row)

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
        enqueue_turn("Old turn", "Old resp", db_path=self.queue_db)
        with sqlite3.connect(self.queue_db) as conn:
            conn.execute("UPDATE turn_queue SET status = 'processed', created_at = datetime('now', '-10 days'), processed_at = datetime('now', '-10 days')")

        args = argparse.Namespace(db_path=self.queue_db, days=7)
        with patch("sys.stdout", new=StringIO()) as fake_out:
            code = cmd_prune(args)
            self.assertEqual(code, 0)
            res = json.loads(fake_out.getvalue())
            self.assertTrue(res["pruned"])
            self.assertEqual(res["deleted_count"], 1)
