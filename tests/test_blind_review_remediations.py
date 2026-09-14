"""
Hermetic test suite for Blind Review Remediations (BR01 through BR05).
Covers:
- BR01: Consolidation identity normalization and target preservation
- BR02: Dashboard token secrecy on unauthenticated GET and mutation gate
- BR03: Queue fresh turn capacity preservation under retry backlog
- BR04: Database generation fencing across snapshot restore
- BR05: Transcript turn continuation updates and revisioning on repeated Stop hook
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import os
import json
import sqlite3
import tempfile
import shutil
import unittest
import threading
import http.client
import io
import re
from http.server import HTTPServer
from unittest.mock import patch
from functools import partial
from contextlib import redirect_stdout

import schema
import agy_memory as memory
import queue_manager as queue
import memory_worker as worker
import dashboard
from scripts import auto_sync_hook as hook


class TestBlindReviewRemediations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="agy_br_test_")
        self.root = Path(self.temp_dir)
        self.memory_db = str(self.root / "memory.db")
        self.queue_db = str(self.root / "queue.db")
        queue.reset_queue_db_guard()

    def tearDown(self):
        queue.reset_queue_db_guard()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_br01_consolidation_whitespace_normalization_and_target_preservation(self):
        """BR01: Padded target ID is normalized during consolidation, target is preserved, and links resolve."""
        with patch.object(schema, "DB_PATH", self.memory_db):
            for fid in ("a", "b", "outside"):
                memory.upsert_fact(fid, "infra", fid)
            memory.link_entities("b", "outside", "uses")

            proposal = {
                "merges": [{
                    "target_id": " a ",
                    "category": "infra",
                    "fact": "merged authoritative fact",
                    "merged_ids": [" a ", "b"],
                    "keywords": "infra,merged",
                    "rationale": "combined redundancy"
                }]
            }
            with patch.object(memory, "_infer_json", return_value=json.dumps(proposal)):
                result = memory.consolidate_memories()

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["target_id"], "a")
            self.assertEqual(result[0]["merged_ids"], ["b"])

            with schema.db_session(self.memory_db) as conn:
                facts = dict(conn.execute("SELECT id, fact FROM memories").fetchall())
                links = conn.execute("SELECT source_id, target_id, relation FROM entity_links").fetchall()

            self.assertIn("a", facts)
            self.assertEqual(facts["a"], "merged authoritative fact")
            self.assertNotIn("b", facts)
            self.assertIn("outside", facts)
            self.assertIn(("a", "outside", "uses"), links)
            self.assertNotIn((" a ", "outside", "uses"), links)

    def test_br01_consolidation_rejects_padded_target_colliding_with_protected_fact(self):
        """BR01: Padded target ID pointing to a protected fact is safely skipped."""
        with patch.object(schema, "DB_PATH", self.memory_db):
            memory.upsert_fact("health_fact", "health", "daily vitamins")
            memory.upsert_fact("temp_fact", "health", "vitamin c")

            proposal = {
                "merges": [{
                    "target_id": " health_fact ",
                    "category": "health",
                    "fact": "overwritten vitamins",
                    "merged_ids": ["temp_fact"],
                    "keywords": "health",
                    "rationale": "merge"
                }]
            }
            with patch.object(memory, "_infer_json", return_value=json.dumps(proposal)):
                result = memory.consolidate_memories()

            self.assertEqual(len(result), 0)
            with schema.db_session(self.memory_db) as conn:
                facts = dict(conn.execute("SELECT id, fact FROM memories").fetchall())
            self.assertEqual(facts["health_fact"], "daily vitamins")

    def test_br02_dashboard_unauthenticated_get_does_not_disclose_token(self):
        """BR02: Unauthenticated GET / returns 401 without embedding token, and POST requires auth."""
        secret_token = "br02-secret-token-xyz-12345"
        with patch.dict(os.environ, {"AGY_MEMORY_DASHBOARD_TOKEN": secret_token}):
            server = HTTPServer(("127.0.0.1", 0), dashboard.MemoryDashboardHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            try:
                # 1. Unauthenticated GET / must return 401
                client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                client.request("GET", "/", headers={"Host": "localhost"})
                resp = client.getresponse()
                body = resp.read().decode()
                client.close()

                self.assertEqual(resp.status, 401)
                self.assertNotIn(secret_token, body)
                self.assertNotIn("const DASHBOARD_TOKEN =", body)

                # 2. Unauthenticated POST must be rejected with 401
                with patch.object(dashboard, "create_snapshot") as mock_create:
                    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    client.request("POST", "/api/create-snapshot", body=b"", headers={"Host": "localhost"})
                    post_resp = client.getresponse()
                    post_resp.read()
                    client.close()

                    self.assertEqual(post_resp.status, 401)
                    self.assertFalse(mock_create.called)

                # 3. Authenticated GET /?token=... returns 200 with embedded token and cookie
                client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                client.request("GET", f"/?token={secret_token}", headers={"Host": "localhost"})
                auth_resp = client.getresponse()
                auth_body = auth_resp.read().decode()
                auth_headers = dict(auth_resp.getheaders())
                client.close()

                self.assertEqual(auth_resp.status, 200)
                self.assertIn(f'const DASHBOARD_TOKEN = "{secret_token}";', auth_body)
                self.assertIn("dashboard_token=", auth_headers.get("Set-Cookie", ""))
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_br03_queue_fresh_turns_not_starved_by_retry_backlog(self):
        """BR03: Eligible retry backlog does not starve fresh unbatched turns."""
        for i in range(50):
            queue.enqueue_turn(f"Broken task {i}", "broken", source="hook", chat_id=f"bad-{i}", db_path=self.queue_db)
            claim = queue.claim_batch(db_path=self.queue_db)
            queue.release_batch(claim.batch_id, claim.lease_token, error="poison", db_path=self.queue_db)

        queue.enqueue_turn("Fresh healthy turn", "healthy", source="hook", chat_id="healthy", db_path=self.queue_db)

        # Make all 50 failed batches eligible for retry
        with sqlite3.connect(self.queue_db) as c:
            c.execute("UPDATE turn_queue SET processed_at=datetime('now', '-120 seconds') WHERE batch_id IS NOT NULL")

        def mock_sync(user_prompt=None, **kw):
            if "healthy" in user_prompt:
                return {"facts": [], "episodes": [], "learnings": [], "entity_links": []}
            raise memory.SyncExtractionError("permanently invalid extraction")

        with patch.object(worker, "sync_turn", side_effect=mock_sync):
            processed = worker.process_queue(batch_size=25, notify=False, db_path=self.queue_db)

        with sqlite3.connect(self.queue_db) as c:
            healthy_status = c.execute("SELECT status, attempt_count FROM turn_queue WHERE chat_id='healthy'").fetchone()

        self.assertGreaterEqual(processed, 1)
        self.assertEqual(healthy_status, ("processed", 1))

    def test_br04_fence_inflight_inference_across_restore_generation(self):
        """BR04: Restore during in-flight inference increments db generation and aborts stale commit."""
        with patch.object(schema, "DB_PATH", self.memory_db):
            memory.upsert_fact("server", "infra", "snapshot value")
            snap = memory.create_snapshot()
            memory.upsert_fact("server", "infra", "pre-inference value")

            def restore_and_edit(*args, **kwargs):
                memory.restore_snapshot(snap["filename"])
                memory.upsert_fact("server", "infra", "manual edit after restore")
                return json.dumps({"facts": [{"id": "server", "category": "infra", "fact": "stale inference overwrites manual edit"}]})

            with patch.object(memory, "_infer_json", side_effect=restore_and_edit):
                with self.assertRaises(memory.SyncExtractionError) as ctx:
                    memory.sync_turn("Remember this server deployment configuration", "saved")
                self.assertIn("generation", str(ctx.exception).lower())

            with schema.db_session(self.memory_db) as conn:
                fact_row = conn.execute("SELECT fact FROM memories WHERE id='server'").fetchone()

            self.assertEqual(fact_row[0], "manual edit after restore")

    def test_br05_repeated_stop_hook_updates_pending_continuation(self):
        """BR05: Repeated Stop hook on continuing response updates pending turn in place."""
        transcript = self.root / "partial.jsonl"
        first = [
            {"type": "USER_INPUT", "content": "Remember the final deployment endpoint"},
            {"type": "MODEL_RESPONSE", "content": "I will inspect the deployment."}
        ]
        transcript.write_text("".join(json.dumps(event) + "\n" for event in first))
        payload = json.dumps({"transcriptPath": str(transcript), "conversationId": "br05-synthetic"})

        with patch.dict(os.environ, {"AGY_INTERNAL_INVOCATION": "0", "AGY_TELEGRAM_CHAT_ID": ""}):
            with patch.object(hook, "enqueue_turn", partial(queue.enqueue_turn, db_path=self.queue_db)):
                # First run: initial chunk
                with patch.object(hook.sys, "stdin", io.StringIO(payload)), redirect_stdout(io.StringIO()):
                    hook.main()

                pending = queue.get_pending_turns(db_path=self.queue_db)
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0]["assistant_response"], "I will inspect the deployment.")

                # Second chunk appended to transcript
                with transcript.open("a") as f:
                    f.write(json.dumps({"type": "MODEL_RESPONSE", "content": "The verified final endpoint is new.example.test."}) + "\n")

                # Second run: hook runs again for continued turn
                with patch.object(hook.sys, "stdin", io.StringIO(payload)), redirect_stdout(io.StringIO()):
                    hook.main()

                updated_pending = queue.get_pending_turns(db_path=self.queue_db)
                self.assertEqual(len(updated_pending), 1)
                expected_response = "I will inspect the deployment.\n\nThe verified final endpoint is new.example.test."
                self.assertEqual(updated_pending[0]["assistant_response"], expected_response)

    def test_br05_continuation_after_batch_claim_creates_new_revision(self):
        """BR05: Expanded response arriving after batch was claimed creates revisioned continuation."""
        event_id = "conv-123:1"
        queue.enqueue_turn("prompt", "chunk 1", source="hook", chat_id="chat1", event_id=event_id, db_path=self.queue_db)

        # Claim the batch
        claim = queue.claim_batch(db_path=self.queue_db)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.turns[0]["assistant_response"], "chunk 1")

        # Response expands while batch is claimed
        queue.enqueue_turn("prompt", "chunk 1 + chunk 2", source="hook", chat_id="chat1", event_id=event_id, db_path=self.queue_db)

        with sqlite3.connect(self.queue_db) as conn:
            rows = conn.execute("SELECT event_id, assistant_response, status FROM turn_queue ORDER BY id").fetchall()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], (event_id, "chunk 1", "claimed"))
        self.assertEqual(rows[1], (f"{event_id}:rev2", "chunk 1 + chunk 2", "pending"))


if __name__ == "__main__":
    unittest.main()
