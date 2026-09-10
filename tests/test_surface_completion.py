"""
Hermetic test suite for external surface hardening (F07, F08, F20):
- agy_memory_mcp.py:
  * Identifier and content validation (rejection of blank/whitespace fields)
  * Finite limit clamping and consistent search envelope
  * FastMCP tool exceptions instead of error-looking success strings
  * Offloading blocking maintenance to bounded concurrency executor
  * Backward-compatible synchronous calling convention
- dashboard.py:
  * Completion reporting and accurate exit code propagation for force-worker & optimize
  * Timeout handling (HTTP 504)
  * Snapshot error status propagation (HTTP 500 on failure)
  * Payload size bounds (HTTP 413 on >1MB payload)
"""

import sys
from pathlib import Path

# Add tests directory and base directory to path
TESTS_DIR = Path(__file__).resolve().parent
BASE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(BASE_DIR))

import _test_environment  # noqa: F401

import asyncio
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import HTTPServer
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import agy_memory_mcp
from agy_memory_mcp import (
    mcp,
    search_memory,
    store_memory,
    record_episode,
    record_learning,
    link_entities_mcp,
    list_memories,
    migrate_memory,
    optimize_memory,
    _MAINTENANCE_EXECUTOR,
)
import config
import dashboard
from dashboard import MemoryDashboardHandler, get_or_create_dashboard_token
import schema


class TestMcpSurfaceHardening(unittest.TestCase):
    """Verify MCP surface hardening per F07 and F20."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="agy_mcp_test_")
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
        self.db_patch = patch.object(schema, 'DB_PATH', self.db_path)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        schema._SCHEMA_INITIALIZED.discard(self.db_path)
        schema._SCHEMA_INITIALIZED.discard(config.DB_PATH)
        self.orig_db = os.environ.get("AGY_MEMORY_DB")
        os.environ["AGY_MEMORY_DB"] = self.db_path
        # Ensure database is initialized on disk for tools like migrate_memory
        for p in (self.db_path, config.DB_PATH):
            with schema.db_session(p) as conn:
                conn.execute("SELECT 1")

    def tearDown(self):
        if self.orig_db is not None:
            os.environ["AGY_MEMORY_DB"] = self.orig_db
        else:
            os.environ.pop("AGY_MEMORY_DB", None)
        schema._SCHEMA_INITIALIZED.discard(self.db_path)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_search_memory_consistent_envelope(self):
        """F20: Empty or tokenless search returns consistent envelope, not []."""
        # Empty string query
        res_empty = json.loads(search_memory(""))
        self.assertIsInstance(res_empty, dict)
        self.assertEqual(res_empty.get("facts"), [])
        self.assertEqual(res_empty.get("episodes"), [])
        self.assertEqual(res_empty.get("learnings"), [])
        self.assertEqual(res_empty.get("entity_links"), [])

        # Whitespace-only query
        res_ws = json.loads(search_memory("   "))
        self.assertIsInstance(res_ws, dict)
        self.assertEqual(res_ws.get("facts"), [])

        # Tokenless symbol query
        res_tokens = json.loads(search_memory("!@#$%^&*()"))
        self.assertIsInstance(res_tokens, dict)
        self.assertEqual(res_tokens.get("facts"), [])

    def test_search_memory_clamped_limits(self):
        """F20: Limit parameter is clamped to finite positive bounds."""
        store_memory("fact.test1", "Alpha fact content", category="infra")
        store_memory("fact.test2", "Beta fact content", category="infra")

        # Negative limit does not disable SQLite limit or cause syntax error
        res_neg = json.loads(search_memory("fact", limit=-1))
        self.assertIsInstance(res_neg, dict)
        self.assertTrue(len(res_neg.get("facts", [])) <= 100)

        # Zero limit is clamped to at least 1
        res_zero = json.loads(search_memory("fact", limit=0))
        self.assertIsInstance(res_zero, dict)

        # Huge limit is clamped to at most 100
        res_huge = json.loads(search_memory("fact", limit=999999))
        self.assertIsInstance(res_huge, dict)

    def test_store_memory_validates_identifiers_and_content(self):
        """F20: store_memory raises ValueError on blank/whitespace id or content."""
        with self.assertRaises(ValueError):
            store_memory("", "Valid content")

        with self.assertRaises(ValueError):
            store_memory("   ", "Valid content")

        with self.assertRaises(ValueError):
            store_memory("valid.id", "")

        with self.assertRaises(ValueError):
            store_memory("valid.id", "   ")

    def test_record_episode_validates_fields(self):
        """F20: record_episode raises ValueError on blank id, title, or narrative."""
        with self.assertRaises(ValueError):
            record_episode("", "home", "Title", "Narrative")

        with self.assertRaises(ValueError):
            record_episode("ep.1", "home", "", "Narrative")

        with self.assertRaises(ValueError):
            record_episode("ep.1", "home", "Title", "")

    def test_record_learning_validates_fields(self):
        """F20: record_learning raises ValueError on blank id or insight."""
        with self.assertRaises(ValueError):
            record_learning("", "workflow", "Valid insight")

        with self.assertRaises(ValueError):
            record_learning("lrn.1", "workflow", "")

    def test_link_entities_validates_fields(self):
        """F20: link_entities_mcp raises ValueError on blank source, target, or relation."""
        with self.assertRaises(ValueError):
            link_entities_mcp("", "target", "hosted_on")

        with self.assertRaises(ValueError):
            link_entities_mcp("source", "", "hosted_on")

        with self.assertRaises(ValueError):
            link_entities_mcp("source", "target", "")

    def test_fastmcp_tool_exceptions_over_protocol(self):
        """F20: FastMCP call_tool raises ToolError for invalid parameters instead of returning error strings."""
        from mcp.server.fastmcp.exceptions import ToolError

        async def run_calls():
            # Calling store_memory with blank id must produce ToolError in FastMCP
            with self.assertRaises(ToolError):
                await mcp.call_tool("store_memory", {"id": "", "fact": "Valid fact"})

            # Calling link_entities_mcp with blank relation must produce ToolError
            with self.assertRaises(ToolError):
                await mcp.call_tool("link_entities_mcp", {"source_id": "a", "target_id": "b", "relation": ""})

        asyncio.run(run_calls())

    def test_maintenance_offloading_and_operation_ids(self):
        """F20: Heavy maintenance operations are offloaded asynchronously with operation IDs."""
        async def run_maintenance():
            # Check optimize_memory tool registration
            tools = mcp._tool_manager.list_tools()
            opt_tool = next((t for t in tools if t.name == "optimize_memory"), None)
            self.assertIsNotNone(opt_tool)
            self.assertTrue(opt_tool.is_async, "optimize_memory tool must be async in FastMCP")

            mig_tool = next((t for t in tools if t.name == "migrate_memory"), None)
            self.assertIsNotNone(mig_tool)
            self.assertTrue(mig_tool.is_async, "migrate_memory tool must be async in FastMCP")

            # Call optimize_memory over FastMCP
            res = await mcp.call_tool("optimize_memory", {"apply_changes": False, "consolidate": False})
            self.assertTrue(len(res) > 0)
            data = json.loads(res[0][0].text)
            self.assertEqual(data.get("status"), "success")
            self.assertIn("operation_id", data)
            self.assertFalse(data.get("stats", {}).get("applied"))

            # Call migrate_memory over FastMCP
            res_mig = await mcp.call_tool("migrate_memory", {"dry_run": True})
            data_mig = json.loads(res_mig[0][0].text)
            self.assertEqual(data_mig.get("status"), "dry_run_complete")
            self.assertIn("operation_id", data_mig)

        asyncio.run(run_maintenance())

    def test_backward_compatible_sync_invocations(self):
        """F20: Calling optimize_memory and migrate_memory synchronously returns valid JSON string."""
        # Synchronous direct calls in Python
        sync_opt = optimize_memory(apply_changes=False, consolidate=False)
        self.assertIsInstance(sync_opt, str)
        data_opt = json.loads(sync_opt)
        self.assertEqual(data_opt["status"], "success")
        self.assertIn("operation_id", data_opt)

        sync_mig = migrate_memory(dry_run=True)
        self.assertIsInstance(sync_mig, str)
        data_mig = json.loads(sync_mig)
        self.assertEqual(data_mig["status"], "dry_run_complete")
        self.assertIn("operation_id", data_mig)


class TestDashboardCompletionReporting(unittest.TestCase):
    """Verify dashboard completion reporting and error statuses per F08 and F20."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="agy_dash_compl_")
        cls.token = "surface-test-dashboard-token-12345"
        os.environ["AGY_MEMORY_DASHBOARD_TOKEN"] = cls.token

        cls.server = HTTPServer(("127.0.0.1", 0), MemoryDashboardHandler)
        cls.port = cls.server.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        os.environ.pop("AGY_MEMORY_DASHBOARD_TOKEN", None)

    def _post(self, path, body=None, headers=None):
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        req_headers = {"X-Dashboard-Token": self.token}
        if headers:
            req_headers.update(headers)
        req = Request(url, data=data, headers=req_headers, method="POST")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers, resp.read().decode("utf-8")
        except HTTPError as e:
            return e.code, e.headers, e.read().decode("utf-8")

    def test_force_worker_reports_nonzero_exit_failure(self):
        """F20: /api/force-worker returns HTTP 500 when child worker exits with non-zero returncode."""
        mock_res = MagicMock()
        mock_res.returncode = 7
        mock_res.stdout = ""
        mock_res.stderr = "Worker fatal lock error"

        with patch("subprocess.run", return_value=mock_res):
            status, _, body = self._post("/api/force-worker")
            self.assertEqual(status, 500)
            data = json.loads(body)
            self.assertEqual(data.get("status"), "error")
            self.assertEqual(data.get("returncode"), 7)
            self.assertIn("Worker fatal lock error", data.get("message"))

    def test_force_worker_handles_timeout(self):
        """F20: /api/force-worker returns HTTP 504 on subprocess timeout."""
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["memory_worker.py"], timeout=180)):
            status, _, body = self._post("/api/force-worker")
            self.assertEqual(status, 504)
            data = json.loads(body)
            self.assertEqual(data.get("status"), "error")
            self.assertIn("timed out", data.get("message"))

    def test_optimize_reports_nonzero_exit_failure(self):
        """F20: /api/optimize returns HTTP 500 when optimize process exits with non-zero code."""
        mock_res = MagicMock()
        mock_res.returncode = 2
        mock_res.stdout = ""
        mock_res.stderr = "VACUUM failed: disk full"

        with patch("subprocess.run", return_value=mock_res):
            status, _, body = self._post("/api/optimize")
            self.assertEqual(status, 500)
            data = json.loads(body)
            self.assertEqual(data.get("status"), "error")
            self.assertEqual(data.get("returncode"), 2)
            self.assertIn("VACUUM failed", data.get("message"))

    def test_restore_snapshot_reports_error_status(self):
        """F20: /api/restore-snapshot returns HTTP 500 when restore returns status != ok."""
        with patch("dashboard.restore_snapshot", return_value={"status": "error", "message": "Corrupted snapshot"}):
            status, _, body = self._post("/api/restore-snapshot", body={"filename": "bad.bak"})
            self.assertEqual(status, 500)
            data = json.loads(body)
            self.assertEqual(data.get("status"), "error")
            self.assertIn("Corrupted snapshot", data.get("message"))

    def test_create_snapshot_reports_error_status(self):
        """F20: /api/create-snapshot returns HTTP 500 when snapshot returns status != ok."""
        with patch("dashboard.create_snapshot", return_value={"status": "error", "message": "Disk write failure"}):
            status, _, body = self._post("/api/create-snapshot")
            self.assertEqual(status, 500)
            data = json.loads(body)
            self.assertEqual(data.get("status"), "error")
            self.assertIn("Disk write failure", data.get("message"))

    def test_oversized_payload_returns_413(self):
        """F08/F20: POST requests with payload > 1MB are rejected with HTTP 413."""
        status, _, body = self._post(
            "/api/restore-snapshot",
            headers={"Content-Length": str(2 * 1024 * 1024)}
        )
        self.assertEqual(status, 413)
        data = json.loads(body)
        self.assertEqual(data.get("status"), "error")


if __name__ == "__main__":
    unittest.main()
