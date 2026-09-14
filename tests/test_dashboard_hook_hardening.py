"""
Hermetic test suite for Dashboard & Hook Hardening (Phases 6 & 7: F08, F20, F11, F12)
Tests:
- F08/F20: Network binding (127.0.0.1 default), token authentication for mutating POST endpoints,
           origin-restricted CORS (no wildcard * on mutations), token usability by browser,
           and safe DOM/inline event handler escaping.
- F11/F12: Multi-root brain transcript resolution and strict turn boundary scanning
           (never pairing unanswered prompt with prior turn's response).
"""

# Set temporary paths before importing modules that capture configuration defaults.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401


import os
import json
import time
import shutil
import tempfile
import threading
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import urllib.parse

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
import dashboard
from dashboard import (
    get_or_create_dashboard_token,
    is_local_origin,
    is_ip_in_private_or_mesh_range,
    is_trusted_host,
    is_trusted_origin,
    MemoryDashboardHandler,
    HTML_TEMPLATE,
)
from scripts.auto_sync_hook import (
    resolve_transcript_path,
    extract_latest_turn,
)


class TestDashboardHardening(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="agy_dash_test_")
        self.orig_home = os.environ.get("HOME")
        self.orig_dash_token = os.environ.get("AGY_MEMORY_DASHBOARD_TOKEN")
        self.orig_dash_path = os.environ.get("AGY_MEMORY_DASHBOARD_TOKEN_PATH")
        self.orig_dash_host = os.environ.get("AGY_MEMORY_DASHBOARD_HOST")

        # Point HOME and token paths to temp directory
        os.environ["HOME"] = self.temp_dir
        self.token_file = Path(self.temp_dir) / ".gemini" / "dashboard.token"
        os.environ["AGY_MEMORY_DASHBOARD_TOKEN_PATH"] = str(self.token_file)
        if "AGY_MEMORY_DASHBOARD_TOKEN" in os.environ:
            del os.environ["AGY_MEMORY_DASHBOARD_TOKEN"]

    def tearDown(self):
        if self.orig_home is not None:
            os.environ["HOME"] = self.orig_home
        else:
            os.environ.pop("HOME", None)

        if self.orig_dash_token is not None:
            os.environ["AGY_MEMORY_DASHBOARD_TOKEN"] = self.orig_dash_token
        else:
            os.environ.pop("AGY_MEMORY_DASHBOARD_TOKEN", None)

        if self.orig_dash_path is not None:
            os.environ["AGY_MEMORY_DASHBOARD_TOKEN_PATH"] = self.orig_dash_path
        else:
            os.environ.pop("AGY_MEMORY_DASHBOARD_TOKEN_PATH", None)

        if self.orig_dash_host is not None:
            os.environ["AGY_MEMORY_DASHBOARD_HOST"] = self.orig_dash_host
        else:
            os.environ.pop("AGY_MEMORY_DASHBOARD_HOST", None)

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_network_binding_is_localhost(self):
        """F08: DASHBOARD_HOST should default to 127.0.0.1, not 0.0.0.0."""
        self.assertEqual(config.DASHBOARD_HOST, "127.0.0.1")
        self.assertEqual(dashboard.DASHBOARD_HOST, "127.0.0.1")

    def test_token_creation_and_persistence(self):
        """F08: Token should be generated and saved with 0600 permissions."""
        token1 = get_or_create_dashboard_token(str(self.token_file))
        self.assertTrue(self.token_file.exists())
        self.assertTrue(len(token1) >= 32)
        # Check permissions (0600)
        mode = self.token_file.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

        # Subsequent call should load the same token
        token2 = get_or_create_dashboard_token(str(self.token_file))
        self.assertEqual(token1, token2)

    def test_token_env_override(self):
        """F08: Environment variable AGY_MEMORY_DASHBOARD_TOKEN takes precedence."""
        os.environ["AGY_MEMORY_DASHBOARD_TOKEN"] = "custom-env-secret-token-12345"
        token = get_or_create_dashboard_token(str(self.token_file))
        self.assertEqual(token, "custom-env-secret-token-12345")

    def test_local_origin_validation(self):
        """F08: CORS origin checking must strictly allow only local loopback origins."""
        self.assertTrue(is_local_origin("http://127.0.0.1:8085"))
        self.assertTrue(is_local_origin("http://127.0.0.1"))
        self.assertTrue(is_local_origin("http://localhost:3000"))
        self.assertTrue(is_local_origin("http://localhost"))
        self.assertTrue(is_local_origin("http://[::1]:8085"))

        # Disallowed cross-origin domains
        self.assertFalse(is_local_origin("https://evil.com"))
        self.assertFalse(is_local_origin("http://evil-localhost.com"))
        self.assertFalse(is_local_origin("http://127.0.0.1.evil.com"))
        self.assertFalse(is_local_origin("null"))
        self.assertFalse(is_local_origin(""))

    def test_is_ip_in_private_or_mesh_range(self):
        """Verify IP detection for private networks, loopback, link-local, and Tailscale mesh."""
        # Loopback
        self.assertTrue(is_ip_in_private_or_mesh_range("127.0.0.1"))
        self.assertTrue(is_ip_in_private_or_mesh_range("::1"))
        self.assertTrue(is_ip_in_private_or_mesh_range("[::1]"))

        # RFC 1918 Private IPv4
        self.assertTrue(is_ip_in_private_or_mesh_range("10.0.0.1"))
        self.assertTrue(is_ip_in_private_or_mesh_range("172.16.0.1"))
        self.assertTrue(is_ip_in_private_or_mesh_range("192.168.1.100"))

        # Tailscale / RFC 6598 CGNAT IPv4 (100.64.0.0/10)
        self.assertTrue(is_ip_in_private_or_mesh_range("100.64.0.1"))
        self.assertTrue(is_ip_in_private_or_mesh_range("100.115.92.5"))
        self.assertTrue(is_ip_in_private_or_mesh_range("100.127.255.254"))

        # IPv6 ULA (Tailscale IPv6) and link-local
        self.assertTrue(is_ip_in_private_or_mesh_range("fd7a:115c:a1e0:ab12:4843:cd96:6276:1234"))
        self.assertTrue(is_ip_in_private_or_mesh_range("fe80::1"))
        self.assertTrue(is_ip_in_private_or_mesh_range("[fe80::1%eth0]"))

        # Non-private public IPs and invalid inputs
        self.assertFalse(is_ip_in_private_or_mesh_range("8.8.8.8"))
        self.assertFalse(is_ip_in_private_or_mesh_range("1.1.1.1"))
        self.assertFalse(is_ip_in_private_or_mesh_range("evil.com"))
        self.assertFalse(is_ip_in_private_or_mesh_range(""))
        self.assertFalse(is_ip_in_private_or_mesh_range(None))

    def test_trusted_host_default_loopback(self):
        """Default loopback host binding strictly rejects foreign and non-local hosts."""
        self.assertTrue(is_trusted_host("localhost", server_host="127.0.0.1", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("127.0.0.1", server_host="127.0.0.1", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("::1", server_host="127.0.0.1", allowed_hosts=[]))

        # Rejects private IPs and foreign domains when bound strictly to loopback
        self.assertFalse(is_trusted_host("192.168.1.50", server_host="127.0.0.1", allowed_hosts=[]))
        self.assertFalse(is_trusted_host("100.64.1.2", server_host="127.0.0.1", allowed_hosts=[]))
        self.assertFalse(is_trusted_host("evil.com", server_host="127.0.0.1", allowed_hosts=[]))
        self.assertFalse(is_trusted_host("attacker.invalid", server_host="127.0.0.1", allowed_hosts=[]))

    def test_trusted_host_wildcard_bind_permits_private_and_mesh(self):
        """When bound to 0.0.0.0, private LAN and Tailscale mesh IPs are permitted while DNS rebinding is blocked."""
        # Permitted on 0.0.0.0
        self.assertTrue(is_trusted_host("127.0.0.1", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("localhost", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("192.168.1.100", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("10.0.0.5", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("100.64.1.2", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("100.115.92.5", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertTrue(is_trusted_host("fd7a:115c:a1e0::1", server_host="0.0.0.0", allowed_hosts=[]))

        # Public IPs and arbitrary domains are still blocked (retaining DNS rebinding protection)
        self.assertFalse(is_trusted_host("8.8.8.8", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertFalse(is_trusted_host("evil.com", server_host="0.0.0.0", allowed_hosts=[]))
        self.assertFalse(is_trusted_host("attacker.invalid", server_host="0.0.0.0", allowed_hosts=[]))

    def test_trusted_host_configured_allowed_hosts(self):
        """Configured allowed hosts (exact and wildcard) are accepted across bindings."""
        allowed = ["my-box.ts.net", "*.tailnet.ts.net", "custom.internal"]
        self.assertTrue(is_trusted_host("my-box.ts.net", server_host="127.0.0.1", allowed_hosts=allowed))
        self.assertTrue(is_trusted_host("node1.tailnet.ts.net", server_host="127.0.0.1", allowed_hosts=allowed))
        self.assertTrue(is_trusted_host("custom.internal", server_host="127.0.0.1", allowed_hosts=allowed))

        # Unmatched domains rejected
        self.assertFalse(is_trusted_host("other.ts.net", server_host="127.0.0.1", allowed_hosts=allowed))
        self.assertFalse(is_trusted_host("evil.com", server_host="127.0.0.1", allowed_hosts=allowed))


class TestDashboardHttpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="agy_dash_srv_")
        cls.token = "test-secret-token-xyz-98765"
        os.environ["AGY_MEMORY_DASHBOARD_TOKEN"] = cls.token

        # Bind to 127.0.0.1 on ephemeral port 0
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

    def _post(self, path, headers=None, body=None):
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        req = Request(url, data=data, headers=headers or {}, method="POST")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers, resp.read().decode("utf-8")
        except HTTPError as e:
            return e.code, e.headers, e.read().decode("utf-8")

    def _get(self, path, headers=None):
        url = f"{self.base_url}{path}"
        req = Request(url, headers=headers or {}, method="GET")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers, resp.read().decode("utf-8")
        except HTTPError as e:
            return e.code, e.headers, e.read().decode("utf-8")

    def _options(self, path, headers=None):
        url = f"{self.base_url}{path}"
        req = Request(url, headers=headers or {}, method="OPTIONS")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers, resp.read().decode("utf-8")
        except HTTPError as e:
            return e.code, e.headers, e.read().decode("utf-8")

    def test_post_endpoints_require_auth(self):
        """F08: Unauthenticated POST requests must be rejected with 401 Unauthorized."""
        endpoints = [
            "/api/force-worker",
            "/api/clear-processed-queue",
            "/api/optimize",
            "/api/create-snapshot",
            "/api/restore-snapshot",
        ]
        for ep in endpoints:
            status, _, body = self._post(ep)
            self.assertEqual(status, 401, f"Endpoint {ep} should require auth")
            data = json.loads(body)
            self.assertEqual(data.get("error"), "Unauthorized")

    def test_post_endpoints_reject_invalid_token(self):
        """F08: POST requests with invalid token must receive 401."""
        status, _, body = self._post(
            "/api/force-worker",
            headers={"X-Dashboard-Token": "wrong-token"}
        )
        self.assertEqual(status, 401)

    def test_post_auth_methods_accepted(self):
        """F08: Valid token via X-Dashboard-Token, Bearer auth, or query param succeeds."""
        # 1. Via X-Dashboard-Token header
        status1, _, _ = self._post(
            "/api/clear-processed-queue",
            headers={"X-Dashboard-Token": self.token}
        )
        self.assertEqual(status1, 200)

        # 2. Via Authorization Bearer header
        status2, _, _ = self._post(
            "/api/clear-processed-queue",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        self.assertEqual(status2, 200)

        # 3. Via query param
        status3, _, _ = self._post(
            f"/api/clear-processed-queue?token={self.token}"
        )
        self.assertEqual(status3, 200)

    def test_cors_blocks_cross_origin(self):
        """F08: Cross-origin requests must NOT receive Access-Control-Allow-Origin."""
        status, headers, _ = self._get(
            "/api/stats",
            headers={"Origin": "https://evil.com"}
        )
        self.assertNotIn("Access-Control-Allow-Origin", headers)

        # Cross-origin preflight OPTIONS must be rejected with 403
        opt_status, _, _ = self._options(
            "/api/optimize",
            headers={"Origin": "https://evil.com"}
        )
        self.assertEqual(opt_status, 403)

    def test_cors_allows_local_origin(self):
        """F08: Local origins receive explicit Access-Control-Allow-Origin matching origin."""
        local_origin = f"http://127.0.0.1:{self.port}"
        status, headers, _ = self._get(
            "/api/stats",
            headers={"Origin": local_origin}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), local_origin)
        self.assertNotEqual(headers.get("Access-Control-Allow-Origin"), "*")

        # Local OPTIONS preflight succeeds with 204
        opt_status, opt_headers, _ = self._options(
            "/api/optimize",
            headers={"Origin": local_origin}
        )
        self.assertEqual(opt_status, 204)
        self.assertEqual(opt_headers.get("Access-Control-Allow-Origin"), local_origin)

    def test_foreign_host_and_origin_cannot_obtain_token_or_mutate(self):
        status, _, body = self._get("/", headers={"Host": "attacker.invalid"})
        self.assertEqual(status, 403)
        self.assertNotIn(self.token, body)
        status, _, _ = self._post("/api/clear-processed-queue", headers={
            "X-Dashboard-Token": self.token, "Origin": "https://attacker.invalid"})
        self.assertEqual(status, 403)

    def test_server_allowed_hosts_and_private_network(self):
        """Verify server responds to allowed hosts and private IPs when configured."""
        try:
            self.server.allowed_hosts = ["my-mac.ts.net"]
            self.server.allow_private = True

            # Configured allowed host accepted (reaches auth check, returns 401 instead of 403)
            status, _, _ = self._get("/", headers={"Host": f"my-mac.ts.net:{self.port}"})
            self.assertEqual(status, 401)

            # Private IP accepted
            status, _, _ = self._get("/", headers={"Host": f"192.168.1.100:{self.port}"})
            self.assertEqual(status, 401)

            # Tailscale CGNAT IP accepted
            status, _, _ = self._get("/", headers={"Host": f"100.64.1.2:{self.port}"})
            self.assertEqual(status, 401)

            # Foreign domain rejected with 403 Forbidden
            status, _, _ = self._get("/", headers={"Host": f"attacker.invalid:{self.port}"})
            self.assertEqual(status, 403)
        finally:
            self.server.allowed_hosts = []
            self.server.allow_private = False

    def test_token_is_safe_inside_script(self):
        payload = '</script><img src=x onerror=alert(1)>'
        with patch.dict(os.environ, {"AGY_MEMORY_DASHBOARD_TOKEN": payload}):
            status, _, body = self._get(f"/?token={urllib.parse.quote(payload)}")
        self.assertEqual(status, 200)
        self.assertNotIn(payload, body)
        self.assertIn('\\u003c/script>', body)

    def test_unauthenticated_get_does_not_disclose_token(self):
        """BR02: Unauthenticated GET / must return 401 and never disclose secret token."""
        status, _, body = self._get("/")
        self.assertEqual(status, 401)
        self.assertNotIn(self.token, body)
        self.assertNotIn("const DASHBOARD_TOKEN =", body)

    def test_browser_token_usability(self):
        """F08 / BR02: Authenticated GET / serves HTML with embedded token and fetch POST calls supply token."""
        status, _, html = self._get(f"/?token={self.token}")
        self.assertEqual(status, 200)
        # Token must be embedded for authenticated dashboard browser scripts
        self.assertIn(f'const DASHBOARD_TOKEN = "{self.token}";', html)
        self.assertNotIn("{{DASHBOARD_TOKEN}}", html)

        # Mutating endpoints in script must pass 'X-Dashboard-Token': DASHBOARD_TOKEN
        self.assertIn("'X-Dashboard-Token': DASHBOARD_TOKEN", html)


class TestDashboardXssEscaping(unittest.TestCase):
    def test_template_escapes_metadata_fields(self):
        """F20: Ensure DOM rendering templates escape title, id, topic, status, period, category."""
        # Verify escapeHtml is defined and handles quotes and tags
        self.assertIn("function escapeHtml(text)", HTML_TEMPLATE)

        # Verify key metadata interpolations use escapeHtml in the client code
        # Facts
        self.assertIn("${escapeHtml(f.id)}", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(f.category)}", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(f.fact)}", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(f.updated_at)}", HTML_TEMPLATE)

        # Episodes
        self.assertIn("${escapeHtml(e.title || e.id)}", HTML_TEMPLATE)
        self.assertIn("(${escapeHtml(e.topic)})", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(e.status)}", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(e.period || '-')}", HTML_TEMPLATE)

        # Learnings
        self.assertIn("${escapeHtml(l.id)}", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(l.category)}", HTML_TEMPLATE)

        # Queue & Batches
        self.assertIn("Turn #${escapeHtml(t.id)}", HTML_TEMPLATE)
        self.assertIn("[${escapeHtml(t.source || 'telegram')}]", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(b.batch_id)}", HTML_TEMPLATE)

        # Audit
        self.assertIn("${escapeHtml(a.target_id)}", HTML_TEMPLATE)
        self.assertIn("${escapeHtml(a.category)}", HTML_TEMPLATE)

    def test_inline_handlers_use_safe_indexing(self):
        """F20: Inline event handlers must not interpolate untrusted strings directly."""
        # Snapshots restore button must use numeric index, not raw filename interpolation
        self.assertIn("onclick=\"handleRestoreSnapshotClick(${idx})\"", HTML_TEMPLATE)
        self.assertNotIn("onclick=\"promptRestoreSnapshot('${escapeHtml(s.filename)}'", HTML_TEMPLATE)

        # Turn toggle must not execute raw string expressions
        self.assertIn("data-turn-id=\"${escapeHtml(t.id)}\"", HTML_TEMPLATE)
        self.assertIn("parseInt(this.dataset.turnId, 10)", HTML_TEMPLATE)


class TestAutoSyncHookHardening(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="agy_hook_test_")
        self.orig_home = os.environ.get("HOME")
        os.environ["HOME"] = self.temp_dir

    def tearDown(self):
        if self.orig_home is not None:
            os.environ["HOME"] = self.orig_home
        else:
            os.environ.pop("HOME", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multi_root_brain_resolution(self):
        """F11: Check both ~/.gemini/antigravity/brain and ~/.gemini/antigravity-cli/brain."""
        conv_id = "conv-multi-root-123"

        # Root 1: antigravity
        root1_file = Path(self.temp_dir) / ".gemini" / "antigravity" / "brain" / conv_id / ".system_generated" / "logs" / "transcript.jsonl"
        root1_file.parent.mkdir(parents=True, exist_ok=True)
        root1_file.write_text('{"type": "USER_INPUT", "content": "hello root 1"}\n{"type": "PLANNER_RESPONSE", "content": "hi 1"}\n')

        res1 = resolve_transcript_path({"conversationId": conv_id})
        self.assertIsNotNone(res1)
        self.assertEqual(res1.resolve(), root1_file.resolve())

        # Remove root 1 and create Root 2: antigravity-cli
        root1_file.unlink()
        root2_file = Path(self.temp_dir) / ".gemini" / "antigravity-cli" / "brain" / conv_id / ".system_generated" / "logs" / "transcript.jsonl"
        root2_file.parent.mkdir(parents=True, exist_ok=True)
        root2_file.write_text('{"type": "USER_INPUT", "content": "hello root 2"}\n{"type": "PLANNER_RESPONSE", "content": "hi 2"}\n')

        res2 = resolve_transcript_path({"conversationId": conv_id})
        self.assertIsNotNone(res2)
        self.assertEqual(res2.resolve(), root2_file.resolve())

    def test_hook_preserves_non_telegram_conversation_identity(self):
        from scripts import auto_sync_hook as hook
        transcript = Path(self.temp_dir) / 'transcript.jsonl'
        transcript.write_text(json.dumps({"type": "USER_INPUT", "content": "Remember this deployment host"}) + '\n' +
                              json.dumps({"type": "MODEL_RESPONSE", "content": "Stored deployment host"}) + '\n')
        payload = json.dumps({"conversationId": "conversation-a", "transcriptPath": str(transcript)})
        with patch.dict(os.environ, {"AGY_INTERNAL_INVOCATION": "0", "AGY_TELEGRAM_CHAT_ID": ""}), patch.object(hook.sys, 'stdin', io.StringIO(payload)), patch.object(hook, 'enqueue_turn') as enqueue, redirect_stdout(io.StringIO()):
            hook.main()
        self.assertEqual(enqueue.call_args.kwargs['source'], 'hook')
        self.assertEqual(enqueue.call_args.kwargs['chat_id'], 'conversation-a')

    def test_turn_boundary_scan_completed_turn(self):
        """F12: Latest turn with user input and model response is extracted cleanly."""
        t_file = Path(self.temp_dir) / "transcript.jsonl"
        lines = [
            json.dumps({"type": "USER_INPUT", "content": "Turn 1 question"}),
            json.dumps({"type": "PLANNER_RESPONSE", "content": "Turn 1 answer"}),
            json.dumps({"type": "USER_INPUT", "content": "<USER_REQUEST>Turn 2 question</USER_REQUEST>"}),
            json.dumps({"type": "PLANNER_RESPONSE", "content": "Turn 2 answer"}),
        ]
        t_file.write_text("\n".join(lines) + "\n")

        prompt, response = extract_latest_turn(t_file)
        self.assertEqual(prompt, "Turn 2 question")
        self.assertEqual(response, "Turn 2 answer")

    def test_turn_boundary_scan_never_pairs_unanswered_prompt_with_prior_response(self):
        """F12: An unanswered prompt at the end must NEVER be paired with prior turn's response."""
        t_file = Path(self.temp_dir) / "transcript.jsonl"
        lines = [
            json.dumps({"type": "USER_INPUT", "content": "Turn 1 question"}),
            json.dumps({"type": "PLANNER_RESPONSE", "content": "Turn 1 answer"}),
            json.dumps({"type": "USER_INPUT", "content": "Turn 2 unanswered question"}),
            # No PLANNER_RESPONSE after Turn 2
        ]
        t_file.write_text("\n".join(lines) + "\n")

        prompt, response = extract_latest_turn(t_file)
        self.assertEqual(prompt, "Turn 2 unanswered question")
        # Must be empty - NEVER "Turn 1 answer"
        self.assertEqual(response, "")

    def test_turn_boundary_scan_multi_chunk_response(self):
        """F12: Multi-step model responses for the current turn are concatenated."""
        t_file = Path(self.temp_dir) / "transcript.jsonl"
        lines = [
            json.dumps({"type": "USER_INPUT", "content": "Prior prompt"}),
            json.dumps({"type": "PLANNER_RESPONSE", "content": "Prior answer"}),
            json.dumps({"type": "USER_INPUT", "content": "Current prompt"}),
            json.dumps({"type": "PLANNER_RESPONSE", "content": "Step 1"}),
            json.dumps({"type": "PLANNER_RESPONSE", "content": "Step 2"}),
        ]
        t_file.write_text("\n".join(lines) + "\n")

        prompt, response = extract_latest_turn(t_file)
        self.assertEqual(prompt, "Current prompt")
        self.assertEqual(response, "Step 1\n\nStep 2")


if __name__ == "__main__":
    unittest.main()
