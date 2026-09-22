"""Jev relevance gate: one call per retrieval, floor filtering, fail-open."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import unittest
from unittest.mock import patch, Mock

import jev_gate


def answers(*probs):
    return {"answers": {"m" + str(i + 1): {"type": "boolean", "probability": p}
                        for i, p in enumerate(probs)}}


class JevGateTests(unittest.TestCase):
    def setUp(self):
        self.stack = patch.multiple(
            jev_gate,
            JEV_GATE_ENABLED=True,
            JEV_GATE_API_KEY="test-key",
            JEV_GATE_FLOOR=0.75,
            JEV_GATE_MIN_ITEMS=3,
            JEV_GATE_MIN_CHARS=600,
        )
        self.stack.start()
        self.addCleanup(self.stack.stop)

    def gate(self, query, texts, response=None, side_effect=None):
        mock = Mock(return_value=response if response is not None else answers(*[0.9] * len(texts)),
                    side_effect=side_effect)
        with patch.object(jev_gate, "_call_jev", mock):
            mask = jev_gate.gate_relevant(query, texts)
        return mask, mock

    def test_single_call_for_many_candidates(self):
        texts = ["alpha " * 40, "beta " * 40, "gamma " * 40, "delta " * 40]
        mask, mock = self.gate("how do I ship the release?", texts, response=answers(0.9, 0.2, 0.8, 0.01))
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mask, [True, False, True, False])
        state, questions = mock.call_args[0][0], mock.call_args[0][1]
        self.assertEqual(sorted(questions), ["m1", "m2", "m3", "m4"])
        self.assertIn("[m1]", state["criterion"])

    def test_many_small_candidates_still_gate(self):
        texts = ["a", "b", "c", "d"]
        mask, mock = self.gate("ship it", texts, response=answers(0.1, 0.1, 0.9, 0.1))
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mask, [False, False, True, False])

    def test_tiny_and_few_pass_through(self):
        texts = ["a", "b"]
        mask, mock = self.gate("ship it", texts)
        self.assertEqual(mock.call_count, 0)
        self.assertEqual(mask, [True, True])

    def test_fail_open_on_transport_error(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        mask, mock = self.gate("q", texts, side_effect=OSError("down"))
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mask, [True, True, True])

    def test_fail_open_on_malformed_answers(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        mask, _ = self.gate("q", texts, response={"answers": {}})
        self.assertEqual(mask, [True, True, True])

    def test_unusable_answer_fails_open_whole_response(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        resp = {"answers": {"m1": {"type": "boolean", "probability": 0.1},
                            "m2": {"type": "boolean", "probability": "high"},
                            "m3": {"type": "boolean", "probability": 0.4}}}
        mask, _ = self.gate("q", texts, response=resp)
        self.assertEqual(mask, [True, True, True])

    def test_partial_answers_fail_open(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        resp = {"answers": {"m1": {"type": "boolean", "probability": 0.1}}}
        mask, _ = self.gate("q", texts, response=resp)
        self.assertEqual(mask, [True, True, True])

    def test_huge_int_probability_fails_open(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        resp = {"answers": {"m1": {"type": "boolean", "probability": 10 ** 400},
                            "m2": {"type": "boolean", "probability": 0.9},
                            "m3": {"type": "boolean", "probability": 0.9}}}
        mask, _ = self.gate("q", texts, response=resp)
        self.assertEqual(mask, [True, True, True])

    def test_wrong_type_answer_fails_open(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        resp = {"answers": {"m1": {"type": "choice", "probability": 0.9},
                            "m2": {"type": "boolean", "probability": 0.9},
                            "m3": {"type": "boolean", "probability": 0.9}}}
        mask, _ = self.gate("q", texts, response=resp)
        self.assertEqual(mask, [True, True, True])

    def test_disabled_gate_never_calls(self):
        with patch.multiple(jev_gate, JEV_GATE_ENABLED=False):
            mask, mock = self.gate("q", ["x " * 200] * 3)
        self.assertEqual(mock.call_count, 0)
        self.assertEqual(mask, [True, True, True])

    def test_missing_key_never_calls(self):
        with patch.multiple(jev_gate, JEV_GATE_API_KEY=""):
            mask, mock = self.gate("q", ["x " * 200] * 3)
        self.assertEqual(mock.call_count, 0)
        self.assertEqual(mask, [True, True, True])

    def test_query_tail_survives_snippet(self):
        query = "ignore this preamble " * 200 + "FINAL TASK: gate the env config"
        texts = ["x " * 200, "y " * 200, "z " * 200]
        mock = Mock(return_value=answers(0.9, 0.9, 0.9))
        with patch.object(jev_gate, "_call_jev", mock):
            jev_gate.gate_relevant(query, texts)
        criterion = mock.call_args[0][0]["criterion"]
        self.assertIn("FINAL TASK", criterion)

    def test_deadline_bounds_trickling_response(self):
        import json
        import threading
        import time
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class Slow(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.dumps({"answers": {"m1": {"type": "boolean", "probability": 0.9}}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for off in range(0, len(body), 4):
                    self.wfile.write(body[off:off + 4])
                    self.wfile.flush()
                    time.sleep(0.05)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Slow)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        with patch.multiple(jev_gate, JEV_GATE_ENABLED=True, JEV_GATE_API_KEY="k",
                            JEV_GATE_TIMEOUT=0.2, JEV_GATE_URL="http://127.0.0.1:{0}/".format(server.server_port),
                            JEV_GATE_MIN_ITEMS=3, JEV_GATE_MIN_CHARS=0):
            started = time.monotonic()
            mask = jev_gate.gate_relevant("release", ["one one one", "two two two", "three three"])
            elapsed = time.monotonic() - started
        thread.join(timeout=2)
        server.server_close()
        self.assertEqual(mask, [True, True, True])
        self.assertLess(elapsed, 1.0)

    def test_response_size_cap_fails_open(self):
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class Big(BaseHTTPRequestHandler):
            def do_POST(self):
                body = b'{"answers": {' + b'x' * 5000
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Big)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        with patch.multiple(jev_gate, JEV_GATE_ENABLED=True, JEV_GATE_API_KEY="k",
                            JEV_GATE_TIMEOUT=2.0, JEV_GATE_URL="http://127.0.0.1:{0}/".format(server.server_port),
                            JEV_GATE_MIN_ITEMS=3, JEV_GATE_MIN_CHARS=0, RESPONSE_BYTES_CAP=100):
            mask = jev_gate.gate_relevant("release", ["one one one", "two two two", "three three"])
        thread.join(timeout=2)
        server.server_close()
        self.assertEqual(mask, [True, True, True])


class JevGateConfigTests(unittest.TestCase):
    """Bad env values disable the gate instead of breaking engine import."""

    def import_config(self, extra):
        import os
        import subprocess
        import sys
        env = dict(os.environ)
        env.update(extra)
        proc = subprocess.run(
            [sys.executable, "-c",
             "import config; print(config.JEV_GATE_ENABLED, config.JEV_GATE_FLOOR, config.JEV_GATE_TIMEOUT)"],
            capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parent.parent))
        return proc

    def test_valid_values_enable(self):
        proc = self.import_config({"AGY_MEMORY_JEV_GATE_FLOOR": "0.8", "AGY_MEMORY_JEV_GATE_TIMEOUT": "9"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split()[:2], ["True", "0.8"])

    def test_non_numeric_floor_disables(self):
        proc = self.import_config({"AGY_MEMORY_JEV_GATE_FLOOR": "oops"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split()[0], "False")

    def test_nan_floor_disables(self):
        proc = self.import_config({"AGY_MEMORY_JEV_GATE_FLOOR": "nan"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split()[0], "False")

    def test_out_of_range_timeout_disables(self):
        proc = self.import_config({"AGY_MEMORY_JEV_GATE_TIMEOUT": "0.01"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.split()[0], "False")


if __name__ == "__main__":
    unittest.main()
