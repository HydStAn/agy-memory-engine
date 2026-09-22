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

    def test_unparsable_probability_keeps_candidate(self):
        texts = ["x " * 200, "y " * 200, "z " * 200]
        resp = {"answers": {"m1": {"type": "boolean", "probability": 0.1},
                            "m2": {"type": "boolean", "probability": "high"},
                            "m3": {"type": "boolean", "probability": 0.4}}}
        mask, _ = self.gate("q", texts, response=resp)
        self.assertEqual(mask, [False, True, False])

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


if __name__ == "__main__":
    unittest.main()
