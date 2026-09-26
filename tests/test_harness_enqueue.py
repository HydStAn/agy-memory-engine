"""Transcript parsing for the shared harness enqueue hook (scripts/harness/memory-enqueue.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_environment  # noqa: F401

import importlib.util
import json
import tempfile
import unittest

HOOK_PATH = Path(__file__).resolve().parent.parent / "scripts" / "harness" / "memory-enqueue.py"
_spec = importlib.util.spec_from_file_location("memory_enqueue_hook", HOOK_PATH)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _write_transcript(entries):
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return Path(handle.name)


def _claude_user(content, **extra):
    return {"type": "user", "message": {"role": "user", "content": content}, **extra}


def _claude_assistant(*blocks, **extra):
    return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}, **extra}


class ClaudeCodeTranscriptTests(unittest.TestCase):
    def test_pairs_real_prompt_with_text_reply_across_tool_calls(self):
        path = _write_transcript([
            _claude_user("older prompt that should be ignored"),
            _claude_assistant({"type": "text", "text": "older reply"}),
            {"type": "attachment", "attachment": {"type": "hook_additional_context"}},
            _claude_user("wire the memory engine into claude code"),
            _claude_assistant({"type": "thinking", "thinking": "private reasoning"}),
            _claude_assistant({"type": "text", "text": "Checking the hooks first."}),
            _claude_assistant({"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}),
            _claude_user([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"}]),
            _claude_assistant({"type": "text", "text": "Registered the MCP server."}),
        ])
        user, assistant = hook.extract_from_transcript(path)
        self.assertEqual(user, "wire the memory engine into claude code")
        self.assertEqual(assistant, "Checking the hooks first.\n\nRegistered the MCP server.")

    def test_reads_text_blocks_from_list_prompt(self):
        path = _write_transcript([
            _claude_user([
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                {"type": "text", "text": "what is in this screenshot"},
            ]),
            _claude_assistant({"type": "text", "text": "A dashboard."}),
        ])
        self.assertEqual(hook.extract_from_transcript(path), ("what is in this screenshot", "A dashboard."))

    def test_skips_meta_and_sidechain_entries(self):
        path = _write_transcript([
            _claude_user("the real prompt from the user"),
            _claude_assistant({"type": "text", "text": "main reply"}),
            _claude_user("subagent task prompt", isSidechain=True),
            _claude_assistant({"type": "text", "text": "subagent reply"}, isSidechain=True),
            _claude_user("Caveat: local command output follows", isMeta=True),
        ])
        self.assertEqual(hook.extract_from_transcript(path), ("the real prompt from the user", "main reply"))


class AntigravityTranscriptTests(unittest.TestCase):
    def test_unwraps_user_request_and_joins_planner_responses(self):
        path = _write_transcript([
            {"type": "USER_INPUT", "content": "<USER_REQUEST>deploy the worker</USER_REQUEST>"},
            {"type": "PLANNER_RESPONSE", "content": "Planning."},
            {"type": "MODEL_RESPONSE", "content": [{"text": "Deployed."}]},
        ])
        self.assertEqual(hook.extract_from_transcript(path), ("deploy the worker", "Planning.\n\nDeployed."))


if __name__ == "__main__":
    unittest.main()
