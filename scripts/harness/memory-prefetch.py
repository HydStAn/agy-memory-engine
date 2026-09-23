#!/usr/bin/env python3
"""Harness hook: inject AGY long-term memory context (Turn 0 prefetch).

Works with Codex / Claude Code / Qoder / Gemini-style hook payloads on stdin.
Emits hookSpecificOutput.additionalContext. Fails open: any error exits 0 with no output.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(os.environ.get("AGY_MEMORY_ENGINE_DIR", "/Users/__blitzzz/Documents/GitHub/agy-memory-engine"))
ENGINE_PYTHON = os.environ.get("AGY_MEMORY_PYTHON") or str(ENGINE_DIR / ".venv" / "bin" / "python")
ENGINE_SCRIPT = str(ENGINE_DIR / "agy_memory.py")
QUERY_MAX_CHARS = 8000
TIMEOUT_SEC = 20
DEBUG_LOG = os.environ.get("AGY_MEMORY_HOOK_DEBUG_LOG") or "/tmp/shared-memory-prefetch.log"
DEBUG = os.environ.get("AGY_MEMORY_HOOK_DEBUG") == "1"


def debug_log(message):
    if not DEBUG:
        return
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def emit(event_name, text):
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event_name or "SessionStart",
            "additionalContext": text,
        }
    }
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def build_query(payload):
    event = (payload.get("hook_event_name") or payload.get("event") or "").strip()
    prompt = (
        payload.get("prompt")
        or payload.get("user_prompt")
        or payload.get("query")
        or payload.get("message")
        or ""
    )
    if isinstance(prompt, dict):
        prompt = prompt.get("text") or prompt.get("content") or ""
    prompt = str(prompt).strip()
    if prompt and event.lower() in ("userpromptsubmit", "beforesubmitprompt", "usersubmit"):
        return prompt
    if prompt and not event:
        return prompt
    if event in ("SessionStart", "sessionStart", "session.start"):
        cwd = payload.get("cwd") or payload.get("workspace") or os.getcwd()
        return os.path.basename(os.path.normpath(str(cwd))).strip()
    if prompt:
        return prompt
    cwd = payload.get("cwd") or ""
    return os.path.basename(os.path.normpath(str(cwd))).strip() if cwd else ""


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError:
        debug_log("payload parse failure")
        return
    if not isinstance(payload, dict):
        payload = {}
    if os.environ.get("AGY_INTERNAL_INVOCATION") == "1":
        return
    query = build_query(payload)[:QUERY_MAX_CHARS]
    if len(query) < 3:
        debug_log("skip: short query")
        return
    if not os.path.isfile(ENGINE_SCRIPT):
        debug_log("missing engine script")
        return
    try:
        proc = subprocess.run(
            [ENGINE_PYTHON, ENGINE_SCRIPT, "prefetch", query],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except Exception as error:
        debug_log("prefetch failed: {0}".format(error))
        return
    text = (proc.stdout or "").strip()
    if not text:
        debug_log("skip: empty prefetch (rc={0})".format(proc.returncode))
        return
    event = payload.get("hook_event_name") or payload.get("event") or "SessionStart"
    debug_log("emit {0} bytes for {1}".format(len(text), event))
    emit(event, text)


if __name__ == "__main__":
    main()
