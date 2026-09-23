#!/usr/bin/env python3
"""Harness hook: enqueue the latest turn into the shared AGY memory queue.

Reads a Claude/Qoder/Gemini/Codex-style hook payload from stdin on Stop /
SessionEnd / PostInvocation. Parses a transcript.jsonl when present; otherwise
uses any prompt/response fields the host provides. Fails open: never blocks the host.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ENGINE_DIR = Path(os.environ.get("AGY_MEMORY_ENGINE_DIR", "/Users/__blitzzz/Documents/GitHub/agy-memory-engine"))
sys.path.insert(0, str(ENGINE_DIR))

DEFAULT_SOURCE = os.environ.get("AGY_MEMORY_SOURCE") or "agent"
DEBUG = os.environ.get("AGY_MEMORY_HOOK_DEBUG") == "1"
DEBUG_LOG = os.environ.get("AGY_MEMORY_HOOK_DEBUG_LOG") or "/tmp/shared-memory-enqueue.log"
MIN_PROMPT = 8


def debug_log(message):
    if not DEBUG:
        return
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        pass


def resolve_source(payload):
    for key in ("source", "agent", "agentName", "host", "harness"):
        value = payload.get(key)
        if value:
            return str(value).strip().lower().replace(" ", "-")[:40]
    env = os.environ.get("AGY_MEMORY_SOURCE")
    if env:
        return env.strip().lower()[:40]
    # Infer from hook path / common fields
    blob = json.dumps(payload)[:2000].lower()
    for token in ("qoder", "codex", "claude", "cursor", "mimocode", "gemini", "antigravity", "opencode"):
        if token in blob:
            return token
    return DEFAULT_SOURCE


def resolve_transcript(payload):
    candidates = []
    for key in (
        "transcript_path",
        "transcriptPath",
        "transcript",
        "log_path",
        "session_path",
        "sessionPath",
    ):
        value = payload.get(key)
        if value:
            candidates.append(Path(str(value)).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    conv_id = payload.get("conversationId") or payload.get("session_id") or payload.get("sessionId")
    if conv_id and Path(str(conv_id)).name == str(conv_id):
        roots = [
            Path.home() / ".gemini" / "antigravity",
            Path.home() / ".gemini" / "antigravity-cli",
            Path.home() / ".claude" / "projects",
        ]
        for root in roots:
            for path in root.rglob("transcript.jsonl"):
                if str(conv_id) in str(path):
                    return path
    return None


def extract_from_transcript(path: Path):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", ""
    assistant_parts = []
    last_user = ""
    user_index = None
    for offset, line in enumerate(reversed(lines)):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        msg_type = data.get("type") or data.get("role") or ""
        content = data.get("content") or data.get("text") or data.get("message") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text") or item.get("content") or "")
                else:
                    parts.append(str(item))
            content = "\n".join(p for p in parts if p)
        content = str(content).strip()
        if not last_user:
            if msg_type in ("USER_INPUT", "user", "human") or data.get("role") == "user":
                if "<USER_REQUEST>" in content:
                    start = content.find("<USER_REQUEST>") + len("<USER_REQUEST>")
                    end = content.find("</USER_REQUEST>")
                    if end != -1:
                        content = content[start:end].strip()
                last_user = content
                user_index = len(lines) - 1 - offset
                break
            if msg_type in ("PLANNER_RESPONSE", "MODEL_RESPONSE", "assistant") or data.get("role") == "assistant":
                if content:
                    assistant_parts.append(content)
        else:
            break
    if not last_user:
        # reverse direction fallback: last user in chronological order
        for line in lines:
            try:
                data = json.loads(line)
            except ValueError:
                continue
            msg_type = data.get("type") or data.get("role") or ""
            if msg_type in ("USER_INPUT", "user") or data.get("role") == "user":
                content = data.get("content") or data.get("text") or ""
                if isinstance(content, str) and content.strip():
                    last_user = content.strip()
        if last_user and not assistant_parts:
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                msg_type = data.get("type") or data.get("role") or ""
                if msg_type in ("MODEL_RESPONSE", "PLANNER_RESPONSE", "assistant") or data.get("role") == "assistant":
                    content = data.get("content") or data.get("text") or ""
                    if isinstance(content, str) and content.strip():
                        assistant_parts.append(content.strip())
                        break
    assistant = "\n\n".join(reversed(assistant_parts)).strip()
    return last_user, assistant


def extract_from_payload(payload):
    user = ""
    assistant = ""
    for key in ("prompt", "user_prompt", "userPrompt", "query", "lastUserMessage", "input"):
        value = payload.get(key)
        if value:
            if isinstance(value, dict):
                value = value.get("text") or value.get("content") or ""
            user = str(value).strip()
            if user:
                break
    for key in ("last_assistant_message", "assistant_response", "assistantResponse", "response", "finalText", "output"):
        value = payload.get(key)
        if value:
            if isinstance(value, dict):
                value = value.get("text") or value.get("content") or ""
            assistant = str(value).strip()
            if assistant:
                break
    return user, assistant


def main():
    try:
        if os.environ.get("AGY_INTERNAL_INVOCATION") == "1":
            print("{}")
            return
        try:
            from queue_manager import enqueue_turn
        except ImportError:
            debug_log("queue_manager import failed")
            print("{}")
            return
        raw = sys.stdin.read()
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except ValueError:
            print("{}")
            return
        if not isinstance(payload, dict):
            payload = {}
        print("{}")
        sys.stdout.flush()

        source = resolve_source(payload)
        chat_id = (
            payload.get("session_id")
            or payload.get("sessionId")
            or payload.get("conversationId")
            or payload.get("cwd")
            or source
        )
        transcript = resolve_transcript(payload)
        if transcript:
            user_prompt, assistant_response = extract_from_transcript(transcript)
        else:
            user_prompt, assistant_response = extract_from_payload(payload)
        if not user_prompt and not assistant_response:
            debug_log("no extractable turn for source={0}".format(source))
            return
        if not user_prompt or len(user_prompt.strip()) < MIN_PROMPT:
            debug_log("prompt too short")
            return
        event_id = "{0}:{1}".format(
            chat_id,
            hashlib.sha1((user_prompt[:200] + assistant_response[:200]).encode("utf-8", "ignore")).hexdigest()[:16],
        )
        queued = enqueue_turn(
            user_prompt=user_prompt,
            assistant_response=assistant_response or "",
            source=source,
            chat_id=str(chat_id),
            event_id=event_id,
        )
        debug_log("queued={0} source={1}".format(queued, source))
    except Exception as error:
        debug_log("enqueue error: {0}".format(error))
        try:
            print("{}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
