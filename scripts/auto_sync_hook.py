#!/usr/bin/env python3
"""
Antigravity Memory Auto-Sync Hook (Stop Lifecycle Hook)
Parses the session transcript and non-blockingly enqueues the conversation turn
into ~/.gemini/turn_queue.db, then triggers the background memory worker.
Returns in < 2ms to ensure zero latency for the user.
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add memory engine directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from queue_manager import enqueue_turn
except ImportError:
    enqueue_turn = None


def resolve_transcript_path(payload: dict) -> Path | None:
    """Resolve transcript.jsonl path across multiple possible brain directory roots."""
    transcript_path = payload.get("transcriptPath")
    if transcript_path and os.path.exists(transcript_path):
        return Path(transcript_path)

    conv_id = payload.get("conversationId")
    if conv_id:
        candidates = [
            Path.home() / ".gemini" / "antigravity" / "brain" / conv_id / ".system_generated" / "logs" / "transcript.jsonl",
            Path.home() / ".gemini" / "antigravity-cli" / "brain" / conv_id / ".system_generated" / "logs" / "transcript.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def extract_latest_turn(transcript_path: Path | str) -> tuple[str, str]:
    """
    Extract latest user prompt and associated assistant response.
    Stops backward scan at the latest USER_INPUT boundary and only associates
    assistant responses that belong to that user turn.
    Returns (user_prompt, assistant_response). If unanswered, returns (user_prompt, '').
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return "", ""

    assistant_parts = []
    saw_assistant_response = False
    last_user_prompt = ""

    for line in reversed(lines):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            data = json.loads(line_str)
            msg_type = data.get("type")

            if not last_user_prompt:
                if msg_type in ("PLANNER_RESPONSE", "MODEL_RESPONSE"):
                    saw_assistant_response = True
                    resp = (data.get("content") or "").strip()
                    if resp:
                        assistant_parts.append(resp)
                elif msg_type == "USER_INPUT":
                    content = data.get("content", "")
                    if "<USER_REQUEST>" in content:
                        start = content.find("<USER_REQUEST>") + len("<USER_REQUEST>")
                        end = content.find("</USER_REQUEST>")
                        if end != -1:
                            content = content[start:end].strip()
                    last_user_prompt = content.strip()
                    # Stop backward scan immediately at latest USER_INPUT boundary
                    break
        except Exception:
            continue

    if not last_user_prompt or not saw_assistant_response:
        return last_user_prompt, ""

    last_model_response = "\n\n".join(reversed(assistant_parts)).strip()

    return last_user_prompt, last_model_response


def main():
    try:
        payload_raw = sys.stdin.read()
        if not payload_raw.strip():
            print(json.dumps({}))
            return
        payload = json.loads(payload_raw)
    except Exception:
        print(json.dumps({}))
        return

    # Always respond immediately to satisfy the Stop hook contract
    print(json.dumps({}))
    sys.stdout.flush()

    t_path = resolve_transcript_path(payload)
    if not t_path:
        return

    last_user_prompt, last_model_response = extract_latest_turn(t_path)
    if not last_user_prompt or not last_model_response:
        return

    # Guard 0: If AGY is running as part of an internal memory worker or extraction script, ignore
    if os.environ.get("AGY_INTERNAL_INVOCATION") == "1":
        return

    # Guard 1: Filter internal prompts and automated background jobs
    internal_markers = [
        "Multi-Layer Cognitive Memory Engine",
        "Du bist Stephans persönlicher autonomer KI-Assistent in Zürich für das Paket",
        "PROFIL Stephan:",
        "TPA BOT REPORT",
        "STATUS-SNAPSHOT [Paket:",
        "AGY Bot Integrity Watchdog"
    ]
    if any(m in last_user_prompt for m in internal_markers):
        return

    # Resolve chat_id if originated from Telegram
    conv_id = payload.get("conversationId")
    chat_id = os.environ.get("AGY_TELEGRAM_CHAT_ID")
    if not chat_id and conv_id:
        state_file = Path.home() / ".local" / "state" / "agy-telegram" / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                for session_chat_id, session in state_data.get("sessions", {}).items():
                    if session.get("conversationId") == conv_id:
                        chat_id = str(session_chat_id)
                        break
            except Exception:
                pass

    source = "telegram" if chat_id else "hook"

    # 1. Enqueue turn in local SQLite queue (< 1ms)
    if enqueue_turn:
        enqueue_turn(
            user_prompt=last_user_prompt[:4000],
            assistant_response=last_model_response[:4000] if last_model_response else "Action executed successfully.",
            source=source,
            chat_id=chat_id or conv_id or str(t_path.resolve())
        )


if __name__ == "__main__":
    main()
