#!/usr/bin/env python3
"""
Antigravity Memory Auto-Sync Hook (Stop Lifecycle Hook)
Parses the session transcript and enqueues the latest completed conversation turn.
A separate periodic worker handles debounce and recovery; SQLite contention is bounded.
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
    if conv_id and Path(conv_id).name == conv_id and conv_id not in ('.', '..'):
        roots = []
        recorded_root = payload.get('app_data_dir') or payload.get('appDataDir')
        if recorded_root:
            roots.append(Path(recorded_root).expanduser())
        roots.extend([Path.home() / '.gemini' / 'antigravity', Path.home() / '.gemini' / 'antigravity-cli'])
        candidates = [root / 'brain' / conv_id / '.system_generated' / 'logs' / 'transcript.jsonl' for root in roots]
        candidates = [path for path in candidates if path.is_file()]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    return None


def extract_latest_turn(transcript_path: Path | str, with_event_index=False):
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
        return ("", "", None) if with_event_index else ("", "")

    assistant_parts = []
    saw_assistant_response = False
    last_user_prompt = ""

    user_index = None
    for offset, line in enumerate(reversed(lines)):
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
                    user_index = len(lines) - 1 - offset
                    # Stop backward scan immediately at latest USER_INPUT boundary
                    break
        except Exception:
            continue

    if not last_user_prompt or not saw_assistant_response:
        return (last_user_prompt, "", user_index) if with_event_index else (last_user_prompt, "")

    last_model_response = "\n\n".join(reversed(assistant_parts)).strip()

    return (last_user_prompt, last_model_response, user_index) if with_event_index else (last_user_prompt, last_model_response)


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

    # The empty hook result is protocol output, not a latency guarantee.
    print(json.dumps({}))
    sys.stdout.flush()

    t_path = resolve_transcript_path(payload)
    if not t_path:
        return

    last_user_prompt, last_model_response, user_index = extract_latest_turn(t_path, with_event_index=True)
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

    # A stable transcript user-event identity permits identical text on later turns.
    event_id = f"{conv_id or str(t_path.resolve())}:{user_index}"

    # Enqueue once; keep full text so extraction does not silently lose context.
    if enqueue_turn:
        queued = enqueue_turn(
            user_prompt=last_user_prompt,
            assistant_response=last_model_response,
            source=source,
            chat_id=chat_id or conv_id or str(t_path.resolve()),
            event_id=event_id
        )
        if not queued:
            print("Memory queue busy or input rejected; turn was not confirmed queued", file=sys.stderr)


if __name__ == "__main__":
    main()
