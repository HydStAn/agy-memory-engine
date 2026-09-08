#!/usr/bin/env python3
"""
Autonomous Calm Memory Worker for Antigravity (AGY).
Processes conversation batches from ~/.gemini/turn_queue.db only when:
1. The last message is at least 5 minutes old (inactivity debounce), OR
2. The oldest pending message has waited for 30 minutes (max timeout), OR
3. Explicitly forced via --force (e.g. /remember command).
"""

import os
import sys
import fcntl
import argparse
import datetime
import uuid
import subprocess
from pathlib import Path

# Add memory engine directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from queue_manager import (
    get_pending_stats,
    get_pending_turns,
    mark_turn_status,
    prune_processed_turns,
    QUEUE_DB_PATH
)
from agy_memory import is_trivial_prompt, sync_turn
from config import (
    INACTIVITY_THRESHOLD_SECONDS,
    MAX_WAIT_THRESHOLD_SECONDS,
    SEND_TELEGRAM_BIN
)

from agy_memory import SyncBusyError, SyncExtractionError

LOCK_FILE = Path(os.environ.get("AGY_WORKER_LOCK_PATH", str(Path.home() / ".gemini" / "memory_worker.lock")))


def send_telegram_notification(message: str, chat_id: str = None) -> bool:
    """Send an informative notification to Telegram via send_telegram.py."""
    if SEND_TELEGRAM_BIN.exists():
        try:
            cmd = ["python3", str(SEND_TELEGRAM_BIN)]
            if chat_id:
                cmd.extend(["--chat-id", str(chat_id)])
            else:
                cmd.append("--reports")
            cmd.append(message)
            res = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=15
            )
            return res.returncode == 0
        except Exception as e:
            sys.stderr.write(f"Failed to send telegram notification: {e}\n")
            return False
    return False


def format_notification(changes: dict) -> str:
    """Format extracted memory changes into an ultra-compact Telegram notification with descriptive texts."""
    facts = changes.get("facts", [])
    episodes = changes.get("episodes", [])
    learnings = changes.get("learnings", [])
    links = changes.get("entity_links", [])

    total_count = len(facts) + len(episodes) + len(learnings) + len(links)
    if total_count == 0:
        return ""

    counts_summary = []
    if facts: counts_summary.append(f"{len(facts)} fact{'s' if len(facts) > 1 else ''}")
    if episodes: counts_summary.append(f"{len(episodes)} episode{'s' if len(episodes) > 1 else ''}")
    if learnings: counts_summary.append(f"{len(learnings)} learning{'s' if len(learnings) > 1 else ''}")
    if links: counts_summary.append(f"{len(links)} link{'s' if len(links) > 1 else ''}")

    lines = [f"🧠 *Autonomous memory updated* (`{', '.join(counts_summary)}`)"]

    items = []
    for f in facts:
        icon = "🔄" if f.get("is_update") else "➕"
        text = f.get("fact", "").replace("\n", " ").strip()
        short_text = (text[:80] + "…") if len(text) > 80 else text
        items.append(f"{icon} {short_text}")

    for ep in episodes:
        icon = "🔄" if ep.get("is_update") else "➕"
        title = ep.get("title", "").strip() or ep.get("id")
        items.append(f"{icon} {title}")

    for lr in learnings:
        icon = "🔄" if lr.get("is_update") else "➕"
        text = lr.get("insight", "").replace("\n", " ").strip()
        short_text = (text[:80] + "…") if len(text) > 80 else text
        items.append(f"{icon} {short_text}")

    if items:
        lines.append("• " + "\n• ".join(items))

    return "\n\n".join(lines).strip()


def should_process_queue(force: bool = False, db_path: str = QUEUE_DB_PATH) -> tuple[bool, str]:
    """Check if the calm-memory threshold conditions are satisfied."""
    if force:
        return True, "Forced run"

    stats = get_pending_stats(db_path=db_path, retry_delay_seconds=60)
    count = stats["count"]
    if count == 0:
        return False, "Queue is empty"

    newest_age = stats["newest_age_seconds"]
    oldest_age = stats["oldest_age_seconds"]

    # Condition 1: Inactivity debounce (5 min of silence)
    if newest_age >= INACTIVITY_THRESHOLD_SECONDS:
        return True, f"Inactivity threshold met (idle for {newest_age}s, {count} turns)"

    # Condition 2: Max wait time (30 min timeout)
    if oldest_age >= MAX_WAIT_THRESHOLD_SECONDS:
        return True, f"Max wait threshold met (oldest turn {oldest_age}s, {count} turns)"

    return False, f"Chat actively in progress (last message {newest_age}s ago, waiting for 5m idle)"


def process_queue(batch_size: int = 25, notify: bool = True, db_path: str = QUEUE_DB_PATH) -> int:
    """Process pending conversation turns partitioned strictly by (source, chat_id)."""
    pending = get_pending_turns(limit=batch_size, db_path=db_path, retry_delay_seconds=60)
    if not pending:
        return 0

    # Group pending turns by (source, chat_id) to avoid cross-chat dialogue mixing
    groups: dict[tuple[str, str | None], list[dict]] = {}
    for turn in pending:
        source_key = turn.get("source")
        raw_chat = turn.get("chat_id")
        chat_key = str(raw_chat) if raw_chat is not None else None
        groups.setdefault((source_key, chat_key), []).append(turn)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for (source, chat_id), group_turns in groups.items():
        group_turn_ids = [t["id"] for t in group_turns]
        batch_id = f"batch_{now_str}_{uuid.uuid4().hex}"

        dialogue_blocks = []
        for turn in group_turns:
            u = turn["user_prompt"].strip()
            a = turn["assistant_response"].strip()
            if is_trivial_prompt(u):
                continue
            dialogue_blocks.append(f"User: {u}\nAssistant: {a}")

        if not dialogue_blocks:
            mark_turn_status(group_turn_ids, status="skipped", summary="All turns trivial", batch_id=batch_id, db_path=db_path)
            continue

        combined_dialogue = "\n\n---\n\n".join(dialogue_blocks)

        try:
            changes = sync_turn(
                user_prompt=combined_dialogue,
                assistant_response="Conversation batch complete.",
                dry_run=False
            )

            keys = ("facts", "episodes", "learnings", "entity_links")
            if not isinstance(changes, dict) or any(not isinstance(changes.get(key), list) for key in keys):
                raise SyncExtractionError("sync_turn did not confirm a successful extraction")

            has_changes = any(bool(changes.get(k)) for k in ["facts", "episodes", "learnings", "entity_links"])
            summary_parts = []
            if changes.get("facts"): summary_parts.append(f"{len(changes['facts'])} facts")
            if changes.get("episodes"): summary_parts.append(f"{len(changes['episodes'])} episodes")
            if changes.get("learnings"): summary_parts.append(f"{len(changes['learnings'])} learnings")
            if changes.get("entity_links"): summary_parts.append(f"{len(changes['entity_links'])} links")

            summary = ", ".join(summary_parts) if summary_parts else "No persistent entities found"
            mark_turn_status(group_turn_ids, status="processed", summary=summary, batch_id=batch_id, db_path=db_path)

            # Notification is sent ONLY for telegram source and when chat_id is present
            if has_changes and notify and source == "telegram" and chat_id:
                msg = format_notification(changes)
                if msg:
                    try:
                        send_telegram_notification(msg, chat_id=str(chat_id))
                    except Exception as error:
                        sys.stderr.write(f"Notification delivery failed: {error}\n")

        except Exception as e:
            sys.stderr.write(f"Error during batch sync for ({source}, {chat_id}): {e}\n")
            # F03: Retain pending status with error message and timestamp; do NOT discard or mark processed
            mark_turn_status(group_turn_ids, status="pending", error=str(e), batch_id=batch_id, db_path=db_path)

    prune_processed_turns(days=7, db_path=db_path)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Calm AGY Memory Queue Worker")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of turns to process per run")
    parser.add_argument("--force", action="store_true", help="Force processing regardless of 5m idle or 30m timer")
    parser.add_argument("--no-notify", action="store_true", help="Disable Telegram notification")
    args = parser.parse_args()

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(0)

    try:
        can_run, reason = should_process_queue(force=args.force)
        if not can_run:
            sys.exit(0)

        count = process_queue(batch_size=args.batch_size, notify=not args.no_notify)
        if count > 0:
            print(f"Memory Worker: Processed batch of {count} turn(s) ({reason}).")
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
