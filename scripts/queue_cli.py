#!/usr/bin/env python3
"""CLI interface for managing turn queue operations and atomic memory extraction commits."""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from queue_manager import (
    get_pending_stats,
    claim_batch,
    acknowledge_batch,
    release_batch,
    prune_processed_turns,
    QUEUE_DB_PATH,
)
from memory_worker import should_process_queue
from agy_memory import (
    upsert_fact,
    upsert_episode,
    upsert_learning,
    link_entities,
    CANONICAL_FACT_CATEGORIES,
    CANONICAL_LEARNING_CATEGORIES,
    CANONICAL_EPISODE_TOPICS,
)
from taxonomy import validate_category
from scripts.migrate_v2_to_v2_1 import map_relation, CANONICAL_EPISODE_STATUSES


def cmd_status(args):
    db_path = args.db_path or QUEUE_DB_PATH
    stats = get_pending_stats(db_path=db_path, retry_delay_seconds=60)
    can_process, reason = should_process_queue(force=args.force, db_path=db_path)
    output = {
        "count": stats["count"],
        "oldest_age_seconds": stats["oldest_age_seconds"],
        "newest_age_seconds": stats["newest_age_seconds"],
        "can_process": can_process,
        "reason": reason,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_claim(args):
    db_path = args.db_path or QUEUE_DB_PATH
    claim = claim_batch(
        batch_size=args.batch_size,
        lease_duration_seconds=args.lease_seconds,
        retry_delay_seconds=60,
        prefer_fresh=True,
        db_path=db_path,
    )
    if not claim:
        print("null")
        return 0
    payload = {
        "batch_id": claim.batch_id,
        "lease_token": claim.lease_token,
        "source": claim.source,
        "chat_id": claim.chat_id,
        "turns": claim.turns,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_ack(args):
    db_path = args.db_path or QUEUE_DB_PATH
    ok = acknowledge_batch(
        batch_id=args.batch_id,
        lease_token=args.lease_token,
        status="processed",
        summary=args.summary,
        db_path=db_path,
    )
    res = {"acknowledged": ok, "batch_id": args.batch_id, "status": "processed"}
    print(json.dumps(res, ensure_ascii=False))
    return 0 if ok else 1


def cmd_skip(args):
    db_path = args.db_path or QUEUE_DB_PATH
    ok = acknowledge_batch(
        batch_id=args.batch_id,
        lease_token=args.lease_token,
        status="skipped",
        summary=args.summary or "No persistent knowledge extracted",
        db_path=db_path,
    )
    res = {"acknowledged": ok, "batch_id": args.batch_id, "status": "skipped"}
    print(json.dumps(res, ensure_ascii=False))
    return 0 if ok else 1


def cmd_release(args):
    db_path = args.db_path or QUEUE_DB_PATH
    ok = release_batch(
        batch_id=args.batch_id,
        lease_token=args.lease_token,
        error=args.error or "Released by worker",
        db_path=db_path,
    )
    res = {"released": ok, "batch_id": args.batch_id}
    print(json.dumps(res, ensure_ascii=False))
    return 0 if ok else 1


def cmd_commit(args):
    db_path = args.db_path or QUEUE_DB_PATH
    raw_data = args.data
    if raw_data == "-" or not raw_data:
        raw_data = sys.stdin.read()
    try:
        data = json.loads(raw_data)
    except Exception as e:
        sys.stderr.write(f"Invalid JSON data: {e}\n")
        return 1

    facts = data.get("facts", [])
    episodes = data.get("episodes", [])
    learnings = data.get("learnings", [])
    entity_links = data.get("entity_links", [])

    committed = {"facts": 0, "episodes": 0, "learnings": 0, "entity_links": 0}

    for f in facts:
        fact_id = str(f.get("id", "")).strip()
        cat = validate_category(f.get("category", "general"), CANONICAL_FACT_CATEGORIES)
        content = str(f.get("fact", "")).strip()
        kw = str(f.get("keywords", "")).strip()
        if fact_id and content:
            upsert_fact(fact_id, cat, content, kw)
            committed["facts"] += 1

    for ep in episodes:
        ep_id = str(ep.get("id", "")).strip()
        topic = validate_category(ep.get("topic", "general"), CANONICAL_EPISODE_TOPICS)
        title = str(ep.get("title", "")).strip()
        narrative = str(ep.get("narrative", "")).strip()
        period = str(ep.get("period", "")).strip()
        status = str(ep.get("status", "active")).strip().lower()
        if status not in CANONICAL_EPISODE_STATUSES:
            status = "active"
        entities = str(ep.get("entities", "")).strip()
        stance = str(ep.get("stance", "")).strip()
        kw = str(ep.get("keywords", "")).strip()
        if ep_id and title and narrative:
            upsert_episode(ep_id, topic, title, narrative, period, status, entities, stance, kw)
            committed["episodes"] += 1

    for lr in learnings:
        lr_id = str(lr.get("id", "")).strip()
        cat = validate_category(lr.get("category", "general"), CANONICAL_LEARNING_CATEGORIES)
        insight = str(lr.get("insight", "")).strip()
        context = str(lr.get("context", "")).strip()
        kw = str(lr.get("keywords", "")).strip()
        if lr_id and insight:
            upsert_learning(lr_id, cat, insight, context, kw)
            committed["learnings"] += 1

    for link in entity_links:
        src = str(link.get("source", "")).strip()
        tgt = str(link.get("target", "")).strip()
        rel = str(link.get("relation", "")).strip()
        if src and tgt and rel:
            can_src, can_tgt, can_rel = map_relation(src, tgt, rel)
            link_entities(can_src, can_tgt, can_rel)
            committed["entity_links"] += 1

    total = sum(committed.values())
    summary_parts = []
    if committed["facts"]:
        summary_parts.append(f"{committed['facts']} facts")
    if committed["episodes"]:
        summary_parts.append(f"{committed['episodes']} episodes")
    if committed["learnings"]:
        summary_parts.append(f"{committed['learnings']} learnings")
    if committed["entity_links"]:
        summary_parts.append(f"{committed['entity_links']} links")
    summary = ", ".join(summary_parts) if summary_parts else "No persistent entities found"

    status = "processed" if total > 0 else "skipped"
    ok = acknowledge_batch(
        batch_id=args.batch_id,
        lease_token=args.lease_token,
        status=status,
        summary=summary,
        db_path=db_path,
    )
    result = {
        "acknowledged": ok,
        "batch_id": args.batch_id,
        "status": status,
        "summary": summary,
        "committed": committed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def cmd_prune(args):
    db_path = args.db_path or QUEUE_DB_PATH
    prune_processed_turns(days=args.days, db_path=db_path)
    print(json.dumps({"pruned": True, "days": args.days}))
    return 0


def main():
    parser = argparse.ArgumentParser(description="AGY Turn Queue CLI")
    parser.add_argument("--db-path", default=None, help="Custom queue database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_p = subparsers.add_parser("status", help="Check queue status and debounce condition")
    status_p.add_argument("--force", action="store_true", help="Ignore debounce thresholds")
    status_p.set_defaults(func=cmd_status)

    claim_p = subparsers.add_parser("claim", help="Claim a batch of pending turns")
    claim_p.add_argument("--batch-size", type=int, default=25, help="Batch size")
    claim_p.add_argument("--lease-seconds", type=int, default=300, help="Lease duration in seconds")
    claim_p.set_defaults(func=cmd_claim)

    ack_p = subparsers.add_parser("ack", help="Acknowledge a processed batch")
    ack_p.add_argument("--batch-id", required=True, help="Batch ID")
    ack_p.add_argument("--lease-token", required=True, help="Lease token")
    ack_p.add_argument("--summary", default="Batch processed successfully", help="Summary message")
    ack_p.set_defaults(func=cmd_ack)

    skip_p = subparsers.add_parser("skip", help="Mark a batch as skipped")
    skip_p.add_argument("--batch-id", required=True, help="Batch ID")
    skip_p.add_argument("--lease-token", required=True, help="Lease token")
    skip_p.add_argument("--summary", default="No persistent knowledge extracted", help="Summary message")
    skip_p.set_defaults(func=cmd_skip)

    rel_p = subparsers.add_parser("release", help="Release a claimed batch back to pending")
    rel_p.add_argument("--batch-id", required=True, help="Batch ID")
    rel_p.add_argument("--lease-token", required=True, help="Lease token")
    rel_p.add_argument("--error", default="Released by worker", help="Error reason")
    rel_p.set_defaults(func=cmd_release)

    commit_p = subparsers.add_parser("commit", help="Commit extracted memory JSON and acknowledge batch")
    commit_p.add_argument("--batch-id", required=True, help="Batch ID")
    commit_p.add_argument("--lease-token", required=True, help="Lease token")
    commit_p.add_argument("--data", default="-", help="JSON data string or '-' to read from stdin")
    commit_p.set_defaults(func=cmd_commit)

    prune_p = subparsers.add_parser("prune", help="Prune processed and skipped turns older than N days")
    prune_p.add_argument("--days", type=int, default=7, help="Age in days")
    prune_p.set_defaults(func=cmd_prune)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
