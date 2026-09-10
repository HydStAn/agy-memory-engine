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
    _get_connection,
)
from config import get_config
from schema import db_session
from memory_worker import should_process_queue
from agy_memory import (
    upsert_fact,
    upsert_episode,
    upsert_learning,
    link_entities,
    _VOCABULARY_CACHE,
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


def _validate_payload(data):
    """Strictly validate schema and content of extraction payload upfront.

    Returns tuple of (validated_facts, validated_episodes, validated_learnings, validated_links)
    or raises ValueError on malformed input.
    """
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object.")

    for key in ("facts", "episodes", "learnings", "entity_links"):
        val = data.get(key, [])
        if not isinstance(val, list):
            raise ValueError(f"Field '{key}' must be a list, got {type(val).__name__}.")

    validated_facts = []
    for i, f in enumerate(data.get("facts", [])):
        if not isinstance(f, dict):
            raise ValueError(f"facts[{i}] must be an object.")
        fid = f.get("id")
        fact = f.get("fact")
        if not isinstance(fid, str) or not fid.strip():
            raise ValueError(f"facts[{i}] has missing or invalid 'id'.")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError(f"facts[{i}] has missing or invalid 'fact'.")
        raw_cat = f.get("category", "general")
        if not isinstance(raw_cat, str) or not raw_cat.strip():
            raise ValueError(f"facts[{i}] has invalid 'category'.")
        norm_cat = validate_category(raw_cat.strip(), CANONICAL_FACT_CATEGORIES)
        kw = str(f.get("keywords") or "").strip()
        validated_facts.append((fid.strip(), norm_cat, fact.strip(), kw))

    validated_episodes = []
    for i, ep in enumerate(data.get("episodes", [])):
        if not isinstance(ep, dict):
            raise ValueError(f"episodes[{i}] must be an object.")
        epid = ep.get("id")
        title = ep.get("title")
        narrative = ep.get("narrative")
        if not isinstance(epid, str) or not epid.strip():
            raise ValueError(f"episodes[{i}] has missing or invalid 'id'.")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"episodes[{i}] has missing or invalid 'title'.")
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError(f"episodes[{i}] has missing or invalid 'narrative'.")
        raw_topic = ep.get("topic", "general")
        if not isinstance(raw_topic, str) or not raw_topic.strip():
            raise ValueError(f"episodes[{i}] has invalid 'topic'.")
        norm_topic = validate_category(raw_topic.strip(), CANONICAL_EPISODE_TOPICS)
        raw_status = str(ep.get("status") or "active").strip().lower()
        if raw_status not in CANONICAL_EPISODE_STATUSES:
            raw_status = "active"
        period = str(ep.get("period") or "").strip()
        entities = str(ep.get("entities") or "").strip()
        stance = str(ep.get("stance") or "").strip()
        kw = str(ep.get("keywords") or "").strip()
        validated_episodes.append((epid.strip(), norm_topic, title.strip(), narrative.strip(), period, raw_status, entities, stance, kw))

    validated_learnings = []
    for i, lr in enumerate(data.get("learnings", [])):
        if not isinstance(lr, dict):
            raise ValueError(f"learnings[{i}] must be an object.")
        lrid = lr.get("id")
        insight = lr.get("insight")
        if not isinstance(lrid, str) or not lrid.strip():
            raise ValueError(f"learnings[{i}] has missing or invalid 'id'.")
        if not isinstance(insight, str) or not insight.strip():
            raise ValueError(f"learnings[{i}] has missing or invalid 'insight'.")
        raw_cat = lr.get("category", "general")
        if not isinstance(raw_cat, str) or not raw_cat.strip():
            raise ValueError(f"learnings[{i}] has invalid 'category'.")
        norm_cat = validate_category(raw_cat.strip(), CANONICAL_LEARNING_CATEGORIES)
        context = str(lr.get("context") or "").strip()
        kw = str(lr.get("keywords") or "").strip()
        validated_learnings.append((lrid.strip(), norm_cat, insight.strip(), context, kw))

    validated_links = []
    for i, link in enumerate(data.get("entity_links", [])):
        if not isinstance(link, dict):
            raise ValueError(f"entity_links[{i}] must be an object.")
        src = link.get("source")
        tgt = link.get("target")
        rel = link.get("relation")
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"entity_links[{i}] has missing or invalid 'source'.")
        if not isinstance(tgt, str) or not tgt.strip():
            raise ValueError(f"entity_links[{i}] has missing or invalid 'target'.")
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError(f"entity_links[{i}] has missing or invalid 'relation'.")
        can_src, can_tgt, can_rel = map_relation(src.strip(), tgt.strip(), rel.strip())
        validated_links.append((can_src, can_tgt, can_rel))

    return validated_facts, validated_episodes, validated_learnings, validated_links


def cmd_commit(args):
    db_path = args.db_path or QUEUE_DB_PATH
    m_db_path = getattr(args, "memory_db", None) or get_config("AGY_MEMORY_DB", None)

    # 1. Batch Receipt Replay Check: If already committed, return existing receipt (idempotent replay)
    with db_session(db_path=m_db_path) as m_conn:
        receipt = m_conn.execute("SELECT result_json FROM batch_receipts WHERE batch_id = ?", (args.batch_id,)).fetchone()
        if receipt:
            summary = "Replay from existing batch receipt"
            ok = acknowledge_batch(
                batch_id=args.batch_id,
                lease_token=args.lease_token,
                status="processed",
                summary=summary,
                db_path=db_path,
            )
            res = {
                "acknowledged": True,
                "batch_id": args.batch_id,
                "status": "processed",
                "summary": summary,
                "committed": json.loads(receipt[0]),
                "replay": True,
            }
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0

    # 2. Lease Fencing: Check that the batch is claimed by this worker and lease has not expired BEFORE touching memory.db
    with _get_connection(db_path, timeout=5.0) as q_conn:
        row = q_conn.execute("""
            SELECT status, lease_expires_at
            FROM turn_queue
            WHERE batch_id = ? AND lease_token = ? AND status = 'claimed'
        """, (args.batch_id, args.lease_token)).fetchone()
        if not row:
            sys.stderr.write(f"Commit rejected: batch '{args.batch_id}' is not claimed with the provided lease token.\n")
            return 1

        exp_check = q_conn.execute("""
            SELECT 1
            FROM turn_queue
            WHERE batch_id = ? AND lease_token = ? AND status = 'claimed'
              AND datetime(lease_expires_at) >= datetime('now')
        """, (args.batch_id, args.lease_token)).fetchone()
        if not exp_check:
            sys.stderr.write(f"Commit rejected: lease for batch '{args.batch_id}' has expired.\n")
            return 1

    # 3. Read payload from data-file, data argument, or stdin
    if getattr(args, "data_file", None):
        try:
            with open(args.data_file, "r", encoding="utf-8") as f:
                raw_data = f.read()
        except Exception as e:
            sys.stderr.write(f"Cannot read data file '{args.data_file}': {e}\n")
            return 1
    else:
        raw_data = args.data
        if raw_data == "-" or not raw_data:
            raw_data = sys.stdin.read()

    # 4. Strict Validation: Parse JSON and validate all layers upfront before any writes
    try:
        data = json.loads(raw_data)
    except Exception as e:
        sys.stderr.write(f"Invalid JSON data: {e}\n")
        return 1

    try:
        facts, episodes, learnings, links = _validate_payload(data)
    except ValueError as e:
        sys.stderr.write(f"Commit validation error: {e}\n")
        return 1

    total_items = len(facts) + len(episodes) + len(learnings) + len(links)
    committed = {
        "facts": len(facts),
        "episodes": len(episodes),
        "learnings": len(learnings),
        "entity_links": len(links),
    }

    # 5. Atomic Persistence: Single transaction for all memory updates and batch receipt
    if total_items > 0:
        try:
            with db_session(db_path=m_db_path) as m_conn, m_conn:
                m_conn.execute("BEGIN IMMEDIATE")
                for fid, cat, content, kw in facts:
                    upsert_fact(fid, cat, content, kw, connection=m_conn)
                for epid, topic, title, narrative, period, status, entities, stance, kw in episodes:
                    upsert_episode(epid, topic, title, narrative, period, status, entities, stance, kw, connection=m_conn)
                for lrid, cat, insight, context, kw in learnings:
                    upsert_learning(lrid, cat, insight, context, kw, connection=m_conn)
                for src, tgt, rel in links:
                    link_entities(src, tgt, rel, connection=m_conn)
                m_conn.execute(
                    "INSERT INTO batch_receipts (batch_id, result_json) VALUES (?, ?)",
                    (args.batch_id, json.dumps(committed))
                )
            _VOCABULARY_CACHE.clear()
        except Exception as e:
            sys.stderr.write(f"Transaction rollback: failed to persist memory batch: {e}\n")
            return 1

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

    status = "processed" if total_items > 0 else "skipped"
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
    deleted_count = prune_processed_turns(days=args.days, db_path=db_path)
    print(json.dumps({"pruned": True, "days": args.days, "deleted_count": deleted_count}))
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
    commit_p.add_argument("--data-file", default=None, help="Path to JSON data file")
    commit_p.add_argument("--memory-db", default=None, help="Custom memory database path")
    commit_p.set_defaults(func=cmd_commit)

    prune_p = subparsers.add_parser("prune", help="Prune processed and skipped turns older than N days")
    prune_p.add_argument("--days", type=int, default=7, help="Age in days")
    prune_p.set_defaults(func=cmd_prune)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
