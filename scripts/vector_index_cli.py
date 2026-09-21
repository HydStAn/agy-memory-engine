#!/usr/bin/env python3
"""
CLI tool for inspecting, draining, and reconciling the AGY Vector Index.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from vector_index import get_vector_sync_status, drain_vector_jobs, reconcile_vector_index
from config import DB_PATH


def cmd_status(args):
    db_path = args.db_path or DB_PATH
    status = get_vector_sync_status(db_path=db_path)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args):
    db_path = args.db_path or DB_PATH
    processed = drain_vector_jobs(db_path=db_path, batch_size=args.batch_size, max_batches=args.max_batches)
    print(json.dumps({"status": "success", "processed_jobs": processed}, ensure_ascii=False, indent=2))
    return 0


def cmd_reconcile(args):
    db_path = args.db_path or DB_PATH
    apply = bool(args.apply)
    report = reconcile_vector_index(db_path=db_path, apply=apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="AGY Vector Index Reliability CLI")
    parser.add_argument("--db-path", type=str, default=None, help="Target SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Inspect vector synchronization status and coverage")
    p_status.set_defaults(func=cmd_status)

    # run
    p_run = subparsers.add_parser("run", help="Drain pending vector jobs with bounded batch and iteration limits")
    p_run.add_argument("--batch-size", type=int, default=25, help="Batch size per claim")
    p_run.add_argument("--max-batches", type=int, default=10, help="Maximum number of batches to drain per run")
    p_run.set_defaults(func=cmd_run)

    # reconcile
    p_rec = subparsers.add_parser("reconcile", help="Reconcile source rows against vector index metadata")
    p_rec.add_argument("--apply", action="store_true", help="Apply missing/stale repair by enqueuing jobs (default: dry-run)")
    p_rec.set_defaults(func=cmd_reconcile)

    args = parser.parse_args()
    code = args.func(args)
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
