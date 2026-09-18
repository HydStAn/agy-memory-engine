#!/usr/bin/env python3
"""
Vector Indexing & Rebuild Script for AGY Memory Engine.
Populates and synchronizes vec_memories, vec_episodes, and vec_learnings via fenced outbox jobs.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from schema import HAS_SQLITE_VEC
from config import DB_PATH, VECTOR_SEARCH_ENABLED
from vector_index import reconcile_vector_index, drain_vector_jobs, get_vector_sync_status


def reindex_all(db_path: str = None, verbose: bool = True) -> dict:
    """Enqueues all records and processes them via the atomic, fenced vector outbox worker."""
    target_db = db_path or DB_PATH
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        if verbose:
            print("[WARN] sqlite-vec or vector search not enabled. Skipping vector reindexing.")
        return {"status": "skipped", "reason": "vector_search_disabled"}

    if verbose:
        print(f"Reconciling and enqueuing vector jobs for {target_db}...")
    report = reconcile_vector_index(db_path=target_db, apply=True)

    total_processed = 0
    while True:
        processed = drain_vector_jobs(db_path=target_db, batch_size=64, max_batches=1)
        if processed == 0:
            break
        total_processed += processed
        if verbose:
            print(f"  Processed {total_processed} vector jobs...")

    final_status = get_vector_sync_status(db_path=target_db)
    stats = {
        "facts": final_status["sources"]["memories"]["eligible"],
        "episodes": final_status["sources"]["episodes"]["eligible"],
        "learnings": final_status["sources"]["learnings"]["eligible"],
    }
    if verbose:
        print(f"[SUCCESS] Reindexed vector store: {total_processed} jobs published. Eligible: {final_status['total_eligible_vectors']}, Missing: {final_status['total_missing_vectors']}, Stale: {final_status['total_stale_vectors']}.")

    return {"status": "success", "processed_jobs": total_processed, "stats": stats, "coverage": final_status}


if __name__ == "__main__":
    reindex_all()
