#!/usr/bin/env python3
"""
Initial Vector Indexing & Rebuild Script for AGY Memory Engine.
Populates vec_memories, vec_episodes, and vec_learnings from existing records.
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from schema import db_session, HAS_SQLITE_VEC
from config import DB_PATH, VECTOR_SEARCH_ENABLED
import embedder

def reindex_all(db_path: str = None, verbose: bool = True) -> dict:
    target_db = db_path or DB_PATH
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        if verbose:
            print("[WARN] sqlite-vec or vector search not enabled. Skipping vector reindexing.")
        return {"status": "skipped", "reason": "vector_search_disabled"}

    stats = {"facts": 0, "episodes": 0, "learnings": 0}

    with db_session(target_db) as conn:
        cursor = conn.cursor()

        # 1. Facts
        cursor.execute("SELECT id, category, fact, keywords FROM memories")
        facts = cursor.fetchall()
        if facts:
            if verbose:
                print(f"Indexing {len(facts)} facts into vec_memories...")
            for fid, cat, fact, kws in facts:
                text = f"[{cat or 'general'}] {fact} {kws or ''}".strip()
                if embedder.upsert_vector(conn, "vec_memories", fid, text):
                    stats["facts"] += 1

        # 2. Episodes
        cursor.execute("SELECT id, topic, title, narrative, stance, keywords FROM episodes")
        episodes = cursor.fetchall()
        if episodes:
            if verbose:
                print(f"Indexing {len(episodes)} episodes into vec_episodes...")
            for eid, topic, title, narrative, stance, kws in episodes:
                text = f"[{topic}] {title}: {narrative} (Stance: {stance or 'neutral'}) {kws or ''}".strip()
                if embedder.upsert_vector(conn, "vec_episodes", eid, text):
                    stats["episodes"] += 1

        # 3. Learnings
        cursor.execute("SELECT id, category, insight, context, keywords FROM learnings")
        learnings = cursor.fetchall()
        if learnings:
            if verbose:
                print(f"Indexing {len(learnings)} learnings into vec_learnings...")
            for lid, cat, insight, ctx, kws in learnings:
                text = f"[{cat or 'general'}] {insight} (Context: {ctx or ''}) {kws or ''}".strip()
                if embedder.upsert_vector(conn, "vec_learnings", lid, text):
                    stats["learnings"] += 1

        conn.commit()

    if verbose:
        print(f"[SUCCESS] Reindexed {stats['facts']} facts, {stats['episodes']} episodes, {stats['learnings']} learnings.")
    return {"status": "success", "stats": stats}

if __name__ == "__main__":
    reindex_all()
