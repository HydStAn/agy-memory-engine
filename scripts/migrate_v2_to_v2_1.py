#!/usr/bin/env python3
"""
Migration Script: AGY Memory Engine v2.0 -> v2.1

Migrates database entities, topics, categories, and relationship links to
comply with the canonical taxonomy standards introduced in v2.1.0:
- Maps legacy relationship link types into the 26 canonical relationship types
  (including inverse relationship resolution like 'hosts' -> 'hosted_on').
- Prunes orphan entity links where source or target records do not exist.
- Normalizes categories for Facts and Learnings.
- Normalizes topics and status for Episodes.
- Rebuilds FTS5 search indexes and optimizes tables.
- Supports dry-run inspection and creates a safety backup before applying changes.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Ensure parent directory is in path to import engine modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from config import DB_PATH, archive_path
import schema
from schema import db_session

from agy_memory import (
    CANONICAL_FACT_CATEGORIES, CANONICAL_LEARNING_CATEGORIES,
    CANONICAL_EPISODE_TOPICS, CANONICAL_RELATIONS,
    _CATEGORY_ALIASES as CATEGORY_ALIASES, _normalize_category as normalize_category,
    online_backup, rebuild_fts, _readonly_db, create_snapshot,
)
from taxonomy import CANONICAL_EPISODE_STATUSES, RELATION_MAPPINGS, map_relation, validate_category


def create_safety_backup(db_path: str) -> str:
    """Create a safety snapshot backup before applying migration changes."""
    snapshot = create_snapshot('migration', db_path)
    return str(archive_path(db_path) / snapshot['filename'])


def normalize_category(value, allowed):
    try:
        return validate_category(value, allowed)
    except ValueError:
        return value  # Preserve unmapped legacy meaning for explicit review.


def run_migration(db_path: str = None, dry_run: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Execute migration from v2.0 to v2.1.
    """
    target_db = db_path or schema.DB_PATH
    if not os.path.exists(target_db):
        raise FileNotFoundError(f"Database file not found: {target_db}")

    report = {
        "backup_file": None,
        "facts_migrated": 0,
        "episodes_migrated": 0,
        "learnings_migrated": 0,
        "links_mapped": 0,
        "orphan_links_pruned": 0,
        "details": {
            "facts": [],
            "episodes": [],
            "learnings": [],
            "links": [],
            "orphans": []
        }
    }

    if not dry_run:
        backup_file = create_safety_backup(target_db)
        report["backup_file"] = backup_file
        if verbose:
            print(f"[BACKUP] Safety snapshot created: {backup_file}")

    with (_readonly_db(target_db) if dry_run else db_session(target_db)) as conn:
        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()

        # 1. Fact Categories Normalization
        cursor.execute("SELECT id, category FROM memories")
        for fid, cat in cursor.fetchall():
            norm_cat = normalize_category(cat, CANONICAL_FACT_CATEGORIES)
            if norm_cat != cat:
                report["facts_migrated"] += 1
                report["details"]["facts"].append({"id": fid, "old": cat, "new": norm_cat})
                if not dry_run:
                    cursor.execute("UPDATE memories SET category = ? WHERE id = ?", (norm_cat, fid))

        # 2. Episode Topics and Status Normalization
        cursor.execute("SELECT id, topic, status FROM episodes")
        for eid, topic, status in cursor.fetchall():
            norm_topic = normalize_category(topic, CANONICAL_EPISODE_TOPICS)
            norm_status = status.strip().lower() if status else "active"
            if norm_status not in CANONICAL_EPISODE_STATUSES:
                norm_status = status

            if norm_topic != topic or norm_status != status:
                report["episodes_migrated"] += 1
                report["details"]["episodes"].append({
                    "id": eid,
                    "old_topic": topic, "new_topic": norm_topic,
                    "old_status": status, "new_status": norm_status
                })
                if not dry_run:
                    cursor.execute("UPDATE episodes SET topic = ?, status = ? WHERE id = ?", (norm_topic, norm_status, eid))

        # 3. Learning Categories Normalization
        cursor.execute("SELECT id, category FROM learnings")
        for lid, cat in cursor.fetchall():
            norm_cat = normalize_category(cat, CANONICAL_LEARNING_CATEGORIES)
            if norm_cat != cat:
                report["learnings_migrated"] += 1
                report["details"]["learnings"].append({"id": lid, "old": cat, "new": norm_cat})
                if not dry_run:
                    cursor.execute("UPDATE learnings SET category = ? WHERE id = ?", (norm_cat, lid))

        # 4. Collect all known IDs across tables for Orphan Link detection
        known_ids = set()
        for row in cursor.execute("SELECT id FROM memories"):
            known_ids.add(row[0])
        for row in cursor.execute("SELECT id FROM episodes"):
            known_ids.add(row[0])
        for row in cursor.execute("SELECT id FROM learnings"):
            known_ids.add(row[0])

        # 5. Entity Link Migration & Orphan Pruning
        cursor.execute("SELECT source_id, target_id, relation FROM entity_links")
        all_links = cursor.fetchall()

        for src, tgt, rel in all_links:
            # Check for orphans: either side missing from database
            if src not in known_ids or tgt not in known_ids:
                report["orphan_links_pruned"] += 1
                report["details"]["orphans"].append({"source": src, "target": tgt, "relation": rel})
                if not dry_run:
                    cursor.execute(
                        "DELETE FROM entity_links WHERE source_id = ? AND target_id = ? AND relation = ?",
                        (src, tgt, rel)
                    )
                continue

            # Check if relation needs mapping / canonicalization
            try:
                new_src, new_tgt, new_rel = map_relation(src, tgt, rel)
            except ValueError:
                report.setdefault('unmapped_relations', []).append((src, tgt, rel))
                continue
            if (new_src, new_tgt, new_rel) != (src, tgt, rel):
                report["links_mapped"] += 1
                report["details"]["links"].append({
                    "old": (src, tgt, rel),
                    "new": (new_src, new_tgt, new_rel)
                })
                if not dry_run:
                    cursor.execute(
                        "DELETE FROM entity_links WHERE source_id = ? AND target_id = ? AND relation = ?",
                        (src, tgt, rel)
                    )
                    cursor.execute(
                        "INSERT OR IGNORE INTO entity_links (source_id, target_id, relation) VALUES (?, ?, ?)",
                        (new_src, new_tgt, new_rel)
                    )

        if not dry_run:
            # Set schema version in PRAGMA user_version to 210 (v2.1.0)
            cursor.execute(f"PRAGMA user_version = {schema.SCHEMA_VERSION};")
            
            # Rebuild FTS5 virtual tables to ensure index coherence
            rebuild_fts(conn)
            conn.commit()

            report['data_committed'] = True
            try:
                cursor.execute("VACUUM;")
                report['compacted'] = True
            except sqlite3.Error as error:
                report['compacted'] = False
                report['compaction_error'] = str(error)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Migrate AGY Memory Engine from v2.0 to v2.1 (Taxonomies, Relations, Graph Pruning)"
    )
    parser.add_argument("--db", type=str, default=None, help=f"Path to SQLite database (default: {DB_PATH})")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and simulate migration without applying changes")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    args = parser.parse_args()

    db_target = args.db or DB_PATH
    print(f"=== AGY Memory Engine: Migration v2.0 -> v2.1 ===")
    print(f"Target Database: {db_target}")
    print(f"Mode: {'DRY RUN (simulation only)' if args.dry_run else 'LIVE MIGRATION'}\n")

    try:
        report = run_migration(db_path=db_target, dry_run=args.dry_run, verbose=not args.quiet)
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Migration Summary ---")
    print(f"• Facts categories normalized:      {report['facts_migrated']}")
    print(f"• Episodes topics/status normalized: {report['episodes_migrated']}")
    print(f"• Learnings categories normalized:   {report['learnings_migrated']}")
    print(f"• Entity links mapped to canonical:  {report['links_mapped']}")
    print(f"• Orphan entity links pruned:        {report['orphan_links_pruned']}")

    if report["details"]["links"]:
        print("\nTop Entity Link Mappings:")
        for item in report["details"]["links"][:10]:
            old_s, old_t, old_r = item["old"]
            new_s, new_t, new_r = item["new"]
            print(f"  {old_s} -[{old_r}]-> {old_t}  ==>  {new_s} -[{new_r}]-> {new_t}")
        if len(report["details"]["links"]) > 10:
            print(f"  ... and {len(report['details']['links']) - 10} more.")

    if report["details"]["orphans"]:
        print("\nSample Orphan Links Pruned:")
        for item in report["details"]["orphans"][:5]:
            print(f"  {item['source']} -[{item['relation']}]-> {item['target']}")
        if len(report["details"]["orphans"]) > 5:
            print(f"  ... and {len(report['details']['orphans']) - 5} more.")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No changes were written to the database. Run without --dry-run to apply.")
    else:
        print(f"\n[MIGRATION COMPLETE] Successfully migrated to v2.1.0 (PRAGMA user_version = 210).")
        if report["backup_file"]:
            print(f"Safety backup retained at: {report['backup_file']}")


if __name__ == "__main__":
    main()
