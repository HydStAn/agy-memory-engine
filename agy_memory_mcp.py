#!/usr/bin/env python3
"""
AGY Memory Engine - MCP Server Layer (FastMCP)
Allows AGY to explicitly query, store, link entities, and manage multi-layer memories (Facts, Episodes, Learnings).
"""

import atexit
import asyncio
import threading
from taxonomy import validate_category
from agy_memory import upsert_fact, upsert_episode, upsert_learning, link_entities
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import sys
import uuid
from typing import Optional

from mcp.server.fastmcp import FastMCP

from schema import db_session
from config import VECTOR_SEARCH_ENABLED
from agy_memory import (
    extract_multilingual_tokens,
    get_all_vocabulary,
    optimize_db,
    CANONICAL_FACT_CATEGORIES,
    CANONICAL_LEARNING_CATEGORIES,
    CANONICAL_EPISODE_TOPICS,
)
from scripts.migrate_v2_to_v2_1 import (
    normalize_category,
    map_relation,
    run_migration,
    CANONICAL_EPISODE_STATUSES,
)
try:
    from embedder import embed_text, upsert_vector, reciprocal_rank_fusion, build_text_repr, log_vec_query_failure
    HAS_EMBEDDER = True
except ImportError:
    embed_text = None
    upsert_vector = None
    reciprocal_rank_fusion = None
    build_text_repr = None
    log_vec_query_failure = None
    HAS_EMBEDDER = False

logger = logging.getLogger("agy_memory_mcp")

mcp = FastMCP("memory")

# F20: Explicit bounded concurrency for heavy maintenance operations
_MAINTENANCE_SLOT = threading.Lock()
_MAINTENANCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-maintenance")
atexit.register(_MAINTENANCE_EXECUTOR.shutdown, wait=False)


def search_memory(query: str, limit: int = 5) -> str:
    """Search personal persistent memories, facts, narrative chronicles, learnings, and linked relations by keyword.

    Args:
        query: Search terms or keywords to query the memory store.
        limit: Maximum number of results to return per category (default: 5).
    """
    empty_envelope = {
        "facts": [],
        "episodes": [],
        "learnings": [],
        "entity_links": []
    }

    # F20: Clamp finite limits (avoid negative limit disabling SQLite bound or huge bounds)
    try:
        limit_val = int(limit)
    except (ValueError, TypeError):
        limit_val = 5
    clamped_limit = max(1, min(limit_val, 100))

    if not query or not str(query).strip():
        return json.dumps(empty_envelope, ensure_ascii=False, indent=2)

    with db_session() as conn:
        cursor = conn.cursor()
        vocab = get_all_vocabulary(cursor)
        words = extract_multilingual_tokens(str(query).strip(), vocab)
        # --- Lexical Search (FTS5) ---
        fts_facts = []
        fts_episodes = []
        fts_learnings = []
        fts_query = None
        if words:
            fts_terms = [f'"{w}"*' if len(w) >= 4 else f'"{w}"' for w in words]
            fts_query = " OR ".join(fts_terms)

            # Facts via JOIN (ranked by relevance)
            cursor.execute("""
                SELECT m.id, m.category, m.fact 
                FROM memories m
                JOIN memories_fts f ON m.id = f.id
                WHERE memories_fts MATCH ?
                ORDER BY f.rank
                LIMIT ?
            """, (fts_query, clamped_limit))
            fts_facts = [{"type": "fact", "id": r[0], "category": r[1], "content": r[2]} for r in cursor.fetchall()]

            # Episodes via JOIN (ranked with status weighting: active > cooling > historic/resolved)
            cursor.execute("""
                SELECT e.id, e.topic, e.title, e.period, e.status, e.narrative, e.stance
                FROM episodes e
                JOIN episodes_fts f ON e.id = f.id
                WHERE episodes_fts MATCH ?
                ORDER BY 
                    CASE e.status 
                        WHEN 'active' THEN 1 
                        WHEN 'cooling' THEN 2 
                        ELSE 3 
                    END ASC,
                    f.rank ASC
                LIMIT ?
            """, (fts_query, clamped_limit))
            fts_episodes = [{
                "type": "episode",
                "id": r[0],
                "topic": r[1],
                "title": r[2],
                "period": r[3],
                "status": r[4],
                "narrative": r[5],
                "stance": r[6]
            } for r in cursor.fetchall()]

            # Learnings via JOIN (ranked by relevance)
            cursor.execute("""
                SELECT l.id, l.category, l.insight, l.context
                FROM learnings l
                JOIN learnings_fts f ON l.id = f.id
                WHERE learnings_fts MATCH ?
                ORDER BY f.rank
                LIMIT ?
            """, (fts_query, clamped_limit))
            fts_learnings = [{
                "type": "learning",
                "id": r[0],
                "category": r[1],
                "insight": r[2],
                "context": r[3]
            } for r in cursor.fetchall()]

        # --- Semantic Vector Search (sqlite-vec) ---
        vec_facts = []
        vec_episodes = []
        vec_learnings = []
        if HAS_EMBEDDER and VECTOR_SEARCH_ENABLED and embed_text:
            query_emb = embed_text(str(query).strip())
            if query_emb is not None:
                # Semantic search in vec_memories
                try:
                    cursor.execute("""
                        SELECT m.id, m.category, m.fact, v.distance
                        FROM vec_memories v
                        JOIN memories m ON m.id = v.id
                        WHERE v.embedding MATCH ? AND k = ?
                        ORDER BY v.distance ASC
                    """, (query_emb, clamped_limit))
                    vec_facts = [{"type": "fact", "id": r[0], "category": r[1], "content": r[2]} for r in cursor.fetchall()]
                except Exception as e:
                    if log_vec_query_failure:
                        log_vec_query_failure("vec_memories", e)

                # Semantic search in vec_episodes
                try:
                    cursor.execute("""
                        SELECT e.id, e.topic, e.title, e.period, e.status, e.narrative, e.stance, v.distance
                        FROM vec_episodes v
                        JOIN episodes e ON e.id = v.id
                        WHERE v.embedding MATCH ? AND k = ?
                        ORDER BY v.distance ASC
                    """, (query_emb, clamped_limit))
                    vec_episodes = [{
                        "type": "episode",
                        "id": r[0],
                        "topic": r[1],
                        "title": r[2],
                        "period": r[3],
                        "status": r[4],
                        "narrative": r[5],
                        "stance": r[6]
                    } for r in cursor.fetchall()]
                except Exception as e:
                    if log_vec_query_failure:
                        log_vec_query_failure("vec_episodes", e)

                # Semantic search in vec_learnings
                try:
                    cursor.execute("""
                        SELECT l.id, l.category, l.insight, l.context, v.distance
                        FROM vec_learnings v
                        JOIN learnings l ON l.id = v.id
                        WHERE v.embedding MATCH ? AND k = ?
                        ORDER BY v.distance ASC
                    """, (query_emb, clamped_limit))
                    vec_learnings = [{
                        "type": "learning",
                        "id": r[0],
                        "category": r[1],
                        "insight": r[2],
                        "context": r[3]
                    } for r in cursor.fetchall()]
                except Exception as e:
                    if log_vec_query_failure:
                        log_vec_query_failure("vec_learnings", e)

        # --- Reciprocal Rank Fusion (RRF) ---
        if HAS_EMBEDDER and reciprocal_rank_fusion:
            facts = reciprocal_rank_fusion(fts_facts, vec_facts, limit=clamped_limit)
            episodes = reciprocal_rank_fusion(fts_episodes, vec_episodes, limit=clamped_limit)
            learnings = reciprocal_rank_fusion(fts_learnings, vec_learnings, limit=clamped_limit)
        else:
            facts = fts_facts[:clamped_limit]
            episodes = fts_episodes[:clamped_limit]
            learnings = fts_learnings[:clamped_limit]

        # Entity links (lexical FTS query)
        entity_links = []
        if words and fts_query:
            cursor.execute("""
                SELECT l.source_id, l.target_id, l.relation
                FROM entity_links l
                JOIN entity_links_fts f ON l.source_id = f.source_id AND l.target_id = f.target_id AND l.relation = f.relation
                WHERE entity_links_fts MATCH ?
                LIMIT ?
            """, (fts_query, clamped_limit))
            entity_links = [{"source": r[0], "target": r[1], "relation": r[2]} for r in cursor.fetchall()]

        return json.dumps({
            "facts": facts,
            "episodes": episodes,
            "learnings": learnings,
            "entity_links": entity_links
        }, ensure_ascii=False, indent=2)


def store_memory(id: str, fact: str, category: str = "general", keywords: str = "") -> str:
    """Store or update an atomic persistent fact or configuration parameter.

    Args:
        id: Unique identifier / key for this memory (e.g. 'infra.server.ip').
        fact: Fact content or description.
        category: Category classification (normalized to canonical taxonomy).
        keywords: Optional search keywords or synonyms.
    """
    clean_id = (id or "").strip()
    clean_fact = (fact or "").strip()
    if not clean_id:
        raise ValueError("Fact identifier 'id' must be a non-empty string.")
    if not clean_fact:
        raise ValueError("Fact content 'fact' must be a non-empty string.")

    norm_category = validate_category(category, CANONICAL_FACT_CATEGORIES)
    upsert_fact(clean_id, norm_category, clean_fact, (keywords or "").strip())
    return f"Successfully stored fact '{clean_id}' (category: {norm_category})"


def record_episode(
    id: str,
    topic: str,
    title: str,
    narrative: str,
    period: str = "",
    status: str = "active",
    entities: str = "",
    stance: str = "",
    keywords: str = ""
) -> str:
    """Record or update a narrative chronicle, background story, relationship context, or ongoing topic dossier.

    Args:
        id: Unique identifier (e.g. 'home.sent.sanierung', 'health.abbie.epilepsie').
        topic: Topic domain (normalized to canonical taxonomy: family, health, travel, finance, home, dev, infra, insurance, music, work, realestate, trading, general).
        title: Human-readable title of this chronicle.
        narrative: Rich multi-sentence narrative summary of history, events, and background context.
        period: Time period (e.g. '2020 - laufend', 'Sommer 2026').
        status: Current status ('active', 'cooling', 'historic', 'resolved').
        entities: Involved people, organizations, or places.
        stance: User's stance, attitude, sentiments, or approach to this subject.
        keywords: Multilingual search terms and synonyms.
    """
    clean_id = (id or "").strip()
    clean_title = (title or "").strip()
    clean_narrative = (narrative or "").strip()
    if not clean_id:
        raise ValueError("Episode identifier 'id' must be a non-empty string.")
    if not clean_title:
        raise ValueError("Episode 'title' must be a non-empty string.")
    if not clean_narrative:
        raise ValueError("Episode 'narrative' must be a non-empty string.")

    norm_topic = validate_category(topic, CANONICAL_EPISODE_TOPICS)
    norm_status = (status or "").strip().lower() or "active"
    if norm_status not in CANONICAL_EPISODE_STATUSES:
        raise ValueError(f"Invalid episode status: {status}")

    upsert_episode(clean_id, norm_topic, clean_title, clean_narrative, period or "", norm_status, entities or "", stance or "", keywords or "")
    return f"Successfully recorded narrative episode '{clean_id}' (topic: {norm_topic}, status: {norm_status})"


def record_learning(id: str, category: str, insight: str, context: str = "", keywords: str = "") -> str:
    """Record a practical learning, rule of thumb, heuristic, or tested opinion.

    Args:
        id: Unique key (e.g. 'travel.fewo_dog', 'automation.systemd_decouple').
        category: Category (normalized to canonical taxonomy: workflow, communication, finance, health, shopping, travel, hardware, safety, architecture, security, automation, general).
        insight: The lesson learned or heuristic.
        context: Context of how/when this was learned.
        keywords: Search terms and synonyms.
    """
    clean_id = (id or "").strip()
    clean_insight = (insight or "").strip()
    if not clean_id:
        raise ValueError("Learning identifier 'id' must be a non-empty string.")
    if not clean_insight:
        raise ValueError("Learning 'insight' must be a non-empty string.")

    norm_category = validate_category(category, CANONICAL_LEARNING_CATEGORIES)
    upsert_learning(clean_id, norm_category, clean_insight, context or "", keywords or "")
    return f"Successfully recorded learning '{clean_id}' (category: {norm_category})"

def link_entities_mcp(source_id: str, target_id: str, relation: str) -> str:
    """Link two memory entities with a canonical semantic relationship.

    Args:
        source_id: Source ID (e.g. 'service.immich').
        target_id: Target ID (e.g. 'infra.beelink').
        relation: Canonical relation type (e.g. 'hosted_on', 'runs_on', 'depends_on', 'part_of', 'member_of', 'monitors', 'uses', 'stores', 'related_to').
                  Legacy relations (e.g. 'hosts', 'runs_in') are automatically mapped and directionally inverted if needed.
    """
    src = (source_id or "").strip()
    tgt = (target_id or "").strip()
    rel = (relation or "").strip()
    if not src:
        raise ValueError("Source entity id 'source_id' must be a non-empty string.")
    if not tgt:
        raise ValueError("Target entity id 'target_id' must be a non-empty string.")
    if not rel:
        raise ValueError("Relation 'relation' must be a non-empty string.")

    canonical_src, canonical_tgt, canonical_rel = map_relation(src, tgt, rel)

    link_entities(canonical_src, canonical_tgt, canonical_rel)
    return f"Successfully linked '{canonical_src}' --[{canonical_rel}]--> '{canonical_tgt}'"


def list_memories() -> str:
    """List all stored semantic facts, narrative chronicles, learnings, and entity links."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, category, fact FROM memories ORDER BY category, id")
        facts = [{"id": r[0], "category": r[1], "fact": r[2]} for r in cursor.fetchall()]

        cursor.execute("SELECT id, topic, title, period, status, narrative, stance FROM episodes ORDER BY topic, id")
        episodes = [{
            "id": r[0],
            "topic": r[1],
            "title": r[2],
            "period": r[3],
            "status": r[4],
            "narrative": r[5],
            "stance": r[6]
        } for r in cursor.fetchall()]

        cursor.execute("SELECT id, category, insight, context FROM learnings ORDER BY category, id")
        learnings = [{"id": r[0], "category": r[1], "insight": r[2], "context": r[3]} for r in cursor.fetchall()]

        cursor.execute("SELECT source_id, target_id, relation FROM entity_links ORDER BY source_id, target_id")
        links = [{"source": r[0], "target": r[1], "relation": r[2]} for r in cursor.fetchall()]

        return json.dumps({
            "facts": facts,
            "episodes": episodes,
            "learnings": learnings,
            "entity_links": links
        }, ensure_ascii=False, indent=2)


def migrate_memory(dry_run: bool = True) -> str:
    """Migrate database to canonical v2.1 taxonomies, map relations, prune orphan links, and rebuild FTS indexes.

    Args:
        dry_run: If True (default), simulates the migration and returns proposed changes without modifying the database.
                 Set to False to apply the migration live.
    """
    report = run_migration(dry_run=dry_run, verbose=False)
    return json.dumps({
        "status": "dry_run_complete" if dry_run else "migration_complete",
        "operation_id": uuid.uuid4().hex[:8],
        "backup_file": report["backup_file"],
        "facts_migrated": report["facts_migrated"],
        "episodes_migrated": report["episodes_migrated"],
        "learnings_migrated": report["learnings_migrated"],
        "links_mapped": report["links_mapped"],
        "orphan_links_pruned": report["orphan_links_pruned"],
        "details": report["details"],
        "data_committed": report.get("data_committed", False),
        "compacted": report.get("compacted"),
        "compaction_error": report.get("compaction_error")
    }, ensure_ascii=False, indent=2)


def optimize_memory(apply_changes: bool = True, consolidate: bool = False) -> str:
    """Run database optimization: episode aging decay, orphan link pruning, FTS index rebuild, and VACUUM.

    Args:
        apply_changes: Whether to apply changes to disk (default: True).
        consolidate: Run semantic LLM deduplication across facts (default: False).
    """
    stats = optimize_db(apply_changes=apply_changes, age_decay=True, consolidate=consolidate)
    return json.dumps({
        "status": "success",
        "operation_id": uuid.uuid4().hex[:8],
        "message": "Database optimization completed." if apply_changes else "Optimization preview; no maintenance changes applied.",
        "stats": stats
    }, ensure_ascii=False)


# -----------------------------------------------------------------------------
# FastMCP Tool Registrations (Async offloaded handlers for clean transport)
# -----------------------------------------------------------------------------

@mcp.tool(name="search_memory")
async def _search_memory_mcp(query: str, limit: int = 5) -> str:
    """Search personal persistent memories, facts, narrative chronicles, learnings, and linked relations by keyword.

    Args:
        query: Search terms or keywords to query the memory store.
        limit: Maximum number of results to return per category (default: 5).
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: search_memory(query=query, limit=limit))


@mcp.tool(name="store_memory")
async def _store_memory_mcp(id: str, fact: str, category: str = "general", keywords: str = "") -> str:
    """Store or update an atomic persistent fact or configuration parameter.

    Args:
        id: Unique identifier / key for this memory (e.g. 'infra.server.ip').
        fact: Fact content or description.
        category: Category classification (normalized to canonical taxonomy: infra, hardware, software, contacts, family, health, fitness, finance, insurance, travel, home, media, music, work, dev, preferences, communication, cloud, security, general).
        keywords: Optional search keywords or synonyms.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: store_memory(id=id, fact=fact, category=category, keywords=keywords))


@mcp.tool(name="record_episode")
async def _record_episode_mcp(
    id: str,
    topic: str,
    title: str,
    narrative: str,
    period: str = "",
    status: str = "active",
    entities: str = "",
    stance: str = "",
    keywords: str = ""
) -> str:
    """Record or update a narrative chronicle, background story, relationship context, or ongoing topic dossier.

    Args:
        id: Unique identifier (e.g. 'home.sent.sanierung', 'health.abbie.epilepsie').
        topic: Topic domain (normalized to canonical taxonomy: family, health, travel, finance, home, dev, infra, insurance, music, work, realestate, trading, general).
        title: Human-readable title of this chronicle.
        narrative: Rich multi-sentence narrative summary of history, events, and background context.
        period: Time period (e.g. '2020 - laufend', 'Sommer 2026').
        status: Current status ('active', 'cooling', 'historic', 'resolved').
        entities: Involved people, organizations, or places.
        stance: User's stance, attitude, sentiments, or approach to this subject.
        keywords: Multilingual search terms and synonyms.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: record_episode(
            id=id,
            topic=topic,
            title=title,
            narrative=narrative,
            period=period,
            status=status,
            entities=entities,
            stance=stance,
            keywords=keywords
        )
    )


@mcp.tool(name="record_learning")
async def _record_learning_mcp(id: str, category: str, insight: str, context: str = "", keywords: str = "") -> str:
    """Record a practical learning, rule of thumb, heuristic, or tested opinion.

    Args:
        id: Unique key (e.g. 'travel.fewo_dog', 'automation.systemd_decouple').
        category: Category (normalized to canonical taxonomy: workflow, communication, finance, health, shopping, travel, hardware, safety, architecture, security, automation, general).
        insight: The lesson learned or heuristic.
        context: Context of how/when this was learned.
        keywords: Search terms and synonyms.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: record_learning(id=id, category=category, insight=insight, context=context, keywords=keywords)
    )


@mcp.tool(name="link_entities_mcp")
async def _link_entities_mcp(source_id: str, target_id: str, relation: str) -> str:
    """Link two memory entities with a canonical semantic relationship.

    Args:
        source_id: Source ID (e.g. 'service.immich').
        target_id: Target ID (e.g. 'infra.beelink').
        relation: Canonical relation type (e.g. 'hosted_on', 'runs_on', 'depends_on', 'part_of', 'member_of', 'monitors', 'uses', 'stores', 'related_to').
                  Legacy relations (e.g. 'hosts', 'runs_in') are automatically mapped and directionally inverted if needed.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: link_entities_mcp(source_id=source_id, target_id=target_id, relation=relation))


@mcp.tool(name="list_memories")
async def _list_memories_mcp() -> str:
    """List all stored semantic facts, narrative chronicles, learnings, and entity links."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, list_memories)


async def _run_maintenance(function):
    if not _MAINTENANCE_SLOT.acquire(blocking=False):
        raise RuntimeError('Memory maintenance is busy; retry after the active operation completes')
    def run():
        try:
            return function()
        finally:
            _MAINTENANCE_SLOT.release()
    # Cancellation detaches the caller; the slot stays occupied until the actual
    # worker completes. A cancelled request is never a rollback guarantee.
    future = asyncio.get_running_loop().run_in_executor(_MAINTENANCE_EXECUTOR, run)
    return await asyncio.shield(future)


@mcp.tool(name="migrate_memory")
async def _migrate_memory_mcp(dry_run: bool = True) -> str:
    """Migrate database to canonical v2.1 taxonomies, map relations, prune orphan links, and rebuild FTS indexes.

    Args:
        dry_run: If True (default), simulates the migration and returns proposed changes without modifying the database.
                 Set to False to apply the migration live.
    """
    loop = asyncio.get_running_loop()
    return await _run_maintenance(lambda: migrate_memory(dry_run=dry_run))


@mcp.tool(name="optimize_memory")
async def _optimize_memory_mcp(apply_changes: bool = True, consolidate: bool = False) -> str:
    """Run database optimization: episode aging decay, orphan link pruning, FTS index rebuild, and VACUUM.

    Args:
        apply_changes: Whether to apply changes to disk (default: True).
        consolidate: Run semantic LLM deduplication across facts (default: False).
    """
    loop = asyncio.get_running_loop()
    return await _run_maintenance(lambda: optimize_memory(apply_changes=apply_changes, consolidate=consolidate))


if __name__ == "__main__":
    mcp.run()
