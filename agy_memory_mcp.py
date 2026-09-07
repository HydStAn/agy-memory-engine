#!/usr/bin/env python3
"""
AGY Memory Engine - MCP Server Layer (FastMCP)
Allows AGY to explicitly query, store, link entities, and manage multi-layer memories (Facts, Episodes, Learnings).
"""

import json
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

mcp = FastMCP("memory")


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """Search personal persistent memories, facts, narrative chronicles, learnings, and linked relations by keyword.
    
    Args:
        query: Search terms or keywords to query the memory store.
        limit: Maximum number of results to return per category (default: 5).
    """
    try:
        with db_session() as conn:
            cursor = conn.cursor()
            vocab = get_all_vocabulary(cursor)
            words = extract_multilingual_tokens(query, vocab)
            # --- Lexical Search (FTS5) ---
            fts_facts = []
            fts_episodes = []
            fts_learnings = []
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
                """, (fts_query, limit))
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
                """, (fts_query, limit))
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
                """, (fts_query, limit))
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
                query_emb = embed_text(query)
                if query_emb is not None:
                    # Semantic search in vec_memories
                    try:
                        cursor.execute("""
                            SELECT m.id, m.category, m.fact, v.distance
                            FROM vec_memories v
                            JOIN memories m ON m.id = v.id
                            WHERE v.embedding MATCH ? AND k = ?
                            ORDER BY v.distance ASC
                        """, (query_emb, limit))
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
                        """, (query_emb, limit))
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
                        """, (query_emb, limit))
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
                facts = reciprocal_rank_fusion(fts_facts, vec_facts, limit=limit)
                episodes = reciprocal_rank_fusion(fts_episodes, vec_episodes, limit=limit)
                learnings = reciprocal_rank_fusion(fts_learnings, vec_learnings, limit=limit)
            else:
                facts = fts_facts[:limit]
                episodes = fts_episodes[:limit]
                learnings = fts_learnings[:limit]

            # Entity links (lexical FTS query)
            entity_links = []
            if words:
                cursor.execute("""
                    SELECT l.source_id, l.target_id, l.relation
                    FROM entity_links l
                    JOIN entity_links_fts f ON l.source_id = f.source_id AND l.target_id = f.target_id AND l.relation = f.relation
                    WHERE entity_links_fts MATCH ?
                    LIMIT ?
                """, (fts_query, limit))
                entity_links = [{"source": r[0], "target": r[1], "relation": r[2]} for r in cursor.fetchall()]

            return json.dumps({
                "facts": facts,
                "episodes": episodes,
                "learnings": learnings,
                "entity_links": entity_links
            }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def store_memory(id: str, fact: str, category: str = "general", keywords: str = "") -> str:
    """Store or update an atomic persistent fact or configuration parameter.
    
    Args:
        id: Unique identifier / key for this memory (e.g. 'infra.server.ip').
        fact: Fact content or description.
        category: Category classification (normalized to canonical taxonomy: infra, hardware, software, contacts, family, health, fitness, finance, insurance, travel, home, media, music, work, dev, preferences, communication, cloud, security, general).
        keywords: Optional search keywords or synonyms.
    """
    try:
        norm_category = normalize_category(category, CANONICAL_FACT_CATEGORIES)
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, category, fact, keywords, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    category = excluded.category,
                    fact = excluded.fact,
                    keywords = excluded.keywords,
                    updated_at = CURRENT_TIMESTAMP;
            """, (id.strip(), norm_category, fact.strip(), keywords.strip()))
            if HAS_EMBEDDER and VECTOR_SEARCH_ENABLED and upsert_vector:
                text_repr = build_text_repr("fact", {"category": norm_category, "fact": fact.strip(), "keywords": keywords.strip()}) if build_text_repr else f"[{norm_category}] {fact.strip()} {keywords.strip()}".strip()
                upsert_vector(conn, "vec_memories", id.strip(), text_repr)
            conn.commit()
        return f"Successfully stored fact '{id}' (category: {norm_category})"
    except Exception as e:
        return f"Error storing memory: {e}"

@mcp.tool()
def record_episode(id: str, topic: str, title: str, narrative: str, period: str = "", status: str = "active", entities: str = "", stance: str = "", keywords: str = "") -> str:
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
    try:
        norm_topic = normalize_category(topic, CANONICAL_EPISODE_TOPICS)
        norm_status = status.strip().lower() if status else "active"
        if norm_status not in CANONICAL_EPISODE_STATUSES:
            norm_status = "active"

        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodes (id, topic, title, period, status, narrative, entities, stance, keywords, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    topic = excluded.topic,
                    title = excluded.title,
                    period = excluded.period,
                    status = excluded.status,
                    narrative = excluded.narrative,
                    entities = excluded.entities,
                    stance = excluded.stance,
                    keywords = excluded.keywords,
                    updated_at = CURRENT_TIMESTAMP;
            """, (id.strip(), norm_topic, title.strip(), period.strip(), norm_status, narrative.strip(), entities.strip(), stance.strip(), keywords.strip()))
            if HAS_EMBEDDER and VECTOR_SEARCH_ENABLED and upsert_vector:
                text_repr = build_text_repr("episode", {"topic": norm_topic, "title": title.strip(), "narrative": narrative.strip(), "stance": stance.strip(), "keywords": keywords.strip()}) if build_text_repr else f"[{norm_topic}] {title.strip()}: {narrative.strip()} (Stance: {stance.strip() or 'neutral'}) {keywords.strip()}".strip()
                upsert_vector(conn, "vec_episodes", id.strip(), text_repr)
            conn.commit()
        return f"Successfully recorded narrative episode '{id}' (topic: {norm_topic}, status: {norm_status})"
    except Exception as e:
        return f"Error recording episode: {e}"

@mcp.tool()
def record_learning(id: str, category: str, insight: str, context: str = "", keywords: str = "") -> str:
    """Record a practical learning, rule of thumb, heuristic, or tested opinion.
    
    Args:
        id: Unique key (e.g. 'travel.fewo_dog', 'automation.systemd_decouple').
        category: Category (normalized to canonical taxonomy: workflow, communication, finance, health, shopping, travel, hardware, safety, architecture, security, automation, general).
        insight: The lesson learned or heuristic.
        context: Context of how/when this was learned.
        keywords: Search terms and synonyms.
    """
    try:
        norm_category = normalize_category(category, CANONICAL_LEARNING_CATEGORIES)
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learnings (id, category, insight, context, keywords, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    category = excluded.category,
                    insight = excluded.insight,
                    context = excluded.context,
                    keywords = excluded.keywords,
                    updated_at = CURRENT_TIMESTAMP;
            """, (id.strip(), norm_category, insight.strip(), context.strip(), keywords.strip()))
            if HAS_EMBEDDER and VECTOR_SEARCH_ENABLED and upsert_vector:
                text_repr = build_text_repr("learning", {"category": norm_category, "insight": insight.strip(), "context": context.strip(), "keywords": keywords.strip()}) if build_text_repr else f"[{norm_category}] {insight.strip()} (Context: {context.strip() or ''}) {keywords.strip()}".strip()
                upsert_vector(conn, "vec_learnings", id.strip(), text_repr)
            conn.commit()
        return f"Successfully recorded learning '{id}' (category: {norm_category})"
    except Exception as e:
        return f"Error recording learning: {e}"

@mcp.tool()
def link_entities_mcp(source_id: str, target_id: str, relation: str) -> str:
    """Link two memory entities with a canonical semantic relationship.
    
    Args:
        source_id: Source ID (e.g. 'service.immich').
        target_id: Target ID (e.g. 'infra.beelink').
        relation: Canonical relation type (e.g. 'hosted_on', 'runs_on', 'depends_on', 'part_of', 'member_of', 'monitors', 'uses', 'stores', 'related_to').
                  Legacy relations (e.g. 'hosts', 'runs_in') are automatically mapped and directionally inverted if needed.
    """
    try:
        src = source_id.strip()
        tgt = target_id.strip()
        canonical_src, canonical_tgt, canonical_rel = map_relation(src, tgt, relation)

        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO entity_links (source_id, target_id, relation)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id, target_id, relation) DO NOTHING;
            """, (canonical_src, canonical_tgt, canonical_rel))
            conn.commit()
        return f"Successfully linked '{canonical_src}' --[{canonical_rel}]--> '{canonical_tgt}'"
    except Exception as e:
        return f"Error linking entities: {e}"

@mcp.tool()
def list_memories() -> str:
    """List all stored semantic facts, narrative chronicles, learnings, and entity links."""
    try:
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
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def migrate_memory(dry_run: bool = True) -> str:
    """Migrate database to canonical v2.1 taxonomies, map relations, prune orphan links, and rebuild FTS indexes.
    
    Args:
        dry_run: If True (default), simulates the migration and returns proposed changes without modifying the database.
                 Set to False to apply the migration live.
    """
    try:
        report = run_migration(dry_run=dry_run, verbose=False)
        return json.dumps({
            "status": "dry_run_complete" if dry_run else "migration_complete",
            "backup_file": report["backup_file"],
            "facts_migrated": report["facts_migrated"],
            "episodes_migrated": report["episodes_migrated"],
            "learnings_migrated": report["learnings_migrated"],
            "links_mapped": report["links_mapped"],
            "orphan_links_pruned": report["orphan_links_pruned"],
            "details": report["details"]
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@mcp.tool()
def optimize_memory(apply_changes: bool = True, consolidate: bool = False) -> str:
    """Run database optimization: episode aging decay, orphan link pruning, FTS index rebuild, and VACUUM.
    
    Args:
        apply_changes: Whether to apply changes to disk (default: True).
        consolidate: Run semantic LLM deduplication across facts (default: False).
    """
    try:
        optimize_db(apply_changes=apply_changes, age_decay=True, consolidate=consolidate)
        return json.dumps({
            "status": "success",
            "message": "Database optimization and FTS index rebuild completed successfully."
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
