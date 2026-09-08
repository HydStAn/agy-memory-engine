#!/usr/bin/env python3
"""
AGY Memory Engine - Multi-Layer Cognitive Memory Component
- Layer 1: Semantic Fact-Store (SQLite FTS5: IPs, configs, hardware, master data)
- Layer 2: Narrative & Episodic Store (Themen-Dossiers, Chroniken, Beziehungsdynamiken, Verläufe)
- Layer 3: Experiential & Learnings Store (Erkenntnisse, Heuristiken, Urteile, Haltungen)
- Feature: Entity Graph & Relationships (Entity Linking)
- Feature: Automatic Episode Aging & State Decay (active -> cooling -> historic)
"""

import sys
import os
import sqlite3
import argparse
import re
import subprocess
import json
import difflib
import datetime
import logging
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager, nullcontext, closing

import schema
from schema import db_session, DB_PATH, PROTECTED_CATEGORIES
from config import (
    MODEL_NAME,
    DEFAULT_MODEL,
    CACHE_PATH,
    AGY_BIN
)

__version__ = "2.1.0"

logger = logging.getLogger("agy_memory")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)
logger.propagate = False
_VOCABULARY_CACHE = {}


class SyncBusyError(RuntimeError):
    """Extraction could not acquire its lock; the turn must be retried."""


class SyncExtractionError(RuntimeError):
    """Extraction or persistence failed; the turn must not be acknowledged."""


@contextmanager
def _writer(connection=None):
    """Reuse the caller's transaction, or commit one standalone operation."""
    if connection is not None:
        yield connection
    else:
        with db_session() as conn, conn:
            yield conn
    _VOCABULARY_CACHE.clear()


# Canonical taxonomies — single source of truth for extraction prompt AND runtime validation
CANONICAL_FACT_CATEGORIES = frozenset({
    "infra", "hardware", "software", "contacts", "family", "health", "fitness",
    "finance", "insurance", "travel", "home", "media", "music", "work", "dev",
    "preferences", "communication", "cloud", "security", "general"
})

CANONICAL_LEARNING_CATEGORIES = frozenset({
    "workflow", "communication", "finance", "health", "shopping", "travel",
    "hardware", "safety", "architecture", "security", "automation", "preferences", "insurance", "general"
})

CANONICAL_EPISODE_TOPICS = frozenset({
    "family", "health", "travel", "finance", "home", "dev", "infra",
    "insurance", "music", "work", "realestate", "trading", "general"
})

CANONICAL_RELATIONS = frozenset({
    "hosted_on", "runs_on", "depends_on", "part_of", "member_of",
    "owned_by", "managed_by", "monitors", "treats", "prescribed_for",
    "insured_by", "finances", "communicates_via", "located_at", "uses",
    "stores", "connects_to", "related_to", "maintains", "created_by",
    "delivers_to", "advises", "works_at", "lives_at", "travels_to",
    "subscribed_to", "prescribes"
})


STOPWORDS = {
    # German (de)
    "wie", "was", "wer", "wo", "wann", "warum", "welches", "welche", "welcher", "woher", "wohin",
    "ist", "sind", "war", "waren", "wird", "werden", "habe", "hat", "haben", "auf", "mit", "von",
    "aus", "für", "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem", "einen",
    "und", "oder", "aber", "auch", "noch", "nur", "schon", "immer", "wieder", "heute", "gestern",
    "morgen", "mein", "meine", "meinen", "meiner", "meinem", "unser", "unsere", "unserem", "unseren",
    "lautet", "läuft", "geht", "bekommt", "macht", "gibt", "zeigt", "registriert", "geregelt",
    "schau", "sag", "zeig", "prüfe", "checke", "bitte", "mal", "uns", "ihr", "ihre", "ihrem", "ihren",

    # English (en)
    "the", "and", "is", "are", "was", "were", "what", "where", "when", "how", "who", "why",
    "which", "with", "from", "for", "about", "can", "could", "would", "should", "have", "has",
    "had", "our", "my", "your", "his", "her", "their", "its", "show", "tell", "check", "find",
    "that", "this", "then", "them", "they", "been", "into", "some", "more", "most", "such",
    "will", "shall", "does", "did", "done", "give", "look", "here", "there",

    # French (fr)
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais", "donc", "car", "ni",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "ce", "cet", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "votre", "leur",
    "leurs", "nous", "vous", "ils", "elles", "qui", "que", "quoi", "quand", "comment", "pourquoi",
    "est", "sont", "ete", "être", "avoir", "fait", "faire", "montre", "donne",

    # Italian (it)
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "ed", "o", "ma", "anche", "se",
    "per", "con", "su", "tra", "fra", "da", "in", "del", "dello", "della", "dei", "degli", "delle",
    "mio", "mia", "miei", "mie", "tuo", "tua", "tuoi", "tue", "suo", "sua", "suoi", "sue",
    "nostro", "nostra", "nostri", "nostre", "vostro", "vostra", "vostri", "vostre", "loro",
    "che", "chi", "cosa", "quando", "come", "dove", "perche", "perché", "sono", "siamo", "siete",
    "stato", "stata", "fare", "mostra", "dimmi",

    # Spanish (es)
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "sino", "de", "del",
    "a", "al", "en", "con", "por", "para", "sobre", "sin", "mi", "mis", "tu", "tus", "su", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras", "que", "qué", "quien", "quién", "como", "cómo",
    "cuando", "cuándo", "donde", "dónde", "porque", "porqué", "es", "son", "fue", "eran", "sido",
    "ser", "estar", "haber", "hacer", "muestra", "dime",

    # Dutch (nl)
    "de", "het", "een", "en", "of", "maar", "want", "dus", "in", "op", "van", "met", "voor",
    "naar", "uit", "over", "aan", "bij", "om", "door", "mijn", "jouw", "zijn", "haar", "onze",
    "jullie", "hun", "wie", "wat", "waar", "wanneer", "waarom", "hoe", "zijn", "was", "waren",
    "heeft", "hebben", "worden", "wordt", "toon", "laat",

    # Scandinavian (sv / da / no)
    "den", "det", "ett", "og", "och", "eller", "men", "som", "på", "till", "från", "fra",
    "mitt", "mina", "ditt", "dina", "hans", "hennes", "vår", "vårt", "våra", "vad", "hvem",
    "vem", "hvor", "var", "när", "når", "har", "hade", "haft", "bli", "blive", "blivit",

    # Latin / Generic
    "est", "non", "sed", "et", "aut", "cum", "per"
}

STEM_SUFFIXES = (
    "ungen", "heiten", "keiten", "schaft", "lichen", "ischen", "enden", "ation", "ations", "azioni", "azione",
    "aciones", "acion", "menti", "mento", "mente", "ings", "ing", "ies", "ied", "ers", "ung", "heit", "keit",
    "lich", "isch", "end", "ern", "en", "er", "es", "em", "ed", "ly", "os", "as", "es", "ando", "endo", "anti", "ante"
)

FUGEN_MORPHEMES = ("s", "en", "n", "er", "e")

def extract_multilingual_tokens(query: str, vocab: set = None) -> list[str]:
    """Extract search terms, perform multilingual morphological suffix stemming,
    and decompose compound nouns with Fugenmorphemes & database vocabulary validation.
    """
    raw_words = re.findall(r'[\w]+', query.lower(), re.UNICODE)
    base_words = [w for w in raw_words if len(w) >= 2 and w not in STOPWORDS]

    expanded_words = set(base_words)

    for w in base_words:
        # 1. Morphological Stemming / Suffix Normalization
        if len(w) >= 5:
            for sfx in STEM_SUFFIXES:
                if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                    stem = w[:-len(sfx)]
                    if stem not in STOPWORDS and len(stem) >= 3:
                        expanded_words.add(stem)
                        break

        # 2. Multilingual Compound Sub-Token Splitting with Vocabulary Validation
        if len(w) >= 8:
            vocab_matches = set()
            unmatched_candidates = set()

            for split_len in range(4, len(w) - 3):
                p1, p2 = w[:split_len], w[split_len:]
                pairs = [(p1, p2)]

                # Interfix / Fugenmorpheme Handling
                for f in FUGEN_MORPHEMES:
                    if p1.endswith(f) and len(p1) - len(f) >= 3:
                        pairs.append((p1[:-len(f)], p2))

                for part1, part2 in pairs:
                    if part1 not in STOPWORDS and len(part1) >= 4 and part2 not in STOPWORDS and len(part2) >= 4:
                        if vocab and (part1 in vocab or part2 in vocab):
                            if part1 in vocab:
                                vocab_matches.add(part1)
                            if part2 in vocab:
                                vocab_matches.add(part2)
                        else:
                            unmatched_candidates.add(part1)
                            unmatched_candidates.add(part2)

            if vocab_matches:
                expanded_words.update(vocab_matches)
            elif not vocab:
                expanded_words.update(unmatched_candidates)

    return sorted(expanded_words)

TRIVIAL_PROMPT_RE = re.compile(
    r'^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|'
    r'hi|hey|hello|yo|sup|1|2|3|4|5|'
    r'continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)'
    r'[\s!?.:;,"\'' + r'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]*$',
    re.IGNORECASE,
)

def get_existing_database_inventory() -> dict:
    """Retrieve the full inventory of existing keys/IDs across all three layers and entity links."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM memories")
        fact_keys = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT id, title, topic, status FROM episodes")
        episode_keys = [{"id": row[0], "title": row[1], "topic": row[2], "status": row[3]} for row in cursor.fetchall()]
        cursor.execute("SELECT id, category FROM learnings")
        learning_keys = [{"id": row[0], "category": row[1]} for row in cursor.fetchall()]
        cursor.execute("SELECT source_id, target_id, relation FROM entity_links")
        link_keys = [{"source": row[0], "target": row[1], "relation": row[2]} for row in cursor.fetchall()]
        return {
            "fact_keys": fact_keys,
            "episodes": episode_keys,
            "learnings": learning_keys,
            "entity_links": link_keys
        }

def get_cached_model() -> str:
    """Read model from config (.env/env var) or cached file on disk, otherwise return default."""
    if MODEL_NAME and MODEL_NAME != DEFAULT_MODEL:
        return MODEL_NAME
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                val = f.read().strip()
                if val:
                    return val
        except Exception:
            pass
    return MODEL_NAME or DEFAULT_MODEL

def discover_and_cache_latest_flash_low_model() -> str:
    """Scan agy models list for the newest Gemini Flash model and persist it to disk."""
    try:
        res = subprocess.run([AGY_BIN, "models"], capture_output=True, text=True, timeout=10,
                             env=dict(os.environ, AGY_INTERNAL_INVOCATION="1", AGY_SAGE_DISABLED="1"))
        lines = res.stdout.splitlines()
        for line in lines:
            parts = line.strip().split()
            if parts and "flash" in parts[0].lower() and "low" in parts[0].lower():
                model_id = parts[0]
                try:
                    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
                    with open(CACHE_PATH, "w", encoding="utf-8") as f:
                        f.write(model_id)
                except Exception:
                    pass
                return model_id
    except Exception:
        pass
    return DEFAULT_MODEL

def is_trivial_prompt(text: str) -> bool:
    """Check if a prompt is too trivial to warrant memory processing."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    if stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped))

def get_all_vocabulary(cursor) -> set:
    """Retrieve indexed vocabulary tokens from all tables for fast fuzzy/typo correction and compound validation."""
    path = cursor.execute("PRAGMA database_list").fetchone()[2]
    # File identity prevents cache reuse after a DB is replaced at the same path.
    identity = (path, os.stat(path).st_ino) if path else None
    cached = _VOCABULARY_CACHE.get(identity) if identity else None
    if cached and time.monotonic() - cached[0] < 60:
        return set(cached[1])
    vocab = set()
    cursor.execute("SELECT id, category, fact, keywords FROM memories")
    for fid, cat, fact, kws in cursor.fetchall():
        tokens = re.findall(r'[\w]{2,}', f"{fid} {cat or ''} {fact} {kws or ''}".lower(), re.UNICODE)
        vocab.update(tokens)
    
    cursor.execute("SELECT id, topic, title, narrative, entities, stance, keywords FROM episodes")
    for row in cursor.fetchall():
        text = " ".join([str(x) for x in row if x])
        tokens = re.findall(r'[\w]{2,}', text.lower(), re.UNICODE)
        vocab.update(tokens)

    cursor.execute("SELECT id, category, insight, context, keywords FROM learnings")
    for row in cursor.fetchall():
        text = " ".join([str(x) for x in row if x])
        tokens = re.findall(r'[\w]{2,}', text.lower(), re.UNICODE)
        vocab.update(tokens)

    cursor.execute("SELECT source_id, target_id, relation FROM entity_links")
    for row in cursor.fetchall():
        tokens = re.findall(r'[\w]{2,}', f"{row[0]} {row[1]} {row[2]}".lower(), re.UNICODE)
        vocab.update(tokens)

    if identity:
        _VOCABULARY_CACHE[identity] = (time.monotonic(), frozenset(vocab))
    return vocab

def link_entities(source_id: str, target_id: str, relation: str, connection=None):
    """Create or update a directional entity link / relationship."""
    with _writer(connection) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO entity_links (source_id, target_id, relation)
            VALUES (?, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO NOTHING;
        """, (source_id.strip(), target_id.strip(), relation.strip()))

def unlink_entities(source_id: str, target_id: str, relation: str = None):
    """Remove entity link(s) between two IDs."""
    with db_session() as conn:
        cursor = conn.cursor()
        if relation:
            cursor.execute("DELETE FROM entity_links WHERE source_id = ? AND target_id = ? AND relation = ?", (source_id, target_id, relation))
        else:
            cursor.execute("DELETE FROM entity_links WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)", (source_id, target_id, target_id, source_id))
        conn.commit()

def list_entity_links(entity_id: str = None) -> list:
    """Retrieve entity links, optionally filtered by entity."""
    with db_session() as conn:
        cursor = conn.cursor()
        if entity_id:
            cursor.execute("""
                SELECT source_id, target_id, relation FROM entity_links
                WHERE source_id = ? OR target_id = ?
                ORDER BY source_id, target_id
            """, (entity_id, entity_id))
        else:
            cursor.execute("SELECT source_id, target_id, relation FROM entity_links ORDER BY source_id, target_id")
        return cursor.fetchall()

def age_episodes(days_to_cooling: int = 30, days_to_historic: int = 90) -> dict:
    """Evaluate and transition episode lifecycles based on updated_at age:
    - 'active' -> 'cooling' if not updated for > days_to_cooling (default: 30 days)
    - 'cooling' -> 'historic' if not updated for > days_to_historic (default: 90 days)
    - 'resolved' remains 'resolved'.
    """
    cooled = []
    historied = []
    with db_session() as conn:
        cursor = conn.cursor()
        
        # 1. active -> cooling
        cursor.execute("""
            SELECT id, title, updated_at FROM episodes
            WHERE status = 'active'
            AND updated_at < datetime('now', '-' || ? || ' days')
        """, (days_to_cooling,))
        to_cool = cursor.fetchall()
        for eid, title, updated in to_cool:
            cursor.execute("UPDATE episodes SET status = 'cooling' WHERE id = ?", (eid,))
            cooled.append((eid, title, updated))

        # 2. cooling -> historic
        cursor.execute("""
            SELECT id, title, updated_at FROM episodes
            WHERE status = 'cooling'
            AND updated_at < datetime('now', '-' || ? || ' days')
        """, (days_to_historic,))
        to_historic = cursor.fetchall()
        for eid, title, updated in to_historic:
            cursor.execute("UPDATE episodes SET status = 'historic' WHERE id = ?", (eid,))
            historied.append((eid, title, updated))

        conn.commit()

    return {"cooled": cooled, "historied": historied}

def prefetch(query: str, limit_facts: int = 3, limit_episodes: int = 2, limit_learnings: int = 2, quiet: bool = False):
    """Multi-layer prefetch with:
    1. Persistent preferences & rules (Layer 1)
    2. FTS5 exact + typo fuzzy search (Facts, Episodes with status weighting, Learnings)
    3. Entity Graph Expansion (1-hop linked facts/episodes/learnings)
    """
    if is_trivial_prompt(query):
        return {} if quiet else None

    with db_session() as conn:
        cursor = conn.cursor()

        # 1. Always load persistent preferences and rules
        cursor.execute("""
            SELECT id, category, fact 
            FROM memories 
            WHERE category IN ('preference', 'rule', 'preferences')
            ORDER BY id ASC
        """)
        pref_rows = cursor.fetchall()
        seen_fact_ids = {r[0] for r in pref_rows}
        seen_episode_ids = set()
        seen_learning_ids = set()

        # 2. Extract search terms & multilingual compound sub-tokens with vocabulary validation
        vocab = get_all_vocabulary(cursor)
        words = extract_multilingual_tokens(query, vocab)

        fact_rows = []
        episode_rows = []
        learning_rows = []

        if words:
            fts_terms = [f'"{w}"*' if len(w) >= 4 else f'"{w}"' for w in words]
            fts_query = " OR ".join(fts_terms)

            # Query Memories FTS via JOIN
            try:
                cursor.execute("""
                    SELECT m.id, m.category, m.fact 
                    FROM memories m
                    JOIN memories_fts f ON m.id = f.id
                    WHERE memories_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?;
                """, (fts_query, limit_facts + len(seen_fact_ids)))
                for r in cursor.fetchall():
                    if r[0] not in seen_fact_ids and r[1] not in ('preference', 'rule', 'preferences'):
                        fact_rows.append(r)
                        seen_fact_ids.add(r[0])
                        if len(fact_rows) >= limit_facts:
                            break
            except sqlite3.OperationalError:
                fact_rows = []

            # Query Episodes FTS via JOIN (with Status Aging Weighting: active > cooling > historic/resolved)
            try:
                cursor.execute("""
                    SELECT e.id, e.topic, e.title, e.period, e.status, e.narrative, e.entities, e.stance 
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
                    LIMIT ?;
                """, (fts_query, limit_episodes))
                for r in cursor.fetchall():
                    episode_rows.append(r)
                    seen_episode_ids.add(r[0])
            except sqlite3.OperationalError:
                episode_rows = []

            # Query Learnings FTS via JOIN
            try:
                cursor.execute("""
                    SELECT l.id, l.category, l.insight, l.context 
                    FROM learnings l
                    JOIN learnings_fts f ON l.id = f.id
                    WHERE learnings_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?;
                """, (fts_query, limit_learnings))
                for r in cursor.fetchall():
                    learning_rows.append(r)
                    seen_learning_ids.add(r[0])
            except sqlite3.OperationalError:
                learning_rows = []

            # Fuzzy / Typo Fallback if no matching results found
            if not fact_rows and not episode_rows and any(len(w) >= 4 for w in words):
                vocab = get_all_vocabulary(cursor)
                corrected_words = []
                for w in words:
                    if len(w) >= 4 and w not in vocab:
                        cutoff = 0.75 if len(w) >= 5 else 0.80
                        closest = difflib.get_close_matches(w, vocab, n=1, cutoff=cutoff)
                        if closest:
                            corrected_words.append(closest[0])
                    elif w in vocab:
                        corrected_words.append(w)

                if corrected_words:
                    fuzzy_fts = " OR ".join([f'"{cw}"*' if len(cw) >= 4 else f'"{cw}"' for cw in corrected_words])
                    try:
                        cursor.execute("""
                            SELECT m.id, m.category, m.fact 
                            FROM memories m
                            JOIN memories_fts f ON m.id = f.id
                            WHERE memories_fts MATCH ?
                            ORDER BY f.rank
                            LIMIT ?;
                        """, (fuzzy_fts, limit_facts + len(seen_fact_ids)))
                        for r in cursor.fetchall():
                            if r[0] not in seen_fact_ids and r[1] not in ('preference', 'rule', 'preferences'):
                                fact_rows.append(r)
                                seen_fact_ids.add(r[0])
                                if len(fact_rows) >= limit_facts:
                                    break
                    except sqlite3.OperationalError:
                        pass

                    try:
                        cursor.execute("""
                            SELECT e.id, e.topic, e.title, e.period, e.status, e.narrative, e.entities, e.stance 
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
                            LIMIT ?;
                        """, (fuzzy_fts, limit_episodes))
                        for r in cursor.fetchall():
                            if r[0] not in seen_episode_ids:
                                episode_rows.append(r)
                                seen_episode_ids.add(r[0])
                    except sqlite3.OperationalError:
                        pass

            # 3. Entity Graph Expansion (1-hop Linked Entities)
            matched_ids = list(seen_fact_ids | seen_episode_ids | seen_learning_ids)
            linked_context = []
            if matched_ids:
                placeholders = ",".join("?" * len(matched_ids))
                cursor.execute(f"""
                    SELECT source_id, target_id, relation FROM entity_links
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                    LIMIT 5
                """, matched_ids + matched_ids)
                links = cursor.fetchall()
                for src, tgt, rel in links:
                    linked_target = tgt if src in matched_ids else src
                    if linked_target not in seen_fact_ids | seen_episode_ids | seen_learning_ids:
                        cursor.execute("SELECT id, category, fact FROM memories WHERE id = ?", (linked_target,))
                        mf = cursor.fetchone()
                        if mf:
                            linked_context.append(f"Linked Fact via '{rel}': ({mf[1]}) {mf[2]}")
                            seen_fact_ids.add(mf[0])
                        else:
                            cursor.execute("SELECT id, title, status, narrative FROM episodes WHERE id = ?", (linked_target,))
                            me = cursor.fetchone()
                            if me:
                                linked_context.append(f"Linked Episode via '{rel}': [{me[1]} | {me[2]}] {me[3]}")
                                seen_episode_ids.add(me[0])
                            else:
                                ml = cursor.execute("SELECT id, category, insight, context FROM learnings WHERE id = ?", (linked_target,)).fetchone()
                                if ml:
                                    linked_context.append(f"Linked Learning via '{rel}': ({ml[1]}) {ml[2]} [Context: {ml[3] or ''}]")
                                    seen_learning_ids.add(ml[0])

        # Output Generation
        total_facts = pref_rows + fact_rows
        if quiet:
            return {
                "facts": total_facts,
                "episodes": episode_rows,
                "learnings": learning_rows,
                "linked_context": linked_context if 'linked_context' in locals() else []
            }

        if total_facts or episode_rows or learning_rows or (words and 'linked_context' in locals() and linked_context):
            if total_facts:
                logger.info("[🧠 Memory Context - Facts]")
                for _, cat, fact in total_facts:
                    prefix = f"({cat}) " if cat else ""
                    logger.info(f"• {prefix}{fact}")

            if episode_rows:
                logger.info("\n[📖 Narrative Context - Episodic Memory]")
                for eid, topic, title, period, status, narrative, entities, stance in episode_rows:
                    period_str = f" | {period}" if period else ""
                    status_str = f" | {status}" if status else ""
                    logger.info(f"• [{title}{period_str}{status_str}]")
                    logger.info(f"  Kontext: {narrative}")
                    if stance:
                        logger.info(f"  Haltung/Stance: {stance}")
                    if entities:
                        logger.info(f"  Beteiligte/Entitäten: {entities}")

            if learning_rows:
                logger.info("\n[💡 Learnings & Heuristics]")
                for lid, cat, insight, context in learning_rows:
                    prefix = f"({cat}) " if cat else ""
                    ctx_str = f" [Kontext: {context}]" if context else ""
                    logger.info(f"• {prefix}{insight}{ctx_str}")

            if 'linked_context' in locals() and linked_context:
                logger.info("\n[🔗 Linked Entity Relations]")
                for lc in linked_context:
                    logger.info(f"• {lc}")

# Common LLM-generated category variants → canonical mapping
_CATEGORY_ALIASES = {
    "contact": "contacts", "kontakte": "contacts",
    "infrastructure": "infra", "system_architecture": "infra", "system_config": "infra",
    "admin": "infra", "config": "infra",
    "pref": "preferences", "preference": "preferences", "rule": "preferences", "user": "preferences",
    "pension": "finance", "trading": "finance", "stweg": "home",
    "gear": "hardware", "tesla": "hardware",
    "devsecops": "dev", "dev.cron": "automation",
    "heuristics": "general", "ai_tools": "software", "ai": "dev",
    "ui_ux": "architecture", "network": "infra",
    "realestate": "home", "calendar": "general",
}


def _normalize_category(category: str, allowed: frozenset) -> str:
    """Normalize a category string to the closest canonical category."""
    if not category:
        return "general"
    cat = category.strip().lower()
    if cat in allowed:
        return cat
    if cat in _CATEGORY_ALIASES:
        alias = _CATEGORY_ALIASES[cat]
        if alias in allowed:
            return alias
    return "general"


def upsert_fact(fact_id: str, category: str, fact: str, keywords: str = "", connection=None):
    """Insert or update an atomic fact in the memories table."""
    category = _normalize_category(category, CANONICAL_FACT_CATEGORIES)
    with _writer(connection) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, category, fact, keywords, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                category = excluded.category,
                fact = excluded.fact,
                keywords = excluded.keywords,
                updated_at = CURRENT_TIMESTAMP;
        """, (fact_id, category, fact, keywords))

def upsert_episode(episode_id: str, topic: str, title: str, narrative: str, period: str = "", status: str = "active", entities: str = "", stance: str = "", keywords: str = "", connection=None):
    """Insert or update a narrative chronicle/episode."""
    with _writer(connection) as conn:
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
        """, (episode_id, topic, title, period, status, narrative, entities, stance, keywords))

def upsert_learning(learning_id: str, category: str, insight: str, context: str = "", keywords: str = "", connection=None):
    """Insert or update an experiential learning/heuristic."""
    category = _normalize_category(category, CANONICAL_LEARNING_CATEGORIES)
    with _writer(connection) as conn:
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
        """, (learning_id, category, insight, context, keywords))

def list_all():
    """Print a formatted overview of all stored facts, episodes, learnings, and entity relations."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, category, fact FROM memories ORDER BY category, id")
        facts = cursor.fetchall()
        cursor.execute("SELECT id, topic, title, period, status, narrative, stance FROM episodes ORDER BY topic, id")
        episodes = cursor.fetchall()
        cursor.execute("SELECT id, category, insight, context FROM learnings ORDER BY category, id")
        learnings = cursor.fetchall()
        cursor.execute("SELECT source_id, target_id, relation FROM entity_links ORDER BY source_id, target_id")
        links = cursor.fetchall()

        logger.info("=" * 80)
        logger.info(f"SEMANTIC FACTS ({len(facts)})")
        logger.info("=" * 80)
        for fid, cat, fact in facts:
            logger.info(f"{fid:<25} | {(cat or ''):<12} | {fact}")

        logger.info("\n" + "=" * 80)
        logger.info(f"NARRATIVE CHRONICLES & EPISODES ({len(episodes)})")
        logger.info("=" * 80)
        for eid, topic, title, period, status, narrative, stance in episodes:
            p_str = f" ({period})" if period else ""
            s_str = f" [{status}]" if status else ""
            logger.info(f"\n▶ [{eid}] {title}{p_str}{s_str} (Topic: {topic})")
            logger.info(f"  Narrative: {narrative}")
            if stance:
                logger.info(f"  Stance:    {stance}")

        logger.info("\n" + "=" * 80)
        logger.info(f"EXPERIENTIAL LEARNINGS & HEURISTICS ({len(learnings)})")
        logger.info("=" * 80)
        for lid, cat, insight, context in learnings:
            ctx = f" (Context: {context})" if context else ""
            logger.info(f"{lid:<25} | {(cat or ''):<12} | {insight}{ctx}")

        logger.info("\n" + "=" * 80)
        logger.info(f"ENTITY GRAPH LINKS & RELATIONS ({len(links)})")
        logger.info("=" * 80)
        for src, tgt, rel in links:
            logger.info(f"{src:<30} --[{rel}]--> {tgt}")

def _is_protected_key(key_id: str, existing: dict = None) -> bool:
    """Protect persisted categories as well as category tokens in an ID."""
    if any(part in PROTECTED_CATEGORIES for part in re.split(r"[._-]", key_id.lower())):
        return True
    if existing is not None:
        return (existing.get("category") or existing.get("topic") or "").lower() in PROTECTED_CATEGORIES
    with db_session() as conn:
        for table, column in (("memories", "category"), ("learnings", "category"), ("episodes", "topic")):
            row = conn.execute(f"SELECT {column} FROM {table} WHERE id = ?", (key_id,)).fetchone()
            if row and (row[0] or "").lower() in PROTECTED_CATEGORIES:
                return True
    return False

def _get_existing_value(key_id: str, table: str, connection=None) -> dict | None:
    """Retrieve existing entry from a table by ID for diff comparison."""
    with nullcontext(connection) if connection is not None else db_session() as conn:
        cursor = conn.cursor()
        if table == "memories":
            cursor.execute("SELECT id, category, fact, keywords FROM memories WHERE id = ?", (key_id,))
            row = cursor.fetchone()
            return {"id": row[0], "category": row[1], "fact": row[2], "keywords": row[3]} if row else None
        elif table == "episodes":
            cursor.execute("SELECT id, topic, title, period, status, narrative, entities, stance, keywords FROM episodes WHERE id = ?", (key_id,))
            row = cursor.fetchone()
            return {"id": row[0], "topic": row[1], "title": row[2], "period": row[3], "status": row[4], "narrative": row[5], "entities": row[6], "stance": row[7], "keywords": row[8]} if row else None
        elif table == "learnings":
            cursor.execute("SELECT id, category, insight, context, keywords FROM learnings WHERE id = ?", (key_id,))
            row = cursor.fetchone()
            return {"id": row[0], "category": row[1], "insight": row[2], "context": row[3], "keywords": row[4]} if row else None
    return None

def _format_diff(old: dict | None, new: dict, label: str) -> str:
    """Format a human-readable diff between old and new values."""
    if old is None:
        return f"  [NEW] {label}: {new.get('id', '?')}"
    
    changes = []
    for key in new:
        old_val = old.get(key, "")
        new_val = new.get(key, "")
        if str(old_val) != str(new_val):
            changes.append(f"    {key}: \"{old_val}\" → \"{new_val}\"")
    
    if changes:
        protected_marker = " 🔒 PROTECTED" if _is_protected_key(new.get("id", ""), old) else ""
        return f"  [UPDATE{protected_marker}] {label}: {new.get('id', '?')}\n" + "\n".join(changes)
    return ""

def sync_turn(user_prompt: str, assistant_response: str, dry_run: bool = False) -> dict:
    """Extract persistent information from a conversation turn and sync to memory.
    
    Args:
        user_prompt: The user's message text.
        assistant_response: The assistant's response text.
        dry_run: If True, only show what would change without writing to DB.

    Returns:
        dict: Summary of applied changes with keys 'facts', 'episodes', 'learnings', 'entity_links'.
    """
    import fcntl
    import tempfile

    lock_path = os.path.join(tempfile.gettempdir(), "agy_memory_sync_turn.lock")
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Preserve the turn for retry rather than reporting an empty extraction.
        sys.stderr.write("agy_memory: sync-turn already running; retry required.\n")
        lock_fd.close()
        raise SyncBusyError("sync-turn already running")

    try:
        return _sync_turn_inner(user_prompt, assistant_response, dry_run)
    except (SyncBusyError, SyncExtractionError):
        raise
    except sqlite3.OperationalError as error:
        if getattr(error, "sqlite_errorcode", 0) & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise SyncBusyError(str(error)) from error
        raise SyncExtractionError(str(error)) from error
    except Exception as error:
        raise SyncExtractionError(str(error)) from error
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _validate_extraction(data):
    """Reject malformed items before any database write or queue acknowledgement."""
    required = {"facts": ("id", "fact"), "episodes": ("id", "narrative"),
                "learnings": ("id", "insight"), "entity_links": ("source", "target", "relation")}
    if not isinstance(data, dict) or not any(key in data for key in required):
        raise SyncExtractionError("Extraction must contain recognized entity lists")
    for key, fields in required.items():
        items = data.get(key, [])
        if not isinstance(items, list):
            raise SyncExtractionError(f"{key} must be a list")
        for item in items:
            if not isinstance(item, dict) or any(not isinstance(item.get(f), str) or not item[f].strip() for f in fields):
                raise SyncExtractionError(f"Invalid {key} item")
            if any(not isinstance(value, str) for value in item.values()):
                raise SyncExtractionError(f"Non-text field in {key}")
            for field in ("id", "source", "target"):
                if field in item:
                    item[field] = item[field].strip()


def _sync_turn_inner(user_prompt: str, assistant_response: str, dry_run: bool = False) -> dict:
    """Inner implementation of sync_turn, called only when lock is held."""
    empty_res = {"facts": [], "episodes": [], "learnings": [], "entity_links": []}
    if is_trivial_prompt(user_prompt):
        return empty_res

    inv = get_existing_database_inventory()
    inv_context = json.dumps(inv, ensure_ascii=False, indent=2)

    prompt = f"""You are the Multi-Layer Cognitive Memory Engine for Stephan Bolten.
Analyze the conversation turn below and extract ONLY genuinely persistent, reusable information.

## Layer Definitions

1. ATOMIC FACTS ("facts"): Hard facts, IPs, specs, master data, device IDs, account names, medications, config parameters, definite dates/appointments, contact details.
   ALLOWED CATEGORIES: infra, hardware, software, contacts, family, health, fitness, finance, insurance, travel, home, media, music, work, dev, preferences, communication, cloud, security, general

2. NARRATIVE CHRONICLES & EPISODES ("episodes"): Background histories, disputes, social/relationship dynamics, sentiment/stances, multi-event story arcs.
   - Status: "active" (ongoing), "cooling" (cooling down), "historic" (concluded past), "resolved" (fixed/completed).
   ALLOWED TOPICS: family, health, travel, finance, home, dev, infra, insurance, music, work, realestate, trading, general

3. EXPERIENTIAL LEARNINGS ("learnings"): ONLY personal heuristics, behavioral insights, and reusable rules of thumb that will help in FUTURE similar situations.
   ALLOWED CATEGORIES: workflow, communication, finance, health, shopping, travel, hardware, safety, architecture, security, automation, general

   ✅ GOOD learnings (store these):
   - "Karin prefers bullet-point summaries for financial topics" (personal communication insight)
   - "E-Mails an Dr. Bär müssen extrem kurz sein" (reusable behavioral rule)
   - "PK-Einkäufe nicht ins Depot, sondern liquide halten" (financial heuristic)
   - "Garmin shows 'Detraining' even with daily exercise if no GPS activity is recorded" (reusable device knowledge)
   - "When self-restarting a systemd service from within, always decouple via sleep+nohup" (reusable ops pattern)

   🚫 DO NOT store as learnings:
   - One-time bug fixes or debugging sessions ("Fixed SQLite UTC conversion in dashboard.py")
   - Implementation details of a specific codebase ("Baileys emits protocolMessages during sync")
   - Version-specific migration notes ("mcp 2.x breaks FastMCP imports")
   - Configuration changes made once ("Set model to gemini-3.8-flash-low")
   - API quirks of a specific service ("Spotify 403 on playlist endpoint")
   These belong in code comments, commit messages, or facts — NOT learnings.

4. ENTITY LINKS ("entity_links"): Structural relationships between existing entities.
   You MUST use ONLY these canonical relation types:
   hosted_on, runs_on, depends_on, part_of, member_of, owned_by, managed_by, monitors,
   treats, prescribed_for, insured_by, finances, communicates_via, located_at, uses,
   stores, connects_to, related_to, maintains, created_by, delivers_to, advises,
   works_at, lives_at, travels_to, subscribed_to
   DO NOT invent new relation types. If none of the above fits, use "related_to".

## Existing Database Keys & Topics:
{inv_context}

## Rules:
- If updating an existing item, reuse its EXACT existing ID from the inventory.
- Keep facts atomic and dense.
- Keep episodes rich with context and stance (3-5 sentences).
- Include multilingual keywords (German/English synonyms, related query terms).
- CRITICAL: Prefer EMPTY arrays over low-value entries. When in doubt, do NOT store it.
- A learning must pass this test: "Would this insight help me handle a SIMILAR situation in the future?" If no, don't store it.
- Entity links must use ONLY the canonical relation types listed above.

User: {user_prompt}
Assistant: {assistant_response}

Output ONLY a single valid JSON object (or {{"facts":[], "episodes":[], "learnings":[], "entity_links":[]}} if nothing worth persisting):
{{
  "facts": [
    {{
      "id": "...",
      "category": "...",
      "fact": "...",
      "keywords": "..."
    }}
  ],
  "episodes": [
    {{
      "id": "...",
      "topic": "...",
      "title": "...",
      "period": "...",
      "status": "active|cooling|historic|resolved",
      "narrative": "...",
      "entities": "...",
      "stance": "...",
      "keywords": "..."
    }}
  ],
  "learnings": [
    {{
      "id": "...",
      "category": "...",
      "insight": "...",
      "context": "...",
      "keywords": "..."
    }}
  ],
  "entity_links": [
    {{
      "source": "...",
      "target": "...",
      "relation": "..."
    }}
  ]
}}
"""

    model_name = get_cached_model()
    out = ""
    run_env = dict(os.environ)
    run_env["AGY_INTERNAL_INVOCATION"] = "1"
    run_env["AGY_SAGE_DISABLED"] = "1"

    try:
        res = subprocess.run(
            [AGY_BIN, "--print", prompt, "--model", model_name, "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=90, env=run_env
        )
        out = res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        out = ""

    if not out:
        model_name = discover_and_cache_latest_flash_low_model()
        try:
            res = subprocess.run(
                [AGY_BIN, "--print", prompt, "--model", model_name, "--dangerously-skip-permissions"],
                capture_output=True, text=True, timeout=90, env=run_env
            )
            out = res.stdout.strip() if res.returncode == 0 else ""
        except Exception as e:
            sys.stderr.write(f"Sync failed: {e}\n")
            raise SyncExtractionError("Model invocation failed") from e

    if not out:
        raise SyncExtractionError("Model returned no successful output")

    applied_changes = {
        "facts": [],
        "episodes": [],
        "learnings": [],
        "entity_links": []
    }

    json_match = re.search(r'\{.*\}', out, re.DOTALL)
    if not json_match:
        raise SyncExtractionError("Model output contains no JSON object")
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            _validate_extraction(data)
            with db_session() as transaction, transaction:
                if not dry_run:
                    transaction.execute("BEGIN IMMEDIATE")
                diff_lines = []
                skipped_protected = []

                # --- Facts ---
                for f in data.get("facts", []):
                    if isinstance(f, dict) and "id" in f and "fact" in f:
                        existing = _get_existing_value(f["id"], "memories", connection=transaction)
                        diff = _format_diff(existing, f, "Fact")
                        if diff:
                            diff_lines.append(diff)
                        
                        if dry_run:
                            continue
                        
                        # Protected category guard: skip updates to existing protected entries
                        if existing and _is_protected_key(f["id"], existing):
                            skipped_protected.append(f["id"])
                            sys.stderr.write(f"[PROTECTED] Skipping update to protected fact '{f['id']}' — use manual 'add' to update.\n")
                            continue
                        
                        norm_cat = _normalize_category(f.get("category", "general"), CANONICAL_FACT_CATEGORIES)
                        upsert_fact(f["id"], norm_cat, f["fact"], f.get("keywords", ""), connection=transaction)
                        applied_changes["facts"].append({
                            "id": f["id"],
                            "category": norm_cat,
                            "fact": f["fact"],
                            "is_update": existing is not None
                        })

                # --- Episodes ---
                for ep in data.get("episodes", []):
                    if isinstance(ep, dict) and "id" in ep and "narrative" in ep:
                        existing = _get_existing_value(ep["id"], "episodes", connection=transaction)
                        diff = _format_diff(existing, ep, "Episode")
                        if diff:
                            diff_lines.append(diff)
                        
                        if dry_run:
                            continue
                        
                        if existing and _is_protected_key(ep["id"], existing):
                            skipped_protected.append(ep["id"])
                            sys.stderr.write(f"[PROTECTED] Skipping update to protected episode '{ep['id']}' — use manual 'add-episode' to update.\n")
                            continue
                        
                        norm_topic = _normalize_category(ep.get("topic", "general"), CANONICAL_EPISODE_TOPICS)
                        upsert_episode(
                            ep["id"],
                            norm_topic,
                            ep.get("title", ep["id"]),
                            ep["narrative"],
                            period=ep.get("period", ""),
                            status=ep.get("status", "active"),
                            entities=ep.get("entities", ""),
                            stance=ep.get("stance", ""),
                            keywords=ep.get("keywords", ""), connection=transaction
                        )
                        applied_changes["episodes"].append({
                            "id": ep["id"],
                            "topic": norm_topic,
                            "title": ep.get("title", ep["id"]),
                            "status": ep.get("status", "active"),
                            "is_update": existing is not None
                        })

                # --- Learnings ---
                for lr in data.get("learnings", []):
                    if isinstance(lr, dict) and "id" in lr and "insight" in lr:
                        existing = _get_existing_value(lr["id"], "learnings", connection=transaction)
                        diff = _format_diff(existing, lr, "Learning")
                        if diff:
                            diff_lines.append(diff)
                        
                        if dry_run:
                            continue
                        
                        if existing and _is_protected_key(lr["id"], existing):
                            skipped_protected.append(lr["id"])
                            continue
                        norm_lcat = _normalize_category(lr.get("category", "general"), CANONICAL_LEARNING_CATEGORIES)
                        upsert_learning(
                            lr["id"],
                            norm_lcat,
                            lr["insight"],
                            context=lr.get("context", ""),
                            keywords=lr.get("keywords", ""), connection=transaction
                        )
                        applied_changes["learnings"].append({
                            "id": lr["id"],
                            "category": norm_lcat,
                            "insight": lr["insight"],
                            "is_update": existing is not None
                        })

                # --- Entity Links ---
                for el in data.get("entity_links", []):
                    if isinstance(el, dict) and "source" in el and "target" in el and "relation" in el:
                        relation = el["relation"].strip().lower()
                        # Normalize non-canonical relations to 'related_to'
                        if relation not in CANONICAL_RELATIONS:
                            sys.stderr.write(f"[NORM] Non-canonical relation '{el['relation']}' normalized to 'related_to' for {el['source']} -> {el['target']}\n")
                            relation = "related_to"
                        if dry_run:
                            diff_lines.append(f"  [NEW] Entity Link: {el['source']} --[{relation}]--> {el['target']}")
                            continue
                        for endpoint in (el["source"], el["target"]):
                            if not any(transaction.execute(f"SELECT 1 FROM {table} WHERE id = ?", (endpoint,)).fetchone()
                                       for table in ("memories", "episodes", "learnings")):
                                raise SyncExtractionError(f"Unknown graph endpoint: {endpoint}")
                        link_entities(el["source"], el["target"], relation, connection=transaction)
                        applied_changes["entity_links"].append({
                            "source": el["source"],
                            "target": el["target"],
                            "relation": relation
                        })

                # Print diff summary
                if dry_run and diff_lines:
                    logger.info("\n[DRY-RUN] Proposed changes:")
                    for dl in diff_lines:
                        logger.info(dl)
                    if not diff_lines:
                        logger.info("  (no changes detected)")
                elif dry_run:
                    logger.info("[DRY-RUN] No changes detected.")

                if skipped_protected:
                    logger.info(f"\n[INFO] {len(skipped_protected)} protected entry/entries skipped: {', '.join(skipped_protected)}")

        except json.JSONDecodeError as e:
            raise SyncExtractionError("Invalid extraction JSON") from e
        except sqlite3.OperationalError as e:
            if getattr(e, "sqlite_errorcode", 0) & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise SyncBusyError(f"Database contention: {e}") from e
            raise SyncExtractionError(f"Extraction transaction failed: {e}") from e
        except Exception as e:
            raise SyncExtractionError(f"Extraction transaction failed: {e}") from e

    _VOCABULARY_CACHE.clear()
    if not dry_run:
        logger.info("Memory sync committed: %s", {key: len(items) for key, items in applied_changes.items()})
    return applied_changes

def consolidate_memories(dry_run: bool = False) -> list:
    """Analyze all stored atomic facts per category with Gemini LLM,
    detect duplicate/overlapping facts, merge them cleanly into a single authoritative record,
    delete the redundant entries, and log everything to consolidation_log.
    """
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, category, fact, keywords FROM memories ORDER BY category, id")
        all_facts = cursor.fetchall()

    if not all_facts or len(all_facts) < 2:
        logger.info("[CONSOLIDATE] Less than 2 facts in database. Nothing to consolidate.")
        return []

    # Group facts by category with at least 2 entries
    categories_to_check = {}
    for fid, cat, fact, kws in all_facts:
        cat_name = cat or "general"
        categories_to_check.setdefault(cat_name, []).append({
            "id": fid,
            "category": cat_name,
            "fact": fact,
            "keywords": kws or ""
        })

    # Only include categories that actually have multiple facts
    categories_to_check = {k: v for k, v in categories_to_check.items() if len(v) >= 2}

    if not categories_to_check:
        logger.info("[CONSOLIDATE] No categories with 2+ facts to consolidate.")
        return []

    canonical_cats = ", ".join(sorted(CANONICAL_FACT_CATEGORIES))
    facts_json = json.dumps(categories_to_check, ensure_ascii=False, indent=2)
    prompt = f"""You are the Memory Consolidation Engine for Stephan Bolten.
Review the following atomic facts grouped by category.
Identify any facts within each category that are duplicates, heavily overlapping, redundant, or represent the same information across different keys.

Categories and Facts:
{facts_json}

INSTRUCTIONS:
1. If two or more facts describe the exact same topic/entity/preference/routine, combine them into ONE authoritative, complete, concise fact.
2. Choose the best, most structured primary ID (target_id) from the existing IDs, or suggest a clean canonical ID.
3. List the redundant IDs that must be DELETED (merged_ids).
4. Combine the keywords/tags cleanly without duplicates.
5. Provide a short rationale explaining why these were merged.
6. If no facts need to be merged in a category, do not create a merge entry for it.
7. The category MUST be one of: {canonical_cats}
   If the existing category is not in this list, map it to the closest canonical one.

Respond ONLY with valid JSON in this exact structure:
{{
  "merges": [
    {{
      "target_id": "authoritative.key.name",
      "category": "health",
      "fact": "Authoritative consolidated fact text...",
      "keywords": "combined keyword tags",
      "merged_ids": ["redundant.key.1", "redundant.key.2"],
      "rationale": "Merged daily intake and brand information into single routine entry."
    }}
  ]
}}
"""
    model_name = get_cached_model()
    out = ""
    run_env = dict(os.environ, AGY_INTERNAL_INVOCATION="1", AGY_SAGE_DISABLED="1")
    try:
        res = subprocess.run(
            [AGY_BIN, "--print", prompt, "--model", model_name, "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=120, env=run_env
        )
        out = res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        out = ""

    if not out:
        model_name = discover_and_cache_latest_flash_low_model()
        try:
            res = subprocess.run(
                [AGY_BIN, "--print", prompt, "--model", model_name, "--dangerously-skip-permissions"],
                capture_output=True, text=True, timeout=120, env=run_env
            )
            out = res.stdout.strip() if res.returncode == 0 else ""
        except Exception as e:
            sys.stderr.write(f"Consolidation LLM call failed: {e}\n")
            return []

    if not out:
        return []

    consolidations = []
    json_match = re.search(r'\{.*\}', out, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            for merge in data.get("merges", []):
                target_id = merge.get("target_id")
                raw_cat = merge.get("category", "general")
                cat_name = _normalize_category(raw_cat, CANONICAL_FACT_CATEGORIES)
                merged_ids = [m for m in merge.get("merged_ids", []) if m != target_id]
                fact_text = merge.get("fact")
                kws = merge.get("keywords", "")
                rationale = merge.get("rationale", "")

                if not target_id or not merged_ids or not fact_text:
                    continue

                category_facts = categories_to_check.get(raw_cat, categories_to_check.get(cat_name, []))
                existing_merged = [m for m in merged_ids if any(f["id"] == m for f in category_facts)]
                if not existing_merged:
                    continue

                diff_summary = f"Merged [{', '.join(existing_merged)}] into [{target_id}]"

                with db_session() as conn, conn:
                    if not dry_run:
                        conn.execute("BEGIN IMMEDIATE")
                    target = conn.execute("SELECT id, category, fact, keywords FROM memories WHERE id = ?", (target_id,)).fetchone()
                    allowed_ids = {f["id"] for f in category_facts}
                    if target and (target_id not in allowed_ids or _is_protected_key(target_id, {"category": target[1]})):
                        continue
                    # A new fact ID must not collide with another entity layer.
                    if not target and any(conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (target_id,)).fetchone() for table in ("episodes", "learnings")):
                        continue
                    sources = []
                    for mid in existing_merged:
                        row = conn.execute("SELECT id, category, fact, keywords FROM memories WHERE id = ?", (mid,)).fetchone()
                        if not row or row[1] != raw_cat or _is_protected_key(mid, {"category": row[1]}):
                            break
                        sources.append(row)
                    if len(sources) != len(existing_merged) or (target and target[1] != raw_cat):
                        continue
                    # Reject stale model proposals if a fact changed during inference.
                    proposed = {f["id"]: f for f in category_facts}
                    if any(row[2] != proposed[row[0]]["fact"] or (row[3] or "") != proposed[row[0]]["keywords"] for row in sources + ([target] if target else [])):
                        continue
                    placeholders = ",".join("?" for _ in existing_merged)
                    links = conn.execute(f"SELECT source_id, target_id, relation FROM entity_links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", existing_merged * 2).fetchall()
                    if not dry_run:
                        upsert_fact(target_id, cat_name, fact_text, kws, connection=conn)
                        for src, tgt, relation in links:
                            conn.execute("DELETE FROM entity_links WHERE source_id=? AND target_id=? AND relation=?", (src, tgt, relation))
                            new_src = target_id if src in existing_merged else src
                            new_tgt = target_id if tgt in existing_merged else tgt
                            if new_src != new_tgt:
                                conn.execute("INSERT OR IGNORE INTO entity_links VALUES (?, ?, ?)",
                                             (new_src, new_tgt, relation))
                        for mid in existing_merged:
                            conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
                        preimage = json.dumps({"summary": diff_summary, "facts": sources, "target": target, "entity_links": links}, ensure_ascii=False)
                        conn.execute("""INSERT INTO consolidation_log
                            (action, category, target_id, merged_ids, diff_summary, rationale)
                            VALUES ('merge', ?, ?, ?, ?, ?)""",
                            (cat_name, target_id, json.dumps(existing_merged), preimage, rationale))

                consolidations.append({
                    "category": cat_name,
                    "target_id": target_id,
                    "merged_ids": existing_merged,
                    "fact": fact_text,
                    "rationale": rationale,
                    "diff_summary": diff_summary
                })
                logger.info(f"[CONSOLIDATE] {diff_summary} (Rationale: {rationale})")
        except Exception as e:
            sys.stderr.write(f"Error parsing consolidation JSON: {e}\n")

    return consolidations


def prune_orphan_links(dry_run: bool = False) -> int:
    """Remove entity links where source or target IDs don't exist in any main table."""
    with db_session() as conn:
        cursor = conn.cursor()

        # Collect all known entity IDs across all tables
        known_ids = set()
        for row in cursor.execute("SELECT id FROM memories"):
            known_ids.add(row[0])
        for row in cursor.execute("SELECT id FROM episodes"):
            known_ids.add(row[0])
        for row in cursor.execute("SELECT id FROM learnings"):
            known_ids.add(row[0])

        # Find orphan links
        cursor.execute("SELECT source_id, target_id, relation FROM entity_links")
        all_links = cursor.fetchall()
        orphans = []
        for src, tgt, rel in all_links:
            if src not in known_ids or tgt not in known_ids:
                orphans.append((src, tgt, rel))

        if orphans and not dry_run:
            for src, tgt, rel in orphans:
                cursor.execute(
                    "DELETE FROM entity_links WHERE source_id = ? AND target_id = ? AND relation = ?",
                    (src, tgt, rel)
                )
            conn.commit()

        return len(orphans)


def normalize_existing_categories() -> dict:
    """Batch-normalize all categories/topics to canonical taxonomies. Returns counts."""
    counts = {"facts": 0, "learnings": 0, "episodes": 0, "links": 0}
    with db_session() as conn:
        cursor = conn.cursor()

        # Facts
        for fid, cat in cursor.execute("SELECT id, category FROM memories").fetchall():
            norm = _normalize_category(cat, CANONICAL_FACT_CATEGORIES)
            if norm != cat:
                cursor.execute("UPDATE memories SET category = ? WHERE id = ?", (norm, fid))
                counts["facts"] += 1

        # Learnings
        for lid, cat in cursor.execute("SELECT id, category FROM learnings").fetchall():
            norm = _normalize_category(cat, CANONICAL_LEARNING_CATEGORIES)
            if norm != cat:
                cursor.execute("UPDATE learnings SET category = ? WHERE id = ?", (norm, lid))
                counts["learnings"] += 1

        # Episodes
        for eid, topic in cursor.execute("SELECT id, topic FROM episodes").fetchall():
            norm = _normalize_category(topic, CANONICAL_EPISODE_TOPICS)
            if norm != topic:
                cursor.execute("UPDATE episodes SET topic = ? WHERE id = ?", (norm, eid))
                counts["episodes"] += 1

        # Entity link relations
        for src, tgt, rel in cursor.execute("SELECT source_id, target_id, relation FROM entity_links").fetchall():
            rel_lower = rel.strip().lower()
            if rel_lower not in CANONICAL_RELATIONS:
                cursor.execute(
                    "DELETE FROM entity_links WHERE source_id = ? AND target_id = ? AND relation = ?",
                    (src, tgt, rel)
                )
                cursor.execute(
                    "INSERT OR IGNORE INTO entity_links (source_id, target_id, relation) VALUES (?, ?, ?)",
                    (src, tgt, "related_to")
                )
                counts["links"] += 1

        conn.commit()
    return counts


FTS_COLUMNS = {
    "memories": "id, category, fact, keywords",
    "episodes": "id, topic, title, narrative, entities, stance, keywords",
    "learnings": "id, category, insight, context, keywords",
    "entity_links": "source_id, target_id, relation",
}


def rebuild_fts(conn):
    """Resynchronize standalone FTS tables from authoritative base rows."""
    for table, columns in FTS_COLUMNS.items():
        conn.execute(f"DELETE FROM {table}_fts")
        conn.execute(f"INSERT INTO {table}_fts ({columns}) SELECT {columns} FROM {table}")
    _VOCABULARY_CACHE.clear()


@contextmanager
def _readonly_db(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
        yield conn


def optimize_db(apply_changes: bool = False, age_decay: bool = True, consolidate: bool = False):
    """Preview without writes, or back up and apply maintenance explicitly."""
    target_db = schema.DB_PATH
    if not os.path.exists(target_db) and apply_changes:
        with db_session(target_db):
            pass
    if not os.path.exists(target_db):
        stats = {table: 0 for table in FTS_COLUMNS}
        stats["aging"] = {"cooling": 0, "historic": 0}
    else:
        with _readonly_db(target_db) as conn:
            stats = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                     for table in FTS_COLUMNS}
            stats["aging"] = {
                "cooling": conn.execute("SELECT COUNT(*) FROM episodes WHERE status = 'active' AND updated_at < datetime('now', '-30 days')").fetchone()[0] if age_decay else 0,
                "historic": conn.execute("SELECT COUNT(*) FROM episodes WHERE status IN ('active', 'cooling') AND updated_at < datetime('now', '-90 days')").fetchone()[0] if age_decay else 0,
            }
    stats["applied"] = apply_changes
    stats["planned"] = ["normalize", "prune_orphan_links", "prune_queue", "rebuild_fts", "vacuum"]
    if age_decay:
        stats["planned"].append("age_episodes")
    if consolidate:
        stats["planned"].append("consolidate")
    if not apply_changes:
        logger.info("[PREVIEW] Planned optimization: %s", stats)
        return stats

    snapshot = create_snapshot(tag="optimization", db_path=target_db)
    logger.info("[BACKUP] Snapshot created: %s", snapshot["filename"])
    archive = Path(os.path.expanduser("~/.gemini/archive"))
    backups = sorted(archive.glob("memory_db_backup_*_optimization.bak"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[20:]:
        old.unlink()
    if age_decay:
        age_episodes()
    if consolidate:
        stats["merges"] = consolidate_memories()
    stats["normalized"] = normalize_existing_categories()
    stats["orphan_links"] = prune_orphan_links()
    from queue_manager import prune_processed_turns
    prune_processed_turns(days=7)
    with db_session() as conn, conn:
        rebuild_fts(conn)
    with closing(sqlite3.connect(target_db, timeout=5)) as conn:
        conn.execute("VACUUM")
    logger.info("[SUCCESS] Optimized! Stats: %s", stats)
    return stats

# Keep backward compatibility alias
compact_all = optimize_db


def list_snapshots() -> list:
    """List all available snapshots from ~/.gemini/archive with file metadata and record counts."""
    import glob
    archive_dir = os.path.expanduser("~/.gemini/archive")
    if not os.path.exists(archive_dir):
        return []

    files = sorted(
        glob.glob(os.path.join(archive_dir, "memory_db_backup_*.bak")),
        key=os.path.getmtime,
        reverse=True
    )

    snapshots = []
    for f in files:
        fname = os.path.basename(f)
        sz = os.path.getsize(f)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")

        tag = "optimization"
        if "_manual" in fname:
            tag = "manual"
        elif "_pre_restore" in fname:
            tag = "pre-restore"

        stats = {"facts": 0, "episodes": 0, "learnings": 0, "links": 0}
        try:
            conn = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories")
            stats["facts"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM episodes")
            stats["episodes"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM learnings")
            stats["learnings"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM entity_links")
            stats["links"] = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass

        snapshots.append({
            "filename": fname,
            "created_at": mtime,
            "size_kb": f"{sz / 1024:.1f} KB",
            "tag": tag,
            "stats": stats
        })
    return snapshots


def _backup_connections(source, destination, timeout=30):
    """Bound SQLite's internal busy retries, which outlive connect(timeout)."""
    deadline = time.monotonic() + timeout

    def progress(status, remaining, total):
        if time.monotonic() > deadline:
            raise TimeoutError("SQLite online backup timed out")

    source.backup(destination, pages=256, progress=progress, sleep=0.05)


def online_backup(source_path: str, destination_path: str):
    """Capture committed WAL frames, validate, and atomically publish a backup."""
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".memory-backup-", dir=destination.parent)
    os.close(fd)
    try:
        with _readonly_db(source_path) as src, closing(sqlite3.connect(temporary, timeout=5)) as dst:
            _backup_connections(src, dst)
            _verify_snapshot(dst)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_snapshot(conn):
    result = conn.execute("PRAGMA integrity_check").fetchall()
    if result != [("ok",)]:
        raise ValueError(f"Snapshot integrity check failed: {result}")
    for table, columns in FTS_COLUMNS.items():
        conn.execute(f"SELECT {columns} FROM {table} LIMIT 0")


def create_snapshot(tag: str = "manual", db_path: str = None) -> dict:
    """Create a consistent online snapshot, including committed WAL data."""
    target_db = db_path or schema.DB_PATH
    if not re.fullmatch(r"[a-zA-Z0-9_-]*", tag):
        raise ValueError("Invalid snapshot tag")
    archive_dir = os.path.expanduser("~/.gemini/archive")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    suffix = f"_{tag}" if tag else ""
    backup_file = os.path.join(archive_dir, f"memory_db_backup_{ts}{suffix}.bak")
    if not os.path.exists(target_db):
        with db_session(target_db):
            pass
    online_backup(target_db, backup_file)
    return {"status": "ok", "filename": os.path.basename(backup_file),
            "message": f"Snapshot created: {os.path.basename(backup_file)}"}


def restore_snapshot(filename: str, db_path: str = None) -> dict:
    """Restore through SQLite locking, preserving active connection coherence."""
    target_db = db_path or schema.DB_PATH
    archive_dir = Path(os.path.expanduser("~/.gemini/archive"))
    if not re.fullmatch(r"memory_db_backup_[a-zA-Z0-9_.-]+\.bak", filename):
        raise ValueError("Invalid snapshot filename format.")
    source_path = archive_dir / filename
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(f"Snapshot '{filename}' not found in archive directory.")
    safety_backup = None
    with _readonly_db(source_path) as src:
        _verify_snapshot(src)
        if os.path.exists(target_db):
            try:
                safety_backup = create_snapshot("pre_restore", target_db)["filename"]
            except (sqlite3.DatabaseError, ValueError) as error:
                raise RuntimeError("Cannot safely back up the current database. Stop all database clients and perform offline recovery; no live files were replaced.") from error
        Path(target_db).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(target_db, timeout=5)) as dst:
            _backup_connections(src, dst)
            with dst:
                rebuild_fts(dst)
            counts = {key: dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for key, table in (("facts", "memories"), ("episodes", "episodes"),
                                         ("learnings", "learnings"), ("links", "entity_links"))}
    schema._SCHEMA_INITIALIZED.discard(target_db)
    return {"status": "ok", "message": f"Successfully restored to snapshot {filename}.",
            "restored_snapshot": filename, "safety_backup": safety_backup,
            "db_size": f"{os.path.getsize(target_db) / 1024:.1f} KB", "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="AGY Multi-Layer Cognitive Memory Engine")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    pf = subparsers.add_parser("prefetch", help="Multi-layer FTS5 prefetch for a query with entity linking & status weighting")
    pf.add_argument("query", type=str, help="User query text")

    st = subparsers.add_parser("sync-turn", help="Extract & sync persistent info from a conversation turn")
    st.add_argument("--user", type=str, required=True)
    st.add_argument("--assistant", type=str, required=True)
    st.add_argument("--dry-run", action="store_true", help="Show proposed changes without writing to DB")

    ad = subparsers.add_parser("add", help="Manually add/update an atomic fact")
    ad.add_argument("--id", type=str, required=True)
    ad.add_argument("--category", type=str, default="general")
    ad.add_argument("--fact", type=str, required=True)
    ad.add_argument("--keywords", type=str, default="")

    ae = subparsers.add_parser("add-episode", help="Manually add/update a narrative episode")
    ae.add_argument("--id", type=str, required=True)
    ae.add_argument("--topic", type=str, required=True)
    ae.add_argument("--title", type=str, required=True)
    ae.add_argument("--narrative", type=str, required=True)
    ae.add_argument("--period", type=str, default="")
    ae.add_argument("--status", type=str, default="active", choices=["active", "cooling", "historic", "resolved"])
    ae.add_argument("--entities", type=str, default="")
    ae.add_argument("--stance", type=str, default="")
    ae.add_argument("--keywords", type=str, default="")

    al = subparsers.add_parser("add-learning", help="Manually add/update an experiential learning")
    al.add_argument("--id", type=str, required=True)
    al.add_argument("--category", type=str, default="general")
    al.add_argument("--insight", type=str, required=True)
    al.add_argument("--context", type=str, default="")
    al.add_argument("--keywords", type=str, default="")

    # Entity link CLI commands
    lk = subparsers.add_parser("link", help="Create a relationship link between two entities / memory IDs")
    lk.add_argument("--source", type=str, required=True, help="Source memory ID or entity")
    lk.add_argument("--target", type=str, required=True, help="Target memory ID or entity")
    lk.add_argument("--relation", type=str, required=True, help="Relationship type (e.g. 'hosts', 'member_of', 'depends_on', 'owns')")

    unlk = subparsers.add_parser("unlink", help="Remove relationship link between two entities")
    unlk.add_argument("--source", type=str, required=True)
    unlk.add_argument("--target", type=str, required=True)
    unlk.add_argument("--relation", type=str, default=None)

    # Episode aging CLI command
    ag = subparsers.add_parser("age-episodes", help="Run automatic state decay for episodes (active -> cooling -> historic)")
    ag.add_argument("--days-to-cooling", type=int, default=30)
    ag.add_argument("--days-to-historic", type=int, default=90)

    # Semantic consolidation CLI command
    cs = subparsers.add_parser("consolidate", help="Run LLM semantic deduplication & consolidation of atomic facts")
    cs.add_argument("--apply", action="store_true", help="Apply consolidations to database")
    cs.add_argument("--dry-run", action="store_true", help="Show proposed consolidations without writing")

    op = subparsers.add_parser("optimize", help="Run episode aging, rebuild FTS indexes, VACUUM, report stats")
    op.add_argument("--apply", action="store_true", help="Apply optimization")
    op.add_argument("--no-age", action="store_true", help="Skip episode aging")
    op.add_argument("--consolidate", action="store_true", default=True, help="Run semantic deduplication (default: True)")
    op.add_argument("--no-consolidate", action="store_false", dest="consolidate", help="Skip semantic deduplication")

    # Keep backward compatibility
    cp = subparsers.add_parser("compact", help="(Alias for 'optimize') Rebuild FTS indexes & VACUUM")
    cp.add_argument("--apply", action="store_true")
    cp.add_argument("--consolidate", action="store_true", default=True, help="Run semantic deduplication (default: True)")
    cp.add_argument("--no-consolidate", action="store_false", dest="consolidate", help="Skip semantic deduplication")

    # Snapshot management CLI commands
    sn = subparsers.add_parser("snapshots", help="List database backup snapshots")
    sn.add_argument("--create", action="store_true", help="Create a new manual snapshot")

    rs = subparsers.add_parser("restore", help="Restore database from a backup snapshot")
    rs.add_argument("filename", type=str, help="Snapshot filename to restore (e.g. memory_db_backup_2026-09-02_092632.bak)")

    subparsers.add_parser("list", help="List all stored facts, episodes, learnings, and entity links")

    ui_p = subparsers.add_parser("ui", help="Launch real-time debug web dashboard")
    ui_p.add_argument("--port", type=int, default=None, help="Port to listen on (default from .env)")
    ui_p.add_argument("--host", type=str, default=None, help="Host to bind to (default from .env)")

    mg = subparsers.add_parser("migrate", help="Run database migrations (e.g. v2.0 -> v2.1)")
    mg.add_argument("--dry-run", action="store_true", help="Simulate migration without modifying database")
    mg.add_argument("--db", type=str, default=None, help="Path to SQLite database")

    args = parser.parse_args()

    if args.command in ("prefetch", "list"):
        for handler in logger.handlers:
            handler.setStream(sys.stdout)

    if args.command == "prefetch":
        prefetch(args.query)
    elif args.command == "sync-turn":
        sync_turn(args.user, args.assistant, dry_run=args.dry_run)
    elif args.command == "add":
        upsert_fact(args.id, args.category, args.fact, args.keywords)
        print(f"Added fact {args.id}")
    elif args.command == "add-episode":
        upsert_episode(args.id, args.topic, args.title, args.narrative, args.period, args.status, args.entities, args.stance, args.keywords)
        print(f"Added episode {args.id}")
    elif args.command == "add-learning":
        upsert_learning(args.id, args.category, args.insight, args.context, args.keywords)
        print(f"Added learning {args.id}")
    elif args.command == "link":
        link_entities(args.source, args.target, args.relation)
        print(f"Linked: {args.source} --[{args.relation}]--> {args.target}")
    elif args.command == "unlink":
        unlink_entities(args.source, args.target, args.relation)
        print(f"Unlinked: {args.source} <-> {args.target}")
    elif args.command == "age-episodes":
        res = age_episodes(args.days_to_cooling, args.days_to_historic)
        print(f"Cooled: {len(res['cooled'])}, Historic: {len(res['historied'])}")
    elif args.command == "consolidate":
        apply_flag = args.apply or (not args.dry_run)
        consolidate_memories(dry_run=not apply_flag)
    elif args.command in ("compact", "optimize"):
        optimize_db(
            apply_changes=getattr(args, 'apply', True),
            age_decay=not getattr(args, 'no_age', False),
            consolidate=getattr(args, 'consolidate', True)
        )
    elif args.command == "snapshots":
        if args.create:
            res = create_snapshot(tag="manual")
            print(f"[SUCCESS] {res['message']}")
        else:
            snaps = list_snapshots()
            print(f"Found {len(snaps)} snapshot(s) in ~/.gemini/archive:")
            for s in snaps:
                st = s["stats"]
                print(f"  • {s['filename']} ({s['size_kb']}, {s['created_at']}) [{s['tag']}] -> {st['facts']} facts, {st['episodes']} eps, {st['learnings']} lrn, {st['links']} links")
    elif args.command == "restore":
        res = restore_snapshot(args.filename)
        print(f"[SUCCESS] {res['message']}")
        print(f"Safety backup: {res['safety_backup']}")
        st = res["counts"]
        print(f"Database state: {st['facts']} facts, {st['episodes']} eps, {st['learnings']} lrn, {st['links']} links (size: {res['db_size']})")
    elif args.command == "list":
        list_all()
    elif args.command in ("ui", "dashboard"):
        from dashboard import run_dashboard
        from config import DASHBOARD_HOST, DASHBOARD_PORT
        port = args.port or DASHBOARD_PORT
        host = args.host or DASHBOARD_HOST
        run_dashboard(host=host, port=port)
    elif args.command == "migrate":
        from scripts.migrate_v2_to_v2_1 import run_migration
        db_target = args.db or DB_PATH
        print(f"=== AGY Memory Engine: Migration v2.0 -> v2.1 ===")
        print(f"Target Database: {db_target}")
        print(f"Mode: {'DRY RUN (simulation only)' if args.dry_run else 'LIVE MIGRATION'}\n")
        report = run_migration(db_path=db_target, dry_run=args.dry_run, verbose=True)
        print("\n--- Migration Summary ---")
        print(f"• Facts categories normalized:      {report['facts_migrated']}")
        print(f"• Episodes topics/status normalized: {report['episodes_migrated']}")
        print(f"• Learnings categories normalized:   {report['learnings_migrated']}")
        print(f"• Entity links mapped to canonical:  {report['links_mapped']}")
        print(f"• Orphan entity links pruned:        {report['orphan_links_pruned']}")
        if not args.dry_run:
            print(f"\n[MIGRATION COMPLETE] Successfully migrated to v2.1.0.")
            if report["backup_file"]:
                print(f"Safety backup retained at: {report['backup_file']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
