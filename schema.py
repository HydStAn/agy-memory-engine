"""
AGY Memory Engine - Shared Schema & Database Initialization Module

Single source of truth for all table definitions, FTS5 virtual tables,
and synchronization triggers. Used by both the CLI engine and the MCP server.
"""

import os
import sqlite3
from contextlib import contextmanager

from config import DB_PATH, EMBEDDING_DIM, VECTOR_SEARCH_ENABLED

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except (ImportError, ModuleNotFoundError):
    HAS_SQLITE_VEC = False

# Protected categories that require explicit confirmation before overwrite in sync-turn
PROTECTED_CATEGORIES = frozenset({"health", "finance", "pension", "insurance", "user"})

_SCHEMA_INITIALIZED = set()  # Track which DB paths have been initialized this process


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, FTS5 virtual tables, and triggers if they don't exist."""

    # --- Layer 1: Atomic Facts (memories) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            category TEXT,
            fact TEXT NOT NULL,
            keywords TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id, category, fact, keywords
        );
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts (id, category, fact, keywords)
            VALUES (new.id, new.category, new.fact, new.keywords);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_memories_ad AFTER DELETE ON memories BEGIN
            DELETE FROM memories_fts WHERE id = old.id;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_memories_au AFTER UPDATE ON memories BEGIN
            DELETE FROM memories_fts WHERE id = old.id;
            INSERT INTO memories_fts (id, category, fact, keywords)
            VALUES (new.id, new.category, new.fact, new.keywords);
        END;
    """)

    # --- Layer 2: Narrative Chronicles & Episodes ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            title TEXT NOT NULL,
            period TEXT,
            status TEXT DEFAULT 'active',
            narrative TEXT NOT NULL,
            entities TEXT,
            stance TEXT,
            keywords TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            id, topic, title, narrative, entities, stance, keywords
        );
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_episodes_ai AFTER INSERT ON episodes BEGIN
            INSERT INTO episodes_fts (id, topic, title, narrative, entities, stance, keywords)
            VALUES (new.id, new.topic, new.title, new.narrative, new.entities, new.stance, new.keywords);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_episodes_ad AFTER DELETE ON episodes BEGIN
            DELETE FROM episodes_fts WHERE id = old.id;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_episodes_au AFTER UPDATE ON episodes BEGIN
            DELETE FROM episodes_fts WHERE id = old.id;
            INSERT INTO episodes_fts (id, topic, title, narrative, entities, stance, keywords)
            VALUES (new.id, new.topic, new.title, new.narrative, new.entities, new.stance, new.keywords);
        END;
    """)

    # --- Layer 3: Experiential Learnings & Heuristics ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learnings (
            id TEXT PRIMARY KEY,
            category TEXT,
            insight TEXT NOT NULL,
            context TEXT,
            keywords TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS learnings_fts USING fts5(
            id, category, insight, context, keywords
        );
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_learnings_ai AFTER INSERT ON learnings BEGIN
            INSERT INTO learnings_fts (id, category, insight, context, keywords)
            VALUES (new.id, new.category, new.insight, new.context, new.keywords);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_learnings_ad AFTER DELETE ON learnings BEGIN
            DELETE FROM learnings_fts WHERE id = old.id;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_learnings_au AFTER UPDATE ON learnings BEGIN
            DELETE FROM learnings_fts WHERE id = old.id;
            INSERT INTO learnings_fts (id, category, insight, context, keywords)
            VALUES (new.id, new.category, new.insight, new.context, new.keywords);
        END;
    """)

    # --- Feature 4: Entity Graph / Relations (Entity Linking) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_links (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            PRIMARY KEY (source_id, target_id, relation)
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entity_links_source ON entity_links(source_id);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entity_links_target ON entity_links(target_id);
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entity_links_fts USING fts5(
            source_id, target_id, relation
        );
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_entity_links_ai AFTER INSERT ON entity_links BEGIN
            INSERT INTO entity_links_fts (source_id, target_id, relation)
            VALUES (new.source_id, new.target_id, new.relation);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_entity_links_ad AFTER DELETE ON entity_links BEGIN
            DELETE FROM entity_links_fts WHERE source_id = old.source_id AND target_id = old.target_id AND relation = old.relation;
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_entity_links_au AFTER UPDATE ON entity_links BEGIN
            DELETE FROM entity_links_fts WHERE source_id = old.source_id AND target_id = old.target_id AND relation = old.relation;
            INSERT INTO entity_links_fts (source_id, target_id, relation)
            VALUES (new.source_id, new.target_id, new.relation);
        END;
    """)

    # --- Feature 5: Consolidation & Deduplication Audit Log ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action TEXT NOT NULL,
            category TEXT,
            target_id TEXT NOT NULL,
            merged_ids TEXT NOT NULL,
            diff_summary TEXT,
            rationale TEXT
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_consolidation_log_ts ON consolidation_log(timestamp DESC);
    """)

    # --- Feature 6: Vector Tables & Triggers (sqlite-vec) ---
    if HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED:
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[{EMBEDDING_DIM}]
                );
            """)
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[{EMBEDDING_DIM}]
                );
            """)
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_learnings USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[{EMBEDDING_DIM}]
                );
            """)

            # Automatic deletion cascade from parent tables
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_vec_memories_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM vec_memories WHERE id = old.id;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_vec_episodes_ad AFTER DELETE ON episodes BEGIN
                    DELETE FROM vec_episodes WHERE id = old.id;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_vec_learnings_ad AFTER DELETE ON learnings BEGIN
                    DELETE FROM vec_learnings WHERE id = old.id;
                END;
            """)
        except Exception:
            # Non-blocking if extension cannot be loaded in current context
            pass


@contextmanager
def db_session(db_path: str = None):
    """Open a SQLite connection with WAL mode and ensure schema is initialized.

    Schema DDL is only executed once per process per database path to avoid
    unnecessary overhead on every call.
    """
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")

    if HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED:
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass

    if path not in _SCHEMA_INITIALIZED:
        _init_schema(conn)
        _SCHEMA_INITIALIZED.add(path)

    try:
        yield conn
    finally:
        conn.close()
