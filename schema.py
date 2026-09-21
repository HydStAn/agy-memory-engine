"""
AGY Memory Engine - Shared Schema & Database Initialization Module

Single source of truth for all table definitions, FTS5 virtual tables,
and synchronization triggers. Used by both the CLI engine and the MCP server.
"""

import os
import sqlite3
import fcntl
import time
import uuid
import hashlib
from pathlib import Path
from contextlib import contextmanager

from config import DB_PATH, EMBEDDING_DIM, EMBEDDING_MODEL_NAME, VECTOR_SEARCH_ENABLED

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except (ImportError, ModuleNotFoundError):
    HAS_SQLITE_VEC = False

# Protected categories that require explicit confirmation before overwrite in sync-turn
PROTECTED_CATEGORIES = frozenset({"health", "finance", "pension", "insurance", "preferences", "user"})

_SCHEMA_IDENTITIES = {}
SCHEMA_VERSION = 212

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

    # --- Feature 6: Database Generation Fencing across Restores ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta_generation (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.execute("INSERT OR IGNORE INTO meta_generation(key, value) VALUES ('generation', ?);", (uuid.uuid4().hex,))

    # --- Feature 7: Vector Tables (sqlite-vec) ---
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
        except Exception:
            # Non-blocking if extension cannot be loaded in current context
            pass

    # --- Feature 8: Vector Reliability & Outbox Synchronization ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_name TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            repr_version TEXT NOT NULL DEFAULT 'v1',
            fingerprint TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_state (
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            indexed_revision INTEGER NOT NULL,
            generation TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            text_hash TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_type, entity_id)
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_jobs (
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            op TEXT NOT NULL DEFAULT 'upsert',
            desired_revision INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            lease_token TEXT,
            lease_expires_at TIMESTAMP,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_type, entity_id)
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_vector_index_jobs_claim ON vector_index_jobs(status, next_attempt_at);
    """)
    default_fp = compute_model_fingerprint(EMBEDDING_MODEL_NAME, EMBEDDING_DIM, "v1")
    conn.execute("""
        INSERT OR IGNORE INTO vector_index_config (id, model_name, dimension, repr_version, fingerprint)
        VALUES (1, ?, ?, 'v1', ?);
    """, (EMBEDDING_MODEL_NAME, EMBEDDING_DIM, default_fp))


def compute_model_fingerprint(model_name: str = None, dim: int = None, repr_version: str = "v1") -> str:
    """Compute deterministic fingerprint for active embedding model and text representation contract."""
    m = model_name or EMBEDDING_MODEL_NAME
    d = dim or EMBEDDING_DIM
    raw = f"{m}:{d}:{repr_version}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def get_db_generation(conn: sqlite3.Connection) -> str:
    """Retrieve current database generation token for fencing stale extractions across restores."""
    try:
        row = conn.execute("SELECT value FROM meta_generation WHERE key='generation'").fetchone()
        if row and row[0]:
            return row[0]
    except sqlite3.OperationalError:
        pass
    gen = uuid.uuid4().hex
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS meta_generation (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        conn.execute("INSERT OR REPLACE INTO meta_generation(key, value) VALUES ('generation', ?);", (gen,))
    except Exception:
        pass
    return gen


def bump_db_generation(conn: sqlite3.Connection) -> str:
    """Increment/regenerate database generation token on database restore."""
    gen = uuid.uuid4().hex
    conn.execute("CREATE TABLE IF NOT EXISTS meta_generation (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
    conn.execute("INSERT OR REPLACE INTO meta_generation(key, value) VALUES ('generation', ?);", (gen,))
    return gen


def _upgrade_schema(conn):
    """Versioned, transactional migration to rowid mirrors, durable receipts, and vector outbox."""
    conn.execute("CREATE TABLE IF NOT EXISTS batch_receipts (batch_id TEXT PRIMARY KEY, result_json TEXT NOT NULL, committed_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS entity_revisions (entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(entity_type, entity_id))")
    conn.execute("CREATE TABLE IF NOT EXISTS meta_generation (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT OR IGNORE INTO meta_generation(key, value) VALUES ('generation', ?)", (uuid.uuid4().hex,))

    # Feature 8 tables: Vector reliability & outbox jobs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_name TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            repr_version TEXT NOT NULL DEFAULT 'v1',
            fingerprint TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_state (
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            indexed_revision INTEGER NOT NULL,
            generation TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            text_hash TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_type, entity_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_jobs (
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            op TEXT NOT NULL DEFAULT 'upsert',
            desired_revision INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            lease_token TEXT,
            lease_expires_at TIMESTAMP,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_type, entity_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_index_jobs_claim ON vector_index_jobs(status, next_attempt_at)")

    default_fp = compute_model_fingerprint(EMBEDDING_MODEL_NAME, EMBEDDING_DIM, "v1")
    conn.execute("""
        INSERT OR IGNORE INTO vector_index_config (id, model_name, dimension, repr_version, fingerprint)
        VALUES (1, ?, ?, 'v1', ?)
    """, (EMBEDDING_MODEL_NAME, EMBEDDING_DIM, default_fp))

    # Drop legacy triggers that cascade delete directly on virtual tables (causes crashes if sqlite-vec is absent)
    for table in ('memories', 'episodes', 'learnings'):
        conn.execute(f"DROP TRIGGER IF EXISTS trg_vec_{table}_ad")

    columns = {
        'memories': 'id, category, fact, keywords',
        'episodes': 'id, topic, title, narrative, entities, stance, keywords',
        'learnings': 'id, category, insight, context, keywords',
        'entity_links': 'source_id, target_id, relation',
    }
    for table, names in columns.items():
        for suffix in ('ai', 'au', 'ad'):
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_{suffix}")
        values = ', '.join('new.' + name.strip() for name in names.split(','))
        conn.execute(f"CREATE TRIGGER trg_{table}_ai AFTER INSERT ON {table} BEGIN INSERT INTO {table}_fts(rowid, {names}) VALUES(new.rowid, {values}); END")
        conn.execute(f"CREATE TRIGGER trg_{table}_ad AFTER DELETE ON {table} BEGIN DELETE FROM {table}_fts WHERE rowid=old.rowid; END")
        conn.execute(f"CREATE TRIGGER trg_{table}_au AFTER UPDATE ON {table} BEGIN DELETE FROM {table}_fts WHERE rowid=old.rowid; INSERT INTO {table}_fts(rowid, {names}) VALUES(new.rowid, {values}); END")
        conn.execute(f"DELETE FROM {table}_fts")
        conn.execute(f"INSERT INTO {table}_fts(rowid, {names}) SELECT rowid, {names} FROM {table}")

        if table == 'entity_links':
            continue

        conn.execute(f"INSERT OR IGNORE INTO entity_revisions SELECT '{table}', id, 1 FROM {table}")

        # Drop old revision triggers before recreating unified revision + outbox triggers
        for suffix in ('ai', 'au', 'ad'):
            conn.execute(f"DROP TRIGGER IF EXISTS rev_{table}_{suffix}")
        conn.execute(f"DROP TRIGGER IF EXISTS rev_{table}_rename")

        for suffix, event, ref, op in (
            ('ai', 'INSERT', 'new', 'upsert'),
            ('au', 'UPDATE', 'new', 'upsert'),
            ('ad', 'DELETE', 'old', 'delete')
        ):
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS rev_{table}_{suffix} AFTER {event} ON {table} BEGIN
                    INSERT INTO entity_revisions VALUES('{table}', {ref}.id, 1)
                        ON CONFLICT(entity_type, entity_id) DO UPDATE SET revision=revision+1;
                    INSERT INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at, updated_at)
                        VALUES('{table}', {ref}.id, '{op}', (SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id={ref}.id), 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                            op='{op}',
                            desired_revision=(SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id={ref}.id),
                            status='pending',
                            lease_token=NULL,
                            lease_expires_at=NULL,
                            updated_at=CURRENT_TIMESTAMP;
                END;
            """)

        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS rev_{table}_rename AFTER UPDATE OF id ON {table} WHEN old.id != new.id BEGIN
                INSERT INTO entity_revisions VALUES('{table}', old.id, 1)
                    ON CONFLICT(entity_type, entity_id) DO UPDATE SET revision=revision+1;
                INSERT INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at, updated_at)
                    VALUES('{table}', old.id, 'delete', (SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id=old.id), 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                        op='delete',
                        desired_revision=(SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id=old.id),
                        status='pending',
                        lease_token=NULL,
                        lease_expires_at=NULL,
                        updated_at=CURRENT_TIMESTAMP;
                INSERT INTO entity_revisions VALUES('{table}', new.id, 1)
                    ON CONFLICT(entity_type, entity_id) DO UPDATE SET revision=revision+1;
                INSERT INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at, updated_at)
                    VALUES('{table}', new.id, 'upsert', (SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id=new.id), 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                        op='upsert',
                        desired_revision=(SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id=new.id),
                        status='pending',
                        lease_token=NULL,
                        lease_expires_at=NULL,
                        updated_at=CURRENT_TIMESTAMP;
            END;
        """)

        # Seed initial pending jobs for legacy rows
        conn.execute(f"""
            INSERT OR IGNORE INTO vector_index_jobs (entity_type, entity_id, op, desired_revision, status, attempts, next_attempt_at)
            SELECT '{table}', id, 'upsert', COALESCE((SELECT revision FROM entity_revisions WHERE entity_type='{table}' AND entity_id={table}.id), 1), 'pending', 0, CURRENT_TIMESTAMP
            FROM {table}
        """)

    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


@contextmanager
def maintenance_lock(db_path, exclusive=False, timeout=5):
    """Coordinate restore with all current engine connections across processes."""
    lock_path = Path(db_path).expanduser().resolve().with_suffix('.maintenance.lock')
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        pass

    handle = None
    try:
        try:
            handle = lock_path.open('a')
        except PermissionError:
            if not exclusive and lock_path.exists():
                handle = lock_path.open('r')
            else:
                raise
    except PermissionError:
        if not exclusive:
            # Multi-user or read-only profile: allow shared access if lock file cannot be created/opened
            yield
            return
        raise

    with handle:
        deadline = time.monotonic() + timeout
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(handle, operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Memory maintenance busy; retry after active clients finish')
                time.sleep(0.025)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def db_session(db_path: str = None):
    """Close on every path, and serialize schema bootstrap across processes."""
    path = os.path.abspath(os.path.expanduser(db_path or DB_PATH))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = maintenance_lock(path)
    lock.__enter__()
    conn = None
    try:
        conn = sqlite3.connect(path, timeout=5)
        if HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                pass
        identity = (os.stat(path).st_dev, os.stat(path).st_ino)
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f'Unsupported memory schema version {version}')
        if path not in _SCHEMA_INITIALIZED or _SCHEMA_IDENTITIES.get(path) != identity or version < SCHEMA_VERSION:
            conn.execute('PRAGMA journal_mode=WAL')
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                version = conn.execute('PRAGMA user_version').fetchone()[0]
                _init_schema(conn)
                if version < SCHEMA_VERSION:
                    _upgrade_schema(conn)
            _SCHEMA_INITIALIZED.add(path)
            _SCHEMA_IDENTITIES[path] = identity
        yield conn
    finally:
        if conn is not None:
            conn.close()
        lock.__exit__(None, None, None)
