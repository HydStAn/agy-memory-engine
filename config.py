"""
AGY Memory Engine - Central Configuration Module
Zero-dependency loader for .env and environment variables.
"""

import os
import shutil
import hashlib
from pathlib import Path


def _load_env_file(filepath: Path) -> dict:
    """Parse a simple KEY=VALUE .env file without external dependencies."""
    env_vars = {}
    if not filepath.exists():
        return env_vars
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    env_vars[key] = val
    except Exception:
        pass
    return env_vars


# Load from project .env first, then ~/.gemini/memory.env fallback
_PROJECT_ENV = Path(__file__).resolve().parent / ".env"
_USER_ENV = Path.home() / ".gemini" / "memory.env"

_FILE_VARS = {}
_FILE_VARS.update(_load_env_file(_USER_ENV))
_FILE_VARS.update(_load_env_file(_PROJECT_ENV))


def get_config(key: str, default: str = "") -> str:
    """Retrieve config value prioritizing OS env, then .env file, then default."""
    return os.environ.get(key) or _FILE_VARS.get(key) or default


# --- Model Configuration ---
DEFAULT_MODEL = "gemini-3.8-flash-low"
MODEL_NAME = get_config("AGY_MEMORY_MODEL", DEFAULT_MODEL)
CACHE_PATH = os.path.expanduser(get_config("AGY_MEMORY_CACHE", str(Path.home() / ".gemini" / "memory_model_cache.txt")))
MODEL_EXPLICIT = bool(get_config("AGY_MEMORY_MODEL"))

# --- Vector Embedding Configuration ---
EMBEDDING_MODEL_NAME = get_config("AGY_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIM = int(get_config("AGY_EMBEDDING_DIM", "384"))
VECTOR_SEARCH_ENABLED = get_config("AGY_VECTOR_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes", "on")
STRICT_GRAPH = get_config("AGY_MEMORY_STRICT_GRAPH", "false").lower() in ("true", "1", "yes", "on")

# --- Binary Paths ---
AGY_BIN = (
    get_config("AGY_BIN")
    or shutil.which("agy")
    or str(Path.home() / ".local" / "bin" / "agy")
)

# --- Database Paths ---
DB_PATH = os.path.expanduser(get_config("AGY_MEMORY_DB", str(Path.home() / ".gemini" / "memory.db")))
QUEUE_DB_PATH = os.path.expanduser(get_config("AGY_TURN_QUEUE_DB", str(Path.home() / ".gemini" / "turn_queue.db")))

# --- Debounce Timers (in seconds) ---
INACTIVITY_THRESHOLD_SECONDS = int(get_config("AGY_MEMORY_INACTIVITY_SECONDS", "300"))
MAX_WAIT_THRESHOLD_SECONDS = int(get_config("AGY_MEMORY_MAX_WAIT_SECONDS", "900"))

# --- Queue Batch Sizing ---
# WORKER caps total turns drained per run; CLAIM caps turns per single claim, i.e. one LLM extraction prompt.
WORKER_BATCH_SIZE = int(get_config("AGY_MEMORY_WORKER_BATCH_SIZE", "150"))
CLAIM_BATCH_SIZE = int(get_config("AGY_MEMORY_CLAIM_BATCH_SIZE", "25"))

# Per-turn character cap applied at enqueue. An agent turn accumulates every
# intermediate output since the last user message, so a deploy session can
# produce hundreds of KB of build logs in one "assistant response".
MAX_TURN_CHARS = int(get_config("AGY_MEMORY_MAX_TURN_CHARS", "20000"))

# After this many attempts on the same batch, retries peel off one turn at a
# time. Without it a batch keeps its original membership forever and a single
# unparseable turn takes its healthy neighbours down with it.
RETRY_SPLIT_AFTER = int(get_config("AGY_MEMORY_RETRY_SPLIT_AFTER", "3"))

# --- Jev Relevance Gate (one call per retrieval, fail-open) ---
_JEV_KEY = get_config("AGY_JEV_API_KEY") or get_config("JEV_API_KEY")
if not _JEV_KEY:
    # Key lives with the agy hook credentials; same secret, no new contract.
    _JEV_KEY = _load_env_file(Path.home() / ".config" / "agy" / "sage.env").get("AGY_JEV_API_KEY", "")
JEV_GATE_API_KEY = _JEV_KEY
JEV_GATE_ENABLED = get_config("AGY_MEMORY_JEV_GATE", "true").lower() in ("true", "1", "yes", "on")
JEV_GATE_URL = get_config("AGY_JEV_GATE_URL", "https://ai-gateway.vercel.sh/v4/ai/evaluation-model")
JEV_GATE_MODEL_ID = get_config("AGY_JEV_GATE_MODEL_ID", "typesafe-ai/jev")


def _jev_number(name, default, low, high):
    """Finite number in range, or None; a bad value disables the gate instead of raising."""
    raw = get_config(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (low <= value <= high):  # NaN fails every comparison, infinities fail the range
        return None
    return value


_JEV_FLOOR = _jev_number("AGY_MEMORY_JEV_GATE_FLOOR", 0.75, 0.0, 1.0)
_JEV_TIMEOUT = _jev_number("AGY_MEMORY_JEV_GATE_TIMEOUT", 6.0, 0.1, 120.0)
_JEV_MIN_ITEMS = _jev_number("AGY_MEMORY_JEV_GATE_MIN_ITEMS", 3, 1, 100000)
_JEV_MIN_CHARS = _jev_number("AGY_MEMORY_JEV_GATE_MIN_CHARS", 600, 0, 10000000)
if None in (_JEV_FLOOR, _JEV_TIMEOUT, _JEV_MIN_ITEMS, _JEV_MIN_CHARS):
    JEV_GATE_ENABLED = False
JEV_GATE_FLOOR = _JEV_FLOOR if _JEV_FLOOR is not None else 0.75
JEV_GATE_TIMEOUT = _JEV_TIMEOUT if _JEV_TIMEOUT is not None else 6.0
JEV_GATE_MIN_ITEMS = int(_JEV_MIN_ITEMS) if _JEV_MIN_ITEMS is not None else 3
JEV_GATE_MIN_CHARS = int(_JEV_MIN_CHARS) if _JEV_MIN_CHARS is not None else 600

# --- Telegram Notifications ---
DEFAULT_TELEGRAM_CHAT_ID = get_config("AGY_MEMORY_TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM_BIN = Path(os.path.expanduser(
    get_config(
        "AGY_MEMORY_TELEGRAM_BIN",
        shutil.which("send_telegram.py") or str(Path.home() / "bin" / "send_telegram.py")
    )
))

# --- Real-Time Debug Dashboard ---
DASHBOARD_ENABLED = get_config("AGY_MEMORY_DEBUG_DASHBOARD", "false").lower() in ("true", "1", "yes", "on")
DASHBOARD_PORT = int(get_config("AGY_MEMORY_DASHBOARD_PORT", "8085"))
DASHBOARD_HOST = get_config("AGY_MEMORY_DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_ALLOWED_HOSTS = [
    h.strip().lower()
    for h in get_config("AGY_MEMORY_DASHBOARD_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]
_raw_allow_private = get_config("AGY_MEMORY_DASHBOARD_ALLOW_PRIVATE_NETWORKS", "").strip().lower()
if _raw_allow_private in ("true", "1", "yes", "on"):
    DASHBOARD_ALLOW_PRIVATE_NETWORKS = True
elif _raw_allow_private in ("false", "0", "no", "off"):
    DASHBOARD_ALLOW_PRIVATE_NETWORKS = False
else:
    DASHBOARD_ALLOW_PRIVATE_NETWORKS = None
DASHBOARD_TOKEN = get_config("AGY_MEMORY_DASHBOARD_TOKEN", "")
DASHBOARD_TOKEN_PATH = os.path.expanduser(
    get_config("AGY_MEMORY_DASHBOARD_TOKEN_PATH", str(Path.home() / ".gemini" / "dashboard.token"))
)



def database_namespace(db_path):
    return hashlib.sha256(str(Path(db_path).expanduser().resolve()).encode()).hexdigest()[:16]


def archive_path(db_path):
    """Keep legacy snapshots visible for the default DB; isolate other stores."""
    root = Path(os.path.expanduser(get_config('AGY_MEMORY_ARCHIVE', str(Path.home() / '.gemini' / 'archive'))))
    default_db = Path.home() / '.gemini' / 'memory.db'
    return root if Path(db_path).resolve() == default_db.resolve() else root / database_namespace(db_path)


def sync_lock_path(db_path):
    override = get_config('AGY_MEMORY_SYNC_LOCK')
    return Path(os.path.expanduser(override)) if override else Path(db_path).resolve().with_suffix('.sync.lock')
