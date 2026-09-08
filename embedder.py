"""
AGY Memory Engine - Vector Embedding & Hybrid Search Module
Lightweight ONNX-based embedding inference using FastEmbed and sqlite-vec.
"""

import sys
from typing import List, Optional
import numpy as np

from config import EMBEDDING_MODEL_NAME, VECTOR_SEARCH_ENABLED
from schema import HAS_SQLITE_VEC

_EMBEDDER_INSTANCE = None


def get_embedder():
    """Lazy-load the FastEmbed embedding model singleton."""
    global _EMBEDDER_INSTANCE
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return None
    if _EMBEDDER_INSTANCE is None:
        try:
            from fastembed import TextEmbedding
            # Suppress user warnings for clean logs
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _EMBEDDER_INSTANCE = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to initialize TextEmbedding ({EMBEDDING_MODEL_NAME}): {e}\n")
            _EMBEDDER_INSTANCE = False
    return _EMBEDDER_INSTANCE if _EMBEDDER_INSTANCE is not False else None


from functools import lru_cache

_VEC_QUERY_FAILED_WARNED = False

_TABLE_SQL = {
    "vec_memories": {
        "delete": "DELETE FROM vec_memories WHERE id = ?",
        "insert": "INSERT INTO vec_memories (id, embedding) VALUES (?, ?)",
    },
    "vec_episodes": {
        "delete": "DELETE FROM vec_episodes WHERE id = ?",
        "insert": "INSERT INTO vec_episodes (id, embedding) VALUES (?, ?)",
    },
    "vec_learnings": {
        "delete": "DELETE FROM vec_learnings WHERE id = ?",
        "insert": "INSERT INTO vec_learnings (id, embedding) VALUES (?, ?)",
    },
}


def log_vec_query_failure(table: str, error: Exception) -> None:
    """Log vec0 query failure on first occurrence instead of silently suppressing."""
    global _VEC_QUERY_FAILED_WARNED
    if not _VEC_QUERY_FAILED_WARNED:
        sys.stderr.write(f"[WARN] sqlite-vec query failed on {table}: {error}\n")
        _VEC_QUERY_FAILED_WARNED = True


def build_text_repr(item_type: str, item_data: dict) -> str:
    """
    Build canonical text representation for vector embedding across upsert and reindex operations.
    Supported types: 'fact', 'episode', 'learning'.
    """
    if item_type == "fact":
        cat = item_data.get("category") or "general"
        fact = item_data.get("fact") or ""
        keywords = item_data.get("keywords") or ""
        return f"[{cat}] {fact} {keywords}".strip()
    elif item_type == "episode":
        topic = item_data.get("topic") or "general"
        title = item_data.get("title") or ""
        narrative = item_data.get("narrative") or ""
        stance = item_data.get("stance") or "neutral"
        keywords = item_data.get("keywords") or ""
        return f"[{topic}] {title}: {narrative} (Stance: {stance}) {keywords}".strip()
    elif item_type == "learning":
        cat = item_data.get("category") or "general"
        insight = item_data.get("insight") or ""
        ctx = item_data.get("context") or ""
        keywords = item_data.get("keywords") or ""
        return f"[{cat}] {insight} (Context: {ctx}) {keywords}".strip()
    return ""


def embed_texts(texts: List[str]) -> Optional[List[np.ndarray]]:
    """Compute dense 384-dim embeddings for a list of texts."""
    model = get_embedder()
    if model is None or not texts:
        return None
    try:
        cleaned = [t.strip() if t and t.strip() else "empty" for t in texts]
        embeddings = list(model.embed(cleaned))
        return [np.array(e, dtype=np.float32) for e in embeddings]
    except Exception as e:
        sys.stderr.write(f"[WARN] Embedding generation failed: {e}\n")
        return None


@lru_cache(maxsize=256)
def _cached_embed_text(text: str) -> Optional[bytes]:
    """Helper caching raw bytes of single text embedding."""
    res = embed_texts([text])
    if res and len(res) > 0:
        return res[0].tobytes()
    return None


def embed_text(text: str) -> Optional[np.ndarray]:
    """Compute dense 384-dim embedding for a single text, utilizing an LRU cache."""
    raw_bytes = _cached_embed_text(text)
    if raw_bytes is not None:
        return np.frombuffer(raw_bytes, dtype=np.float32)
    return None


def upsert_vector(conn, table: str, item_id: str, text: str) -> bool:
    """Compute and store an embedding for an item in the corresponding vec table."""
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return False
    queries = _TABLE_SQL.get(table)
    if not queries:
        return False
    emb = embed_text(text)
    if emb is None:
        return False
    try:
        # vec0 delete-then-insert pattern
        conn.execute(queries["delete"], (item_id,))
        conn.execute(queries["insert"], (item_id, emb))
        return True
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to upsert vector in {table} for {item_id}: {e}\n")
        return False


def upsert_vectors_batch(conn, table: str, items: List[tuple]) -> int:
    """
    Batch compute and store embeddings for items in the corresponding vec table.
    items: List of (item_id, text) tuples.
    Returns count of successfully indexed items.
    """
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED) or not items:
        return 0
    queries = _TABLE_SQL.get(table)
    if not queries:
        return 0

    texts = [text for _, text in items]
    embeddings = embed_texts(texts)
    if embeddings is None or len(embeddings) != len(items):
        return 0

    success_count = 0
    try:
        cursor = conn.cursor()
        del_sql = queries["delete"]
        ins_sql = queries["insert"]
        for (item_id, _), emb in zip(items, embeddings):
            cursor.execute(del_sql, (item_id,))
            cursor.execute(ins_sql, (item_id, emb))
            success_count += 1
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed during batch upsert into {table}: {e}\n")
    return success_count


def delete_vector(conn, table: str, item_id: str) -> bool:
    """Delete an item embedding from a vec table."""
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return False
    queries = _TABLE_SQL.get(table)
    if not queries:
        return False
    try:
        conn.execute(queries["delete"], (item_id,))
        return True
    except Exception:
        return False


def reciprocal_rank_fusion(fts_results: list, vec_results: list, k: int = 60, limit: int = 5) -> list:
    """
    Combine FTS BM25 results and Vector Cosine Similarity results using Reciprocal Rank Fusion (RRF).
    
    RRF Score: sum(1 / (k + rank))
    Each input list is a list of dicts with at least 'id'.
    Returns merged and deduplicated list sorted by highest RRF score.
    """
    scores = {}
    item_map = {}

    for rank, item in enumerate(fts_results, start=1):
        iid = item.get("id")
        if not iid:
            continue
        scores[iid] = scores.get(iid, 0.0) + (1.0 / (k + rank))
        if iid not in item_map:
            item_map[iid] = item

    for rank, item in enumerate(vec_results, start=1):
        iid = item.get("id")
        if not iid:
            continue
        scores[iid] = scores.get(iid, 0.0) + (1.0 / (k + rank))
        if iid not in item_map:
            item_map[iid] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [item_map[iid] for iid in sorted_ids[:limit]]
