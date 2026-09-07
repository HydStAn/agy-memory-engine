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


def embed_text(text: str) -> Optional[np.ndarray]:
    """Compute dense 384-dim embedding for a single text."""
    res = embed_texts([text])
    return res[0] if res else None


def upsert_vector(conn, table: str, item_id: str, text: str) -> bool:
    """Compute and store an embedding for an item in the corresponding vec table."""
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return False
    emb = embed_text(text)
    if emb is None:
        return False
    try:
        # vec0 delete-then-insert pattern
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))
        conn.execute(f"INSERT INTO {table} (id, embedding) VALUES (?, ?)", (item_id, emb))
        return True
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to upsert vector in {table} for {item_id}: {e}\n")
        return False


def delete_vector(conn, table: str, item_id: str) -> bool:
    """Delete an item embedding from a vec table."""
    if not (HAS_SQLITE_VEC and VECTOR_SEARCH_ENABLED):
        return False
    try:
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))
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
