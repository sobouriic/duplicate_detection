from typing import List, Tuple, Optional
import numpy as np
from .embedder import embed_text_query
from .storage import fetch_embeddings

def similar_posts(
    query_text: str,
    top_k: int = 10,
    min_score: float = 0.45,
    topic: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Returns sorted list of (post_id, cosine_similarity) for the query.
    Uses dot product on normalized vectors (cosine).
    """
    q = embed_text_query(query_text).astype(np.float32)
    items = fetch_embeddings(topic=topic)
    if not items:
        return []

    ids, vecs = zip(*items)
    M = np.vstack(vecs)
    sims = M @ q
    order = np.argsort(-sims)

    out: List[Tuple[str, float]] = []
    for idx in order[:max(top_k, 0)]:
        score = float(sims[idx])
        if score >= min_score:
            out.append((ids[idx], score))
    return out
