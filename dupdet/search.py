from typing import List, Tuple, Optional
import numpy as np
from .embedder import embed_text_query
from .storage import fetch_embeddings
from .calibration import calibrate

def similar_posts_old(
    query_text: str,
    top_k: Optional[int] = 10,
    min_score: Optional[float] = 0.80,
    topic: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Returns sorted list of (post_id, cosine_similarity) for the query.
    Uses dot product on normalized vectors (cosine). Filters by raw cosine >= min_score.
    """
    q = embed_text_query(query_text).astype(np.float32)
    items = fetch_embeddings(topic=topic)
    if not items:
        return []

    ids, vecs = zip(*items)
    M = np.vstack(vecs)  # (N, D)
    sims = M @ q         # cosine on L2-normalized vectors
    order = np.argsort(-sims)

    out: List[Tuple[str, float]] = []
    for idx in order[:max(top_k, 0)]:
        score = float(sims[idx])
        if score >= min_score:
            out.append((ids[idx], score))
    return out



def similar_posts(
    query_text: str,
    top_k: Optional[int] = 10,
    min_score: Optional[float] = 0.80,
    topic: Optional[str] = None
) -> List[Tuple[str, float, float]]:
    """
    RETURNS (post_id, calibrated_score, raw_score).
    """
    # Treat "" as None (match-all)
    effective_topic = None if (topic is None or str(topic).strip() == "") else topic

    q = embed_text_query(query_text).astype(np.float32)
    items = fetch_embeddings(topic=effective_topic)
    if not items:
        return []

    ids, vecs = zip(*items)
    M = np.vstack(vecs)
    sims = M @ q
    order = np.argsort(-sims)

    out: List[Tuple[str, float, float]] = []
    for idx in order[:max(top_k, 0)]:
        raw = float(sims[idx])
        if raw >= min_score:
            out.append((ids[idx], calibrate(raw), raw))
    return out
