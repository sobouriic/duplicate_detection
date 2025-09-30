from functools import lru_cache
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import CFG

def _l2(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-12)).astype(np.float32)

def _resolve_instructions(model_name: str):
    qi = CFG.query_instruction
    ti = CFG.text_instruction
    m = (model_name or "").lower()

    if "e5" in m:
        if not qi:
            qi = "query: "
        if not ti:
            ti = "query: "
    return qi, ti

@lru_cache(maxsize=1)
def _get_embedder() -> HuggingFaceEmbedding:
    model_name = CFG.model_name
    try:
        qi, ti = _resolve_instructions(model_name)
        return HuggingFaceEmbedding(
            model_name=model_name,
            query_instruction=qi,
            text_instruction=ti,
            device=CFG.device,
        )
    except Exception as e:
        print("[dupdet] Warning: could not load primary model:", model_name, "->", e)
        fallback = "intfloat/multilingual-e5-base"
        print("[dupdet] Falling back to", fallback)
        qi, ti = _resolve_instructions(fallback)
        return HuggingFaceEmbedding(
            model_name=fallback,
            query_instruction=qi,
            text_instruction=ti,
            device=CFG.device,
        )

def embed_text_document(text: str) -> np.ndarray:
    """Embedding for posts (documents). Returns L2-normalized float32 vector."""
    emb = _get_embedder().get_text_embedding(text)
    return _l2(emb)

def embed_text_query(text: str) -> np.ndarray:
    """Embedding for search queries. Returns L2-normalized float32 vector."""
    emb = _get_embedder().get_query_embedding(text)
    return _l2(emb)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def self_similarity_doc(text: str) -> float:
    """cos(embed_doc(text), embed_doc(text)) -> should be ~1.0"""
    v = embed_text_document(text)
    return _cos(v, v)

def doc_vs_query_same_text(text: str) -> float:
    """cos(embed_doc(text), embed_query(text)) -> should be ~1.0 for symmetric setup"""
    vd = embed_text_document(text)
    vq = embed_text_query(text)
    return _cos(vd, vq)

