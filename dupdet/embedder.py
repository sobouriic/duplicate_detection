from functools import lru_cache
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import CFG


@lru_cache(maxsize=1)
def _get_embedder() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=CFG.model_name,
        query_instruction=CFG.query_instruction,
        text_instruction=CFG.text_instruction,
        device=CFG.device,
    )

def embed_text_document(text: str) -> np.ndarray:
    """Embedding for posts (documents). Returns L2-normalized float32 vector."""
    emb = _get_embedder().get_text_embedding(text)
    v = np.asarray(emb, dtype=np.float32)
    # L2 normalize to make cosine == dot
    n = np.linalg.norm(v)
    return (v / max(n, 1e-12)).astype(np.float32)

def embed_text_query(text: str) -> np.ndarray:
    """Embedding for search queries. Returns L2-normalized float32 vector."""
    emb = _get_embedder().get_query_embedding(text)
    v = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / max(n, 1e-12)).astype(np.float32)


@lru_cache(maxsize=1)
def _get_embedder() -> HuggingFaceEmbedding:
    try:
        return HuggingFaceEmbedding(
            model_name=CFG.model_name,
            query_instruction=CFG.query_instruction,
            text_instruction=CFG.text_instruction,
            device=CFG.device,
        )
    except Exception as e:
        # Fallback: open multilingual model (no gated access needed)
        print("[dupdet] Warning: could not load EmbeddingGemma ->", e)
        print("[dupdet] Falling back to 'intfloat/multilingual-e5-base'.")
        return HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-base",
            device=CFG.device,
        )
