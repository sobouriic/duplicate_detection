from functools import lru_cache
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import CFG

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None  # optional translation


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-12)).astype(np.float32)


def _resolve_instructions(model_name: str):
    m = (model_name or "").lower()
    qi = CFG.query_instruction or ""
    ti = CFG.text_instruction or ""

    if "bge" in m:
        return "query: ", "passage: "

    if "e5" in m:
        qi = qi or "query: "
        ti = ti or "passage: "
        return qi, ti

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
        print("[dupdet] Warning: could not load", model_name, "->", e)
        fallback = "intfloat/multilingual-e5-base"
        print("[dupdet] Falling back to", fallback)
        qi, ti = _resolve_instructions(fallback)
        return HuggingFaceEmbedding(
            model_name=fallback,
            query_instruction=qi,
            text_instruction=ti,
            device=CFG.device,
        )


def _maybe_translate(text: str) -> str:
    if getattr(CFG, "translate_to_english", False) and GoogleTranslator:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            print("[dupdet] Translation failed:", e)
    return text


def embed_text_document(text: str) -> np.ndarray:
    text = _maybe_translate(text)
    emb = _get_embedder().get_text_embedding(text)
    return _l2(emb)


def embed_text_query(text: str) -> np.ndarray:
    text = _maybe_translate(text)
    emb = _get_embedder().get_query_embedding(text)
    return _l2(emb)


# --- Debug helpers ---
def _cos(a, b) -> float:
    return float(np.dot(a, b))

def self_similarity_doc(text: str) -> float:
    v = embed_text_document(text)
    return _cos(v, v)

def doc_vs_query_same_text(text: str) -> float:
    return _cos(embed_text_document(text), embed_text_query(text))
