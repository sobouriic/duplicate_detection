from typing import Optional
import numpy as np
from .embedder import embed_text_document
from .storage import upsert_post, upsert_embedding, delete_post_and_embedding

def record_post(post_id: str, text: str, topic: Optional[str] = None) -> None:
    try:
        delete_post_and_embedding(post_id)
    except Exception as e:
        print("[dupdet] record_post: pre-delete failed:", e)

    upsert_post(post_id, text, topic)
    vec: np.ndarray = embed_text_document(text)
    upsert_embedding(post_id, vec)
