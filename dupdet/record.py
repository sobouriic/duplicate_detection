from typing import Optional
import numpy as np
from .embedder import embed_text_document
from .storage import upsert_post, upsert_embedding

def record_post(post_id: str, text: str, topic: Optional[str] = None) -> None:
    """
    Create/update a post and persist its normalized embedding.
    Call this whenever a post is created or edited.
    """
    upsert_post(post_id, text, topic)
    vec: np.ndarray = embed_text_document(text)
    upsert_embedding(post_id, vec)
