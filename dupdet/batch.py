from typing import Iterable, Tuple
import numpy as np
from .storage import upsert_post, upsert_embedding, missing_embedding_posts
from .embedder import embed_text_document

def batch_fill(topic: str, posts: Iterable[Tuple[str, str]]) -> None:
    """
    Ingest/refresh a topic's posts:
      posts = iterable of (post_id, text)
    1) Upsert posts
    2) Embed only those missing embeddings (
    """
    # 1) we Persist metadata
    for pid, txt in posts:
        upsert_post(pid, txt, topic)

    # 2) we  Find un-embedded
    pending = missing_embedding_posts(topic)
    if not pending:
        return

    post_ids, texts = zip(*pending)
    # 3) we Embed in mini-batches
    B = 32
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        embs = [embed_text_document(t) for t in chunk]
        for pid, vec in zip(post_ids[i:i+B], embs):
            upsert_embedding(pid, vec)
