from typing import Iterable, Tuple, Optional
import numpy as np
from .storage import (
    upsert_post,
    upsert_embedding,
    missing_embedding_posts,
    delete_post_and_embedding,
)
from .embedder import embed_text_document


def batch_fill(topic: str, posts: Optional[Iterable[Tuple[str, str]]] = None) -> None:
    if posts:  # only if posts were passed in
        for pid, txt in posts:
            upsert_post(pid, txt, topic)

    pending = missing_embedding_posts(topic)
    if not pending:
        return

    post_ids, texts = zip(*pending)
    B = 32
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        embs = [embed_text_document(t) for t in chunk]
        for pid, vec in zip(post_ids[i:i + B], embs):
            upsert_embedding(pid, vec)


def delete_post(post_id: str) -> bool:
    return delete_post_and_embedding(post_id)
