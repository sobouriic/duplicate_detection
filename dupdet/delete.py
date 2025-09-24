from .storage import delete_post_and_embedding

def delete_post(post_id: str) -> bool:
    return delete_post_and_embedding(post_id)
