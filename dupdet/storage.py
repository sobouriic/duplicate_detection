import sqlite3
from contextlib import contextmanager
from typing import List, Optional, Tuple
import numpy as np
from .config import CFG

@contextmanager
def _conn():
    con = sqlite3.connect(CFG.db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    try:
        yield con
    finally:
        con.commit()
        con.close()

def init_db():
    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS posts(
          post_id TEXT PRIMARY KEY,
          topic   TEXT,
          text    TEXT NOT NULL,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings(
          post_id TEXT PRIMARY KEY,
          dim     INTEGER NOT NULL,
          vec     BLOB NOT NULL,
          FOREIGN KEY(post_id) REFERENCES posts(post_id) ON DELETE CASCADE
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_posts_updated_at ON posts(updated_at);")

def _to_blob(vec: np.ndarray) -> bytes:
    assert vec.dtype == np.float32 and vec.ndim == 1
    return vec.tobytes(order="C")

def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)

def upsert_post(post_id: str, text: str, topic: Optional[str]) -> None:
    init_db()
    with _conn() as con:
        con.execute("""
        INSERT INTO posts (post_id, topic, text)
        VALUES (?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
          topic=excluded.topic,
          text=excluded.text,
          updated_at=CURRENT_TIMESTAMP;
        """, (post_id, topic, text))

def upsert_embedding(post_id: str, vec: np.ndarray) -> None:
    init_db()
    dim = int(vec.shape[0])
    with _conn() as con:
        con.execute("""
        INSERT INTO embeddings (post_id, dim, vec)
        VALUES (?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
          dim=excluded.dim,
          vec=excluded.vec;
        """, (post_id, dim, _to_blob(vec)))

def delete_post_and_embedding(post_id: str) -> bool:
    """Deletes the post and its embedding. Returns True if a row was deleted."""
    init_db()
    with _conn() as con:
        cur = con.cursor()
        cur.execute("DELETE FROM posts WHERE post_id = ?;", (post_id,))
        return cur.rowcount > 0

def fetch_embeddings(topic: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
    """Returns list of (post_id, vector) filtered by topic if provided."""
    init_db()
    with _conn() as con:
        cur = con.cursor()
        if topic is None:
            cur.execute("""
                SELECT e.post_id, e.dim, e.vec
                FROM embeddings e JOIN posts p ON p.post_id = e.post_id;
            """)
        else:
            cur.execute("""
                SELECT e.post_id, e.dim, e.vec
                FROM embeddings e JOIN posts p ON p.post_id = e.post_id
                WHERE p.topic = ?;
            """, (topic,))
        rows = cur.fetchall()

    out = []
    for post_id, dim, blob in rows:
        out.append((post_id, _from_blob(blob, int(dim))))
    return out

def missing_embedding_posts(topic: Optional[str]) -> List[Tuple[str, str]]:
    """Returns [(post_id, text)] where posts exist but no embedding yet."""
    init_db()
    with _conn() as con:
        cur = con.cursor()
        if topic is None:
            cur.execute("""
              SELECT p.post_id, p.text
              FROM posts p
              LEFT JOIN embeddings e ON e.post_id = p.post_id
              WHERE e.post_id IS NULL;
            """)
        else:
            cur.execute("""
              SELECT p.post_id, p.text
              FROM posts p
              LEFT JOIN embeddings e ON e.post_id = p.post_id
              WHERE p.topic = ? AND e.post_id IS NULL;
            """, (topic,))
        return cur.fetchall()
