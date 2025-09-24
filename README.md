# Duplicate Detection

Duplicate Detection provides a set of utilities for detecting similar or duplicate content using text embeddings and cosine similarity, with persistent storage in SQLite. It is designed to efficiently store, search, and manage posts or documents, supporting batch operations and topic-based grouping.

## Features

- **Add or update posts** with embeddings stored persistently.
- **Delete posts** and their embeddings.
- **Search for similar posts** using cosine similarity.
- **Batch fill** for efficient bootstrapping.
- **Topic-based grouping** for organization and focused search.

## Functions Overview

| Function         | Parameters                                                                 | Returns                 | Description                                                                                       |
|------------------|----------------------------------------------------------------------------|-------------------------|---------------------------------------------------------------------------------------------------|
| `record_post`    | `post_id` (str), `text` (str), `topic` (str, optional)                     | None                    | Create or update a post and store its embedding in SQLite. Recomputes embedding if text changes.   |
| `delete_post`    | `post_id` (str)                                                            | bool                    | Delete a post and its embedding. Returns `True` if a post was deleted.                            |
| `similar_posts`  | `query_text` (str), `top_k` (int, default 10), `min_score` (float, default 0.45), `topic` (str, optional) | List[(post_id, score)]  | Embed the query and return top-k similar posts with cosine similarity scores.                     |
| `batch_fill`     | `topic` (str), `posts` (iterable of (post_id, text))                       | None                    | Add posts in batch, only embedding missing ones. Useful for bootstrapping a topic.                |

---

## Installation

### 1. Clone or Copy the Project

```
git clone https://github.com/sobouriic/duplicate_detection.git
cd duplicate_detection
```

Project structure:
```
.
├── dupdet
│   ├── __init__.py
│   ├── batch.py
│   ├── config.py
│   ├── delete.py
│   ├── embedder.py
│   ├── record.py
│   ├── search.py
│   └── storage.py
└── run_tests.py
```

### 2. Set Up Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -U llama-index-core
pip install -U llama-index-embeddings-huggingface
pip install -U transformers
pip install -U numpy

python -m pip install --upgrade pip wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.1.2"
python -m pip install "llama-index-embeddings-huggingface==0.2.0"
python -m pip install "transformers==4.43.3" "accelerate<1.0" sentencepiece "huggingface-hub>=0.19.0"
```

### 4. Hugging Face Access Setup

You need authentication to use the `google/embeddinggemma-300m` model:

```bash
huggingface-cli login
```
Then, enter the Hugging Face token when prompted. (create your own acount)

---

## Usage

### 1. Record or Update a Post

```python
from dupdet import record_post

record_post(post_id="123", text="Sample content", topic="news")
```

### 2. Delete a Post

```python
from dupdet import delete_post

success = delete_post(post_id="123")
```

### 3. Find Similar Posts

```python
from dupdet import similar_posts

results = similar_posts("Find similar to this text", top_k=5, min_score=0.5, topic="news")
# results: list of (post_id, similarity_score)
```

### 4. Batch Fill for a Topic

```python
from dupdet import batch_fill

batch_fill("news", [("id1", "Text 1"), ("id2", "Text 2")])
```

---

## Testing

- **First test:**  
  ```bash
  python3 -m dupdet.test
  ```
- **Second test:**  
  ```bash
  python3 test_each_function.py
  ```

---

## Notes

- All embeddings and post data are persisted in an SQLite database.
- Topic grouping is optional but useful for narrowing searches.

## License

[MIT](LICENSE) 
