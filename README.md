# Duplicate Detection






Duplicate Detection is a Python package for identifying similar or duplicate text content using state-of-the-art text embeddings. It uses the BGE-m3 model by default, provides score calibration, and supports persistent storage with SQLite. The package is designed for flexible batch operations and supports multiple languages.





## Features





- **Modern Embeddings:** Uses the BGE-m3 embedding model for robust multilingual similarity detection.


- **Score Calibration:** Raw cosine similarities can be calibrated (min-max or logistic) for more interpretable results.


- **Persistent Storage:** All posts and their embeddings are stored in an SQLite database.


- **Batch Operations:** Efficiently ingest, embed, and search large numbers of posts.


- **Flexible API:** Functions for adding/updating posts, searching for similar posts, and batch ingestion.


- **Demo & Testing Scripts:** Includes pipeline and embedding comparison scripts to demonstrate functionality.





## Installation





Clone the repository and set up your Python environment:


```bash


git clone https://github.com/sobouriic/duplicate_detection.git


cd duplicate_detection


python -m venv .venv


source .venv/bin/activate


pip install -U pip wheel


pip install -U llama-index-core llama-index-embeddings-huggingface transformers numpy


pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.1.2"


```





## Usage





### 1. Ingest and Embed Posts


```python


from dupdet import record_post, batch_fill





record_post("p1", "This is a sample post.", topic="news")


batch_fill("news", [("p2", "Another post."), ("p3", "Un autre post.")])


```





### 2. Search for Similar Posts


```python


from dupdet import similar_posts





results = similar_posts("Find related content here", topic="news", min_score=0.80, top_k=5)


for post_id, calibrated_score, raw_score in results:


    print(post_id, calibrated_score, raw_score)


```











### 3. Run Embedding Test


```bash


python embedding_test.py


```





## Configuration





- The default embedding model is `BAAI/bge-m3`.


- Calibration method can be set to `minmax` or `logistic` in `dupdet/config.py`.


- Device selection (`cpu`/`cuda`) is available in configuration.





## Project Structure





```


.


├── dupdet/


│   ├── __init__.py


│   ├── batch.py


│   ├── calibration.py


│   ├── config.py


│   ├── emb.py


│   ├── embedder.py


│   ├── record.py


│   ├── search.py


│   └── storage.py


├── pipeline_test.py


├── embedding_compare.py


├── embedding_test.py


├── pipeline_results.md


├── pipeline_results.csv


└── embedding_results.md


```




## License





MIT License. See LICENSE file for details.





## Acknowledgments





- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)


- [llama-index](https://github.com/run-llama/llama_index)
