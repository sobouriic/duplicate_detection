from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    db_path: Path = Path("./dupdet.sqlite")
    # Full-size EmbeddingGemma embeddings
    model_name: str = "google/embeddinggemma-300m"
    # LlamaIndex prompt instructions
    query_instruction: str = "task: search result | query: "
    text_instruction: str = "title: none | text: "
    # i have to  set to  "cuda" if we use  a GPU
    device: str = "cpu"

CFG = Config()
