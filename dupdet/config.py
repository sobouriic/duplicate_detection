from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    db_path: Path = Path("./dupdet.sqlite")

    model_name: str = "BAAI/bge-m3"
    query_instruction: str = ""
    text_instruction: str = ""
    device: str = "cpu"

    translate_to_english: bool = False

    # Choices: "minmax" or "logistic" or "" (disabled)
    calibration_method: str = "minmax"

    # Min-Max parameters (map raw ~[min_raw, max_raw] -> [0, 1])
    cal_min_raw: float = 0.45
    cal_max_raw: float = 0.95

    # Logistic parameters  (y = 1 / (1 + exp(-k*(x - x0))))
    cal_logistic_k: float = 10.0
    cal_logistic_x0: float = 0.75

CFG = Config()
