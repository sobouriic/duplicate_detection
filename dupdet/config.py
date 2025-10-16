from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    db_path: Path = Path("./dupdet.sqlite")

    model_name: str = "BAAI/bge-m3"
    query_instruction: str = "query: "
    text_instruction: str = "passage: "
    device: str = "cpu"

    translate_to_english: bool = False

    # === Calibration controls ===
    # "none" | "minmax" | "logistic"
    calibration_method: str = "logistic"

    # for minmax (only used if calibration_method == "minmax")
    cal_min_raw: float = 0.55
    cal_max_raw: float = 0.95

    cal_logistic_k: float = 15
    cal_logistic_x0: float = 0.71

CFG = Config()
