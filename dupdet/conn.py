from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple  

@dataclass(frozen=True)
class Config:
    db_path: Path = Path("./dupdet.sqlite")

    model_name: str = "BAAI/bge-m3"
    query_instruction: str = "query: "
    text_instruction: str = "passage: "
    device: str = "cpu"

    translate_to_english: bool = False

    # "none" | "minmax" | "logistic" | "isotonic"
    calibration_method: str = "isotonic"

    cal_min_raw: float = 0.55
    cal_max_raw: float = 0.95
    cal_logistic_k: float = 15.6
    cal_logistic_x0: float = 0.711

    # anchors for isotonic (raw -> target)
    calibration_anchors: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.00,        0.00),
        (0.553433895, 0.030716175),
        (0.617143035, 0.08653682),
        (0.621735096, 0.19181731),
        (0.641734958, 0.08118228),
        (0.656753778, 0.18967533),
        (0.659280658, 0.43775898),
        (1.00,        1.00),
    ])

CFG = Config()
