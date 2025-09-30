import math
from .config import CFG

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _minmax(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clip01((x - lo) / (hi - lo))

def _logistic(x: float, k: float, x0: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if (k * (x - x0)) < 0 else 1.0

def calibrate(raw: float) -> float:
    m = (CFG.calibration_method or "").lower()
    if m == "minmax":
        return _minmax(raw, CFG.cal_min_raw, CFG.cal_max_raw)
    if m == "logistic":
        return _logistic(raw, CFG.cal_logistic_k, CFG.cal_logistic_x0)
    return raw
