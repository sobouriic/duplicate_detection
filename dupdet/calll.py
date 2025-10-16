import math
import numpy as np
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

# -------- Isotonic (monotone piecewise-linear) --------
# We enforce monotonicity via a tiny Pool Adjacent Violators (PAV) step
# then use np.interp for fast evaluation.
_isotonic_x = None
_isotonic_y = None

def _build_isotonic():
    global _isotonic_x, _isotonic_y
    anchors = list(CFG.calibration_anchors or [])
    if not anchors:
        # sane default: identity
        _isotonic_x = np.array([0.0, 1.0], dtype=np.float32)
        _isotonic_y = np.array([0.0, 1.0], dtype=np.float32)
        return

    # sort anchors by raw x
    anchors.sort(key=lambda p: p[0])
    xs = np.array([a for a, _ in anchors], dtype=np.float32)
    ys = np.array([b for _, b in anchors], dtype=np.float32)

    # Pool Adjacent Violators (equal weights) to enforce monotone increasing ys
    blocks = [[i, i, ys[i]] for i in range(len(ys))]
    idx = 0
    while idx < len(blocks) - 1:
        if blocks[idx][2] <= blocks[idx + 1][2]:
            idx += 1
        else:
            s = blocks[idx][0]
            e = blocks[idx + 1][1]
            # mean over merged block
            val = ys[s:e + 1].mean()
            blocks[idx][0] = s
            blocks[idx][1] = e
            blocks[idx][2] = val
            blocks.pop(idx + 1)
            if idx > 0:
                idx -= 1

    # Write back the piecewise-constant monotone solution, then keep breakpoints
    iso_y = np.zeros_like(ys)
    iso_x = xs.copy()
    for s, e, val in blocks:
        iso_y[s:e + 1] = val

    # Optionally, compress consecutive duplicates to keep the interp array small
    keep = [0]
    for i in range(1, len(iso_x)):
        if not (abs(iso_y[i] - iso_y[i - 1]) < 1e-12 and abs(iso_x[i] - iso_x[i - 1]) < 1e-12):
            keep.append(i)
    _isotonic_x = iso_x[keep]
    _isotonic_y = iso_y[keep]

# Build once at import
_build_isotonic()

def _iso_eval(x: float) -> float:
    # fast linear interpolation on the isotonic curve
    xarr = np.array([x], dtype=np.float32)
    y = np.interp(xarr, _isotonic_x, _isotonic_y)
    return float(_clip01(float(y[0])))

def calibrate(raw: float) -> float:
    # identical vectors should map to 1.0
    if abs(raw - 1.0) < 1e-6:
        return 1.0

    m = (CFG.calibration_method or "").lower()
    if m == "minmax":
        return _minmax(raw, CFG.cal_min_raw, CFG.cal_max_raw)
    if m == "logistic":
        return _logistic(raw, CFG.cal_logistic_k, CFG.cal_logistic_x0)
    if m == "isotonic":
        return _iso_eval(raw)
    return raw
