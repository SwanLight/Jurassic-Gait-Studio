from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import savgol_filter


def interpolate_missing(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(float)
    xs = np.arange(arr.shape[0])
    for dim in range(arr.shape[1]):
        good = mask & np.isfinite(arr[:, dim])
        if good.sum() < 2:
            fill = np.nanmean(arr[:, dim]) if np.isfinite(arr[:, dim]).any() else 0.0
            out[:, dim] = fill
            continue
        out[:, dim] = np.interp(xs, xs[good], arr[good, dim])
    return out


def smooth_xy(arr: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    if arr.shape[0] < 3:
        return arr.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > arr.shape[0]:
        window = arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1
    if window < 3:
        return arr.copy()
    out = np.empty_like(arr, dtype=float)
    for dim in range(arr.shape[1]):
        out[:, dim] = savgol_filter(arr[:, dim], window_length=window, polyorder=min(polyorder, window - 1), mode="interp")
    return out


def smooth_dict(points: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], window: int = 7) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in points.items():
        interp = interpolate_missing(value, masks[key])
        out[key] = smooth_xy(interp, window=window)
    return out
