from __future__ import annotations

from typing import Dict

import numpy as np

from .schema import SpeciesMorphology


def infer_direction(points: Dict[str, np.ndarray]) -> int:
    torso = points["torso_front"] - points["torso_back"]
    mean_dx = float(np.nanmedian(torso[:, 0]))
    if abs(mean_dx) > 1e-6:
        return 1 if mean_dx >= 0 else -1
    tail_dx = float(points["torso_front"][-1, 0] - points["torso_front"][0, 0])
    return 1 if tail_dx >= 0 else -1


def estimate_scale_m_per_px(points: Dict[str, np.ndarray], species: SpeciesMorphology) -> float:
    candidates = []

    def add_candidate(a: str, b: str, target: float) -> None:
        if a in points and b in points and target > 0:
            dist = np.linalg.norm(points[a] - points[b], axis=1)
            dist = dist[np.isfinite(dist)]
            if dist.size:
                px = float(np.nanmedian(dist))
                if px > 1e-6:
                    candidates.append(target / px)

    add_candidate("near_ankle", "near_toe_base", species.metatarsus)
    add_candidate("near_toe_base", "near_toe_tip", species.foot)
    add_candidate("far_ankle", "far_toe_base", species.metatarsus)
    add_candidate("far_toe_base", "far_toe_tip", species.foot)
    add_candidate("torso_back", "torso_front", species.torso)
    add_candidate("tail_base", "tail_tip", max(species.tail * 0.45, 1e-6))
    if not candidates:
        return species.height_m / 500.0
    return float(np.median(np.asarray(candidates, dtype=float)))


def estimate_ground_y_px(points: Dict[str, np.ndarray]) -> float:
    vals = []
    for name in ["near_toe_tip", "near_toe_base", "far_toe_tip", "far_toe_base"]:
        if name in points:
            vals.append(points[name][:, 1])
    toe_samples = np.concatenate(vals) if vals else np.array([0.0])
    toe_samples = toe_samples[np.isfinite(toe_samples)]
    if toe_samples.size == 0:
        return 0.0
    return float(np.percentile(toe_samples, 96))


def make_phase(n_frames: int) -> np.ndarray:
    if n_frames <= 1:
        return np.zeros(1, dtype=float)
    return np.linspace(0.0, 2.0 * np.pi, num=n_frames, endpoint=False, dtype=float)


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return np.zeros_like(x)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order / max(1, x.size - 1)


def _close_small_gaps(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool).copy()
    if mask.size < 3:
        return mask
    for _ in range(2):
        inner = mask.copy()
        inner[1:-1] |= mask[:-2] & mask[2:]
        mask = inner
    return mask


def _expand_neighbors(mask: np.ndarray) -> np.ndarray:
    if mask.size < 3:
        return mask.astype(bool)
    m = mask.astype(int)
    support = np.convolve(m, np.ones(3, dtype=int), mode="same")
    return (support >= 2) | mask


def estimate_stance_mask_side(points_planar: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    toe = points_planar[f"{prefix}_toe_tip"]
    base = points_planar[f"{prefix}_toe_base"]
    ankle = points_planar[f"{prefix}_ankle"]
    n = toe.shape[0]
    if n < 5:
        return np.ones(n, dtype=bool)

    foot_z = 0.55 * toe[:, 1] + 0.45 * base[:, 1]
    toe_speed = np.abs(np.gradient(toe[:, 0])) + 0.45 * np.abs(np.gradient(base[:, 0]))
    ankle_speed = np.abs(np.gradient(ankle[:, 0]))

    score = 0.55 * _rank01(foot_z) + 0.30 * _rank01(toe_speed) + 0.15 * _rank01(ankle_speed)
    thresh = float(np.quantile(score, 0.42))
    mask = score <= thresh
    low_and_slow = (foot_z <= np.quantile(foot_z, 0.55)) & (toe_speed <= np.quantile(toe_speed, 0.60))
    mask |= low_and_slow
    mask = _close_small_gaps(mask)
    mask = _expand_neighbors(mask)

    duty = float(mask.mean())
    if duty < 0.18:
        mask = score <= float(np.quantile(score, 0.50))
        mask = _close_small_gaps(mask)
        mask = _expand_neighbors(mask)
    elif duty > 0.72:
        mask = score <= float(np.quantile(score, 0.30))
        mask = _close_small_gaps(mask)

    if mask.sum() == 0:
        mask[np.argmin(score)] = True
    return mask.astype(bool)
