from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def angle_of(vector: np.ndarray) -> float:
    return float(math.atan2(vector[1], vector[0]))


def normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.array([1.0, 0.0], dtype=float) if vec.shape == (2,) else np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def perp(vec2: np.ndarray) -> np.ndarray:
    return np.array([-vec2[1], vec2[0]], dtype=float)


def circle_intersections(c0: np.ndarray, r0: float, c1: np.ndarray, r1: float) -> Tuple[np.ndarray, np.ndarray] | None:
    x0, y0 = map(float, c0)
    x1, y1 = map(float, c1)
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return None
    if d > (r0 + r1) or d < abs(r0 - r1):
        return None
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h_sq = max(r0 * r0 - a * a, 0.0)
    h = math.sqrt(h_sq)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    p1 = np.array([xm + rx, ym + ry], dtype=float)
    p2 = np.array([xm - rx, ym - ry], dtype=float)
    return p1, p2


def best_effort_knee(c0: np.ndarray, r0: float, c1: np.ndarray, r1: float) -> Tuple[np.ndarray, np.ndarray]:
    pair = circle_intersections(c0, r0, c1, r1)
    if pair is not None:
        return pair
    direction = c1 - c0
    unit = normalize(direction)
    midpoint = c0 + unit * min(r0, max(0.45 * r0, 0.6 * np.linalg.norm(direction)))
    normal = perp(unit)
    offset = min(r0, r1) * 0.35
    return midpoint + normal * offset, midpoint - normal * offset


def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(direction, dtype=float)
    un = np.linalg.norm(u)
    if un < 1e-9:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
        un = 1.0
    u = u / un
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(u, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    v = np.cross(u, ref)
    v = v / max(np.linalg.norm(v), 1e-9)
    w = np.cross(u, v)
    w = w / max(np.linalg.norm(w), 1e-9)
    return u, v, w
