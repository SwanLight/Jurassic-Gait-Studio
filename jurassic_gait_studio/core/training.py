from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def fit_phase_depth_prior(samples: List[Dict[str, np.ndarray]], joint_names: List[str]) -> Dict[str, Dict[str, List[float]]]:
    features = []
    targets = {name: [] for name in joint_names}
    for sample in samples:
        phase = sample["phase"]
        for t, ph in enumerate(phase):
            features.append([1.0, np.sin(ph), np.cos(ph)])
            for name in joint_names:
                targets[name].append(float(sample["lateral_depth"][name][t]))
    x = np.asarray(features, dtype=float)
    result: Dict[str, Dict[str, List[float]]] = {}
    for name in joint_names:
        y = np.asarray(targets[name], dtype=float)
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        result[name] = {"features": ["bias", "sin_phase", "cos_phase"], "weights": coef.tolist()}
    return result


def load_teacher_sample(path: str | Path) -> Dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    phase = np.asarray(payload["phase"], dtype=float)
    lateral_depth = {name: np.asarray(values, dtype=float) for name, values in payload["lateral_depth"].items()}
    return {"phase": phase, "lateral_depth": lateral_depth}
