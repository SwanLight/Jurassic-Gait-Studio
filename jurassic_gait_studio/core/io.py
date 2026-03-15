from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .aliases import canonical_joint_name
from .schema import CANONICAL_VISIBLE, OPTIONAL_VISIBLE, ObservationSequence, SkeletonSequence, SpeciesMorphology


def load_species(path: str | Path) -> SpeciesMorphology:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SpeciesMorphology(**payload)


def _discover_joint_triplets(columns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    columns = list(columns)
    discovered: Dict[str, Dict[str, str]] = {}
    for col in columns:
        stem = None
        axis = None
        if col.endswith("_x"):
            stem, axis = col[:-2], "x"
        elif col.endswith("_y"):
            stem, axis = col[:-2], "y"
        elif col.endswith("_score"):
            stem, axis = col[:-6], "score"
        elif col.endswith(".x"):
            stem, axis = col[:-2], "x"
        elif col.endswith(".y"):
            stem, axis = col[:-2], "y"
        elif col.endswith(".score"):
            stem, axis = col[:-6], "score"
        if stem is None:
            continue
        canonical = canonical_joint_name(stem)
        discovered.setdefault(canonical, {})[axis] = col
    return discovered


def _clean_score(score: np.ndarray | None) -> np.ndarray | None:
    if score is None:
        return None
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).any():
        return None
    return score


def _mask_from(arr: np.ndarray, score: np.ndarray | None) -> np.ndarray:
    mask = np.isfinite(arr).all(axis=1)
    score = _clean_score(score)
    if score is not None:
        finite_score = np.isfinite(score)
        if finite_score.any():
            thresh = max(0.02, float(np.nanpercentile(score[finite_score], 5)))
            mask &= (~finite_score) | (score >= thresh)
    return mask


def load_tracks_csv(path: str | Path, fps: float, source_name: str | None = None) -> ObservationSequence:
    df = pd.read_csv(path)
    frame_col = "frame" if "frame" in df.columns else "frame_idx" if "frame_idx" in df.columns else None
    frame_indices = df[frame_col].to_numpy(dtype=int) if frame_col else np.arange(len(df), dtype=int)
    triplets = _discover_joint_triplets(df.columns)
    if "near_ankle" not in triplets and "visible_ankle" in triplets:
        triplets["near_ankle"] = triplets["visible_ankle"]
    if "near_toe_base" not in triplets and "toe_base" in triplets:
        triplets["near_toe_base"] = triplets["toe_base"]
    if "near_toe_tip" not in triplets and "toe_tip" in triplets:
        triplets["near_toe_tip"] = triplets["toe_tip"]
    missing = [name for name in CANONICAL_VISIBLE if name not in triplets]
    if missing:
        raise ValueError(f"Missing required keypoints after alias mapping: {missing}")

    points: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    scores: Dict[str, np.ndarray] = {}
    for name in list(CANONICAL_VISIBLE) + list(OPTIONAL_VISIBLE) + ["pelvis_hint"]:
        if name not in triplets:
            continue
        cols = triplets[name]
        if "x" not in cols or "y" not in cols:
            continue
        arr = df[[cols["x"], cols["y"]]].to_numpy(dtype=float)
        score = _clean_score(df[cols["score"]].to_numpy(dtype=float) if "score" in cols else None)
        points[name] = arr
        masks[name] = _mask_from(arr, score)
        if score is not None:
            scores[name] = score

    return ObservationSequence(
        fps=float(fps),
        frame_indices=frame_indices,
        points=points,
        masks=masks,
        scores=scores,
        source_name=source_name or Path(path).stem,
    )




def _load_skeleton_meta(path: Path) -> Dict[str, object]:
    meta: Dict[str, object] = {"source_path": str(path)}

    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if isinstance(payload.get("meta"), dict):
                    meta.update(payload["meta"])
                else:
                    meta.update(payload)
        except Exception:
            pass

    # Fallback for older runs that were saved before CSV metadata sidecars existed.
    # When opening a dinosaur run viewer, recover the exact target payload from the
    # adjacent session report so rod diameters and angles match the exported GIF.
    session_path = path.parent / "session_report.json"
    if session_path.exists():
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
            stem_lower = path.stem.lower()
            if "dinosaur" in stem_lower and isinstance(session.get("target_payload"), dict):
                meta.setdefault("style", "gait")
                meta["species_payload"] = session["target_payload"]
            elif "bird" in stem_lower:
                summary = session.get("reconstruction_summary", {})
                if isinstance(summary, dict) and isinstance(summary.get("source_species_payload"), dict):
                    meta.setdefault("style", "gait")
                    meta["species_payload"] = summary["source_species_payload"]
        except Exception:
            pass

    meta.setdefault("style", "gait")
    return meta
def load_skeleton_csv(path: str | Path) -> SkeletonSequence:
    df = pd.read_csv(path)
    stems: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        if col.endswith('_x'):
            stems.setdefault(col[:-2], {})['x'] = col
        elif col.endswith('_y'):
            stems.setdefault(col[:-2], {})['y'] = col
        elif col.endswith('_z'):
            stems.setdefault(col[:-2], {})['z'] = col
        elif col.endswith('.x'):
            stems.setdefault(col[:-2], {})['x'] = col
        elif col.endswith('.y'):
            stems.setdefault(col[:-2], {})['y'] = col
        elif col.endswith('.z'):
            stems.setdefault(col[:-2], {})['z'] = col
    joint_names: List[str] = []
    xyz = []
    for name, cols in stems.items():
        if {'x', 'y', 'z'} <= set(cols):
            joint_names.append(name)
            xyz.append(df[[cols['x'], cols['y'], cols['z']]].to_numpy(dtype=float))
    if not joint_names:
        raise ValueError(f"No xyz joint triplets found in skeleton CSV: {path}")
    order = np.argsort(joint_names)
    joint_names = [joint_names[i] for i in order]
    xyz = [xyz[i] for i in order]
    stacked = np.stack(xyz, axis=1)
    return SkeletonSequence(joint_names=joint_names, xyz=stacked, meta=_load_skeleton_meta(Path(path)))


def save_csv_rows(rows: List[Dict[str, float]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_json(payload: Dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_skeleton_csv(sequence: SkeletonSequence, path: str | Path) -> Path:
    path = save_csv_rows(sequence.to_rows(), path)
    meta_sidecar = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    meta_payload = {"meta": dict(sequence.meta or {})}
    try:
        meta_sidecar.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return path


def save_planar_points(points: Dict[str, np.ndarray], path: str | Path) -> Path:
    rows: List[Dict[str, float]] = []
    keys = sorted(points)
    n = next(iter(points.values())).shape[0]
    for t in range(n):
        row: Dict[str, float] = {"frame": float(t)}
        for key in keys:
            row[f"{key}_x"] = float(points[key][t, 0])
            row[f"{key}_z"] = float(points[key][t, 1])
        rows.append(row)
    return save_csv_rows(rows, path)
