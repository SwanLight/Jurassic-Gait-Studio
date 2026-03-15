from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .io import load_species, load_tracks_csv, save_json
from .reconstruction import PlanarReconstructionResult, reconstruct_planar_cycle
from .schema import PlanarKinematics, SpeciesMorphology


@dataclass
class BirdLibraryEntry:
    name: str
    track_path: Path
    species_path: Path
    species: SpeciesMorphology
    reconstruction: PlanarReconstructionResult
    weight_bias: float = 1.0


@dataclass
class FusionReport:
    target_species: str
    contributions: List[Dict[str, float | str | Dict[str, float]]]
    normalized_weights: Dict[str, float]
    fused_source_name: str
    n_reference_birds: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_species": self.target_species,
            "contributions": self.contributions,
            "normalized_weights": self.normalized_weights,
            "fused_source_name": self.fused_source_name,
            "n_reference_birds": self.n_reference_birds,
        }


_MANIFEST_KEYS = {"entries", "birds", "clips", "library"}


def _resample_xy(arr: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == n_frames:
        return arr.copy()
    old_t = np.linspace(0.0, 1.0, arr.shape[0], endpoint=False)
    new_t = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    x = np.interp(new_t, old_t, arr[:, 0], period=1.0)
    y = np.interp(new_t, old_t, arr[:, 1], period=1.0)
    return np.column_stack([x, y])


def _resample_scalar(arr: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == n_frames:
        return arr.copy()
    old_t = np.linspace(0.0, 1.0, arr.shape[0], endpoint=False)
    new_t = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    return np.interp(new_t, old_t, arr, period=1.0)


def _infer_species_name_from_track(path: str | Path, species_names: Iterable[str] | None = None) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"\.analysis$", "", stem)
    stem = re.sub(r"[^a-z0-9_]+", "_", stem)
    if species_names is not None:
        matches = [name for name in species_names if name.lower() in stem]
        if matches:
            return max(matches, key=len).lower()
    tokens = [tok for tok in stem.split("_") if tok]
    for tok in tokens:
        if tok.isalpha():
            return tok
    if tokens:
        return re.sub(r"\d+$", "", tokens[0]) or tokens[0]
    return stem


def _load_manifest_entries(manifest_path: str | Path | None) -> List[dict]:
    if manifest_path is None:
        return []
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        for key in _MANIFEST_KEYS:
            if key in payload and isinstance(payload[key], list):
                return list(payload[key])
    raise ValueError("Bird library manifest must be a list, or contain one of: entries/birds/clips/library")


def _discover_track_paths(track_paths: Sequence[str | Path] | None = None, track_dir: str | Path | None = None) -> List[Path]:
    discovered: List[Path] = []
    if track_dir is not None:
        discovered.extend(sorted(Path(track_dir).glob("*.csv")))
    if track_paths is not None:
        discovered.extend(Path(p) for p in track_paths)
    unique: List[Path] = []
    seen = set()
    for path in discovered:
        path = path.resolve()
        if path.suffix.lower() != ".csv":
            continue
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _species_map(species_dir: str | Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(Path(species_dir).glob("*.json")):
        out[path.stem.lower()] = path.resolve()
    return out


def morphology_feature_vector(species: SpeciesMorphology) -> np.ndarray:
    leg = max(species.thigh + species.shank + species.metatarsus + species.foot, 1e-6)
    torso = max(species.torso, 1e-6)
    return np.asarray(
        [
            species.thigh / leg,
            species.shank / leg,
            species.metatarsus / leg,
            species.foot / leg,
            species.torso / leg,
            species.neck / torso,
            species.metatarsus / max(species.shank, 1e-6),
            (species.shank + species.metatarsus) / max(species.thigh, 1e-6),
            species.hip_width_ratio,
            species.trunk_depth_ratio,
        ],
        dtype=float,
    )


def posture_feature_vector(species: SpeciesMorphology) -> np.ndarray:
    neck_mid = 0.5 * (species.neck_min_pitch_deg + species.neck_max_pitch_deg)
    head_mid = 0.5 * (species.head_min_pitch_deg + species.head_max_pitch_deg)
    return np.asarray(
        [
            species.trunk_pitch_bias_deg / 25.0,
            species.tail_pitch_deg / 25.0,
            neck_mid / 60.0,
            head_mid / 40.0,
            getattr(species, "level_head_strength", 0.7),
        ],
        dtype=float,
    )


def gait_feature_vector(planar: PlanarKinematics, species: SpeciesMorphology) -> np.ndarray:
    pts = planar.joints_planar_m
    h = max(species.height_m, 1e-6)
    trunk = pts["torso_front"] - pts["torso_back"]
    trunk_pitch = np.degrees(np.arctan2(trunk[:, 1], np.maximum(np.abs(trunk[:, 0]), 1e-6)))
    near_step = float(np.nanmax(pts["near_toe_tip"][:, 0]) - np.nanmin(pts["near_toe_tip"][:, 0])) / h
    far_step = float(np.nanmax(pts["far_toe_tip"][:, 0]) - np.nanmin(pts["far_toe_tip"][:, 0])) / h
    head = pts.get("head", pts["torso_front"])
    head_vec = head - pts["torso_front"]
    head_pitch = np.degrees(np.arctan2(head_vec[:, 1], np.maximum(np.abs(head_vec[:, 0]), 1e-6)))
    return np.asarray(
        [
            0.5 * (near_step + far_step),
            float(np.mean(planar.stance_masks["near"])),
            float(np.mean(planar.stance_masks["far"])),
            float(np.median(trunk_pitch)) / 40.0,
            float(np.median(head_pitch)) / 60.0,
        ],
        dtype=float,
    )


def blend_species_morphologies(entries: Sequence[BirdLibraryEntry], weights: Dict[str, float], out_name: str = "fused_bird") -> SpeciesMorphology:
    if not entries:
        raise ValueError("No bird library entries to blend")
    total = sum(float(weights[e.name]) for e in entries)
    if total <= 0:
        raise ValueError("Bird weights must sum to a positive value")

    def w(entry: BirdLibraryEntry) -> float:
        return float(weights[entry.name]) / total

    limb_names = sorted({k for e in entries for k in e.species.limb_lengths_m})
    limb_lengths = {
        name: float(sum(w(e) * float(e.species.limb_lengths_m.get(name, 0.0)) for e in entries)) for name in limb_names
    }

    return SpeciesMorphology(
        name=out_name,
        species_group="bird",
        mass_kg=float(sum(w(e) * e.species.mass_kg for e in entries)),
        height_m=float(sum(w(e) * e.species.height_m for e in entries)),
        limb_lengths_m=limb_lengths,
        hip_width_ratio=float(sum(w(e) * e.species.hip_width_ratio for e in entries)),
        pelvis_height_ratio=float(sum(w(e) * e.species.pelvis_height_ratio for e in entries)),
        pelvis_length_ratio=float(sum(w(e) * e.species.pelvis_length_ratio for e in entries)),
        trunk_depth_ratio=float(sum(w(e) * e.species.trunk_depth_ratio for e in entries)),
        head_length_ratio=float(sum(w(e) * e.species.head_length_ratio for e in entries)),
        body_yaw_deg=float(sum(w(e) * e.species.body_yaw_deg for e in entries)),
        tail_drop_deg=float(sum(w(e) * e.species.tail_drop_deg for e in entries)),
        neck_blend=float(sum(w(e) * e.species.neck_blend for e in entries)),
        neck_min_pitch_deg=float(sum(w(e) * e.species.neck_min_pitch_deg for e in entries)),
        neck_max_pitch_deg=float(sum(w(e) * e.species.neck_max_pitch_deg for e in entries)),
        neck_forward_bias=float(sum(w(e) * e.species.neck_forward_bias for e in entries)),
        neck_up_bias=float(sum(w(e) * e.species.neck_up_bias for e in entries)),
        head_blend=float(sum(w(e) * e.species.head_blend for e in entries)),
        head_min_pitch_deg=float(sum(w(e) * e.species.head_min_pitch_deg for e in entries)),
        head_max_pitch_deg=float(sum(w(e) * e.species.head_max_pitch_deg for e in entries)),
        head_forward_bias=float(sum(w(e) * e.species.head_forward_bias for e in entries)),
        head_up_bias=float(sum(w(e) * e.species.head_up_bias for e in entries)),
        tail_blend=float(sum(w(e) * e.species.tail_blend for e in entries)),
        tail_pitch_deg=float(sum(w(e) * e.species.tail_pitch_deg for e in entries)),
        trunk_pitch_bias_deg=float(sum(w(e) * e.species.trunk_pitch_bias_deg for e in entries)),
        level_head_strength=float(sum(w(e) * e.species.level_head_strength for e in entries)),
        level_head_pitch_deg=float(sum(w(e) * e.species.level_head_pitch_deg for e in entries)),
        head_tip_tracking=float(sum(w(e) * e.species.head_tip_tracking for e in entries)),
    )


def discover_bird_library(
    fps: float,
    species_dir: str | Path,
    track_paths: Sequence[str | Path] | None = None,
    track_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> List[BirdLibraryEntry]:
    species_lookup = _species_map(species_dir)
    manifest_entries = _load_manifest_entries(manifest_path)
    manifest_base = Path(manifest_path).expanduser().resolve().parent if manifest_path is not None else None
    entries: List[BirdLibraryEntry] = []

    for item in manifest_entries:
        raw_track_path = Path(item["tracks_csv"]).expanduser()
        track_path = ((manifest_base / raw_track_path) if manifest_base is not None and not raw_track_path.is_absolute() else raw_track_path).resolve()
        raw_species_path = item.get("species_json")
        species_path = Path(raw_species_path).expanduser() if raw_species_path else species_lookup[_infer_species_name_from_track(track_path, species_lookup.keys())]
        species_path = ((manifest_base / species_path) if manifest_base is not None and not Path(species_path).is_absolute() else Path(species_path)).resolve()
        species = load_species(species_path)
        if species.species_group != "bird":
            continue
        obs = load_tracks_csv(track_path, fps=float(item.get("fps", fps)))
        result = reconstruct_planar_cycle(obs, species)
        entries.append(
            BirdLibraryEntry(
                name=str(item.get("name") or track_path.stem),
                track_path=track_path,
                species_path=species_path,
                species=species,
                reconstruction=result,
                weight_bias=float(item.get("weight_bias", 1.0)),
            )
        )

    for track_path in _discover_track_paths(track_paths=track_paths, track_dir=track_dir):
        if any(track_path == e.track_path for e in entries):
            continue
        guessed = _infer_species_name_from_track(track_path, species_lookup.keys())
        if guessed not in species_lookup:
            raise ValueError(f"Could not match track file '{track_path.name}' to a species JSON inside {species_dir}")
        species_path = species_lookup[guessed]
        species = load_species(species_path)
        if species.species_group != "bird":
            continue
        obs = load_tracks_csv(track_path, fps=float(fps))
        result = reconstruct_planar_cycle(obs, species)
        entries.append(
            BirdLibraryEntry(
                name=track_path.stem,
                track_path=track_path,
                species_path=species_path,
                species=species,
                reconstruction=result,
            )
        )

    if not entries:
        raise ValueError("No usable bird reference clips were discovered")
    return entries


def compute_bird_weights(entries: Sequence[BirdLibraryEntry], target_species: SpeciesMorphology, temperature: float = 0.18) -> FusionReport:
    if not entries:
        raise ValueError("No bird reference entries available")
    target_morph = morphology_feature_vector(target_species)
    target_posture = posture_feature_vector(target_species)
    raw_scores = []
    rows: List[Dict[str, float | str | Dict[str, float]]] = []
    for entry in entries:
        morph_dist = float(np.linalg.norm((morphology_feature_vector(entry.species) - target_morph) * np.asarray([2.5, 2.2, 3.5, 1.0, 2.0, 2.0, 3.0, 2.8, 0.8, 0.8])))
        posture_dist = float(np.linalg.norm((posture_feature_vector(entry.species) - target_posture) * np.asarray([1.0, 0.9, 0.8, 0.8, 1.2])))
        gait_dist = float(np.linalg.norm((gait_feature_vector(entry.reconstruction.planar, entry.species) - np.asarray([
            0.78 * target_species.metatarsus / max(target_species.height_m, 1e-6),
            0.52,
            0.52,
            target_species.trunk_pitch_bias_deg / 40.0,
            target_species.level_head_pitch_deg / 60.0,
        ])) * np.asarray([1.8, 0.8, 0.8, 0.6, 0.8])))
        total_dist = 0.66 * morph_dist + 0.20 * posture_dist + 0.14 * gait_dist
        biased = total_dist / max(entry.weight_bias, 1e-6)
        raw = math.exp(-biased / max(temperature, 1e-6))
        raw_scores.append(raw)
        rows.append(
            {
                "name": entry.name,
                "species": entry.species.name,
                "morph_distance": morph_dist,
                "posture_distance": posture_dist,
                "gait_distance": gait_dist,
                "compatibility": 1.0 / (1.0 + total_dist),
                "raw_score": raw,
            }
        )
    raw_scores = np.asarray(raw_scores, dtype=float)
    weights = raw_scores / np.maximum(raw_scores.sum(), 1e-9)
    uniform_mix = 0.12
    weights = (1.0 - uniform_mix) * weights + uniform_mix / float(len(entries))
    norm_weights = {entry.name: float(weights[i]) for i, entry in enumerate(entries)}
    for i, row in enumerate(rows):
        row["weight"] = float(weights[i])
    rows = sorted(rows, key=lambda x: float(x["weight"]), reverse=True)
    fused_name = "+".join(f"{row['species']}:{row['weight']:.2f}" for row in rows)
    return FusionReport(
        target_species=target_species.name,
        contributions=rows,
        normalized_weights=norm_weights,
        fused_source_name=fused_name,
        n_reference_birds=len(entries),
    )


def _canonical_keys(entries: Sequence[BirdLibraryEntry]) -> List[str]:
    common = set(entries[0].reconstruction.planar.joints_planar_m)
    for entry in entries[1:]:
        common &= set(entry.reconstruction.planar.joints_planar_m)
    preferred = [
        "pelvis_center",
        "torso_back",
        "torso_front",
        "head",
        "tail_base",
        "tail_tip",
        "near_true_knee",
        "near_ankle",
        "near_toe_base",
        "near_toe_tip",
        "far_true_knee",
        "far_ankle",
        "far_toe_base",
        "far_toe_tip",
    ]
    return [k for k in preferred if k in common]


def fuse_planar_kinematics(
    entries: Sequence[BirdLibraryEntry],
    weights: Dict[str, float],
    output_height_m: float,
    n_frames: int = 101,
) -> PlanarKinematics:
    if not entries:
        raise ValueError("No bird reference entries available for fusion")
    keys = _canonical_keys(entries)
    if not keys:
        raise ValueError("No shared planar keys available across bird references")

    points_unit = {k: np.zeros((n_frames, 2), dtype=float) for k in keys}
    near_stance = np.zeros(n_frames, dtype=float)
    far_stance = np.zeros(n_frames, dtype=float)

    total = sum(float(weights[e.name]) for e in entries)
    if total <= 0:
        raise ValueError("Bird fusion weights must sum to a positive value")

    for entry in entries:
        weight = float(weights[entry.name]) / total
        h = max(entry.species.height_m, 1e-6)
        planar = entry.reconstruction.planar
        for key in keys:
            points_unit[key] += weight * _resample_xy(planar.joints_planar_m[key] / h, n_frames)
        near_stance += weight * _resample_scalar(planar.stance_masks["near"].astype(float), n_frames)
        far_stance += weight * _resample_scalar(planar.stance_masks["far"].astype(float), n_frames)

    points_m = {k: v * float(output_height_m) for k, v in points_unit.items()}
    if "pelvis_center" in points_m:
        pelvis0 = points_m["pelvis_center"][0].copy()
        for key in points_m:
            points_m[key] = points_m[key].copy()
            points_m[key][:, 0] -= pelvis0[0]

    ground_candidates = []
    for key in ["near_toe_tip", "far_toe_tip", "near_toe_base", "far_toe_base"]:
        if key in points_m:
            ground_candidates.append(points_m[key][:, 1])
    if ground_candidates:
        all_ground = np.concatenate(ground_candidates)
        ground_shift = float(np.nanpercentile(all_ground, 3))
        for key in points_m:
            points_m[key] = points_m[key].copy()
            points_m[key][:, 1] -= ground_shift

    phase = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    return PlanarKinematics(
        phase=phase,
        direction=1,
        scale_m_per_px=float("nan"),
        ground_y_px=float("nan"),
        joints_planar_m=points_m,
        joints_px={},
        stance_masks={
            "near": near_stance >= 0.5,
            "far": far_stance >= 0.5,
        },
        meta={
            "fused": True,
            "weights": {k: float(v) for k, v in weights.items()},
            "n_reference_birds": len(entries),
        },
    )


def render_weight_plot(report: FusionReport, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["species"]) for row in report.contributions]
    weights = [float(row["weight"]) for row in report.contributions]
    compat = [float(row["compatibility"]) for row in report.contributions]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=180)
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, weights, width=0.58)
    for i, (w, c) in enumerate(zip(weights, compat)):
        ax.text(i, w + 0.015, f"{w:.2f}\ncompat={c:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(0.35, max(weights) + 0.12))
    ax.set_ylabel("Contribution weight")
    ax.set_title(f"Bird reference weighting for target: {report.target_species}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fusion_report(report: FusionReport, out_path: str | Path) -> Path:
    return save_json(report.to_dict(), out_path)
