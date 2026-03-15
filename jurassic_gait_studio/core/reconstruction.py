from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .gait import estimate_ground_y_px, estimate_scale_m_per_px, estimate_stance_mask_side, infer_direction, make_phase
from .geometry import best_effort_knee, normalize, perp
from .schema import ObservationSequence, PlanarKinematics, SpeciesMorphology
from .signals import smooth_dict, smooth_xy


@dataclass
class PlanarReconstructionResult:
    planar: PlanarKinematics
    quality_flags: List[str]


def _choose_downward_normal(u: np.ndarray, anchor: np.ndarray, ankle_mid: np.ndarray) -> np.ndarray:
    n = perp(u)
    if np.dot(n, ankle_mid - anchor) < 0.0:
        n = -n
    return normalize(n)


def infer_pelvis_center_px(points_px: Dict[str, np.ndarray], species: SpeciesMorphology, scale_m_per_px: float, direction: int) -> np.ndarray:
    torso_back = points_px["torso_back"]
    torso_front = points_px["torso_front"]
    tail_base = points_px["tail_base"]
    ankle_near = points_px["near_ankle"]
    ankle_far = points_px.get("far_ankle", ankle_near)
    thigh_px = species.thigh / max(scale_m_per_px, 1e-8)

    n_frames = torso_back.shape[0]
    pelvis = np.zeros((n_frames, 2), dtype=float)
    ankle_mid = 0.5 * (ankle_near + ankle_far)
    for t in range(n_frames):
        body = torso_front[t] - torso_back[t]
        torso_len = max(float(np.linalg.norm(body)), 1e-6)
        u = normalize(body)
        n = _choose_downward_normal(u, torso_back[t], ankle_mid[t])
        body_anchor = torso_back[t] + u * (0.24 * torso_len) + n * (0.14 * torso_len)
        tail_anchor = tail_base[t] + u * (0.18 * torso_len) + n * (0.10 * torso_len)
        pelvis[t] = 0.78 * body_anchor + 0.22 * tail_anchor

    if "pelvis_hint" in points_px:
        pelvis = 0.55 * pelvis + 0.45 * points_px["pelvis_hint"]

    pelvis = smooth_xy(pelvis, window=9)

    for _ in range(3):
        near_knee = _solve_single_knee_path(pelvis, points_px, species, scale_m_per_px, direction, side="near")
        far_knee = _solve_single_knee_path(pelvis, points_px, species, scale_m_per_px, direction, side="far")
        knee_mid = 0.5 * (near_knee + far_knee)
        desired = np.zeros_like(pelvis)
        desired[:, 0] = knee_mid[:, 0] - direction * 0.48 * thigh_px
        desired[:, 1] = knee_mid[:, 1] - 0.44 * thigh_px
        pelvis = smooth_xy(0.74 * pelvis + 0.26 * desired, window=9)

    return pelvis


def _score_knee_candidate(
    candidate: np.ndarray,
    pelvis: np.ndarray,
    ankle: np.ndarray,
    toe_base: np.ndarray,
    hint: np.ndarray | None,
    prev: np.ndarray | None,
    direction: int,
    thigh_px: float,
    shank_px: float,
) -> float:
    score = 0.0
    drop = candidate[1] - pelvis[1]
    ankle_drop = ankle[1] - candidate[1]
    knee_forward = direction * (candidate[0] - pelvis[0])
    shank_back = direction * (candidate[0] - ankle[0])
    foot_forward = direction * (toe_base[0] - ankle[0])

    min_drop = 0.14 * thigh_px
    pref_drop = 0.34 * thigh_px
    max_drop = 0.82 * thigh_px
    if drop < min_drop:
        score += 12.0 * (min_drop - drop)
    score += 1.25 * abs(drop - pref_drop)
    if drop > max_drop:
        score += 4.0 * (drop - max_drop)
    if ankle_drop < 0.10 * shank_px:
        score += 14.0 * (0.10 * shank_px - ankle_drop)

    if knee_forward < 0.06 * thigh_px:
        score += 10.0 * (0.06 * thigh_px - knee_forward)
    if shank_back < 0.04 * shank_px:
        score += 11.0 * (0.04 * shank_px - shank_back)
    if foot_forward < 0.0:
        score += 8.0 * abs(foot_forward)

    if hint is not None and np.isfinite(hint).all():
        score += 0.45 * float(np.linalg.norm(candidate - hint))
    if prev is not None:
        score += 0.50 * float(np.linalg.norm(candidate - prev))
    score += 0.03 * abs(candidate[0] - 0.5 * (pelvis[0] + ankle[0]))
    return float(score)


def _solve_single_knee_path(
    pelvis_px: np.ndarray,
    points_px: Dict[str, np.ndarray],
    species: SpeciesMorphology,
    scale_m_per_px: float,
    direction: int,
    side: str,
) -> np.ndarray:
    thigh_px = species.thigh / max(scale_m_per_px, 1e-8)
    shank_px = species.shank / max(scale_m_per_px, 1e-8)
    ankle = points_px[f"{side}_ankle"]
    toe_base = points_px[f"{side}_toe_base"]
    hint = points_px.get(f"{side}_knee_hint")

    n = pelvis_px.shape[0]
    all_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for t in range(n):
        c1, c2 = best_effort_knee(pelvis_px[t], thigh_px, ankle[t], shank_px)
        all_candidates.append((c1, c2))

    dp = np.full((n, 2), np.inf, dtype=float)
    back = np.zeros((n, 2), dtype=int)
    for k in range(2):
        dp[0, k] = _score_knee_candidate(
            all_candidates[0][k],
            pelvis_px[0],
            ankle[0],
            toe_base[0],
            hint[0] if hint is not None else None,
            None,
            direction,
            thigh_px,
            shank_px,
        )
    for t in range(1, n):
        for k in range(2):
            unary = _score_knee_candidate(
                all_candidates[t][k],
                pelvis_px[t],
                ankle[t],
                toe_base[t],
                hint[t] if hint is not None else None,
                None,
                direction,
                thigh_px,
                shank_px,
            )
            prev_scores = []
            for j in range(2):
                prev_scores.append(dp[t - 1, j] + unary + 0.55 * float(np.linalg.norm(all_candidates[t][k] - all_candidates[t - 1][j])))
            best_prev = int(np.argmin(prev_scores))
            dp[t, k] = prev_scores[best_prev]
            back[t, k] = best_prev
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    knees = np.stack([all_candidates[t][path[t]] for t in range(n)], axis=0)
    return smooth_xy(knees, window=7)


def recover_bilateral_knees(points_px: Dict[str, np.ndarray], species: SpeciesMorphology, scale_m_per_px: float, direction: int, pelvis_px: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    out["near_true_knee"] = _solve_single_knee_path(pelvis_px, points_px, species, scale_m_per_px, direction, "near")
    if "far_ankle" in points_px and "far_toe_base" in points_px and "far_toe_tip" in points_px:
        out["far_true_knee"] = _solve_single_knee_path(pelvis_px, points_px, species, scale_m_per_px, direction, "far")
    else:
        out["far_true_knee"] = smooth_xy(np.roll(out["near_true_knee"], shift=max(1, len(out["near_true_knee"]) // 2), axis=0), window=7)
    return out


def convert_pixels_to_planar_m(points_px: Dict[str, np.ndarray], scale_m_per_px: float, ground_y_px: float, direction: int, pelvis_px: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    pelvis0 = pelvis_px[0].copy()
    for key, arr in points_px.items():
        x = direction * (arr[:, 0] - pelvis0[0]) * scale_m_per_px
        z = (ground_y_px - arr[:, 1]) * scale_m_per_px
        out[key] = np.column_stack([x, z])
    x = direction * (pelvis_px[:, 0] - pelvis0[0]) * scale_m_per_px
    z = (ground_y_px - pelvis_px[:, 1]) * scale_m_per_px
    out["pelvis_center"] = np.column_stack([x, z])
    return out


def reconstruct_planar_cycle(observation: ObservationSequence, species: SpeciesMorphology, smooth_window: int = 7) -> PlanarReconstructionResult:
    smooth_px = smooth_dict(observation.points, observation.masks, window=smooth_window)
    direction = infer_direction(smooth_px)
    scale = estimate_scale_m_per_px(smooth_px, species)
    ground_y = estimate_ground_y_px(smooth_px)
    pelvis_px = infer_pelvis_center_px(smooth_px, species, scale, direction)
    knees_px = recover_bilateral_knees(smooth_px, species, scale, direction, pelvis_px)
    smooth_px.update(knees_px)
    if "far_ankle" not in smooth_px:
        shift = max(1, observation.n_frames // 2)
        for name in ["ankle", "toe_base", "toe_tip"]:
            smooth_px[f"far_{name}"] = smooth_xy(np.roll(smooth_px[f"near_{name}"], shift=shift, axis=0), window=7)
    planar_m = convert_pixels_to_planar_m(smooth_px, scale, ground_y, direction, pelvis_px)
    phase = make_phase(observation.n_frames)
    stance_masks = {
        "near": estimate_stance_mask_side(planar_m, "near"),
        "far": estimate_stance_mask_side(planar_m, "far") if "far_toe_tip" in planar_m else np.roll(estimate_stance_mask_side(planar_m, "near"), shift=max(1, observation.n_frames // 2)),
    }
    quality_flags: List[str] = []
    knee_drop = planar_m["pelvis_center"][:, 1] - planar_m["near_true_knee"][:, 1]
    if np.nanmedian(knee_drop) < 0.05 * species.thigh:
        quality_flags.append("near_knee_too_high")
    if "far_true_knee" in planar_m:
        far_drop = planar_m["pelvis_center"][:, 1] - planar_m["far_true_knee"][:, 1]
        if np.nanmedian(far_drop) < 0.05 * species.thigh:
            quality_flags.append("far_knee_too_high")
    if "far_ankle" not in observation.points:
        quality_flags.append("far_leg_synthesized")

    meta = {
        "bilateral_source": bool("far_ankle" in observation.points),
        "pelvis_center_px": pelvis_px,
    }
    return PlanarReconstructionResult(
        planar=PlanarKinematics(
            phase=phase,
            direction=direction,
            scale_m_per_px=scale,
            ground_y_px=ground_y,
            joints_planar_m=planar_m,
            joints_px={**smooth_px, "pelvis_center": pelvis_px},
            stance_masks=stance_masks,
            meta=meta,
        ),
        quality_flags=quality_flags,
    )
