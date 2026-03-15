from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .geometry import normalize
from .schema import PlanarKinematics, SkeletonSequence, SpeciesMorphology


def _normalize_rows(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    out = np.zeros_like(vec, dtype=float)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    good = norms[:, 0] > 1e-9
    out[good] = vec[good] / norms[good]
    out[~good] = fallback
    return out


def _unit_planar(planar: Dict[str, np.ndarray], a: str, b: str, fallback: np.ndarray) -> np.ndarray:
    return _normalize_rows(planar[b] - planar[a], fallback)


def _pitch_deg_rows(vec: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))


def _vector_from_pitch_deg(pitch_deg: np.ndarray | float) -> np.ndarray:
    pitch = np.deg2rad(pitch_deg)
    x = np.cos(pitch)
    z = np.sin(pitch)
    return np.stack([x, z], axis=-1) if np.ndim(pitch) else np.array([x, z], dtype=float)


def _vector_from_tail_pitch_deg(pitch_deg: np.ndarray | float) -> np.ndarray:
    pitch = np.deg2rad(pitch_deg)
    x = -np.cos(pitch)
    z = np.sin(pitch)
    return np.stack([x, z], axis=-1) if np.ndim(pitch) else np.array([x, z], dtype=float)


def _clamp_pitch_rows(vec: np.ndarray, min_deg: float, max_deg: float) -> np.ndarray:
    pitch = np.clip(_pitch_deg_rows(vec), min_deg, max_deg)
    return _vector_from_pitch_deg(pitch)


def _blend_dirs(observed: np.ndarray, reference: np.ndarray, observed_weight: float, bias_forward: float = 0.0, bias_up: float = 0.0) -> np.ndarray:
    if reference.ndim == 1:
        reference = np.repeat(reference[None, :], observed.shape[0], axis=0)
    vec = (1.0 - observed_weight) * reference + observed_weight * observed
    vec[:, 0] += bias_forward
    vec[:, 1] += bias_up
    return _normalize_rows(vec, reference[0])


def _constrain_leg_dir(obs: np.ndarray, ref: np.ndarray, x_min: float | None = None, x_max: float | None = None, z_max: float = -0.05, obs_weight: float = 0.75) -> np.ndarray:
    vec = _blend_dirs(obs, ref, observed_weight=obs_weight)
    if x_min is not None:
        vec[:, 0] = np.maximum(vec[:, 0], x_min)
    if x_max is not None:
        vec[:, 0] = np.minimum(vec[:, 0], x_max)
    vec[:, 1] = np.minimum(vec[:, 1], z_max)
    return _normalize_rows(vec, ref)


def _build_body_points(planar: Dict[str, np.ndarray], species: SpeciesMorphology) -> Dict[str, np.ndarray]:
    pelvis = planar["pelvis_center"]
    n = pelvis.shape[0]
    trunk_obs = _unit_planar(planar, "torso_back", "torso_front", np.array([1.0, 0.0]))
    trunk_ref = _vector_from_pitch_deg(species.trunk_pitch_bias_deg)
    trunk_dir = _blend_dirs(trunk_obs, trunk_ref, observed_weight=0.90)
    trunk_dir = _clamp_pitch_rows(trunk_dir, -8.0, 28.0)

    head_tip_obs = planar.get("head")
    head_reach_obs = _unit_planar(planar, "torso_front", "head", trunk_dir[0]) if head_tip_obs is not None else trunk_dir
    if species.neck_angle_deg is not None:
        neck_ref = _vector_from_pitch_deg(species.neck_angle_deg)
        neck_dir = _blend_dirs(head_reach_obs, neck_ref, observed_weight=species.neck_blend)
    else:
        neck_dir = _blend_dirs(head_reach_obs, trunk_dir, observed_weight=species.neck_blend, bias_forward=species.neck_forward_bias, bias_up=species.neck_up_bias)
    neck_dir = _clamp_pitch_rows(neck_dir, species.neck_min_pitch_deg, species.neck_max_pitch_deg)

    level_strength = species.level_head_strength
    if species.species_group == "theropod":
        level_strength = max(level_strength, 0.88)
    if species.head_angle_deg is not None:
        head_ref = _vector_from_pitch_deg(species.head_angle_deg)
        if head_ref.ndim == 1:
            head_ref = np.repeat(head_ref[None, :], n, axis=0)
        head_dir = _blend_dirs(head_reach_obs, head_ref, observed_weight=max(0.04, species.head_blend))
    else:
        horizontal_ref = _vector_from_pitch_deg(species.level_head_pitch_deg)
        if horizontal_ref.ndim == 1:
            horizontal_ref = np.repeat(horizontal_ref[None, :], n, axis=0)
        level_ref = _blend_dirs(neck_dir, horizontal_ref, observed_weight=max(0.05, 1.0 - level_strength), bias_forward=species.head_forward_bias, bias_up=species.head_up_bias)
        head_dir = _blend_dirs(head_reach_obs, level_ref, observed_weight=max(0.04, species.head_blend * (1.0 - level_strength)))
    head_dir = _clamp_pitch_rows(head_dir, species.head_min_pitch_deg, species.head_max_pitch_deg)

    tail_obs = _unit_planar(planar, "tail_base", "tail_tip", _vector_from_tail_pitch_deg(species.tail_pitch_deg)) if "tail_tip" in planar else np.repeat(_vector_from_tail_pitch_deg(species.tail_pitch_deg)[None, :], pelvis.shape[0], axis=0)
    tail_ref = _vector_from_tail_pitch_deg(species.tail_pitch_deg)
    tail_dir = _blend_dirs(tail_obs, tail_ref, observed_weight=species.tail_blend)
    tail_dir[:, 0] = np.minimum(tail_dir[:, 0], -0.25)
    tail_dir = _normalize_rows(tail_dir, tail_ref)

    pelvis_to_back = _unit_planar(planar, "pelvis_center", "torso_back", np.array([-0.55, 0.6]))
    pelvis_back_len = max(0.16 * species.height_m, 0.42 * species.trunk_depth_ratio * species.height_m)
    trunk_back = pelvis + pelvis_to_back * pelvis_back_len
    torso_front = trunk_back + trunk_dir * species.torso
    neck_base = torso_front.copy()
    neck_tip_pred = neck_base + neck_dir * species.neck
    head_tip_tracking = species.head_tip_tracking
    if species.species_group == "theropod":
        head_tip_tracking = min(head_tip_tracking, 0.08)
    if head_tip_obs is not None:
        neck_tip_obs = head_tip_obs - head_dir * species.head
        neck_tip = (1.0 - head_tip_tracking) * neck_tip_pred + head_tip_tracking * neck_tip_obs
    else:
        neck_tip = neck_tip_pred
    neck_mid = neck_base + 0.55 * (neck_tip - neck_base)
    head_center = neck_tip + head_dir * (0.5 * species.head)
    snout_tip = neck_tip + head_dir * species.head
    tail_base = pelvis + _vector_from_tail_pitch_deg(8.0) * max(0.08 * species.height_m, 0.35 * species.pelvis_length_ratio * species.height_m)
    tail_mid = tail_base + tail_dir * (0.42 * species.tail)
    tail_tip = tail_base + tail_dir * species.tail
    return {
        "pelvis_center": pelvis,
        "trunk_back": trunk_back,
        "torso_front": torso_front,
        "neck_base": neck_base,
        "neck_mid": neck_mid,
        "head_center": head_center,
        "snout_tip": snout_tip,
        "tail_base": tail_base,
        "tail_mid": tail_mid,
        "tail_tip": tail_tip,
    }


def _build_leg_points(planar: Dict[str, np.ndarray], species: SpeciesMorphology, side: str) -> Dict[str, np.ndarray]:
    pelvis = planar["pelvis_center"]
    thigh_obs = _unit_planar(planar, "pelvis_center", f"{side}_true_knee", np.array([0.28, -0.96]))
    shank_obs = _unit_planar(planar, f"{side}_true_knee", f"{side}_ankle", np.array([-0.18, -0.98]))
    meta_obs = _unit_planar(planar, f"{side}_ankle", f"{side}_toe_base", np.array([0.30, -0.95]))
    foot_obs = _unit_planar(planar, f"{side}_toe_base", f"{side}_toe_tip", np.array([0.98, -0.08]))

    thigh_u = _constrain_leg_dir(thigh_obs, np.array([0.28, -0.96]), x_min=0.04, x_max=0.82, z_max=-0.16, obs_weight=0.78)
    shank_u = _constrain_leg_dir(shank_obs, np.array([-0.20, -0.98]), x_max=-0.01, z_max=-0.20, obs_weight=0.72)
    meta_u = _constrain_leg_dir(meta_obs, np.array([0.34, -0.94]), x_min=0.02, x_max=0.70, z_max=-0.32, obs_weight=0.76)
    foot_u = _constrain_leg_dir(foot_obs, np.array([0.995, -0.10]), x_min=0.55, x_max=1.0, z_max=-0.30, obs_weight=0.82)

    knee = pelvis + thigh_u * species.thigh
    ankle = knee + shank_u * species.shank
    toe_base = ankle + meta_u * species.metatarsus
    toe_tip = toe_base + foot_u * species.foot
    return {
        f"{side}_knee": knee,
        f"{side}_ankle": ankle,
        f"{side}_toe_base": toe_base,
        f"{side}_toe_tip": toe_tip,
    }


def build_bilateral_skeleton(planar: PlanarKinematics, species: SpeciesMorphology, translate_root: bool = True) -> SkeletonSequence:
    pts = planar.joints_planar_m
    n = pts["pelvis_center"].shape[0]
    body = _build_body_points(pts, species)
    near = _build_leg_points(pts, species, "near")
    far = _build_leg_points(pts, species, "far")

    root_x = body["pelvis_center"][:, 0].copy()
    if not translate_root:
        for block in [body, near, far]:
            for key in block:
                block[key] = block[key].copy()
                block[key][:, 0] -= root_x

    hip_half = 0.5 * species.hip_width
    phase = planar.phase
    near_stance = planar.stance_masks["near"].astype(float)
    far_stance = planar.stance_masks["far"].astype(float)
    sway = 0.16 * hip_half * np.sin(phase) + 0.08 * hip_half * (near_stance - far_stance)
    yaw = np.deg2rad(species.body_yaw_deg) * np.sin(phase) * 0.25

    joint_names = [
        "pelvis_center",
        "left_hip",
        "right_hip",
        "trunk_back",
        "torso_front",
        "neck_base",
        "neck_mid",
        "head_center",
        "snout_tip",
        "tail_base",
        "tail_mid",
        "tail_tip",
        "left_knee",
        "left_ankle",
        "left_toe_base",
        "left_toe_tip",
        "right_knee",
        "right_ankle",
        "right_toe_base",
        "right_toe_tip",
    ]
    xyz = np.zeros((n, len(joint_names), 3), dtype=float)

    for t in range(n):
        points_t = {
            "pelvis_center": np.array([body["pelvis_center"][t, 0], 0.0, body["pelvis_center"][t, 1]], dtype=float),
            "left_hip": np.array([body["pelvis_center"][t, 0], hip_half + sway[t], body["pelvis_center"][t, 1]], dtype=float),
            "right_hip": np.array([body["pelvis_center"][t, 0], -hip_half + sway[t], body["pelvis_center"][t, 1]], dtype=float),
            "trunk_back": np.array([body["trunk_back"][t, 0], 0.32 * sway[t], body["trunk_back"][t, 1]], dtype=float),
            "torso_front": np.array([body["torso_front"][t, 0], -0.10 * sway[t], body["torso_front"][t, 1]], dtype=float),
            "neck_base": np.array([body["neck_base"][t, 0], -0.12 * sway[t], body["neck_base"][t, 1]], dtype=float),
            "neck_mid": np.array([body["neck_mid"][t, 0], -0.14 * sway[t], body["neck_mid"][t, 1]], dtype=float),
            "head_center": np.array([body["head_center"][t, 0], -0.15 * sway[t], body["head_center"][t, 1]], dtype=float),
            "snout_tip": np.array([body["snout_tip"][t, 0], -0.15 * sway[t], body["snout_tip"][t, 1]], dtype=float),
            "tail_base": np.array([body["tail_base"][t, 0], 0.12 * sway[t], body["tail_base"][t, 1]], dtype=float),
            "tail_mid": np.array([body["tail_mid"][t, 0], 0.24 * sway[t], body["tail_mid"][t, 1]], dtype=float),
            "tail_tip": np.array([body["tail_tip"][t, 0], 0.35 * sway[t], body["tail_tip"][t, 1]], dtype=float),
            "left_knee": np.array([near["near_knee"][t, 0], hip_half + 0.40 * sway[t], near["near_knee"][t, 1]], dtype=float),
            "left_ankle": np.array([near["near_ankle"][t, 0], hip_half + 0.18 * sway[t], near["near_ankle"][t, 1]], dtype=float),
            "left_toe_base": np.array([near["near_toe_base"][t, 0], hip_half, near["near_toe_base"][t, 1]], dtype=float),
            "left_toe_tip": np.array([near["near_toe_tip"][t, 0], hip_half, near["near_toe_tip"][t, 1]], dtype=float),
            "right_knee": np.array([far["far_knee"][t, 0], -hip_half + 0.40 * sway[t], far["far_knee"][t, 1]], dtype=float),
            "right_ankle": np.array([far["far_ankle"][t, 0], -hip_half + 0.18 * sway[t], far["far_ankle"][t, 1]], dtype=float),
            "right_toe_base": np.array([far["far_toe_base"][t, 0], -hip_half, far["far_toe_base"][t, 1]], dtype=float),
            "right_toe_tip": np.array([far["far_toe_tip"][t, 0], -hip_half, far["far_toe_tip"][t, 1]], dtype=float),
        }
        c = math.cos(yaw[t])
        s = math.sin(yaw[t])
        rot = np.array([[c, -s], [s, c]], dtype=float)
        body_names = ["pelvis_center", "left_hip", "right_hip", "trunk_back", "torso_front", "neck_base", "neck_mid", "head_center", "snout_tip", "tail_base", "tail_mid", "tail_tip"]
        for name in body_names:
            xy = points_t[name][:2] @ rot.T
            points_t[name][:2] = xy
        for j, name in enumerate(joint_names):
            xyz[t, j] = points_t[name]

    return SkeletonSequence(joint_names=joint_names, xyz=xyz, meta={"species": species.name, "style": "capsule", "translate_root": bool(translate_root), "species_payload": species.to_dict()})


def repeat_skeleton(sequence: SkeletonSequence, repeats: int) -> SkeletonSequence:
    if repeats <= 1:
        return sequence
    xyz_parts = []
    root_idx = sequence.joint_names.index("pelvis_center") if "pelvis_center" in sequence.joint_names else 0
    step_dx = float(sequence.xyz[-1, root_idx, 0] - sequence.xyz[0, root_idx, 0])
    for r in range(repeats):
        block = sequence.xyz.copy()
        if sequence.meta.get("translate_root", False):
            block[:, :, 0] += r * step_dx
        xyz_parts.append(block)
    xyz = np.concatenate(xyz_parts, axis=0)
    return SkeletonSequence(joint_names=sequence.joint_names, xyz=xyz, meta={**sequence.meta, "repeats": repeats})
