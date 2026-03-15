from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .render import render_frame_png
from .schema import SkeletonSequence, SpeciesMorphology


def species_from_payload(payload: Dict[str, Any]) -> SpeciesMorphology:
    return SpeciesMorphology(**payload)


def build_species_pose(species: SpeciesMorphology) -> SkeletonSequence:
    """Neutral one-frame pose used for morphology editing/previewing.

    The preview pose mirrors the rod rig used by the 3D render: tail, two-part
    body, neck, and head. The goal is not a full simulator posture, but a
    stable editable side-view that reacts to the same JSON fields users tweak in
    Morphology Lab.
    """
    h = float(species.height_m)
    hip_half = 0.5 * float(species.hip_width)
    torso_len = max(0.18, float(species.torso))
    pelvis_len = max(0.02, float(species.rear_body_length))
    trunk_depth = max(0.02, float(species.body_depth))
    front_body_angle = np.deg2rad(float(species.trunk_pitch_bias_deg))
    rear_body_angle = np.deg2rad(float(species.effective_rear_body_angle_deg))

    pelvis = np.array([0.0, 0.0, species.pelvis_height_ratio * h + 0.42 * species.thigh], dtype=float)
    pelvis_peak = pelvis + np.array([0.0, 0.0, max(0.02, 0.32 * species.effective_rear_body_diameter_m)], dtype=float)
    rear_axis = np.array([-np.cos(rear_body_angle), 0.0, -np.sin(rear_body_angle)], dtype=float)
    front_axis = np.array([np.cos(front_body_angle), 0.0, np.sin(front_body_angle)], dtype=float)

    tail_base = pelvis_peak + rear_axis * pelvis_len
    trunk_back = pelvis_peak + rear_axis * (0.54 * pelvis_len)
    torso_front = pelvis_peak + front_axis * torso_len

    neck_base = torso_front + np.array([0.02 * h, 0.0, 0.0], dtype=float)
    neck_angle = np.deg2rad(float(species.effective_neck_angle_deg))
    neck_forward = species.neck * np.cos(neck_angle)
    neck_lift = species.neck * np.sin(neck_angle)
    neck_blend = float(np.clip(species.neck_blend, 0.0, 1.0))
    neck_mid = neck_base + np.array([neck_forward * (0.34 + 0.20 * neck_blend), 0.0, neck_lift * (0.44 + 0.20 * neck_blend)], dtype=float)

    head_center = neck_base + np.array([neck_forward, 0.0, neck_lift], dtype=float)
    head_angle = np.deg2rad(float(species.effective_head_angle_deg))
    head_axis = np.array([np.cos(head_angle), 0.0, np.sin(head_angle)], dtype=float)
    head_blend = float(np.clip(species.head_blend, 0.0, 1.0))
    snout_tip = head_center + head_axis * ((0.86 + 0.08 * head_blend) * species.head)

    tail_angle = np.deg2rad(float(species.tail_pitch_deg) - 4.0)
    tail_mid = tail_base + np.array([-0.48 * species.tail * np.cos(tail_angle), 0.0, 0.48 * species.tail * np.sin(tail_angle) + 0.03 * species.tail], dtype=float)
    tail_tip = tail_base + np.array([-1.00 * species.tail * np.cos(tail_angle), 0.0, 1.00 * species.tail * np.sin(tail_angle) + 0.10 * species.tail], dtype=float)

    left_hip = pelvis + np.array([0.0, hip_half, 0.0], dtype=float)
    right_hip = pelvis + np.array([0.0, -hip_half, 0.0], dtype=float)

    def leg(side_sign: float):
        hip = pelvis + np.array([0.0, side_sign * hip_half, 0.0], dtype=float)
        knee = hip + np.array([0.24 * species.thigh, 0.0, -0.97 * species.thigh], dtype=float)
        ankle = knee + np.array([-0.14 * species.shank, 0.0, -0.99 * species.shank], dtype=float)
        toe_base = ankle + np.array([0.34 * species.metatarsus, 0.0, -0.94 * species.metatarsus], dtype=float)
        toe_tip = toe_base + np.array([1.0 * species.foot, 0.0, 0.0], dtype=float)
        return knee, ankle, toe_base, toe_tip

    left_knee, left_ankle, left_toe_base, left_toe_tip = leg(+1.0)
    right_knee, right_ankle, right_toe_base, right_toe_tip = leg(-1.0)

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
    xyz = np.array(
        [[
            pelvis,
            left_hip,
            right_hip,
            trunk_back,
            torso_front,
            neck_base,
            neck_mid,
            head_center,
            snout_tip,
            tail_base,
            tail_mid,
            tail_tip,
            left_knee,
            left_ankle,
            left_toe_base,
            left_toe_tip,
            right_knee,
            right_ankle,
            right_toe_base,
            right_toe_tip,
        ]],
        dtype=float,
    )
    return SkeletonSequence(joint_names=joint_names, xyz=xyz, meta={"species": species.name, "translate_root": False, "style": "editor_preview", "species_payload": species.to_dict()})


def render_species_preview(payload: Dict[str, Any], out_path: str | Path) -> Path:
    return render_frame_png(build_species_pose(species_from_payload(payload)), out_path, frame_index=0)


def summarize_species_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    species = species_from_payload(payload)
    return {
        "name": species.name,
        "group": species.species_group,
        "hip_width_m": round(species.hip_width, 3),
        "rear_body_length_m": round(species.rear_body_length, 3),
        "front_body_length_m": round(species.torso, 3),
        "tail_length_m": round(species.tail, 3),
        "neck_length_m": round(species.neck, 3),
        "head_length_m": round(species.head, 3),
        "rear_body_diameter_m": round(species.effective_rear_body_diameter_m, 3),
        "front_body_diameter_m": round(species.effective_front_body_diameter_m, 3),
        "thigh_diameter_m": round(species.effective_thigh_diameter_m, 3),
        "neck_diameter_m": round(species.effective_neck_diameter_m, 3),
        "head_diameter_m": round(species.effective_head_diameter_m, 3),
        "rear_body_angle_deg": round(float(species.effective_rear_body_angle_deg), 2),
        "front_body_angle_deg": round(float(species.trunk_pitch_bias_deg), 2),
        "tail_angle_deg": round(float(species.tail_pitch_deg), 2),
        "neck_angle_deg": round(float(species.effective_neck_angle_deg), 2),
        "head_angle_deg": round(float(species.effective_head_angle_deg), 2),
    }


def load_preview_payload(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
