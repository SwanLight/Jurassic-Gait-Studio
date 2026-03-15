from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

CANONICAL_VISIBLE = [
    "torso_back",
    "torso_front",
    "tail_base",
    "tail_tip",
    "near_ankle",
    "near_toe_base",
    "near_toe_tip",
]
OPTIONAL_VISIBLE = [
    "head",
    "far_ankle",
    "far_toe_base",
    "far_toe_tip",
    "near_knee_hint",
    "far_knee_hint",
]


@dataclass
class SpeciesMorphology:
    name: str
    species_group: str
    mass_kg: float
    height_m: float
    limb_lengths_m: Dict[str, float]
    hip_width_ratio: float = 0.12
    pelvis_height_ratio: float = 0.11
    pelvis_length_ratio: float = 0.08
    trunk_depth_ratio: float = 0.12
    head_length_ratio: float = 1.0
    body_yaw_deg: float = 4.0
    tail_drop_deg: float = 10.0
    neck_blend: float = 0.55
    neck_min_pitch_deg: float = 20.0
    neck_max_pitch_deg: float = 80.0
    neck_forward_bias: float = 0.0
    neck_up_bias: float = 0.0
    head_blend: float = 0.45
    head_min_pitch_deg: float = 10.0
    head_max_pitch_deg: float = 85.0
    head_forward_bias: float = 0.0
    head_up_bias: float = 0.0
    level_head_strength: float = 0.72
    level_head_pitch_deg: float = 2.0
    head_tip_tracking: float = 0.22
    tail_blend: float = 0.30
    tail_pitch_deg: float = 4.0
    trunk_pitch_bias_deg: float = 0.0
    # Optional direct-edit geometry overrides used by Morphology Lab and the 3D renderer.
    rear_body_diameter_m: float | None = None
    front_body_diameter_m: float | None = None
    thigh_diameter_m: float | None = None
    shank_diameter_m: float | None = None
    metatarsus_diameter_m: float | None = None
    neck_diameter_m: float | None = None
    head_diameter_m: float | None = None
    tail_base_diameter_m: float | None = None
    tail_tip_diameter_m: float | None = None
    rear_body_angle_deg: float | None = None
    neck_angle_deg: float | None = None
    head_angle_deg: float | None = None

    @property
    def thigh(self) -> float:
        return float(self.limb_lengths_m["thigh"])

    @property
    def shank(self) -> float:
        return float(self.limb_lengths_m["shank"])

    @property
    def metatarsus(self) -> float:
        return float(self.limb_lengths_m["metatarsus"])

    @property
    def foot(self) -> float:
        return float(self.limb_lengths_m["foot"])

    @property
    def torso(self) -> float:
        return float(self.limb_lengths_m.get("torso", 0.0))

    @property
    def neck(self) -> float:
        return float(self.limb_lengths_m.get("neck", 0.0))

    @property
    def head(self) -> float:
        return float(self.limb_lengths_m.get("head", 0.0)) * float(self.head_length_ratio)

    @property
    def tail(self) -> float:
        return float(self.limb_lengths_m.get("tail", 0.0))

    @property
    def humerus(self) -> float:
        return float(self.limb_lengths_m.get("humerus", 0.0))

    @property
    def forearm(self) -> float:
        return float(self.limb_lengths_m.get("forearm", 0.0))

    @property
    def manus(self) -> float:
        return float(self.limb_lengths_m.get("manus", 0.0))

    @property
    def hip_width(self) -> float:
        return self.height_m * float(self.hip_width_ratio)

    @property
    def rear_body_length(self) -> float:
        return self.height_m * float(self.pelvis_length_ratio)

    @property
    def body_depth(self) -> float:
        return self.height_m * float(self.trunk_depth_ratio)

    @property
    def effective_rear_body_angle_deg(self) -> float:
        if self.rear_body_angle_deg is not None:
            return float(self.rear_body_angle_deg)
        rise = 0.48 * self.body_depth + 0.02 * self.height_m
        run = max(self.rear_body_length, 1e-6)
        return float(np.degrees(np.arctan2(rise, run)))

    @property
    def effective_neck_angle_deg(self) -> float:
        if self.neck_angle_deg is not None:
            return float(self.neck_angle_deg)
        forward = self.neck * (0.62 + 0.26 * float(self.neck_forward_bias))
        lift = self.neck * (0.24 + 0.24 * float(self.neck_up_bias)) + 0.04 * self.height_m
        return float(np.degrees(np.arctan2(lift, max(forward, 1e-6))))

    @property
    def effective_head_angle_deg(self) -> float:
        if self.head_angle_deg is not None:
            return float(self.head_angle_deg)
        forward = 1.0 + 0.18 * float(self.head_forward_bias)
        lift = 0.04 + 0.16 * float(self.head_up_bias)
        return float(np.degrees(np.arctan2(lift, max(forward, 1e-6))))

    @property
    def effective_rear_body_diameter_m(self) -> float:
        return float(self.rear_body_diameter_m) if self.rear_body_diameter_m is not None else max(0.024, 1.08 * self.body_depth)

    @property
    def effective_front_body_diameter_m(self) -> float:
        return float(self.front_body_diameter_m) if self.front_body_diameter_m is not None else max(0.020, 0.88 * self.body_depth)

    @property
    def effective_thigh_diameter_m(self) -> float:
        return float(self.thigh_diameter_m) if self.thigh_diameter_m is not None else max(0.018, 0.22 * self.thigh)

    @property
    def effective_shank_diameter_m(self) -> float:
        return float(self.shank_diameter_m) if self.shank_diameter_m is not None else max(0.016, 0.17 * self.shank)

    @property
    def effective_metatarsus_diameter_m(self) -> float:
        return float(self.metatarsus_diameter_m) if self.metatarsus_diameter_m is not None else max(0.012, 0.22 * self.metatarsus)

    @property
    def effective_neck_diameter_m(self) -> float:
        return float(self.neck_diameter_m) if self.neck_diameter_m is not None else max(0.012, 0.40 * self.body_depth)

    @property
    def effective_head_diameter_m(self) -> float:
        return float(self.head_diameter_m) if self.head_diameter_m is not None else max(0.012, 0.55 * self.head)

    @property
    def effective_tail_base_diameter_m(self) -> float:
        return float(self.tail_base_diameter_m) if self.tail_base_diameter_m is not None else max(0.012, 0.38 * self.body_depth)

    @property
    def effective_tail_tip_diameter_m(self) -> float:
        return float(self.tail_tip_diameter_m) if self.tail_tip_diameter_m is not None else max(0.006, 0.18 * self.body_depth)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationSequence:
    fps: float
    frame_indices: np.ndarray
    points: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    source_name: str = "unknown"

    @property
    def n_frames(self) -> int:
        return int(self.frame_indices.shape[0])


@dataclass
class PlanarKinematics:
    phase: np.ndarray
    direction: int
    scale_m_per_px: float
    ground_y_px: float
    joints_planar_m: Dict[str, np.ndarray]
    joints_px: Dict[str, np.ndarray]
    stance_masks: Dict[str, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkeletonSequence:
    joint_names: List[str]
    xyz: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for t in range(self.xyz.shape[0]):
            row: Dict[str, float] = {"frame": float(t)}
            for j, name in enumerate(self.joint_names):
                row[f"{name}_x"] = float(self.xyz[t, j, 0])
                row[f"{name}_y"] = float(self.xyz[t, j, 1])
                row[f"{name}_z"] = float(self.xyz[t, j, 2])
            rows.append(row)
        return rows


@dataclass
class ReconstructionSummary:
    source_name: str
    source_species: str
    target_species: str
    n_frames: int
    scale_m_per_px: float
    direction: int
    near_stance_fraction: float
    far_stance_fraction: float
    bilateral_source: bool
    quality_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
