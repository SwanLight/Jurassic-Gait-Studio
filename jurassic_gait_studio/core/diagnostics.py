from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .schema import ObservationSequence, PlanarKinematics


BODY_KEYS = ["torso_back", "torso_front", "tail_base", "tail_tip", "head", "pelvis_center"]


def save_diagnostics(observation: ObservationSequence, planar: PlanarKinematics, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obs = planar.joints_px
    plan = planar.joints_planar_m
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for key in ["torso_back", "torso_front", "tail_base", "pelvis_center", "near_true_knee", "near_ankle", "near_toe_tip"]:
        if key not in obs:
            continue
        ax.plot(obs[key][:, 0], obs[key][:, 1], label=key)
    if "far_true_knee" in obs:
        ax.plot(obs["far_true_knee"][:, 0], obs["far_true_knee"][:, 1], label="far_true_knee", alpha=0.75)
    ax.set_title("Pixel-space trajectories")
    ax.invert_yaxis()
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(plan["pelvis_center"][:, 0], plan["pelvis_center"][:, 1], label="pelvis_center")
    for key in ["near_true_knee", "near_ankle", "near_toe_base", "near_toe_tip", "far_true_knee", "far_ankle", "far_toe_base", "far_toe_tip"]:
        if key in plan:
            ax.plot(plan[key][:, 0], plan[key][:, 1], label=key)
    ax.set_title("Planar reconstruction in meters")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    ax.plot(planar.phase, planar.stance_masks["near"].astype(float), label="near stance")
    ax.plot(planar.phase, planar.stance_masks["far"].astype(float), label="far stance")
    ax.plot(planar.phase, plan["near_toe_tip"][:, 1], label="near toe tip z")
    if "far_toe_tip" in plan:
        ax.plot(planar.phase, plan["far_toe_tip"][:, 1], label="far toe tip z")
    ax.set_title("Phase and stance")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    knee_drop = plan["pelvis_center"][:, 1] - plan["near_true_knee"][:, 1]
    ax.plot(knee_drop, label="near knee drop")
    if "far_true_knee" in plan:
        ax.plot(plan["pelvis_center"][:, 1] - plan["far_true_knee"][:, 1], label="far knee drop")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_title("Knee below pelvis check")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
