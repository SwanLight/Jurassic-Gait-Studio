from __future__ import annotations

from typing import Dict

ALIASES: Dict[str, str] = {
    "torso_back": "torso_back",
    "back": "torso_back",
    "torso_front": "torso_front",
    "shoulder": "torso_front",
    "head": "head",
    "beak_tip": "head",
    "tail_base": "tail_base",
    "tail_tip": "tail_tip",
    "hip": "pelvis_hint",
    "pelvis": "pelvis_hint",
    "root": "pelvis_hint",
    "ankle": "near_ankle",
    "visible_ankle": "near_ankle",
    "hock": "near_ankle",
    "toe_base": "near_toe_base",
    "metatarsal_head": "near_toe_base",
    "toe": "near_toe_tip",
    "toe_tip": "near_toe_tip",
    "knee": "near_knee_hint",
    "knee_hint": "near_knee_hint",
}


def canonical_joint_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    key = key.replace("__", "_")
    if key.startswith("near_") or key.startswith("far_"):
        prefix, suffix = key.split("_", 1)
        if suffix in {"ankle", "visible_ankle", "hock"}:
            return f"{prefix}_ankle"
        if suffix in {"toe_base", "metatarsal_head"}:
            return f"{prefix}_toe_base"
        if suffix in {"toe", "toe_tip"}:
            return f"{prefix}_toe_tip"
        if suffix in {"knee", "knee_hint"}:
            return f"{prefix}_knee_hint"
        return f"{prefix}_{suffix}"
    return ALIASES.get(key, key)
