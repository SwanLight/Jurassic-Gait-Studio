from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .geometry import orthonormal_basis
from .schema import SkeletonSequence, SpeciesMorphology

BG_COLOR = "#f5f5f2"
GRID_COLOR = "#d7d9d3"
GRID_BOLD = "#c4c7c0"
BONE_COLOR = "#98b88e"
JOINT_COLOR = "#90b184"
SHADOW_COLOR = (0.18, 0.20, 0.18, 0.10)

SEGMENT_RADII: Dict[tuple[str, str], tuple[float, float]] = {
    ("left_hip", "left_knee"): (0.022, 0.019),
    ("left_knee", "left_ankle"): (0.019, 0.015),
    ("left_ankle", "left_toe_base"): (0.015, 0.012),
    ("left_toe_base", "left_toe_tip"): (0.012, 0.009),
    ("right_hip", "right_knee"): (0.022, 0.019),
    ("right_knee", "right_ankle"): (0.019, 0.015),
    ("right_ankle", "right_toe_base"): (0.015, 0.012),
    ("right_toe_base", "right_toe_tip"): (0.012, 0.009),
    ("tail_base", "tail_mid"): (0.016, 0.012),
    ("tail_mid", "tail_tip"): (0.012, 0.008),
}

DERIVED_SEGMENT_RADII: Dict[tuple[str, str], tuple[float, float]] = {
    ("tail_base", "pelvis_peak"): (0.022, 0.031),
    ("pelvis_peak", "body_front"): (0.031, 0.024),
    ("body_front", "head_root"): (0.015, 0.012),
    ("head_root", "snout_tip"): (0.020, 0.010),
}

JOINT_RADII: Dict[str, float] = {
    "tail_base": 0.010,
    "tail_mid": 0.009,
    "tail_tip": 0.007,
    "pelvis_peak": 0.011,
    "body_front": 0.010,
    "left_hip": 0.015,
    "right_hip": 0.015,
    "left_knee": 0.014,
    "right_knee": 0.014,
    "left_ankle": 0.011,
    "right_ankle": 0.011,
    "left_toe_base": 0.009,
    "right_toe_base": 0.009,
    "left_toe_tip": 0.007,
    "right_toe_tip": 0.007,
    "head_root": 0.009,
}




def _species_from_sequence(sequence: SkeletonSequence) -> SpeciesMorphology | None:
    payload = sequence.meta.get("species_payload")
    if not isinstance(payload, dict):
        return None
    try:
        return SpeciesMorphology(**payload)
    except Exception:
        return None


def _segment_radii_for_sequence(sequence: SkeletonSequence) -> tuple[Dict[tuple[str, str], tuple[float, float]], Dict[tuple[str, str], tuple[float, float]], Dict[str, float]]:
    seg = dict(SEGMENT_RADII)
    drv = dict(DERIVED_SEGMENT_RADII)
    joints = dict(JOINT_RADII)
    species = _species_from_sequence(sequence)
    if species is None:
        return seg, drv, joints

    thigh_r = 0.5 * species.effective_thigh_diameter_m
    shank_r = 0.5 * species.effective_shank_diameter_m
    meta_r = 0.5 * species.effective_metatarsus_diameter_m
    foot_r = max(0.0035, 0.70 * meta_r)
    tail_base_r = 0.5 * species.effective_tail_base_diameter_m
    tail_tip_r = 0.5 * species.effective_tail_tip_diameter_m
    rear_body_r = 0.5 * species.effective_rear_body_diameter_m
    front_body_r = 0.5 * species.effective_front_body_diameter_m
    neck_r = 0.5 * species.effective_neck_diameter_m
    head_r = 0.5 * species.effective_head_diameter_m

    seg.update({
        ("left_hip", "left_knee"): (thigh_r, max(0.004, 0.90 * thigh_r)),
        ("left_knee", "left_ankle"): (shank_r, max(0.004, 0.86 * shank_r)),
        ("left_ankle", "left_toe_base"): (meta_r, max(0.0035, 0.82 * meta_r)),
        ("left_toe_base", "left_toe_tip"): (foot_r, max(0.003, 0.72 * foot_r)),
        ("right_hip", "right_knee"): (thigh_r, max(0.004, 0.90 * thigh_r)),
        ("right_knee", "right_ankle"): (shank_r, max(0.004, 0.86 * shank_r)),
        ("right_ankle", "right_toe_base"): (meta_r, max(0.0035, 0.82 * meta_r)),
        ("right_toe_base", "right_toe_tip"): (foot_r, max(0.003, 0.72 * foot_r)),
        ("tail_base", "tail_mid"): (tail_base_r, max(0.0035, 0.62 * tail_base_r + 0.38 * tail_tip_r)),
        ("tail_mid", "tail_tip"): (max(0.0035, 0.62 * tail_base_r + 0.38 * tail_tip_r), tail_tip_r),
    })
    drv.update({
        ("tail_base", "pelvis_peak"): (tail_base_r, rear_body_r),
        ("pelvis_peak", "body_front"): (rear_body_r, front_body_r),
        ("body_front", "head_root"): (neck_r, max(0.0035, 0.82 * neck_r)),
        ("head_root", "snout_tip"): (head_r, max(0.0035, 0.52 * head_r)),
    })
    joints.update({
        "tail_base": max(0.0035, 0.50 * tail_base_r),
        "tail_mid": max(0.0030, 0.70 * (0.62 * tail_base_r + 0.38 * tail_tip_r)),
        "tail_tip": max(0.0025, 0.90 * tail_tip_r),
        "pelvis_peak": max(0.0040, 0.34 * rear_body_r),
        "body_front": max(0.0035, 0.34 * front_body_r),
        "left_hip": max(0.0040, 0.60 * thigh_r),
        "right_hip": max(0.0040, 0.60 * thigh_r),
        "left_knee": max(0.0035, 0.60 * shank_r),
        "right_knee": max(0.0035, 0.60 * shank_r),
        "left_ankle": max(0.0030, 0.56 * meta_r),
        "right_ankle": max(0.0030, 0.56 * meta_r),
        "left_toe_base": max(0.0028, 0.55 * foot_r),
        "right_toe_base": max(0.0028, 0.55 * foot_r),
        "left_toe_tip": max(0.0025, 0.42 * foot_r),
        "right_toe_tip": max(0.0025, 0.42 * foot_r),
        "head_root": max(0.0030, 0.36 * head_r),
    })
    return seg, drv, joints
def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.asarray(fallback, dtype=float)
    return v / n


def default_camera_for_sequence(sequence: SkeletonSequence) -> Dict[str, float | str | bool]:
    style = str(sequence.meta.get("style", "gait"))
    if style == "editor_preview":
        return {
            "elev": 18.0,
            "azim": -66.0,
            "distance": 5.8,
            "wall_side": "far",
            "invert_y": False,
        }
    return {
        "elev": 11.0,
        "azim": -82.0,
        "distance": 5.8,
        "wall_side": "far",
        "invert_y": False,
    }


def _cylinder_mesh(p0: np.ndarray, p1: np.ndarray, r0: float, r1: float, n_theta: int = 18, n_len: int = 5):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return None
    u, v, w = orthonormal_basis(axis)
    ts = np.linspace(0.0, 1.0, n_len)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    T, TH = np.meshgrid(ts, theta)
    radii = (1.0 - T) * r0 + T * r1
    centers = p0[None, None, :] + T[..., None] * axis[None, None, :]
    offsets = radii[..., None] * (
        np.cos(TH)[..., None] * v[None, None, :] + np.sin(TH)[..., None] * w[None, None, :]
    )
    pts = centers + offsets
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _sphere_mesh(center: np.ndarray, radius: float, n_theta: int = 18, n_phi: int = 10):
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    x = center[0] + radius * np.cos(TH) * np.sin(PH)
    y = center[1] + radius * np.sin(TH) * np.sin(PH)
    z = center[2] + radius * np.cos(PH)
    return x, y, z


def _shift_chain(points: dict[str, np.ndarray], names: list[str], offset: np.ndarray) -> None:
    for name in names:
        if name in points:
            points[name] = points[name] + offset


def _build_render_points(sequence: SkeletonSequence, frame_index: int) -> dict[str, np.ndarray]:
    xyz = sequence.xyz[int(frame_index)]
    points = {name: xyz[i].astype(float).copy() for i, name in enumerate(sequence.joint_names)}
    species = _species_from_sequence(sequence)

    pelvis = points.get("pelvis_center")
    trunk_back = points.get("trunk_back")
    torso_front = points.get("torso_front")
    tail_base_raw = points.get("tail_base")
    raw_neck_base = points.get("neck_base")
    neck_mid_raw = points.get("neck_mid")
    head_center_raw = points.get("head_center")
    snout_tip_raw = points.get("snout_tip")

    if pelvis is not None and trunk_back is not None and torso_front is not None and tail_base_raw is not None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if species is not None:
            rear_len = max(0.02, float(species.rear_body_length))
            front_len = max(0.04, float(species.torso))
            rear_angle = np.deg2rad(float(species.effective_rear_body_angle_deg))
            front_angle = np.deg2rad(float(species.trunk_pitch_bias_deg))
            rear_axis = np.array([-np.cos(rear_angle), 0.0, -np.sin(rear_angle)], dtype=float)
            front_axis = np.array([np.cos(front_angle), 0.0, np.sin(front_angle)], dtype=float)
            body_len = max(0.14, rear_len + front_len)
            pelvis_peak = pelvis + np.array([0.0, 0.0, max(0.02, 0.32 * species.effective_rear_body_diameter_m)], dtype=float)
            tail_root = pelvis_peak + rear_axis * rear_len
            body_front = pelvis_peak + front_axis * front_len
            points["trunk_back"] = pelvis_peak + rear_axis * (0.54 * rear_len)
            points["torso_front"] = body_front
        else:
            body_forward = _safe_unit(torso_front - trunk_back, np.array([1.0, 0.0, 0.0]))
            body_len = max(0.14, float(np.linalg.norm(torso_front - trunk_back)))
            tail_root = trunk_back + 0.010 * body_len * up - 0.010 * body_len * body_forward
            pelvis_peak = pelvis + 0.145 * body_len * up + 0.020 * body_len * body_forward
            body_front = torso_front - 0.030 * body_len * up + 0.010 * body_len * body_forward

        tail_offset = tail_root - tail_base_raw
        points["tail_base"] = tail_root
        _shift_chain(points, ["tail_mid", "tail_tip"], tail_offset)

        hip_z = pelvis_peak[2] - 0.050 * body_len
        for side in ("left", "right"):
            hip_name = f"{side}_hip"
            if hip_name not in points:
                continue
            raw_hip = points[hip_name]
            hip_target = np.array([pelvis_peak[0], raw_hip[1], hip_z], dtype=float)
            leg_offset = hip_target - raw_hip
            points[hip_name] = hip_target
            _shift_chain(points, [f"{side}_knee", f"{side}_ankle", f"{side}_toe_base", f"{side}_toe_tip"], leg_offset)

        points["pelvis_peak"] = pelvis_peak
        points["body_front"] = body_front

    if raw_neck_base is not None and head_center_raw is not None and snout_tip_raw is not None and "body_front" in points:
        body_front = points["body_front"]
        chain_offset = body_front - raw_neck_base
        neck_mid = neck_mid_raw + chain_offset if neck_mid_raw is not None else None
        head_center_shift = head_center_raw + chain_offset
        snout_tip_shift = snout_tip_raw + chain_offset

        head_center = body_front + 1.12 * (head_center_shift - body_front)
        snout_tip = body_front + 1.16 * (snout_tip_shift - body_front)

        neck_axis = _safe_unit((neck_mid if neck_mid is not None else head_center) - body_front, np.array([1.0, 0.0, 0.2]))
        head_axis = _safe_unit(snout_tip - head_center, neck_axis)
        snout_len = max(1e-6, float(np.linalg.norm(snout_tip - head_center)))

        # Keep Morphology Lab head length directly tied to the rendered head rod.
        # We only retain a very small geometric floor to avoid a degenerate zero-length
        # cylinder; head length is no longer silently inflated by neck length.
        head_len = max(0.008, float(species.head)) if species is not None else max(0.008, 1.05 * snout_len)

        head_root = head_center - 0.50 * head_len * head_axis + 0.02 * head_len * neck_axis
        snout_tip_render = head_center + 0.50 * head_len * head_axis

        points["head_root"] = head_root
        points["head_center"] = head_center
        points["snout_tip"] = snout_tip_render

    return points


def _contact_floor_offset(points: dict[str, np.ndarray]) -> float:
    contact_names = ["left_toe_tip", "right_toe_tip", "left_toe_base", "right_toe_base", "left_ankle", "right_ankle"]
    z_values = [float(points[name][2]) for name in contact_names if name in points]
    if not z_values:
        z_values = [float(p[2]) for p in points.values()]
    return -min(z_values) if z_values else 0.0


def _apply_floor_contact(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    dz = _contact_floor_offset(points)
    if abs(dz) < 1e-9:
        return points
    shift = np.array([0.0, 0.0, dz], dtype=float)
    return {name: p + shift for name, p in points.items()}


def _bounds_from_points(points: dict[str, np.ndarray]) -> Dict[str, float]:
    arr = np.stack(list(points.values()), axis=0)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = np.array([0.10, 0.14, 0.18]) * max(float(np.max(span)), 1.0)
    mins = mins - pad
    maxs = maxs + pad
    return {
        "x_min": float(mins[0]),
        "x_max": float(maxs[0]),
        "y_min": float(mins[1]),
        "y_max": float(maxs[1]),
        "z_min": 0.0,
        "z_max": float(max(maxs[2], 0.18)),
    }


def _sequence_bounds(sequence: SkeletonSequence) -> Dict[str, float]:
    points = _apply_floor_contact(_build_render_points(sequence, 0)) if sequence.xyz.shape[0] == 1 else None
    if points is not None:
        return _bounds_from_points(points)
    xyz = sequence.xyz
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    span = np.maximum(maxs - mins, 1e-6)
    pad = np.array([0.10, 0.12, 0.14]) * max(float(np.max(span)), 1.0)
    mins = mins - pad
    maxs = maxs + pad
    return {
        "x_min": float(mins[0]),
        "x_max": float(maxs[0]),
        "y_min": float(mins[1]),
        "y_max": float(maxs[1]),
        "z_min": 0.0,
        "z_max": float(max(maxs[2], 0.18)),
    }


def _draw_backdrop(ax, bounds: Dict[str, float], wall_side: str = "near") -> None:
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    z_max = bounds["z_max"]
    floor_z = 0.0
    depth_span = max(1e-6, y_max - y_min)
    wall_y = (y_min - 0.04 * depth_span) if wall_side != "far" else (y_max + 0.04 * depth_span)

    floor_front = y_min - 0.02 * depth_span if wall_side == "far" else y_max + 0.02 * depth_span
    floor_back = wall_y
    floor_ys = np.linspace(min(floor_front, floor_back), max(floor_front, floor_back), 13)
    floor_xs = np.linspace(x_min, x_max, 17)
    for i, gx in enumerate(floor_xs):
        ax.plot([gx, gx], [floor_ys[0], floor_ys[-1]], [floor_z, floor_z], color=GRID_BOLD if i % 4 == 0 else GRID_COLOR, linewidth=1.2 if i % 4 == 0 else 0.7, alpha=0.95)
    for i, gy in enumerate(floor_ys):
        ax.plot([x_min, x_max], [gy, gy], [floor_z, floor_z], color=GRID_BOLD if i % 4 == 0 else GRID_COLOR, linewidth=1.2 if i % 4 == 0 else 0.7, alpha=0.95)

    wall_xs = np.linspace(x_min, x_max, 17)
    wall_zs = np.linspace(floor_z, z_max, 13)
    for i, gx in enumerate(wall_xs):
        ax.plot([gx, gx], [wall_y, wall_y], [floor_z, z_max], color=GRID_BOLD if i % 4 == 0 else GRID_COLOR, linewidth=1.2 if i % 4 == 0 else 0.7, alpha=0.95)
    for i, gz in enumerate(wall_zs):
        ax.plot([x_min, x_max], [wall_y, wall_y], [gz, gz], color=GRID_BOLD if i % 4 == 0 else GRID_COLOR, linewidth=1.2 if i % 4 == 0 else 0.7, alpha=0.95)


def _draw_shadow(ax, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        return
    min_x, max_x = float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))
    min_y, max_y = float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1]))
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    rx = max(0.18, 0.30 * (max_x - min_x))
    ry = max(0.10, 0.48 * (max_y - min_y + 0.08))
    t = np.linspace(0.0, 2.0 * np.pi, 60)
    poly = [[cx + rx * np.cos(a), cy + ry * np.sin(a), 0.001] for a in t]
    shadow = Poly3DCollection([poly], facecolors=[SHADOW_COLOR], edgecolors="none")
    ax.add_collection3d(shadow)


def render_scene_3d(
    ax,
    sequence: SkeletonSequence,
    frame_index: int,
    bounds: Dict[str, float] | None = None,
    *,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    distance: Optional[float] = None,
    wall_side: Optional[str] = None,
    invert_y: Optional[bool] = None,
) -> None:
    points = _apply_floor_contact(_build_render_points(sequence, frame_index))
    arr = np.stack(list(points.values()), axis=0)
    bounds = bounds or _bounds_from_points(points)
    camera = default_camera_for_sequence(sequence)
    if elev is None:
        elev = float(camera["elev"])
    if azim is None:
        azim = float(camera["azim"])
    if distance is None:
        distance = float(camera["distance"])
    if wall_side is None:
        wall_side = str(camera.get("wall_side", "near"))
    if invert_y is None:
        invert_y = bool(camera.get("invert_y", False))

    ax.cla()
    ax.set_facecolor(BG_COLOR)
    _draw_backdrop(ax, bounds, wall_side=str(wall_side))
    _draw_shadow(ax, arr)

    segment_radii, derived_segment_radii, joint_radii = _segment_radii_for_sequence(sequence)
    items = []
    for source in (segment_radii, derived_segment_radii):
        for (a, b), (r0, r1) in source.items():
            if a in points and b in points:
                p0, p1 = points[a], points[b]
                depth = 0.55 * (p0[1] + p1[1]) + 0.18 * (p0[2] + p1[2])
                items.append((depth, a, b, r0, r1))
    items.sort(key=lambda row: row[0])

    for _, a, b, r0, r1 in items:
        mesh = _cylinder_mesh(points[a], points[b], r0, r1)
        if mesh is not None:
            ax.plot_surface(*mesh, color=BONE_COLOR, linewidth=0, antialiased=True, shade=True)

    for name, radius in joint_radii.items():
        if name in points:
            sphere = _sphere_mesh(points[name], radius)
            ax.plot_surface(*sphere, color=JOINT_COLOR, linewidth=0, antialiased=True, shade=True)

    ax.set_xlim(bounds["x_min"], bounds["x_max"])
    if bool(invert_y):
        ax.set_ylim(bounds["y_max"], bounds["y_min"])
    else:
        ax.set_ylim(bounds["y_min"], bounds["y_max"])
    ax.set_zlim(bounds["z_min"], bounds["z_max"])
    ax.set_box_aspect((
        max(1e-6, bounds["x_max"] - bounds["x_min"]),
        max(1e-6, bounds["y_max"] - bounds["y_min"]),
        max(1e-6, bounds["z_max"] - bounds["z_min"]),
    ))
    ax.view_init(elev=elev, azim=azim)
    if hasattr(ax, "dist"):
        try:
            ax.dist = distance
        except Exception:
            pass

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    try:
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))
    except Exception:
        pass


def _make_figure(size=(920, 600)):
    dpi = 100
    fig = Figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BG_COLOR)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, canvas, ax


def _canvas_to_rgb(canvas) -> np.ndarray:
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def _render_frame(sequence: SkeletonSequence, t: int, size=(920, 600), bounds: Dict[str, float] | None = None) -> np.ndarray:
    fig, canvas, ax = _make_figure(size=size)
    render_scene_3d(ax, sequence, t, bounds=bounds)
    frame = _canvas_to_rgb(canvas)
    fig.clear()
    return frame


def render_gif(sequence: SkeletonSequence, out_path: str | Path, step: int = 1, duration: float = 0.08) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = _sequence_bounds(sequence)
    with imageio.get_writer(out_path, mode="I", duration=duration, loop=0) as writer:
        for t in range(0, sequence.xyz.shape[0], max(1, int(step))):
            writer.append_data(_render_frame(sequence, t, bounds=bounds))
    return out_path


def render_frame_png(sequence: SkeletonSequence, out_path: str | Path, frame_index: int = 0) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = _sequence_bounds(sequence)
    imageio.imwrite(out_path, _render_frame(sequence, frame_index, bounds=bounds))
    return out_path
