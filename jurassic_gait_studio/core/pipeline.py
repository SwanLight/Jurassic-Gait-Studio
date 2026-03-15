from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from .diagnostics import save_diagnostics
from .fusion import (
    blend_species_morphologies,
    compute_bird_weights,
    discover_bird_library,
    fuse_planar_kinematics,
    render_weight_plot,
    save_fusion_report,
)
from .io import load_species, load_tracks_csv, save_json, save_planar_points, save_skeleton_csv
from .reconstruction import reconstruct_planar_cycle
from .render import render_frame_png, render_gif
from .retarget import build_bilateral_skeleton, repeat_skeleton
from .schema import PlanarKinematics, ReconstructionSummary, SpeciesMorphology


def _materialize_outputs(
    planar: PlanarKinematics,
    source_species: SpeciesMorphology,
    target_species: SpeciesMorphology,
    source_name: str,
    out_dir: str | Path,
    translate_root: bool = True,
    repeat_cycles: int = 3,
    extra_summary: Dict[str, object] | None = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bird_world = build_bilateral_skeleton(planar, source_species, translate_root=True)
    bird_loop = build_bilateral_skeleton(planar, source_species, translate_root=False)
    dino_world = build_bilateral_skeleton(planar, target_species, translate_root=True)
    dino_loop = build_bilateral_skeleton(planar, target_species, translate_root=False)
    bird_render = repeat_skeleton(bird_world if translate_root else bird_loop, repeats=repeat_cycles)
    dino_render = repeat_skeleton(dino_world if translate_root else dino_loop, repeats=repeat_cycles)

    quality_flags = list(planar.meta.get("quality_flags", []))
    near_frac = float(planar.stance_masks["near"].mean())
    far_frac = float(planar.stance_masks["far"].mean())
    if near_frac > 0.90 and far_frac > 0.90:
        quality_flags.append("stance_masks_all_contact")
    if near_frac < 0.08:
        quality_flags.append("near_stance_too_sparse")
    if far_frac < 0.08:
        quality_flags.append("far_stance_too_sparse")

    summary_payload = ReconstructionSummary(
        source_name=source_name,
        source_species=source_species.name,
        target_species=target_species.name,
        n_frames=next(iter(planar.joints_planar_m.values())).shape[0],
        scale_m_per_px=float(planar.scale_m_per_px),
        direction=int(planar.direction),
        near_stance_fraction=near_frac,
        far_stance_fraction=far_frac,
        bilateral_source=bool(planar.meta.get("bilateral_source", False) or planar.meta.get("fused", False)),
        quality_flags=quality_flags,
    ).to_dict()
    if extra_summary:
        summary_payload.update(extra_summary)

    paths = {
        "planar_csv": save_planar_points(planar.joints_planar_m, out_dir / "source_planar_reconstruction.csv"),
        "bird_csv": save_skeleton_csv(bird_world, out_dir / "bird_3d.csv"),
        "dino_csv": save_skeleton_csv(dino_world, out_dir / "dinosaur_3d.csv"),
        "summary": save_json(summary_payload, out_dir / "summary.json"),
    }
    if planar.joints_px:
        class _Obs:
            def __init__(self, name: str, n_frames: int):
                self.source_name = name
                self.n_frames = n_frames
        obs = _Obs(source_name, next(iter(planar.joints_px.values())).shape[0])
        paths["diagnostics"] = save_diagnostics(obs, planar, out_dir / "diagnostics.png")
    paths["bird_frame0"] = render_frame_png(bird_render, out_dir / "bird3d_frame0.png", frame_index=0)
    paths["dino_frame0"] = render_frame_png(dino_render, out_dir / "dinosaur3d_frame0.png", frame_index=0)
    gif_step = max(1, bird_render.xyz.shape[0] // 48)
    paths["bird_gif"] = render_gif(bird_render, out_dir / "bird3d.gif", step=gif_step)
    paths["dino_gif"] = render_gif(dino_render, out_dir / "dinosaur3d.gif", step=gif_step)
    return paths


def run_pipeline(
    tracks_csv: str | Path,
    source_species_json: str | Path,
    target_species_json: str | Path,
    fps: float,
    out_dir: str | Path,
    translate_root: bool = True,
    repeat_cycles: int = 3,
) -> Dict[str, Path]:
    source_species = load_species(source_species_json)
    target_species = load_species(target_species_json)
    obs = load_tracks_csv(tracks_csv, fps=fps)
    result = reconstruct_planar_cycle(obs, source_species)
    planar = result.planar
    planar.meta = {**planar.meta, "quality_flags": list(result.quality_flags)}
    paths = _materialize_outputs(
        planar=planar,
        source_species=source_species,
        target_species=target_species,
        source_name=obs.source_name,
        out_dir=out_dir,
        translate_root=translate_root,
        repeat_cycles=repeat_cycles,
    )
    if planar.joints_px:
        paths["diagnostics"] = save_diagnostics(obs, planar, Path(out_dir) / "diagnostics.png")
    return paths


def run_fused_pipeline(
    target_species_json: str | Path,
    fps: float,
    out_dir: str | Path,
    bird_species_dir: str | Path,
    bird_track_paths: Sequence[str | Path] | None = None,
    bird_tracks_dir: str | Path | None = None,
    bird_manifest_json: str | Path | None = None,
    translate_root: bool = True,
    repeat_cycles: int = 3,
    n_phase_frames: int = 101,
) -> Dict[str, Path]:
    target_species = load_species(target_species_json)
    entries = discover_bird_library(
        fps=fps,
        species_dir=bird_species_dir,
        track_paths=bird_track_paths,
        track_dir=bird_tracks_dir,
        manifest_path=bird_manifest_json,
    )
    report = compute_bird_weights(entries, target_species)
    fused_bird = blend_species_morphologies(entries, report.normalized_weights, out_name="fused_bird_reference")
    fused_planar = fuse_planar_kinematics(entries, report.normalized_weights, output_height_m=target_species.height_m, n_frames=n_phase_frames)
    fused_planar.meta = {
        **fused_planar.meta,
        "quality_flags": ["multi_bird_fusion"],
    }
    paths = _materialize_outputs(
        planar=fused_planar,
        source_species=fused_bird,
        target_species=target_species,
        source_name=report.fused_source_name,
        out_dir=out_dir,
        translate_root=translate_root,
        repeat_cycles=repeat_cycles,
        extra_summary={
            "fused": True,
            "bird_weights": report.normalized_weights,
            "bird_reference_species": [e.species.name for e in entries],
        },
    )
    out_dir = Path(out_dir)
    paths["fusion_report"] = save_fusion_report(report, out_dir / "fusion_report.json")
    paths["fusion_weights_plot"] = render_weight_plot(report, out_dir / "fusion_weights.png")
    return paths
