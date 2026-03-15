from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from .core.fusion import FusionReport, compute_bird_weights, discover_bird_library
from .core.io import load_species
from .core.pipeline import run_fused_pipeline
from .paths import BIRD_SPECIES_DIR
from .registry import (
    get_species_path,
    get_target_path,
    list_bird_clips,
    list_runs,
    next_run_dir,
    now_iso,
    register_run,
    resolve_clip_paths,
)

APP_VERSION = "3.0.0"


def _materialize_effective_inputs(
    target_stem: str,
    selected_clips: List[Dict[str, Any]],
    *,
    target_payload_override: Dict[str, Any] | None = None,
    bird_species_payload_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> tuple[TemporaryDirectory | None, Path, Path, Path | None, List[str]]:
    overrides = bird_species_payload_overrides or {}
    needs_temp = target_payload_override is not None or bool(overrides)
    if not needs_temp:
        return None, get_target_path(target_stem), BIRD_SPECIES_DIR, None, []

    temp_dir = TemporaryDirectory(prefix="jgs_effective_inputs_")
    root = Path(temp_dir.name)
    birds_dir = root / "bird_species"
    birds_dir.mkdir(parents=True, exist_ok=True)

    used_override_stems: List[str] = []
    override_paths: Dict[str, Path] = {}
    for stem, payload in overrides.items():
        override_path = birds_dir / f"{stem}.json"
        override_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        override_paths[stem] = override_path
        used_override_stems.append(stem)

    if target_payload_override is not None:
        target_json = root / f"{target_stem}.json"
        target_json.write_text(json.dumps(target_payload_override, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        target_json = get_target_path(target_stem)

    manifest_entries = []
    for clip in selected_clips:
        species_stem = str(clip.get("species_stem") or "")
        species_json = override_paths.get(species_stem)
        if species_json is None:
            species_json = get_species_path("bird", species_stem)
        manifest_entries.append(
            {
                "name": clip.get("name") or Path(str(clip["absolute_path"])).stem,
                "tracks_csv": str(clip["absolute_path"]),
                "species_json": str(species_json),
                "fps": float(clip.get("fps", 0.0) or 0.0),
            }
        )
    manifest_path = root / "bird_manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return temp_dir, target_json, BIRD_SPECIES_DIR, manifest_path, used_override_stems


def preview_weights(
    target_stem: str,
    clip_ids: List[str] | None = None,
    fps: float = 30.0,
    *,
    target_payload_override: Dict[str, Any] | None = None,
    bird_species_payload_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> FusionReport:
    selected_set = set(clip_ids) if clip_ids else None
    selected_clips = [item for item in list_bird_clips() if selected_set is None or item["clip_id"] in selected_set]
    temp_dir, target_json, species_dir, manifest_path, _ = _materialize_effective_inputs(
        target_stem,
        selected_clips,
        target_payload_override=target_payload_override,
        bird_species_payload_overrides=bird_species_payload_overrides,
    )
    try:
        target = load_species(target_json)
        entries = discover_bird_library(
            fps=float(fps),
            species_dir=species_dir,
            manifest_path=manifest_path,
            track_paths=None if manifest_path is not None else resolve_clip_paths(clip_ids),
        )
        return compute_bird_weights(entries, target)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _relative_paths(paths: Dict[str, Path], base: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in paths.items():
        p = Path(value)
        try:
            out[key] = str(p.relative_to(base))
        except ValueError:
            out[key] = str(p)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def generate_run(
    target_stem: str,
    clip_ids: List[str] | None = None,
    fps: float = 30.0,
    repeat_cycles: int = 3,
    phase_frames: int = 101,
    *,
    target_payload_override: Dict[str, Any] | None = None,
    bird_species_payload_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    run_dir = next_run_dir(target_stem)
    selected_set = set(clip_ids) if clip_ids else None
    selected_clips = [item for item in list_bird_clips() if selected_set is None or item["clip_id"] in selected_set]
    temp_dir, target_json, species_dir, manifest_path, used_override_stems = _materialize_effective_inputs(
        target_stem,
        selected_clips,
        target_payload_override=target_payload_override,
        bird_species_payload_overrides=bird_species_payload_overrides,
    )
    try:
        target_species = load_species(target_json)
        paths = run_fused_pipeline(
            target_species_json=target_json,
            fps=float(fps),
            out_dir=run_dir,
            bird_species_dir=species_dir,
            bird_track_paths=None if manifest_path is not None else resolve_clip_paths(clip_ids),
            bird_manifest_json=manifest_path,
            repeat_cycles=max(1, int(repeat_cycles)),
            n_phase_frames=max(31, int(phase_frames)),
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    fusion_report = _load_json(Path(paths["fusion_report"]))
    summary_payload = _load_json(Path(paths["summary"]))
    session_report = {
        "app_version": APP_VERSION,
        "run_name": run_dir.name,
        "target_stem": target_stem,
        "target_species": target_species.name,
        "target_payload": target_species.to_dict(),
        "created_at": now_iso(),
        "selected_clip_ids": [item["clip_id"] for item in selected_clips],
        "selected_clips": selected_clips,
        "parameters": {
            "fps": float(fps),
            "repeat_cycles": int(repeat_cycles),
            "phase_frames": int(phase_frames),
        },
        "fusion_report": fusion_report,
        "reconstruction_summary": summary_payload,
        "outputs": _relative_paths(paths, run_dir),
        "notes": {
            "run_sequence": run_dir.name,
            "description": "Auto-numbered theropod reconstruction run generated by Jurassic Gait Studio.",
            "used_target_override": bool(target_payload_override is not None),
            "used_bird_overrides": sorted(used_override_stems),
        },
    }
    session_path = run_dir / "session_report.json"
    session_path.write_text(json.dumps(session_report, indent=2, ensure_ascii=False), encoding="utf-8")
    register_run(
        {
            "run_name": run_dir.name,
            "target_stem": target_stem,
            "target_species": session_report["target_species"],
            "created_at": session_report["created_at"],
            "run_dir": str(run_dir),
            "preview_gif": str(run_dir / session_report["outputs"]["dino_gif"]),
            "preview_frame": str(run_dir / session_report["outputs"]["dino_frame0"]),
            "report_path": str(session_path),
        }
    )
    return {
        "run_dir": str(run_dir),
        "session_report_path": str(session_path),
        "session_report": session_report,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def recent_runs(limit: int = 12) -> List[Dict[str, Any]]:
    return list_runs(limit=limit)
