from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .core.io import load_species
from .paths import (
    BIRD_CLIPS_DIR,
    BIRD_SPECIES_DIR,
    CLIP_LIBRARY_JSON,
    DATABASE_ROOT,
    LEGACY_BIRD_CLIPS_DIR,
    LEGACY_BIRD_SPECIES_DIR,
    LEGACY_META_DIR,
    LEGACY_RUNS_DIR,
    LEGACY_TARGET_SPECIES_DIR,
    LEGACY_WORKSPACE_ROOT,
    META_DIR,
    RUN_INDEX_JSON,
    RUNS_DIR,
    TARGET_SPECIES_DIR,
    WORKSPACE_ROOT,
)


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "item"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _migrate_legacy_workspace() -> None:
    if not LEGACY_WORKSPACE_ROOT.exists():
        return
    # If the new layout already contains data, leave it alone.
    if any(BIRD_SPECIES_DIR.glob("*.json")) or any(BIRD_CLIPS_DIR.glob("*.csv")) or any(TARGET_SPECIES_DIR.glob("*.json")):
        return

    migrations = [
        (LEGACY_BIRD_SPECIES_DIR, BIRD_SPECIES_DIR),
        (LEGACY_BIRD_CLIPS_DIR, BIRD_CLIPS_DIR),
        (LEGACY_TARGET_SPECIES_DIR, TARGET_SPECIES_DIR),
        (LEGACY_RUNS_DIR, RUNS_DIR),
    ]
    for src_dir, dst_dir in migrations:
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.iterdir():
            dst = dst_dir / src.name
            if dst.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    if LEGACY_META_DIR.exists():
        META_DIR.mkdir(parents=True, exist_ok=True)
        for src in LEGACY_META_DIR.glob("*.json"):
            dst = META_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def ensure_workspace() -> None:
    DATABASE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [BIRD_SPECIES_DIR, BIRD_CLIPS_DIR, TARGET_SPECIES_DIR, RUNS_DIR, META_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_workspace()
    if not CLIP_LIBRARY_JSON.exists():
        CLIP_LIBRARY_JSON.write_text("[]\n", encoding="utf-8")
    if not RUN_INDEX_JSON.exists():
        RUN_INDEX_JSON.write_text("[]\n", encoding="utf-8")


# Backward-compatible public name.
def bootstrap_workspace() -> None:
    ensure_workspace()
    manifest = _load_clip_manifest()
    if manifest:
        return
    species_lookup = _species_lookup("bird")
    seeded: List[Dict[str, Any]] = []
    for csv_path in sorted(BIRD_CLIPS_DIR.glob("*.csv")):
        species_stem = infer_bird_species_from_filename(csv_path)
        if not species_stem or species_stem not in species_lookup:
            continue
        seeded.append(
            {
                "clip_id": f"clip_{uuid.uuid4().hex[:10]}",
                "name": csv_path.stem,
                "species_stem": species_stem,
                "species_name": species_lookup[species_stem]["name"],
                "csv_path": str(csv_path.relative_to(WORKSPACE_ROOT)),
                "fps": 30.0,
                "imported_at": now_iso(),
                "notes": "Bundled demo sample",
            }
        )
    _save_clip_manifest(seeded)


def list_species(group: str) -> List[Dict[str, Any]]:
    ensure_workspace()
    folder = BIRD_SPECIES_DIR if group == "bird" else TARGET_SPECIES_DIR
    out: List[Dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            species = load_species(path)
        except Exception:
            continue
        out.append(
            {
                "stem": path.stem,
                "name": species.name,
                "species_group": species.species_group,
                "height_m": species.height_m,
                "mass_kg": species.mass_kg,
                "path": str(path),
                "payload": payload,
            }
        )
    return out


def _species_lookup(group: str) -> Dict[str, Dict[str, Any]]:
    items = list_species(group)
    return {item["stem"]: item for item in items}


def import_species_json(src_path: str | Path, group: str) -> Dict[str, Any]:
    ensure_workspace()
    src = Path(src_path)
    payload = json.loads(src.read_text(encoding="utf-8"))
    if payload.get("species_group") != group:
        raise ValueError(f"Expected a {group} JSON but got {payload.get('species_group')!r}")
    stem = slugify(payload.get("name") or src.stem)
    folder = BIRD_SPECIES_DIR if group == "bird" else TARGET_SPECIES_DIR
    dst = folder / f"{stem}.json"
    shutil.copy2(src, dst)
    return {"stem": stem, "path": str(dst), "name": payload.get("name", stem), "payload": payload}



def get_species_path(group: str, stem: str) -> Path:
    folder = BIRD_SPECIES_DIR if group == "bird" else TARGET_SPECIES_DIR
    path = folder / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Species JSON not found: {path}")
    return path


def save_species_payload(payload: Dict[str, Any], group: str, overwrite_stem: str | None = None, save_as_name: str | None = None) -> Dict[str, Any]:
    ensure_workspace()
    if payload.get("species_group") != group:
        raise ValueError(f"Expected species_group='{group}' inside JSON payload")
    stem = slugify(save_as_name or payload.get("name") or overwrite_stem or "species")
    folder = BIRD_SPECIES_DIR if group == "bird" else TARGET_SPECIES_DIR
    if overwrite_stem:
        dst = folder / f"{overwrite_stem}.json"
    else:
        dst = folder / f"{stem}.json"
        suffix = 1
        while dst.exists():
            dst = folder / f"{stem}_{suffix:02d}.json"
            suffix += 1
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"stem": dst.stem, "path": str(dst), "name": payload.get("name", dst.stem), "payload": payload}

def _load_clip_manifest() -> List[Dict[str, Any]]:
    ensure_workspace()
    data = _read_json(CLIP_LIBRARY_JSON, [])
    if not isinstance(data, list):
        raise ValueError("clip_library.json must contain a list")
    return data


def _save_clip_manifest(items: List[Dict[str, Any]]) -> Path:
    return _write_json(CLIP_LIBRARY_JSON, items)


def infer_bird_species_from_filename(path: str | Path) -> str | None:
    species_lookup = _species_lookup("bird")
    stem = slugify(Path(path).stem)
    matches = [name for name in species_lookup if name in stem]
    if matches:
        return max(matches, key=len)
    return None


def list_bird_clips() -> List[Dict[str, Any]]:
    bootstrap_workspace()
    out: List[Dict[str, Any]] = []
    for item in _load_clip_manifest():
        abs_path = WORKSPACE_ROOT / item["csv_path"]
        out.append({**item, "absolute_path": str(abs_path), "exists": abs_path.exists()})
    return out


def import_bird_clip(
    src_path: str | Path,
    species_stem: str | None = None,
    fps: float = 30.0,
    display_name: str | None = None,
    notes: str = "",
) -> Dict[str, Any]:
    ensure_workspace()
    species_lookup = _species_lookup("bird")
    species_stem = species_stem or infer_bird_species_from_filename(src_path)
    if not species_stem:
        raise ValueError(
            "Could not infer bird species from the CSV filename. Import the bird JSON first and pick the species in the import dialog."
        )
    if species_stem not in species_lookup:
        raise ValueError(f"Bird species '{species_stem}' is not in the database")
    src = Path(src_path)
    stem = slugify(display_name or src.stem)
    dst = BIRD_CLIPS_DIR / f"{stem}.csv"
    suffix = 1
    while dst.exists():
        dst = BIRD_CLIPS_DIR / f"{stem}_{suffix:02d}.csv"
        suffix += 1
    shutil.copy2(src, dst)
    entry = {
        "clip_id": f"clip_{uuid.uuid4().hex[:10]}",
        "name": dst.stem,
        "species_stem": species_stem,
        "species_name": species_lookup[species_stem]["name"],
        "csv_path": str(dst.relative_to(WORKSPACE_ROOT)),
        "fps": float(fps),
        "imported_at": now_iso(),
        "notes": notes,
    }
    items = _load_clip_manifest()
    items.append(entry)
    _save_clip_manifest(items)
    return {**entry, "absolute_path": str(dst)}


def resolve_clip_paths(clip_ids: List[str] | None = None) -> List[Path]:
    items = list_bird_clips()
    selected = {item["clip_id"] for item in items} if not clip_ids else set(clip_ids)
    paths = [Path(item["absolute_path"]) for item in items if item["clip_id"] in selected and item["exists"]]
    if not paths:
        raise ValueError("No valid bird CSV clips are available for this run")
    return paths


def get_target_path(target_stem: str) -> Path:
    path = TARGET_SPECIES_DIR / f"{target_stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Target species JSON not found: {path}")
    return path


def next_run_dir(target_stem: str) -> Path:
    ensure_workspace()
    target_dir = RUNS_DIR / slugify(target_stem)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in target_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    next_idx = 1
    if existing:
        next_idx = max(int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()) + 1
    run_dir = target_dir / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def register_run(summary: Dict[str, Any]) -> Path:
    items = _read_json(RUN_INDEX_JSON, [])
    if not isinstance(items, list):
        items = []
    items.append(summary)
    return _write_json(RUN_INDEX_JSON, items)


def list_runs(limit: int | None = None) -> List[Dict[str, Any]]:
    ensure_workspace()
    runs: List[Dict[str, Any]] = []
    for report_path in sorted(RUNS_DIR.glob("*/run_*/session_report.json"), reverse=True):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["report_path"] = str(report_path)
        runs.append(payload)
    return runs[:limit] if limit else runs
