from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent
ASSETS_DIR = PACKAGE_ROOT / "assets"

# New cleaner open-source layout.
DATABASE_ROOT = APP_ROOT / "database"
BIRD_SPECIES_DIR = DATABASE_ROOT / "bird_species"
BIRD_CLIPS_DIR = DATABASE_ROOT / "bird_clips"
TARGET_SPECIES_DIR = DATABASE_ROOT / "target_species"
RUNS_DIR = APP_ROOT / "runs"
META_DIR = APP_ROOT / ".studio"

# Legacy v1 layout kept only for migration.
LEGACY_WORKSPACE_ROOT = APP_ROOT / "workspace"
LEGACY_BIRD_SPECIES_DIR = LEGACY_WORKSPACE_ROOT / "bird_species"
LEGACY_BIRD_CLIPS_DIR = LEGACY_WORKSPACE_ROOT / "bird_clips"
LEGACY_TARGET_SPECIES_DIR = LEGACY_WORKSPACE_ROOT / "target_species"
LEGACY_RUNS_DIR = LEGACY_WORKSPACE_ROOT / "runs"
LEGACY_META_DIR = LEGACY_WORKSPACE_ROOT / ".studio"

CLIP_LIBRARY_JSON = META_DIR / "clip_library.json"
RUN_INDEX_JSON = META_DIR / "run_index.json"

# Backward-compatible alias used by older imports/tests.
WORKSPACE_ROOT = APP_ROOT
