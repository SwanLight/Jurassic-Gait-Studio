# Jurassic Gait Studio

Jurassic Gait Studio is an app for theropod locomotion reconstruction.
It fuses multiple modern bird reference clips, automatically biases them toward the selected dinosaur target, and exports numbered dinosaur gait runs with GIF previews and detailed reports.

## Highlights

- modern `customtkinter` desktop GUI with a Jurassic-inspired dark theme
- clear end-user workflow: import bird JSON, import bird CSV, import target JSON, edit morphology JSON in-app, generate, inspect history
- built-in Morphology Lab for editing bird and dinosaur JSON files with shape preview and save/save-as
- interactive 3D gait viewer with looping playback, orbit, zoom, and frame scrubber
- multi-bird fusion with target-aware weighting
- automatic run numbering in `runs/<target>/run_001`, `run_002`, ...
- per-run `session_report.json` with selected clips, parameters, fusion weights, and reconstruction summary
- clean open-source layout with user-facing `database/` and `runs/` folders
- legacy `workspace/` migration support for earlier prototypes

## Project layout

```text
jurassic_gait_studio/
├── launch_studio.py
├── jurassic_gait_studio/
│   ├── assets/
│   ├── core/
│   ├── ui/
│   ├── paths.py
│   ├── registry.py
│   └── studio.py
├── database/
│   ├── bird_species/
│   ├── bird_clips/
│   └── target_species/
├── runs/
├── docs/
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
pip install -r requirements.txt
python launch_studio.py
```

## End-user workflow

1. Open **Library** and import bird species JSON files, bird motion CSV clips, or target dinosaur JSON files.
2. Open **Morphology Lab** to edit a bird or theropod JSON in-app and preview the body shape before saving.
3. Open **Generate**, choose the target dinosaur, choose the bird clips to contribute, and preview weights.
4. Generate a new run and inspect the GIF, fusion plot, saved files, and session report.
5. Open the built-in **3D viewer** to orbit, zoom, and scrub the looping gait inside a simulated 3D scene.
6. Use **History** to reopen previous numbered runs.

## What gets saved

Each generated run writes a folder like:

```text
runs/ornithomimid_template/run_001/
├── bird3d.gif
├── dinosaur3d.gif
├── bird_3d.csv
├── dinosaur_3d.csv
├── fusion_report.json
├── fusion_weights.png
├── session_report.json
└── summary.json
```

`session_report.json` adds higher-level app metadata on top of the core pipeline outputs so users can review exactly which clips and parameters were used.

## Bundled starter data

The app ships with:

- bird species: ostrich, emu, chicken
- bird clips: `ostrich_01.csv`, `emu_01.csv`
- theropod targets: ornithomimid, qianzhousaurus, rahonavis, microraptor, khaan, heyuannia, eotyrannus, compsognathus, buitreraptor, and a generic theropod template

## Public release note

The included amber logo and poster-style banners are prototype branding to match the Jurassic-style GUI.