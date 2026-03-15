# Open-source release notes

## Packaging goals

This release intentionally presents the project as a user-facing desktop application instead of a research prototype dump.

- `database/` contains user-editable bird and target libraries.
- `runs/` contains numbered generation outputs.
- `.studio/` stores hidden app metadata such as the clip registry and run index.
- older cache folders, prototype outputs, and `__pycache__` folders are not shipped.

## Branding note

The bundled amber logo and poster banners are included only as prototype styling assets to match the requested Jurassic-inspired UI direction. If this repository is published broadly, replacing those assets with original branding is the safer option.

## New in this release

- Morphology Lab lets users edit bird and theropod JSON files without leaving the GUI.
- The built-in 3D viewer gives looping playback with orbit, zoom, and frame scrubbing for generated runs.
- GIF rendering now uses thicker shaded capsules and clearer joint separation for a more volumetric look.
