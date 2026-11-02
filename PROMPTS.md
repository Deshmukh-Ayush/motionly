# Milestone prompts

Paste one at a time into Claude Code from the repo root. Don't start the next until the current one's tests pass and PROGRESS.md is updated. Start each in plan mode (Shift+Tab) and approve the plan before it writes code.

---

## M0: Scaffold

Read CLAUDE.md. Set up the `engine/` Python package with uv (Python 3.11+), the folder layout from CLAUDE.md, pytest, a `motionly` CLI entry point with a `--version` command, `.gitignore` (include `data/`, `.env`), `.env.example`, and an empty PROGRESS.md. Dependencies for now: numpy, scipy, opencv-python-headless, pillow, typer. Done when `uv run motionly --version` and `uv run pytest` both work.

## M1: Easing math

Implement `fit/easing.py`: cubic-bezier evaluation (time fraction → progress, robust for overshooting y values), a damped spring (stiffness, damping, mass 1), and a table of named presets (CSS ease family, Penner sine/quad/cubic/quart/quint/expo/circ/back in/out/in-out as cubic-bezier approximations). Add `curve_distance(a, b)` and `nearest_preset(params)`. Tests: known values (linear midpoint = 0.5, endpoints, an overshooting back-out exceeds 1), spring settles to 1.

## M2: Ground-truth generator

Implement `generate/`: a seeded scene sampler and renderer that produces test clips with known answers. Scenes have 1–4 elements (rounded rects, circles, text) on a flat background, each with 1–2 tweens on x, y, scale or opacity, using named eases, random custom beziers, and springs. 35% of scenes use stagger (same ease, offset starts). Elements must not overlap or leave their area. Render with 2× supersampling, encode H.264 at 30 fps and CRF 26 via FFmpeg, and save `clip_NNN.mp4` + `clip_NNN.json` (the truth). CLI: `motionly generate --count 40 --seed 1 --out data/gt`. Show me 3 frames from one clip so I can check it visually.

## M3: Measurer (simple backgrounds)

Implement `measure/` for the ground-truth clips: decode frames with timestamps, estimate the background, find elements as connected regions of the union of moving pixels, and for each element compute per-frame x, y (intensity-weighted centroid), scale (from second moments) and opacity (mass ÷ area). Output signals as JSON. CLI: `motionly measure <clip>`. Test on 5 generated clips: signals should follow the truth's shape (plot one element's truth vs measured into `data/runs/` as a PNG).

## M4: Fitter

Implement `fit/`: split each signal into tweens (holds, then transitions), then fit start, duration, from/to and easing with `scipy.optimize.least_squares`, multi-starting from the named presets. Also fit a spring model and pick with BIC. Report visual start/end, nearest preset and rmse. Output `timeline.json` in the CLAUDE.md shape. CLI: `motionly fit <clip>`.

## M5: Benchmark

Implement `benchmark/`: run measure + fit over the ground-truth set, match measured tweens to true ones, and report every metric in the CLAUDE.md accuracy table, broken down by ease family, property and duration bucket. Save `data/runs/<id>/report.md` and print a summary. Then tell me which cases fail most and propose fixes, but don't apply them until I approve. Repeat at 1920×1080.

## M6: Real videos

Add element finding for real launch videos behind the same interface: SAM 3 masks via fal on keyframes (env `FAL_KEY`), tracking between keyframes, camera-motion removal, and text handling. Record cost per run. Test on 3 clips I'll put in `data/real/`. Report what was found, with overlay images of masks and tracks.

## M7: Spec writer

Implement `llm/` (provider-agnostic, per CLAUDE.md) and `spec/`: given keyframes + timeline.json, write `motion.md` (palette, type, easing vocabulary, durations, stagger, pacing per scene) and `build-notes.md` (scene-by-scene shot list with exact timings, for the chosen target). Measured numbers must be copied from timeline.json, never re-estimated by the model. CLI: `motionly analyze <clip> --target hyperframes --out <dir>`.

## M8: Rebuild check

Build the output of M7 in HyperFrames, render it, run measure + fit on the render, and compare its timeline with the original's. Report the match per tween. This is the end-to-end proof that the specs work.
