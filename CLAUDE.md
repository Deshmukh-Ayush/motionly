# Motionly

Motionly is a curated gallery of the best motion graphics videos from X. Every video has two copyable files, `motion.md` and `build-notes.md`, that let a user rebuild the video in Claude Code, Claude Motion, HyperFrames or Remotion.

**This repo currently holds only the engine**: the pipeline that turns a reference video into those two files. The website (`web/`, Next.js) comes later, after the engine passes its benchmark.

The engine's edge is **measured** timing and easing. LLMs guess durations and curves from frames; we measure them from pixels first, then hand the numbers to the LLM.

## Environment

- WSL2 Ubuntu on Windows 11. Repo lives in the Linux filesystem (`~/motionly`), never `/mnt/c`.
- Python via `uv` (never bare pip). FFmpeg 6.1. Node LTS (for HyperFrames later).
- The machine struggles with parallel heavy work: run **one** heavy process at a time (no parallel renders or benchmark shards), and run only the tests related to the change unless asked for the full suite.

## Repo layout

```
engine/
  pyproject.toml
  src/motionly_engine/
    generate/    # ground-truth test clips with known timing + easing
    measure/     # video -> per-element signals over time
    fit/         # signals -> tweens with fitted easing
    benchmark/   # compare fitted results against ground truth
    spec/        # timeline.json -> motion.md + build-notes.md (LLM)
    llm/         # provider-agnostic LLM client
    cli.py       # `motionly ...` commands
  tests/
data/            # gitignored: clips, frames, outputs, reports
reference/rem/   # REM (MIT) cloned for reference. Read-only. Never import from it.
web/             # later
PROGRESS.md      # what's done, what's next, latest benchmark numbers
```

## The pipeline

1. **Scenes**: FFmpeg scene detection splits the clip into shots.
2. **Frames**: decode frames with their real timestamps (never assume constant frame rate).
3. **Elements**: find and track each moving element. Simple backgrounds: background subtraction + connected components. Real videos: SAM 3 masks on keyframes (via fal), local tracking in between.
4. **Signals**: per element, per frame: `x`, `y`, `scale`, `rotation`, `opacity`. Use intensity-weighted moments so fades don't move the centroid.
5. **Tweens**: split each signal into moves (holds, then transitions).
6. **Fit**: per tween, fit start time, duration, from/to and easing with subframe precision:
   - cubic-bezier `(x1, y1, x2, y2)`, x in [0,1], y may overshoot;
   - damped spring (stiffness, damping, mass = 1) for overshoot-and-settle;
   - choose between them with BIC; also report the nearest named preset (CSS / Penner / GSAP families) and its distance.
7. **Group**: detect stagger (same ease, offset starts) and shared timing across elements.
8. **timeline.json**: the measured result (schema below).
9. **Spec writer**: LLM reads keyframes + timeline.json and writes `motion.md` and `build-notes.md` for the requested target (`hyperframes`, `remotion`, `claude-motion`).

### timeline.json (minimum shape)

```json
{
  "engine_version": "0.1.0",
  "video": {"id": "...", "fps": 30, "duration_s": 6.2, "width": 1920, "height": 1080},
  "elements": [
    {"id": "el0", "label": "headline text", "bbox": [x, y, w, h],
     "tweens": [
       {"prop": "y", "from": 40, "to": 0, "start_s": 0.43, "duration_s": 0.62,
        "visual_start_s": 0.45, "visual_end_s": 1.01,
        "ease": {"type": "bezier", "params": [0.16, 1, 0.3, 1], "nearest": "easeOutExpo", "nearest_dist": 0.02},
        "confidence": "measured", "rmse": 0.4}
     ]}
  ],
  "groups": [{"type": "stagger", "elements": ["el0", "el1"], "step_s": 0.08}]
}
```

`confidence` is one of `measured` (read from pixels), `fitted` (derived) or `inferred` (LLM judgement). Never present an inferred value as measured.

## Accuracy targets (definition of done for the engine)

Measured on the ground-truth benchmark: 30 fps, H.264 CRF 26–28 (like an X upload), at 960×540 and 1920×1080.

| Metric | Target |
|---|---|
| Visual start and end time | within 1 frame on ≥ 90% of tweens |
| Trajectory error (max progress difference vs truth) | ≤ 3% of the move on ≥ 90% of tweens |
| Named ease correct, or curve distance ≤ 0.03 | ≥ 85% of named-ease tweens |
| Spurious tweens (detected but not in truth) | ≤ 5% |

"Visual start/end" = when the move passes 1% of its change and when it settles within 1%. Compare these instead of raw durations, because long ease-out tails make duration ambiguous.

## Rules

- **Small milestones.** Do one milestone at a time (see PROMPTS.md). Each one ends with passing tests, a CLI command that runs it, and an updated PROGRESS.md.
- **Tests with pytest.** Known-answer tests for the math (bezier evaluation, spring, fitting on clean synthetic curves) before tests on rendered video.
- **Deterministic.** All randomness takes a seed. Same input + same engine_version = same output.
- **LLM access only through `llm/`.** Read `LLM_PROVIDER` (`openrouter` | `anthropic`), `LLM_MODEL` and `LLM_API_KEY` from `.env`. The chosen model must accept images. No provider SDK calls anywhere else.
- **Secrets** live in `.env` (gitignored). Keep `.env.example` updated. Keys: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `FAL_KEY`.
- **Log cost.** Every paid external call (LLM, fal) records `cost_usd` into the run's output.
- **Ask before adding heavy dependencies** (PyTorch, model weights, GPU libraries). Prefer NumPy, SciPy, OpenCV.
- **REM is a reference, not a dependency.** Read its approach; reimplement what we need with attribution in comments.
- **Data hygiene.** Test clips go in `data/`, which is gitignored. Production code never downloads from X; gallery source clips are added manually.
- **Plain output.** CLI prints a short summary; full results go to JSON files in `data/runs/<run-id>/`.

## How the website will use the engine (context only)

- One engine run per gallery video, triggered by the first paying user who clicks Generate; results are cached by `video_id + engine_version` and served to everyone after.
- So the engine must be callable as a single function / CLI: `motionly analyze <clip> --target hyperframes --out <dir>`, returning both files plus timeline.json and cost.
