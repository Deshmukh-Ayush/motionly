# Progress

Reference repo (gitignored): `git clone https://github.com/sidart10/rem reference/rem`

## Done

- **M0 Scaffold**: `engine/` uv package (Python 3.12), `motionly --version`, pytest.
- **M1 Easing math**: `fit/easing.py`: cubic-bezier (Newton + bisection solver, overshoot-safe),
  closed-form damped spring (mass 1) with settle time and normalized curve, CSS + Penner bezier presets,
  exact Penner functions (sine…back), GSAP aliases, `curve_distance` (max progress diff, 201 samples),
  `nearest_preset` (searches exact + bezier, reports kind). CLI: `motionly ease <name | x1,y1,x2,y2>`.
  - Note: the easings.net bezier approximations of several in-out eases are > 0.03 from the exact
    function (InOutQuint 0.032, InOutExpo 0.038, InOutCirc 0.039, InOutBack 0.036). Naming must rely
    on the exact functions; a single bezier can't hit those within the 0.03 target.

- **M2 Ground-truth generator**: `generate/`: seeded sampler (`scene.py`, RNG per `(seed, index)`),
  supersampled renderer (`render.py`, 2x, analytic shapes + Pillow text, subpixel affine placement),
  FFmpeg H.264 encoder (`encode.py`). 1–4 elements (rounded rect / circle / text) in non-overlapping
  slots, 1–2 tweens each on x / y / scale / opacity; eases ~50% named (exact Penner or CSS),
  25% custom bezier (≥ 0.03 from any preset), 25% spring; 35% stagger scenes. Truth has
  `visual_start_s` / `visual_end_s` (1% rule). CLI: `motionly generate --count 40 --seed 1 --out data/gt
  [--size mixed] [--fps 24|30|60] [--crf 26]`.
  - Set `data/gt` (seed 1): 40 clips, 159 tweens, 11 stagger (27.5%, sampling noise around 35%),
    eases bezier 56 / named 72 / spring 31, 121 s of video, about 2 min 15 s to render.
  - Text truth x/y = ink bbox centre; an intensity centroid is offset from it by a constant (scaled by `scale`).
  - TODO (after M5 shows base accuracy): rotation tweens and motion blur aren't generated yet; both are
    common in real launch videos.
  - TODO(M4): elastic and bounce eases.

- **M3 Measurer**: `measure/`: FFmpeg decode with real timestamps (ffprobe), flat background from the
  border strip, union of pixels that ever differ from background → components → merged into elements
  (identical motion joins text glyphs; a frame-to-frame "handoff" joins fast movers whose per-frame
  blobs don't overlap). Per frame: unclipped colour projection weights, summed only in a window around
  the visible pixels; x / y = weighted centroid, scale = √trace(cov) relative to the last well-visible
  frame, opacity = mass / (scale² · robust full-opacity level). CLI: `motionly measure <clip> [--plot]`.
  - Accuracy at 960×540 (5 test clips): shapes x / y < 1 px; text change-over-time < 1–1.5 px; scale < 3%;
    opacity < 0.05. Elements under ~12 px (tiny text at scale ≈ 0.4) lose contrast to compression, which
    biases mass-based scale / opacity by up to ~10%.
  - Assumption: every element is fully opaque at some point (true for the GT set).
  - About 3 s per 960×540 clip, 8–11 s per 1080p clip (three decode passes).

- **M4 Fitter**: `fit/`: segmentation (multi-scale change detection over the noise level, merges
  gaps ≤ 3 frames, minimum change x/y 1.5 px or 0.2% of the dimension, scale 2%, opacity 0.08),
  `least_squares` fits for every exact Penner function + CSS keyword (k = 4), a free bezier (k = 8,
  multi-started from the best named fits) and a spring (k = 5, ζ grid, ω₀ matched to the 10–90% time).
  Pick lowest BIC; a named fit within 2 BIC of the best wins. Visual start/end with the same 1% rule as
  the truth. Stagger / shared-timing groups. `timeline.json` in the CLAUDE.md shape (+ a `fit` block
  with the top-3 BIC ranking). CLI: `motionly fit <clip>`.
  - Known-answer tests (σ = 0.2 px): exact easeOutExpo and easeInOutExpo come back as the exact names,
    CSS `ease` as `ease`, a custom bezier within 0.03, an underdamped spring as a spring with k, c
    within 3–5%; visual start/end within ⅓ frame.
  - `clip_011` (1080p, 4-element spring stagger): springs within 0.3% of truth, starts within 1 ms,
    stagger step 0.182 s exact.
  - x / y / scale samples are ignored when the element's mass is < 30% of its peak (near-invisible).

- **M5 Benchmark**: `benchmark/`: element match by slot, tween match by visual-window overlap, metrics
  per the CLAUDE.md table (timing ±1 frame, normalised-progress trajectory ≤ 3%, named ease or curve
  distance ≤ 0.03, spurious rate) plus missed, value error, stagger detection; breakdowns by ease
  family / prop / duration / resolution / shape. CLI: `motionly benchmark --gt data/gt`.

### Latest benchmark (engine 0.1.0, 40 clips, seed 1, mixed 960×540 / 1920×1080, 30 fps)

| Metric | Target | CRF 26 | CRF 28 |
|---|---|---:|---:|
| Visual start and end within 1 frame | ≥ 90% | 73.2% | 73.9% |
| Trajectory error ≤ 3% | ≥ 90% | 76.5% | 76.5% |
| Named ease correct or dist ≤ 0.03 | ≥ 85% | 63.8% | 63.8% |
| Spurious tweens | ≤ 5% | 6.7% | 8.4% |
| Missed (diagnostic) | – | 6 / 159 | 6 / 159 |

1920×1080 is much better than 960×540 (CRF 26 timing 83% vs 63%, trajectory 84% vs 68%).
Reports: `data/runs/bench_gt_crf26/report.md`, `data/runs/bench_gt_crf28/report.md`.

Failure types (fixes proposed, not applied, awaiting approval):
1. Elements split into several measured elements (fast movers whose hand-off happens at low opacity;
   scaled text whose glyphs no longer move identically) → most spurious tweens + partial moves.
2. x / y / scale gated on mass relative to peak, which conflates small scale with low opacity → missed
   and truncated moves.
3. Free bezier beats the correct exact / CSS name on BIC because measurement error is not white noise.
4. Scale / opacity measurement bias on small elements at 960×540.
5. End time on slow ease-out tails is very sensitive to sub-1% progress error (metric conditioning).
- Truth-set note: 3 of 159 truth tweens move while the element is (nearly) invisible.

## Next

- Decide on the M5 fixes, then re-run both benchmarks.
