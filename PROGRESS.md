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

### Fix batches (CRF 26, seed 1; timing column uses the new 5% / 50% / 95% metric from batch A on)

| Run | Timing | Trajectory | Named | Spurious | Missed | Unobservable |
|---|---:|---:|---:|---:|---:|---:|
| M5 baseline (old 1% start/end metric) | 73.2% | 76.5% | 63.8% | 6.7% | 6 | – |
| A: metric + unobservable + fixed-noise BIC | 84.9% | 77.6% | 60.3% | 6.7% | 3 | 4 |
| B: visibility + element joining | 95.5% | 86.5% | 65.7% | 7.2% | 0 | 4 |
| C: fit starts/caps/end check + core opacity + low-confidence flag | 94.8% | 85.8% | 64.3% | 7.2% | 0 | 4 |

- Batch A: CLAUDE.md timing target is now 5% start / 50% midpoint / 95% end (end = settles within 5%),
  each within 1 frame; 1% times kept in timeline.json. Moves with opacity < 0.1 for over half their
  duration are unobservable (excluded, listed separately). Selection uses BIC with σ from the
  signal's holds, prefers a named ease within 2 BIC or within σ² extra mean squared residual, and
  refits the 6 named eases closest to the bezier from the bezier's timing.
  - The named-ease score went down (63.8% → 60.3%): hold-frame σ is far smaller than the error
    during motion, so RSS/σ² magnifies the bezier's small advantage. 26 of the 27 failures were
    reported as a free bezier; in 16 the correct name was the runner-up.

- Batch B:
  - x / y / scale are used where opacity ≥ 0.1 and mass > 3 px (absolute), weighted by opacity,
    instead of mass relative to the peak.
  - Element joining:
    - A frame-to-frame handoff joins pieces by position (jump ≤ 30% of the frame) and shape
      (size within 25%, density within 1.5×) at any opacity.
    - Parts of one element are grouped when they move as one rigid body that translates and scales
      (vector between them scales with their sizes), compared on frames where both have ≥ 20% of
      their peak mass.
    - Merged pieces are re-expressed against the group's most saturated colour (moments are linear),
      so a piece only seen at low opacity no longer inflates mass. Colours must point the same way
      from the background (cos ≥ 0.85; thin text loses chroma to 4:2:0).
  - Spurious went up (6.7% → 7.2%, 12 tweens):
    - 4 from clip_025 ("Ship it" fades out, then scales while nearly invisible; still splits in two);
    - 5 on other text: y / x drift of the ink centroid while the text scales or fades (2.4–3.4 px),
      a 0.96 → 1 scale during a fade, and a 0.85 → 1 opacity during a fast move;
    - 3 small scale changes (1.4–2.7%) on shapes during their opacity tween.
  - Stagger detection 4 / 11 (was 6 / 11).

- Batch C:
  - Fitting:
    - Named eases get a second start aligned by crossings found from the end; the bezier and the
      spring get more starts (generic shapes, best named timing, a wider ζ grid).
    - Duration is capped at the data window.
    - Fits whose 1% start / end fall outside the window (±2 frames), or whose end value misses the
      following hold (> 3σ and > 3% of the move), are rejected unless every candidate fails.
  - Opacity:
    - Filled shapes (fill = mass / (core × width × height) ≥ 0.6) use the solid core: the 90th
      percentile weight, normalised to its fully-opaque level.
    - Thin shapes (text) keep the mass-based estimate.
    - The per-frame pixel extent is now in signals.json (`width_px`, `height_px`).
  - Tweens on elements under 16 px (median min(width, height) during the move) get confidence
    `low`; none in this set.
  - Net effect is about neutral:
    - Extra starts / caps fixed 5 trajectories and broke 4.
    - Core opacity broke 4 opacity timings (fade-outs).
    - Stagger detection went from 4 / 11 to 6 / 11.

### Holdout (engine after batch C, no changes; CRF 26, 30 fps, mixed sizes)

| Set | Tweens scored | Timing | Trajectory | Named | Spurious | Missed | Unobservable | Stagger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 1 (tuned on) | 155 | 94.8% | 85.8% | 64.3% | 7.2% | 0 | 4 | 6 / 11 |
| seed 2 (holdout) | 133 | 96.2% | 90.2% | 78.0% | 6.3% | 0 | 3 | 13 / 16 |

- Seed 2 scores higher than seed 1 on every metric, so the fixes don't look overfitted to seed 1;
  seed 1 has more hard cases (e.g. clip_025, clip_031).
- Weakest in both: opacity trajectory (66% / 72%) and named eases. Every named failure in both sets
  was reported as a free bezier (25 / 25 and 13 / 13), so the remaining gap is selection, not fitting.
- Reports: `data/runs/bench_gt_crf26_C/report.md`, `data/runs/bench_gt_seed2/report.md`.

## Next

- Waiting for review of the holdout before more changes.
