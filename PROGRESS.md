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

## Next

- M3: measurer (simple backgrounds).
