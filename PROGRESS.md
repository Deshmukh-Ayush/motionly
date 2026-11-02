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
  - TODO(M4): elastic and bounce need their own model.

## Next

- M2: ground-truth generator.
