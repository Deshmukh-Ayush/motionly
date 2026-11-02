"""Visual start / end of a move: when progress passes 1% of the change and when it settles within 1%.

Same rule as the ground-truth generator (generate/scene.py Tween.visual_times), reimplemented
here so the fitter never depends on the generator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

VISUAL_TOL = 0.01


def visual_times(progress_at: Callable[[np.ndarray], np.ndarray], start_s: float, duration_s: float,
                 step: float = 1e-4) -> tuple[float, float]:
    ts = np.arange(start_s, start_s + duration_s + step, step)
    p = np.asarray(progress_at(ts), dtype=float)
    moved = np.nonzero(np.abs(p) >= VISUAL_TOL)[0]
    unsettled = np.nonzero(np.abs(p - 1.0) > VISUAL_TOL)[0]
    vs = ts[moved[0]] if len(moved) else start_s
    ve = ts[min(unsettled[-1] + 1, len(ts) - 1)] if len(unsettled) else start_s
    return float(vs), float(ve)
