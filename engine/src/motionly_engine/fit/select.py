"""Choose the ease for a move.

BIC uses the noise level measured on the signal's holds (RSS / sigma^2 + k ln n). Pick the lowest BIC,
but prefer a named ease when it is within 2 BIC points of the best, or when naming costs no more than
the noise (extra mean squared residual per sample <= sigma^2). Measurement error on real pixels is not
white noise, so a free bezier can always shave a little residual off; a name within noise is more
useful in build notes.
"""

from __future__ import annotations

import numpy as np

from motionly_engine.fit import easing
from motionly_engine.fit.models import Candidate, named_curve
from motionly_engine.fit.visual import visual_times

NAMED_MARGIN = 2.0


def select(cands: list[Candidate]) -> Candidate:
    best = min(cands, key=lambda c: c.bic)
    named = [
        c for c in cands
        if c.kind == "named"
        and (c.bic <= best.bic + NAMED_MARGIN or (c.rss - best.rss) / c.n <= c.sigma * c.sigma)
    ]
    return min(named, key=lambda c: c.bic) if named else best


def ease_curve(c: Candidate):
    """The chosen ease as a curve on u in [0, 1]."""
    if c.kind == "named":
        return named_curve(c.name)
    if c.kind == "bezier":
        p = c.bezier_params
        return lambda u: easing.bezier(u, p)
    return easing.spring_curve(c.stiffness, c.damping)


def ease_dict(c: Candidate) -> dict:
    m = easing.nearest_preset(ease_curve(c))
    if c.kind == "named":
        d = {"type": "named", "name": c.name, "kind": "exact" if c.name in easing.EXACT else "bezier"}
        if c.name in easing.PRESETS:
            d["params"] = list(easing.PRESETS[c.name])
    elif c.kind == "bezier":
        d = {"type": "bezier", "params": [round(v, 4) for v in c.bezier_params]}
    else:
        d = {"type": "spring", "stiffness": round(c.stiffness, 3), "damping": round(c.damping, 3), "mass": 1.0}
    d["nearest"] = m.name
    d["nearest_kind"] = m.kind
    d["nearest_dist"] = round(m.dist, 4)
    return d


def tween_dict(prop: str, c: Candidate, cands: list[Candidate], t: np.ndarray, v: np.ndarray) -> dict:
    dur = c.duration
    vs, ve = visual_times(c.progress, c.start, dur)
    rmse = float(np.sqrt(np.mean((c.value(t) - v) ** 2)))
    ranked = sorted(cands, key=lambda k: k.bic)[:3]
    return {
        "prop": prop,
        "from": round(c.from_, 4),
        "to": round(c.to, 4),
        "start_s": round(c.start, 4),
        "duration_s": round(dur, 4),
        "visual_start_s": round(vs, 4),
        "visual_end_s": round(ve, 4),
        "ease": ease_dict(c),
        "confidence": "measured",
        "rmse": round(rmse, 4),
        "fit": {"bic": round(c.bic, 2),
                "ranked": [{"model": k.name or k.kind, "bic": round(k.bic, 2)} for k in ranked]},
    }
