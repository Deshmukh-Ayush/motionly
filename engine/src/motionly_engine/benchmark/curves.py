"""Evaluate a timeline / truth tween's normalised progress and ease shape."""

from __future__ import annotations

import numpy as np

from motionly_engine.fit import easing
from motionly_engine.fit.models import named_curve


def resolve_name(name: str) -> str:
    return easing.GSAP_ALIASES.get(name, name)


def ease_shape(ease: dict):
    """Ease as a curve on u in [0, 1] (springs normalised to their settle time)."""
    if ease["type"] == "named":
        return named_curve(resolve_name(ease["name"]))
    if ease["type"] == "bezier":
        return tuple(ease["params"])
    return easing.spring_curve(ease["stiffness"], ease["damping"])


def progress(tween: dict, t: np.ndarray) -> np.ndarray:
    """Normalised progress (0 -> 1, may overshoot) at times t; holds outside the move."""
    t = np.asarray(t, dtype=float)
    e = tween["ease"]
    if e["type"] == "spring":
        p = np.asarray(easing.spring(t - tween["start_s"], e["stiffness"], e["damping"]), dtype=float)
        return np.where(t >= tween["start_s"] + tween["duration_s"], 1.0, p)
    u = np.clip((t - tween["start_s"]) / tween["duration_s"], 0.0, 1.0)
    f = ease_shape(e)
    return np.asarray(easing.bezier(u, f) if isinstance(f, tuple) else f(u), dtype=float)


def family(ease: dict) -> str:
    if ease["type"] == "spring":
        return "spring"
    if ease["type"] == "bezier":
        return "custom bezier"
    name = resolve_name(ease["name"])
    if name in ("ease", "ease-in", "ease-out", "ease-in-out"):
        return "CSS"
    if name == "linear":
        return "linear"
    for fam in ("Sine", "Quad", "Cubic", "Quart", "Quint", "Expo", "Circ", "Back"):
        if name.endswith(fam):
            return fam
    return "other"
