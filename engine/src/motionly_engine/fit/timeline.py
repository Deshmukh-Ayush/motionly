"""signals.json -> timeline.json (CLAUDE.md shape)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motionly_engine import __version__
from motionly_engine.fit.group import find_groups
from motionly_engine.fit.models import MoveData, fit_candidates
from motionly_engine.fit.segment import find_moves, noise_sigma
from motionly_engine.fit.select import select, tween_dict
from motionly_engine.measure import PROPS, ElementSignals, Signals

MIN_REL_MASS = 0.3  # x / y / scale are unreliable when the element is nearly invisible
SIGMA_FLOOR = {"x": 0.01, "y": 0.01, "scale": 1e-4, "opacity": 1e-3}


def change_floor(prop: str, width: int, height: int) -> float:
    return {"x": max(1.5, 0.002 * width), "y": max(1.5, 0.002 * height), "scale": 0.02, "opacity": 0.08}[prop]


def fit_signal(t: np.ndarray, v: np.ndarray, w: np.ndarray, prop: str, floor: float, frame: float) -> list[dict]:
    """t, v, w: valid samples of one signal. Returns tween dicts."""
    out = []
    sigma = max(noise_sigma(v), SIGMA_FLOOR[prop])
    for mv in find_moves(t, v, floor, SIGMA_FLOOR[prop]):
        sl = slice(mv.w0, mv.w1 + 1)
        d = MoveData(t[sl], v[sl], w[sl], mv.i0 - mv.w0, mv.i1 - mv.w0, frame, sigma)
        cands = fit_candidates(d)
        out.append(tween_dict(prop, select(cands), cands, d.t, d.v))
    return out


def fit_element(el: ElementSignals, ts: np.ndarray, width: int, height: int, frame: float) -> list[dict]:
    peak = np.nanmax(el.mass) if np.any(np.isfinite(el.mass)) else 0.0
    rel = np.where(np.isfinite(el.mass), el.mass / peak, 0.0) if peak > 0 else np.zeros_like(ts)
    tweens = []
    for prop in PROPS:
        v = el.signals[prop]
        if prop == "opacity":
            ok = np.isfinite(v)
            w = np.ones_like(v)
        else:
            ok = np.isfinite(v) & (rel >= MIN_REL_MASS)
            w = np.clip(rel / 0.6, 0.2, 1.0)
        if ok.sum() < 4:
            continue
        tweens += fit_signal(ts[ok], v[ok], w[ok], prop, change_floor(prop, width, height), frame)
    return sorted(tweens, key=lambda d: (d["start_s"], d["prop"]))


def fit_signals(sig: Signals) -> dict:
    ts = np.asarray(sig.timestamps, dtype=float)
    frame = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30
    elements, flat = [], []
    for el in sig.elements:
        tw = fit_element(el, ts, sig.width, sig.height, frame)
        x0, y0, x1, y1 = el.bbox
        elements.append({"id": el.id, "label": "element", "bbox": [x0, y0, x1 - x0, y1 - y0],
                         "bbox_kind": "swept region", "color": [round(c) for c in el.color], "tweens": tw})
        flat += [(el.id, t) for t in tw]
    return {
        "engine_version": __version__,
        "video": {"id": sig.clip, "fps": round(1 / frame, 3), "duration_s": round(float(ts[-1]) + frame, 4),
                  "width": sig.width, "height": sig.height},
        "background": [round(c) for c in sig.background],
        "elements": elements,
        "groups": find_groups(flat, frame),
    }


def fit_clip(clip: Path, out: Path, reuse: bool = True) -> dict:
    from motionly_engine.measure import measure

    sig_path = out / "signals.json"
    sig = None
    if reuse and sig_path.exists():
        d = json.loads(sig_path.read_text())
        if d.get("engine_version") == __version__:
            sig = Signals.from_dict(d)
    if sig is None:
        sig = measure(clip)
        sig.save(sig_path)
    tl = fit_signals(sig)
    (out / "timeline.json").write_text(json.dumps(tl, indent=2))
    return tl
