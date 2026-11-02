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

MIN_OPACITY = 0.1  # x / y / scale are unreliable when the element is nearly invisible
LOW_CONF_PX = 16  # tweens on elements smaller than this (min of width, height) get confidence "low"
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
    # Visibility from the opacity estimate and the absolute signal (not mass relative to the peak,
    # which would confuse a small element with a faint one).
    from motionly_engine.measure.signals import MIN_MASS

    op = np.nan_to_num(el.signals["opacity"], nan=0.0)
    mass = np.nan_to_num(el.mass, nan=0.0)
    seen = (op >= MIN_OPACITY) & (mass > MIN_MASS)
    tweens = []
    for prop in PROPS:
        v = el.signals[prop]
        if prop == "opacity":
            ok = np.isfinite(v)
            w = np.ones_like(v)
        else:
            ok = np.isfinite(v) & seen
            w = np.clip(op / 0.5, 0.2, 1.0)
        if ok.sum() < 4:
            continue
        tweens += fit_signal(ts[ok], v[ok], w[ok], prop, change_floor(prop, width, height), frame)
    # Unobservable moves (mirror of the benchmark's truth rule): drop tweens during which the element's
    # opacity is below 0.1 for more than half the move; what is measured there is mostly noise.
    kept = []
    for tw in tweens:
        sel = (ts >= tw["start_s"]) & (ts <= tw["start_s"] + tw["duration_s"])
        if sel.sum() >= 2 and np.mean(op[sel] < MIN_OPACITY) > 0.5:
            continue
        kept.append(tw)
    tweens = kept
    # Small elements: compression and edge blur bias their measurements.
    extent = np.fmin(el.signals.get("width_px", np.full_like(ts, np.nan)),
                     el.signals.get("height_px", np.full_like(ts, np.nan)))
    for tw in tweens:
        sel = (ts >= tw["start_s"]) & (ts <= tw["start_s"] + tw["duration_s"]) & np.isfinite(extent)
        px = float(np.median(extent[sel])) if sel.any() else float("nan")
        tw["min_extent_px"] = None if not np.isfinite(px) else round(px, 1)
        if np.isfinite(px) and px < LOW_CONF_PX:
            tw["confidence"] = "low"
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
