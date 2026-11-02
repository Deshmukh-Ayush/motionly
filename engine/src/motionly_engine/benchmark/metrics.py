"""Per-tween metrics for the CLAUDE.md accuracy table."""

from __future__ import annotations

import numpy as np

from motionly_engine.benchmark.curves import ease_shape, family, progress, resolve_name
from motionly_engine.fit import easing

TIMING_FRAMES = 1.0
TRAJ_MAX = 0.03
EASE_DIST = 0.03


def timing_points(tween: dict, step: float = 1e-4) -> tuple[float, float, float]:
    """(start, midpoint, end): first passes 5%, first reaches 50%, settles within 5% of the end."""
    ts = np.arange(tween["start_s"], tween["start_s"] + tween["duration_s"] + step, step)
    p = progress(tween, ts)
    moved = np.nonzero(np.abs(p) >= 0.05)[0]
    half = np.nonzero(p >= 0.5)[0]
    unsettled = np.nonzero(np.abs(p - 1.0) > 0.05)[0]
    t5 = ts[moved[0]] if len(moved) else ts[0]
    t50 = ts[half[0]] if len(half) else ts[-1]
    t95 = ts[min(unsettled[-1] + 1, len(ts) - 1)] if len(unsettled) else ts[0]
    return float(t5), float(t50), float(t95)


def tween_metrics(tt: dict, mt: dict, fps: float, shape: str) -> dict:
    frame = 1.0 / fps
    pt, pm = timing_points(tt), timing_points(mt)
    d5, d50, d95 = ((b - a) / frame for a, b in zip(pt, pm))
    ds = (mt["visual_start_s"] - tt["visual_start_s"]) / frame  # 1% times: reference only
    de = (mt["visual_end_s"] - tt["visual_end_s"]) / frame
    lo = min(tt["start_s"], mt["start_s"]) - 0.1
    hi = max(tt["start_s"] + tt["duration_s"], mt["start_s"] + mt["duration_s"]) + 0.1
    grid = np.arange(lo, hi, 1e-3)
    traj = float(np.max(np.abs(progress(tt, grid) - progress(mt, grid))))

    named = tt["ease"]["type"] == "named"
    name_ok = None
    dist = easing.curve_distance(ease_shape(tt["ease"]), ease_shape(mt["ease"]))
    if named:
        same = mt["ease"]["type"] == "named" and resolve_name(mt["ease"]["name"]) == resolve_name(tt["ease"]["name"])
        name_ok = bool(same or dist <= EASE_DIST)

    p = tt["prop"]
    if p in ("x", "y") and shape != "text":
        value_err = max(abs(mt["from"] - tt["from"]), abs(mt["to"] - tt["to"]))
    elif p in ("x", "y"):  # text: truth is the ink bbox centre, compare the change only
        value_err = abs((mt["to"] - mt["from"]) - (tt["to"] - tt["from"]))
    elif p == "scale":
        value_err = abs(mt["to"] / mt["from"] - tt["to"] / tt["from"]) / (tt["to"] / tt["from"])
    else:
        value_err = max(abs(mt["from"] - tt["from"]), abs(mt["to"] - tt["to"]))

    return {
        "t5_err_frames": round(d5, 3),
        "t50_err_frames": round(d50, 3),
        "t95_err_frames": round(d95, 3),
        "timing_ok": bool(max(abs(d5), abs(d50), abs(d95)) <= TIMING_FRAMES + 1e-9),
        "start_err_frames": round(ds, 3),
        "end_err_frames": round(de, 3),
        "trajectory_err": round(traj, 4),
        "trajectory_ok": bool(traj <= TRAJ_MAX),
        "ease_dist": round(dist, 4),
        "named": named,
        "name_ok": name_ok,
        "value_err": round(float(value_err), 4),
        "duration_err_s": round(mt["duration_s"] - tt["duration_s"], 4),
        "family": family(tt["ease"]),
    }


def duration_bucket(d: float) -> str:
    if d < 0.5:
        return "< 0.5 s"
    if d < 1.0:
        return "0.5–1.0 s"
    if d < 1.5:
        return "1.0–1.5 s"
    return "≥ 1.5 s"


def unobservable(truth_el, tween: dict, n: int = 400) -> bool:
    """Element opacity < 0.1 for more than half of the move (truth_el: generator Element)."""
    ts = np.linspace(tween["start_s"], tween["start_s"] + tween["duration_s"], n)
    return bool(np.mean(np.asarray(truth_el.value_at("opacity", ts)) < 0.1) > 0.5)


def summarize(rows: list[dict], n_spurious: int, n_measured: int, n_missed: int) -> dict:
    n = len(rows)
    named = [r for r in rows if r["named"]]
    return {
        "matched": n,
        "timing_pass": sum(r["timing_ok"] for r in rows) / n if n else 0.0,
        "trajectory_pass": sum(r["trajectory_ok"] for r in rows) / n if n else 0.0,
        "named_pass": sum(r["name_ok"] for r in named) / len(named) if named else 0.0,
        "named_n": len(named),
        "spurious_rate": n_spurious / n_measured if n_measured else 0.0,
        "spurious": n_spurious,
        "measured": n_measured,
        "missed": n_missed,
        "missed_rate": n_missed / (n + n_missed) if n + n_missed else 0.0,
    }


TARGETS = [
    ("Timing: 5% / 50% / 95% points each within 1 frame", "timing_pass", 0.90, ">="),
    ("Trajectory error ≤ 3% of the move", "trajectory_pass", 0.90, ">="),
    ("Named ease correct, or curve distance ≤ 0.03", "named_pass", 0.85, ">="),
    ("Spurious tweens", "spurious_rate", 0.05, "<="),
]


def target_rows(s: dict) -> list[tuple[str, float, str, bool]]:
    out = []
    for label, key, tgt, op in TARGETS:
        v = s[key]
        ok = v >= tgt if op == ">=" else v <= tgt
        out.append((label, v, f"{op.replace('>=', '≥').replace('<=', '≤')} {tgt:.0%}", ok))
    return out
