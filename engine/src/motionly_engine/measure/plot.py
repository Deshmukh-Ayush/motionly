"""Debug plot: truth vs measured signals for one element (Pillow, no plotting dependency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SURFACE = (252, 252, 251)
INK = (11, 11, 11)
MUTED = (110, 110, 105)
GRID = (226, 225, 220)
TRUTH = (42, 120, 214)  # categorical slot 1 (blue)
MEASURED = (235, 104, 52)  # categorical slot 2 (orange)


def plot_element(t: np.ndarray, series: dict[str, tuple[np.ndarray, np.ndarray]], title: str, path: Path) -> Path:
    """series: prop -> (truth, measured). One small-multiple panel per prop, shared time axis."""
    W, ph, left, top, gap = 900, 170, 70, 56, 28
    H = top + len(series) * (ph + gap) + 30
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    d.text((left, 12), title, fill=INK, font=ImageFont.load_default(size=16))
    # Legend (always present for two series).
    lx = W - 260
    d.line([(lx, 22), (lx + 24, 22)], fill=TRUTH, width=2)
    d.text((lx + 30, 15), "truth", fill=INK, font=font)
    d.ellipse([lx + 100, 18, lx + 108, 26], fill=MEASURED)
    d.text((lx + 116, 15), "measured", fill=INK, font=font)
    t0, t1 = float(t.min()), float(t.max())
    pw = W - left - 20
    for k, (prop, (tv, mv)) in enumerate(series.items()):
        y0 = top + k * (ph + gap)
        vals = np.concatenate([tv[np.isfinite(tv)], mv[np.isfinite(mv)]])
        lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def X(v):
            return left + (v - t0) / max(t1 - t0, 1e-9) * pw

        def Y(v):
            return y0 + ph - (v - lo) / (hi - lo) * ph

        for g in np.linspace(lo + pad, hi - pad, 3):
            d.line([(left, Y(g)), (left + pw, Y(g))], fill=GRID, width=1)
            dec = max(0, int(np.ceil(-np.log10(max(hi - lo, 1e-9)))) + 2)
            d.text((6, Y(g) - 7), f"{g:.{dec}f}", fill=MUTED, font=font)
        d.text((left, y0 - 18), prop, fill=INK, font=font)
        pts = [(X(a), Y(b)) for a, b in zip(t, tv) if np.isfinite(b)]
        if len(pts) > 1:
            d.line(pts, fill=TRUTH, width=2)
        for a, b in zip(t, mv):
            if np.isfinite(b):
                d.ellipse([X(a) - 3, Y(b) - 3, X(a) + 3, Y(b) + 3], fill=MEASURED, outline=SURFACE)
    for s in np.arange(np.ceil(t0 * 2) / 2, t1 + 1e-9, 0.5):
        x = left + (s - t0) / max(t1 - t0, 1e-9) * pw
        d.text((x - 8, H - 24), f"{s:.1f}s", fill=MUTED, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def truth_vs_measured(signals, truth_scene, element_index: int = 0):
    """Match a truth element to measured signals by slot; return (t, series, title).

    Text x/y are compared as change from the first visible frame (truth = ink bbox centre,
    measured = intensity centroid). Scale is a ratio to the last visible frame for both.
    """
    te = truth_scene.elements[element_index]
    t = signals.timestamps
    me = None
    for e in signals.elements:
        cx, cy = (e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2
        if te.slot[0] <= cx <= te.slot[2] and te.slot[1] <= cy <= te.slot[3]:
            me = e
    if me is None:
        raise ValueError(f"no measured element inside {te.id}'s slot")
    vis = np.isfinite(me.signals["x"])
    i0 = int(np.argmax(vis))
    series = {}
    for p in ("x", "y", "scale", "opacity"):
        tv, mv = np.asarray(te.value_at(p, t), dtype=float), me.signals[p].copy()
        if p in ("x", "y") and te.shape == "text":
            tv, mv = tv - tv[i0], mv - mv[i0]
            p = f"{p} (change from start)"
        if p == "scale":
            ref = int(np.nonzero(np.isfinite(mv) & (me.mass >= 0.5 * np.nanmax(me.mass)))[0][-1])
            tv, mv = tv / tv[ref], mv / mv[ref]
            p = "scale (ratio to end)"
        series[p] = (tv, mv)
    return t, series, f"{truth_scene.id} {te.id} ({te.label}): truth vs measured", me
