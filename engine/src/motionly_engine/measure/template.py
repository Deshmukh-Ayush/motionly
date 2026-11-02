"""Template opacity: alpha per frame from the element's full-opacity appearance.

Take the element's appearance at a full-opacity, low-motion frame (as coverage weights against the
background), warp it to each frame's measured position and scale, and solve
    m_t(p) ~= alpha_t * C_t(p)   =>   alpha_t = sum(m_t C_t) / sum(C_t^2)
by least squares over a window around the element. Unlike mass / size^2 or the core level, this
uses the element's own shape, so edge blur and thin strokes cancel out. Elements without a usable
full-opacity frame keep the previous estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FULL = 0.9  # current-method opacity at or above which a frame counts as full opacity
PAD = 3


@dataclass
class _Plan:
    el: object  # ElementSignals
    ref: int
    d: np.ndarray  # (c - bg) / |c - bg|^2, float32 (1, 3)
    off: float
    template: np.ndarray | None = None
    cx: float = 0.0  # element centroid in template coordinates
    cy: float = 0.0
    alpha: np.ndarray | None = None


def _ref_frame(el) -> int | None:
    s = el.signals
    x, y, sc, op = s["x"], s["y"], s["scale"], s["opacity"]
    w, h = s.get("width_px"), s.get("height_px")
    if w is None or h is None:
        return None
    ok = np.isfinite(x) & np.isfinite(sc) & np.isfinite(w) & np.isfinite(h) & (op >= FULL)
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return None
    # Least motion: compare with the neighbouring frames.
    def motion(i):
        js = [j for j in (i - 1, i + 1) if 0 <= j < len(x) and np.isfinite(x[j])]
        if not js:
            return 0.0
        return max(abs(x[i] - x[j]) + abs(y[i] - y[j]) + abs(sc[i] - sc[j]) * np.nanmax(w) for j in js)
    return int(min(idx, key=lambda i: (round(motion(i), 3), -op[i])))


def plan(elements, bg: np.ndarray) -> list[_Plan]:
    bg = np.asarray(bg, dtype=np.float64)
    out = []
    for el in elements:
        r = _ref_frame(el)
        if r is None:
            continue
        d = np.asarray(el.color, dtype=np.float64) - bg
        d2 = max(float(d @ d), 1e-9)
        out.append(_Plan(el, r, (d / d2).astype(np.float32).reshape(1, 3), float(bg @ d / d2)))
    return out


def _weights(frame: np.ndarray, p: _Plan, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    return cv2.transform(crop, p.d) - np.float32(p.off)


def capture_templates(frames_iter, plans: list[_Plan], width: int, height: int) -> None:
    need = {}
    for p in plans:
        need.setdefault(p.ref, []).append(p)
    last = max(need) if need else -1
    for i, frame in enumerate(frames_iter):
        for p in need.get(i, []):
            s = p.el.signals
            cx, cy = s["x"][i], s["y"][i]
            hw, hh = s["width_px"][i] / 2 + PAD, s["height_px"][i] / 2 + PAD
            x0, y0 = max(0, int(np.floor(cx - hw))), max(0, int(np.floor(cy - hh)))
            x1, y1 = min(width, int(np.ceil(cx + hw)) + 1), min(height, int(np.ceil(cy + hh)) + 1)
            t = _weights(frame, p, x0, y0, x1, y1)
            t[t < 0.05] = 0.0  # keep the element, drop background noise
            p.template, p.cx, p.cy = t, float(cx - x0), float(cy - y0)
        if i >= last:
            break


def solve_alpha(frames_iter, plans: list[_Plan], width: int, height: int, n_frames: int) -> None:
    plans = [p for p in plans if p.template is not None and float((p.template**2).sum()) > 1e-6]
    for p in plans:
        p.alpha = np.zeros(n_frames)
    for i, frame in enumerate(frames_iter):
        for p in plans:
            s = p.el.signals
            x, y = s["x"][i], s["y"][i]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(s["scale"][i])):
                continue  # not visible: alpha 0
            k = float(s["scale"][i] / s["scale"][p.ref])
            th, tw = p.template.shape
            x0 = max(0, int(np.floor(x - k * p.cx)) - PAD)
            y0 = max(0, int(np.floor(y - k * p.cy)) - PAD)
            x1 = min(width, int(np.ceil(x + k * (tw - p.cx))) + PAD)
            y1 = min(height, int(np.ceil(y + k * (th - p.cy))) + PAD)
            if x1 <= x0 or y1 <= y0:
                continue
            M = np.array([[k, 0, x - k * p.cx - x0], [0, k, y - k * p.cy - y0]], dtype=np.float64)
            c = cv2.warpAffine(p.template, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m = _weights(frame, p, x0, y0, x1, y1)
            # Least squares after the same small blur on both: resampling a thin template and the
            # codec's smearing of moving strokes differ at the pixel level, not at this scale.
            mb = cv2.GaussianBlur(m, (5, 5), 1.0).astype(np.float64)
            cb = cv2.GaussianBlur(c, (5, 5), 1.0).astype(np.float64)
            den = float((cb**2).sum())
            if den > 1e-9:
                p.alpha[i] = max(0.0, float((mb * cb).sum()) / den)
