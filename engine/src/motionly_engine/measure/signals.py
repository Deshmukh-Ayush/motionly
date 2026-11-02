"""Per-element signals: x, y (intensity-weighted centroid), scale (second moments), opacity (mass / area)."""

from __future__ import annotations

import cv2
import numpy as np

from motionly_engine.measure.elements import Component

MOMENTS = ("m0", "mx", "my", "mxx", "myy")


class MomentAccumulator:
    """Per component and frame: raw weighted moments, additive so components can be merged.

    Weights are the unclipped projection m = (p - bg) . (c - bg) / |c - bg|^2 (coverage x opacity),
    so zero-mean compression noise averages out instead of biasing. To keep far-away noise out of
    the moments, each frame only sums a window around the pixels where the (blurred) weight is
    clearly above noise.
    """

    def __init__(self, comps: list[Component], bg: np.ndarray, noise_l1: float = 2.0):
        self.comps = comps
        self.bg = bg.astype(np.float64)
        self.rows: list[list[np.ndarray]] = [[] for _ in comps]
        self._dirs, self._offs, self._grids, self._tau = [], [], [], []
        for c in comps:
            x0, y0, x1, y1 = c.bbox
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            self._grids.append((xx, yy))
            d = c.color - self.bg
            d2 = max(float(d @ d), 1e-9)
            self._dirs.append((d / d2).astype(np.float32).reshape(1, 3))
            self._offs.append(float(self.bg @ d / d2))
            # Noise of m per pixel ~ per-channel noise / |c - bg|; a 5x5 blur divides it by 5.
            self._tau.append(max(0.03, 4 * (noise_l1 / 3) / np.sqrt(d2) / 5))

    def add(self, frame: np.ndarray) -> None:
        for i, c in enumerate(self.comps):
            x0, y0, x1, y1 = c.bbox
            m = cv2.transform(frame[y0:y1, x0:x1].astype(np.float32), self._dirs[i]) - np.float32(self._offs[i])
            m[~c.mask] = 0
            fg = cv2.blur(m, (5, 5)) > self._tau[i]
            rows, cols = np.flatnonzero(fg.any(axis=1)), np.flatnonzero(fg.any(axis=0))
            if len(rows) == 0:
                self.rows[i].append(np.zeros(5))
                continue
            r0, r1 = max(0, rows[0] - 3), rows[-1] + 4
            c0, c1 = max(0, cols[0] - 3), cols[-1] + 4
            w = m[r0:r1, c0:c1].astype(np.float64)
            xx, yy = self._grids[i][0][r0:r1, c0:c1], self._grids[i][1][r0:r1, c0:c1]
            wx, wy = w * xx, w * yy
            self.rows[i].append(np.array([w.sum(), wx.sum(), wy.sum(), (wx * xx).sum(), (wy * yy).sum()]))

    def moments(self) -> list[np.ndarray]:
        return [np.array(r) for r in self.rows]  # each (n_frames, 5)


def signals_from_moments(mom: np.ndarray, vis_frac: float = 0.02) -> dict[str, np.ndarray]:
    m0 = mom[:, 0]
    peak = np.max(m0) if len(m0) else 0.0
    vis = m0 > vis_frac * peak
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(vis, mom[:, 1] / m0, np.nan)
        y = np.where(vis, mom[:, 2] / m0, np.nan)
        var = mom[:, 3] / m0 - x * x + mom[:, 4] / m0 - y * y
        size = np.where(vis & (var > 0), np.sqrt(np.abs(var)), np.nan)
    idx = np.nonzero(np.isfinite(size))[0]
    # Scale is relative to the last well-visible frame (base size is unknowable from pixels).
    strong = np.nonzero(np.isfinite(size) & (m0 >= 0.5 * peak))[0]
    ref_i = strong[-1] if len(strong) else (idx[-1] if len(idx) else None)
    scale = size / size[ref_i] if ref_i is not None else size
    # Opacity: mass / (size^2 * reference), reference = (robust) max over visible frames,
    # i.e. assumes the element is fully opaque at some point.
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = m0 / size**2
    if len(idx):
        d = dens[idx]
        ref = float(np.median(d[d >= 0.9 * np.max(d)]))  # robust "fully opaque" level
        opacity = np.where(np.isfinite(dens), dens / ref, 0.0)
        opacity = np.where(vis, opacity, np.clip(m0 / (ref * size[idx[-1]] ** 2), 0, None))
    else:
        opacity = np.zeros_like(m0)
    return {"x": x, "y": y, "scale": scale, "opacity": opacity, "size_px": size}


def _same_motion(sa: dict, sb: dict, ma: np.ndarray, mb: np.ndarray) -> bool:
    """Visible on the same frames, translating and scaling identically, constant mass ratio."""
    va, vb = np.isfinite(sa["x"]), np.isfinite(sb["x"])
    if va.sum() < 3 or np.any(va != vb):
        return False
    for p in ("x", "y"):
        da, db = sa[p][va] - sa[p][va][0], sb[p][va] - sb[p][va][0]
        if np.max(np.abs(da - db)) >= 0.5:
            return False
    ratio = ma[va, 0] / mb[va, 0]
    if np.std(ratio) / abs(np.mean(ratio)) >= 0.05:
        return False
    za, zb = sa["size_px"][va] / sa["size_px"][va][-1], sb["size_px"][va] / sb["size_px"][va][-1]
    return bool(np.max(np.abs(za - zb)) < 0.03)


def _handoff(ma: np.ndarray, mb: np.ndarray, vis: float = 0.02, strong: float = 0.5) -> bool:
    """A fast mover leaves gaps in the union: the element is in A on frame t and in B on t+1."""
    a, b = ma[:, 0], mb[:, 0]
    pa, pb = a.max(), b.max()
    if pa <= 0 or pb <= 0:
        return False
    va, vb = a > vis * pa, b > vis * pb
    if np.any(va & vb):
        return False
    sa, sb = a >= strong * pa, b >= strong * pb
    for x, y, sx, sy in ((a, b, sa, sb), (b, a, sb, sa)):
        hits = np.nonzero(sx[:-1] & sy[1:])[0]
        for t in hits:
            r = x[t] / y[t + 1]
            if 0.67 <= r <= 1.5:
                return True
    return False


def merge_components(moms: list[np.ndarray], comps: list[Component], min_dim: int) -> list[list[int]]:
    """Group components that are one element: glyphs moving together, or a fast mover split by gaps."""
    groups = [[i] for i in range(len(moms))]

    def bbox(g):
        return (min(comps[i].bbox[0] for i in g), min(comps[i].bbox[1] for i in g),
                max(comps[i].bbox[2] for i in g), max(comps[i].bbox[3] for i in g))

    gap_max = 0.03 * min_dim
    changed = True
    while changed:
        changed = False
        gm = [sum(moms[i] for i in g) for g in groups]
        gs = [signals_from_moments(m) for m in gm]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if _handoff(gm[a], gm[b]):
                    ok = True
                else:
                    ba, bb = bbox(groups[a]), bbox(groups[b])
                    gap = max(bb[0] - ba[2], ba[0] - bb[2], bb[1] - ba[3], ba[1] - bb[3], 0)
                    ok = gap <= gap_max and _same_motion(gs[a], gs[b], gm[a], gm[b])
                if ok:
                    groups[a] = sorted(groups[a] + groups[b])
                    del groups[b]
                    changed = True
                    break
            if changed:
                break
    return sorted(groups, key=lambda g: min(g))
