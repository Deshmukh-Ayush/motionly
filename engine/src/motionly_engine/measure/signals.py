"""Per-element signals: x, y (intensity-weighted centroid), scale (second moments), opacity (solid core, or mass for thin shapes like text)."""

from __future__ import annotations

import warnings

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
                self.rows[i].append(np.array([0.0] * 6 + [np.nan] * 4))
                continue
            r0, r1 = max(0, rows[0] - 3), rows[-1] + 4
            c0, c1 = max(0, cols[0] - 3), cols[-1] + 4
            w = m[r0:r1, c0:c1].astype(np.float64)
            xx, yy = self._grids[i][0][r0:r1, c0:c1], self._grids[i][1][r0:r1, c0:c1]
            wx, wy = w * xx, w * yy
            # Solid core: the 90th percentile of the weight over detected pixels is coverage 1 x opacity
            # inside a filled shape, independent of its size and edge blur.
            core = float(np.percentile(m[fg], 90))
            y0, x0 = c.bbox[1], c.bbox[0]
            self.rows[i].append(np.array([
                w.sum(), wx.sum(), wy.sum(), (wx * xx).sum(), (wy * yy).sum(), core,
                x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1,
            ]))

    def moments(self) -> list[np.ndarray]:
        # each (n_frames, 10): m0, mx, my, mxx, myy (additive), core (max), extent x0, y0, x1, y1
        return [np.array(r) for r in self.rows]


def combine(parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Merge component moments (each with its colour rescale factor)."""
    out = sum(m[:, :5] * k for m, k in parts)
    core = np.max(np.stack([m[:, 5] * k for m, k in parts]), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames: element not visible
        lo = np.nanmin(np.stack([m[:, 6:8] for m, _ in parts]), axis=0)
        hi = np.nanmax(np.stack([m[:, 8:10] for m, _ in parts]), axis=0)
    return np.column_stack([out, core, lo, hi])


MIN_MASS = 3.0  # visible: at least ~3 fully covered pixels' worth of signal (absolute, not relative to peak)


SOLID_CORE = 0.85  # raw core level (in coverage units) at full opacity above which the core is trusted
SOLID_FILL = 0.6  # filled shapes (rect ~1, circle ~0.79) use the core; thin shapes (text) use mass


def _opacity_from_mass(m0: np.ndarray, size: np.ndarray, idx: np.ndarray, peak: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = m0 / size**2
    if not len(idx):
        return np.zeros_like(m0)
    d = dens[idx]
    ref = float(np.median(d[d >= 0.9 * np.max(d)]))  # robust "fully opaque" level
    # When the element is faint its size estimate is noisy: use the size from the nearest frame
    # with a solid signal instead.
    solid = np.nonzero(np.isfinite(size) & (m0 >= 0.05 * peak))[0]
    if len(solid):
        k = np.arange(len(m0))
        near = solid[np.clip(np.searchsorted(solid, k), 0, len(solid) - 1)]
        prev = solid[np.clip(np.searchsorted(solid, k) - 1, 0, len(solid) - 1)]
        pick = np.where(np.abs(prev - k) < np.abs(near - k), prev, near)
        size_for_op = np.where(m0 >= 0.05 * peak, size, size[pick])
    else:
        size_for_op = size
    with np.errstate(divide="ignore", invalid="ignore"):
        opacity = np.clip(m0, 0, None) / (ref * size_for_op**2)
    return np.where(np.isfinite(opacity), opacity, 0.0)


def signals_from_moments(mom: np.ndarray) -> dict[str, np.ndarray]:
    m0 = mom[:, 0]
    peak = np.max(m0) if len(m0) else 0.0
    vis = m0 > MIN_MASS
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(vis, mom[:, 1] / m0, np.nan)
        y = np.where(vis, mom[:, 2] / m0, np.nan)
        var = mom[:, 3] / m0 - x * x + mom[:, 4] / m0 - y * y
        size = np.where(vis & (var > 0), np.sqrt(np.abs(var)), np.nan)
    idx = np.nonzero(np.isfinite(size))[0]
    width = np.where(vis, mom[:, 8] - mom[:, 6], np.nan)
    height = np.where(vis, mom[:, 9] - mom[:, 7], np.nan)
    # Scale is relative to the last well-visible frame (base size is unknowable from pixels).
    strong = np.nonzero(np.isfinite(size) & (m0 >= 0.5 * peak))[0]
    ref_i = strong[-1] if len(strong) else (idx[-1] if len(idx) else None)
    scale = size / size[ref_i] if ref_i is not None else size
    # Opacity. Assumes the element is fully opaque at some point.
    # Elements with a solid core (shapes): the core level (coverage-1 pixels) normalised to its robust
    # "fully opaque" value; it does not drift with size or edge blur.
    # Thin elements (text: few interior pixels, and compression smears moving strokes):
    # mass / (size^2 * reference), as before.
    core = np.where(m0 > 0, np.clip(mom[:, 5], 0, None), 0.0)
    good = core[vis] if vis.any() else core
    core_ref = float(np.median(good[good >= 0.9 * np.max(good)])) if len(good) and np.max(good) > 0 else 0.0
    # Fill = mass / (core x extent area): ~1 for rects, ~0.79 for circles, much lower for text.
    strong = vis & (m0 >= 0.5 * peak) & (core > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = m0[strong] / (core[strong] * width[strong] * height[strong])
    fill = float(np.median(fill)) if strong.any() else 0.0
    if core_ref >= SOLID_CORE and fill >= SOLID_FILL:
        opacity = core / core_ref
    else:
        opacity = _opacity_from_mass(m0, size, idx, peak)
    return {"x": x, "y": y, "scale": scale, "opacity": opacity, "size_px": size,
            "width_px": width, "height_px": height}


def _rigid_scaled(sa: dict, sb: dict, ma: np.ndarray, mb: np.ndarray) -> bool:
    """Parts of one element: visible on the same frames and moving as one rigid body that translates
    and scales about a shared centre (so the vector between them scales with their sizes), with a
    constant mass ratio. Covers glyphs of static, moving and scaling text."""
    va, vb = np.isfinite(sa["x"]), np.isfinite(sb["x"])
    if va.sum() < 3 or np.any(va != vb):
        return False
    # Compare motion only where both parts are clearly visible (faint frames are noisy).
    va = va & (ma[:, 0] >= 0.2 * np.max(ma[:, 0])) & (mb[:, 0] >= 0.2 * np.max(mb[:, 0]))
    if va.sum() < 3:
        return False
    ref = int(np.nonzero(va)[0][-1])
    za, zb = sa["size_px"][va] / sa["size_px"][ref], sb["size_px"][va] / sb["size_px"][ref]
    if not (np.all(np.isfinite(za)) and np.all(np.isfinite(zb))) or np.max(np.abs(za - zb)) >= 0.03:
        return False
    s = (za + zb) / 2
    dx = sa["x"][va] - sb["x"][va]
    dy = sa["y"][va] - sb["y"][va]
    d0x, d0y = sa["x"][ref] - sb["x"][ref], sa["y"][ref] - sb["y"][ref]
    err = np.hypot(dx - s * d0x, dy - s * d0y)
    if np.max(err) >= 0.5 + 0.01 * np.hypot(d0x, d0y):
        return False
    ratio = ma[va, 0] / mb[va, 0]
    return bool(np.std(ratio) / abs(np.mean(ratio)) < 0.05)


def _handoff(sa: dict, sb: dict, ma: np.ndarray, mb: np.ndarray, max_jump: float) -> bool:
    """A fast mover leaves gaps in the union: the element is in A on frame t and in B on t+1.
    Joined by position (a plausible jump) and shape (same size and density), at any opacity."""
    va, vb = np.isfinite(sa["x"]), np.isfinite(sb["x"])
    if not va.any() or not vb.any() or np.any(va & vb):
        return False
    for s1, s2, m1, m2, v1, v2 in ((sa, sb, ma, mb, va, vb), (sb, sa, mb, ma, vb, va)):
        for t in np.nonzero(v1[:-1] & v2[1:])[0]:
            z1, z2 = s1["size_px"][t], s2["size_px"][t + 1]
            if not (np.isfinite(z1) and np.isfinite(z2)) or not 0.8 <= z1 / z2 <= 1.25:
                continue
            d1, d2 = m1[t, 0] / z1**2, m2[t + 1, 0] / z2**2
            if not 0.67 <= d1 / d2 <= 1.5:
                continue
            if np.hypot(s2["x"][t + 1] - s1["x"][t], s2["y"][t + 1] - s1["y"][t]) <= max_jump:
                return True
    return False


def merge_components(moms: list[np.ndarray], comps: list[Component], bg: np.ndarray, min_dim: int,
                     max_dim: int | None = None) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    """Group components that are one element: parts moving as one rigid (scaling) body, or a fast
    mover split by gaps. Returns (component indices, moments, colour) per element.

    Each component's colour is estimated from the most saturated pixels it ever saw, so a component
    only seen at low opacity gets a faded colour and an inflated mass. Moments are linear in the
    colour projection, so a group re-expresses every member against its most saturated colour:
    m_ref = m_i * |c_i - bg| / |c_ref - bg| (colours must point the same way from the background).
    """
    bg = np.asarray(bg, dtype=np.float64)
    dirs = [np.asarray(c.color, dtype=np.float64) - bg for c in comps]
    norms = [max(float(np.linalg.norm(d)), 1e-9) for d in dirs]
    groups = [[i] for i in range(len(moms))]

    def ref_of(g):
        return max(g, key=lambda i: norms[i])

    def moments(g, ref):
        return combine([(moms[i], norms[i] / norms[ref]) for i in g])

    def parallel(i, j):
        # Loose: thin text strokes lose chroma to 4:2:0 subsampling, so colour estimates of the same
        # element can differ; size, density and position checks still have to agree.
        return float(dirs[i] @ dirs[j]) / (norms[i] * norms[j]) >= 0.85

    def bbox(g):
        return (min(comps[i].bbox[0] for i in g), min(comps[i].bbox[1] for i in g),
                max(comps[i].bbox[2] for i in g), max(comps[i].bbox[3] for i in g))

    gap_max = 0.03 * min_dim
    max_jump = 0.3 * (max_dim or min_dim)
    changed = True
    while changed:
        changed = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ra, rb = ref_of(groups[a]), ref_of(groups[b])
                if not parallel(ra, rb):
                    continue
                ref = ra if norms[ra] >= norms[rb] else rb
                ma, mb = moments(groups[a], ref), moments(groups[b], ref)
                sa, sb = signals_from_moments(ma), signals_from_moments(mb)
                if _handoff(sa, sb, ma, mb, max_jump):
                    ok = True
                else:
                    ba, bb = bbox(groups[a]), bbox(groups[b])
                    gap = max(bb[0] - ba[2], ba[0] - bb[2], bb[1] - ba[3], ba[1] - bb[3], 0)
                    ok = gap <= gap_max and _rigid_scaled(sa, sb, ma, mb)
                if ok:
                    groups[a] = sorted(groups[a] + groups[b])
                    del groups[b]
                    changed = True
                    break
            if changed:
                break
    out = []
    for g in sorted(groups, key=lambda g: min(g)):
        ref = ref_of(g)
        out.append((g, moments(g, ref), np.asarray(comps[ref].color, dtype=np.float64)))
    return out
