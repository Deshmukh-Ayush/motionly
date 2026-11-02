"""Tween models and their least-squares fits.

v(t) = from + (to - from) * E((t - start) / duration), holding outside the move.
Candidates: every exact Penner function and CSS keyword (named, k = 4), a free cubic-bezier (k = 8)
and a damped spring (k = 5). All starts are deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from motionly_engine.fit import easing

CSS_NAMES = ("ease", "ease-in", "ease-out", "ease-in-out")
NAMED = tuple(easing.EXACT) + CSS_NAMES
ZETAS = (0.12, 0.2, 0.28, 0.35, 0.5, 0.7, 0.9, 1.0, 1.3)
_U = np.linspace(0.0, 1.0, 4001)


def named_curve(name: str):
    if name in easing.EXACT:
        return easing.EXACT[name]
    params = easing.PRESETS[name]
    return lambda u: easing.bezier(u, params)


def _first_cross(u: np.ndarray, p: np.ndarray, level: float) -> float | None:
    above = np.nonzero(p >= level)[0]
    if len(above) == 0:
        return None
    j = above[0]
    if j == 0:
        return float(u[0])
    a, b = p[j - 1], p[j]
    return float(u[j - 1] + (level - a) / (b - a) * (u[j] - u[j - 1]))


def _last_cross(u: np.ndarray, p: np.ndarray, level: float) -> float | None:
    below = np.nonzero(p < level)[0]
    if len(below) == 0 or below[-1] + 1 >= len(p):
        return None
    j = below[-1] + 1
    a, b = p[j - 1], p[j]
    return float(u[j - 1] + (level - a) / (b - a) * (u[j] - u[j - 1]))


def _crossings(u, p):
    return tuple(_first_cross(u, p, lv) for lv in (0.1, 0.5, 0.9))


def _crossings_last(u, p):
    """Crossings found from the end (differ from the first ones for overshoot or noisy data)."""
    return tuple(_last_cross(u, p, lv) for lv in (0.1, 0.5, 0.9))


_NAMED_X = {n: _crossings(_U, np.asarray(named_curve(n)(_U))) for n in NAMED}
_NAMED_XL = {n: _crossings_last(_U, np.asarray(named_curve(n)(_U))) for n in NAMED}
_TS = np.linspace(0, 60, 24001)
_SPRING_X = {z: _crossings(_TS, np.asarray(easing.spring(_TS, 1.0, 2 * z))) for z in ZETAS}


@dataclass
class Candidate:
    kind: str  # "named" | "bezier" | "spring"
    name: str | None
    x: np.ndarray
    rss: float
    n: int
    sigma: float  # per-sample noise of the signal, estimated from its holds
    valid: bool = True  # inside the data window and consistent with the following hold

    @property
    def k(self) -> int:
        return {"named": 4, "bezier": 8, "spring": 5}[self.kind]

    @property
    def bic(self) -> float:
        """BIC with the noise level fixed from the hold frames: RSS / sigma^2 + k ln n."""
        return self.rss / (self.sigma * self.sigma) + self.k * math.log(self.n)

    @property
    def start(self) -> float:
        return float(self.x[0])

    @property
    def from_(self) -> float:
        return float(self.x[1] if self.kind == "spring" else self.x[2])

    @property
    def to(self) -> float:
        return float(self.x[2] if self.kind == "spring" else self.x[3])

    @property
    def stiffness(self) -> float:
        return float(self.x[3] ** 2)

    @property
    def damping(self) -> float:
        return float(2 * self.x[4] * self.x[3])

    @property
    def duration(self) -> float:
        if self.kind == "spring":
            return easing.spring_settle_time(self.stiffness, self.damping)
        return float(self.x[1])

    @property
    def bezier_params(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.x[4:8])

    def progress(self, t):
        return _progress(self.kind, self.name, self.x, np.asarray(t, dtype=float))

    def value(self, t):
        return self.from_ + (self.to - self.from_) * self.progress(t)


def _progress(kind, name, x, t):
    if kind == "spring":
        start, _, _, w0, z = x
        return np.asarray(easing.spring(t - start, w0 * w0, 2 * z * w0), dtype=float)
    start, dur = x[0], x[1]
    u = np.clip((t - start) / dur, 0.0, 1.0)
    if kind == "named":
        return np.asarray(named_curve(name)(u), dtype=float)
    return np.asarray(easing.bezier(u, tuple(np.clip(x[4:8], [0, -1, 0, -1], [1, 2, 1, 2]))), dtype=float)


def _value(kind, name, x, t):
    f, to = (x[1], x[2]) if kind == "spring" else (x[2], x[3])
    return f + (to - f) * _progress(kind, name, x, t)


@dataclass
class MoveData:
    t: np.ndarray
    v: np.ndarray
    w: np.ndarray  # residual weights
    i0: int  # first moving sample (within t / v)
    i1: int  # last moving sample
    frame: float  # frame interval, seconds
    sigma: float = 1.0  # per-sample noise (prop units), from the signal's holds


def _levels(d: MoveData) -> tuple[float, float]:
    pre = d.v[:max(d.i0, 1)]
    post = d.v[d.i1 + 1:] if d.i1 + 1 < len(d.v) else d.v[-1:]
    return float(np.median(pre)), float(np.median(post))


def _data_crossings(d: MoveData, f0: float, f1: float):
    if abs(f1 - f0) < 1e-12:
        return None
    p = (d.v - f0) / (f1 - f0)
    a = max(d.i0 - 1, 0)
    return _crossings(d.t[a:], p[a:])


def _fit(kind, name, d: MoveData, x0, lb, ub) -> Candidate:
    x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)

    def res(x):
        return d.w * (_value(kind, name, x, d.t) - d.v)

    r = least_squares(res, x0, bounds=(lb, ub), x_scale="jac", method="trf", max_nfev=400)
    rss = float(np.sum(res(r.x) ** 2))
    return Candidate(kind, name, r.x, rss, len(d.t), d.sigma)


def _named_init(name, d, f0, f1, cross, fallback, ux=None):
    ux = ux or _NAMED_X[name]
    if cross and None not in cross and None not in ux and ux[2] - ux[0] > 1e-6:
        dur = (cross[2] - cross[0]) / (ux[2] - ux[0])
        if dur > 0:
            return [cross[0] - ux[0] * dur, dur, f0, f1]
    return [fallback[0], fallback[1], f0, f1]


def _check(c: Candidate, d: MoveData) -> Candidate:
    """Reject fits that run past the data window or end away from the following hold."""
    from motionly_engine.fit.visual import visual_times

    t_lo, t_hi = float(d.t[0]), float(d.t[-1])
    try:
        vs, ve = visual_times(c.progress, c.start, min(c.duration, 60.0), step=1e-3)
    except ValueError:
        c.valid = False
        return c
    ok = vs >= t_lo - 2 * d.frame and ve <= t_hi + 2 * d.frame
    post = d.v[d.i1 + 1:]
    if ok and len(post) >= 3:
        hold = float(np.median(post[-3:]))
        ok = abs(c.to - hold) <= max(3 * d.sigma, 0.03 * abs(c.to - c.from_))
    c.valid = bool(ok)
    return c


def fit_candidates(d: MoveData) -> list[Candidate]:
    f0, f1 = _levels(d)
    cross = _data_crossings(d, f0, f1)
    a = max(d.i0 - 1, 0)
    cross_last = _crossings_last(d.t[a:], (d.v[a:] - f0) / (f1 - f0)) if abs(f1 - f0) > 1e-12 else None
    t_lo, t_hi = float(d.t[0]), float(d.t[-1])
    span = t_hi - t_lo
    fb_start = float(d.t[max(d.i0 - 1, 0)])
    fb = (fb_start, max(float(d.t[min(d.i1 + 1, len(d.t) - 1)]) - fb_start, 2 * d.frame))
    big = np.inf
    # Duration capped at the data window (+2 frames): a move can't be longer than the data shows.
    lb4 = [t_lo - span - 1, 0.5 * d.frame, -big, -big]
    ub4 = [t_hi, span + 2 * d.frame, big, big]

    cands = []
    for n in NAMED:
        # Two starts: align by the first crossings and by the crossings found from the end
        # (they differ for overshooting eases and noisy data).
        c = _fit("named", n, d, _named_init(n, d, f0, f1, cross, fb), lb4, ub4)
        if cross_last and None not in cross_last and _NAMED_XL[n] and None not in _NAMED_XL[n]:
            c2 = _fit("named", n, d, _named_init(n, d, f0, f1, cross_last, fb, _NAMED_XL[n]), lb4, ub4)
            if c2.rss < c.rss:
                c = c2
        cands.append(c)

    # Free bezier: start from the best named fits that have a bezier form, plus a generic one.
    ranked = sorted(cands, key=lambda c: c.rss)
    starts = []
    for c in ranked:
        if c.name in easing.PRESETS and len(starts) < 3:
            starts.append(list(c.x[:4]) + list(easing.PRESETS[c.name]))
    for generic in ([0.25, 0.1, 0.25, 1.0], [0.5, 0.0, 0.75, 0.0], [0.25, 1.0, 0.5, 1.0], [0.4, 0.0, 0.2, 1.0]):
        starts.append(list(ranked[0].x[:4]) + generic)
    lb8, ub8 = lb4 + [0, -1, 0, -1], ub4 + [1, 2, 1, 2]
    bez = [_fit("bezier", None, d, s, lb8, ub8) for s in starts]
    best_bez = min(bez, key=lambda c: c.rss)
    cands.append(best_bez)

    # Refit the named eases closest to the bezier's shape, starting from the bezier's timing:
    # guards against a named fit stuck in a poor local minimum.
    shape = easing.as_samples(best_bez.bezier_params)
    near = sorted(NAMED, key=lambda n: float(np.max(np.abs(easing.as_samples(named_curve(n)) - shape))))[:6]
    by_name = {c.name: i for i, c in enumerate(cands) if c.kind == "named"}
    for n in near:
        c = _fit("named", n, d, list(best_bez.x[:4]), lb4, ub4)
        i = by_name[n]
        if c.rss < cands[i].rss:
            cands[i] = c

    # Spring: (start, from, to, omega0, zeta) from a zeta grid, omega0 matched to the 10-90% time.
    springs = []
    if cross and None not in cross and cross[2] > cross[0]:
        for z, tx in _SPRING_X.items():
            if None in tx:
                continue
            w0 = (tx[2] - tx[0]) / (cross[2] - cross[0])
            springs.append([cross[0] - tx[0] / w0, f0, f1, w0, z])
    best_named = min((c for c in cands if c.kind == "named"), key=lambda c: c.rss)
    for z in (0.3, 0.6):  # timing taken from the best named fit
        springs.append([best_named.start, best_named.from_, best_named.to, 4.0 / max(best_named.duration * z, 1e-3), z])
    if not springs:
        springs.append([fb[0], f0, f1, 10.0, 0.7])
    lb5 = [t_lo - span - 1, -big, -big, 2.0, 0.05]
    ub5 = [t_hi, big, big, 200.0, 3.0]
    sp = [_fit("spring", None, d, s, lb5, ub5) for s in springs]
    cands.append(min(sp, key=lambda c: c.rss))
    return [_check(c, d) for c in cands]
