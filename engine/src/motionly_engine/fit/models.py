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
ZETAS = (0.2, 0.35, 0.5, 0.7, 0.9, 1.0, 1.3)
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


def _crossings(u, p):
    return tuple(_first_cross(u, p, lv) for lv in (0.1, 0.5, 0.9))


_NAMED_X = {n: _crossings(_U, np.asarray(named_curve(n)(_U))) for n in NAMED}
_TS = np.linspace(0, 60, 24001)
_SPRING_X = {z: _crossings(_TS, np.asarray(easing.spring(_TS, 1.0, 2 * z))) for z in ZETAS}


@dataclass
class Candidate:
    kind: str  # "named" | "bezier" | "spring"
    name: str | None
    x: np.ndarray
    rss: float
    n: int

    @property
    def k(self) -> int:
        return {"named": 4, "bezier": 8, "spring": 5}[self.kind]

    @property
    def bic(self) -> float:
        return self.n * math.log(max(self.rss, 1e-12) / self.n) + self.k * math.log(self.n)

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
    return Candidate(kind, name, r.x, rss, len(d.t))


def _named_init(name, d, f0, f1, cross, fallback):
    ux = _NAMED_X[name]
    if cross and None not in cross and None not in ux and ux[2] - ux[0] > 1e-6:
        dur = (cross[2] - cross[0]) / (ux[2] - ux[0])
        if dur > 0:
            return [cross[0] - ux[0] * dur, dur, f0, f1]
    return [fallback[0], fallback[1], f0, f1]


def fit_candidates(d: MoveData) -> list[Candidate]:
    f0, f1 = _levels(d)
    cross = _data_crossings(d, f0, f1)
    t_lo, t_hi = float(d.t[0]), float(d.t[-1])
    span = t_hi - t_lo
    fb_start = float(d.t[max(d.i0 - 1, 0)])
    fb = (fb_start, max(float(d.t[min(d.i1 + 1, len(d.t) - 1)]) - fb_start, 2 * d.frame))
    big = np.inf
    lb4 = [t_lo - span - 1, 0.5 * d.frame, -big, -big]
    ub4 = [t_hi, 4 * span + 1, big, big]

    cands = [_fit("named", n, d, _named_init(n, d, f0, f1, cross, fb), lb4, ub4) for n in NAMED]

    # Free bezier: start from the best named fits that have a bezier form, plus a generic one.
    ranked = sorted(cands, key=lambda c: c.rss)
    starts = []
    for c in ranked:
        if c.name in easing.PRESETS and len(starts) < 3:
            starts.append(list(c.x[:4]) + list(easing.PRESETS[c.name]))
    starts.append(list(ranked[0].x[:4]) + [0.25, 0.1, 0.25, 1.0])
    lb8, ub8 = lb4 + [0, -1, 0, -1], ub4 + [1, 2, 1, 2]
    bez = [_fit("bezier", None, d, s, lb8, ub8) for s in starts]
    cands.append(min(bez, key=lambda c: c.rss))

    # Spring: (start, from, to, omega0, zeta) from a zeta grid, omega0 matched to the 10-90% time.
    springs = []
    if cross and None not in cross and cross[2] > cross[0]:
        for z, tx in _SPRING_X.items():
            if None in tx:
                continue
            w0 = (tx[2] - tx[0]) / (cross[2] - cross[0])
            springs.append([cross[0] - tx[0] / w0, f0, f1, w0, z])
    if not springs:
        springs.append([fb[0], f0, f1, 10.0, 0.7])
    lb5 = [t_lo - span - 1, -big, -big, 0.5, 0.05]
    ub5 = [t_hi, big, big, 200.0, 3.0]
    sp = [_fit("spring", None, d, s, lb5, ub5) for s in springs]
    cands.append(min(sp, key=lambda c: c.rss))
    return cands
