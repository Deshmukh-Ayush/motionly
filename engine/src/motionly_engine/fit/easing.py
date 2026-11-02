"""Easing math: cubic-bezier, damped spring, named presets and curve distance.

All curve functions map a time fraction t in [0, 1] to progress (0 -> 1, may
overshoot). Functions accept scalars or NumPy arrays and return the same shape.

Distances use the same definition as the benchmark's trajectory error: the
maximum absolute progress difference over a uniform grid on [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Union

import numpy as np

Bezier = tuple[float, float, float, float]
CurveFn = Callable[[np.ndarray], np.ndarray]
Curve = Union[Bezier, str, CurveFn, np.ndarray]

N_SAMPLES = 201


def _out(x: np.ndarray, like) -> np.ndarray | float:
    return float(x) if np.ndim(like) == 0 else x


# ---------------------------------------------------------------- cubic-bezier


def _check_bezier(params: Bezier) -> tuple[float, float, float, float]:
    if len(params) != 4:
        raise ValueError(f"bezier needs 4 params, got {len(params)}")
    x1, y1, x2, y2 = (float(p) for p in params)
    if not (0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0):
        raise ValueError(f"bezier x1, x2 must be in [0, 1], got {x1}, {x2}")
    return x1, y1, x2, y2


def _poly(s: np.ndarray, p1: float, p2: float) -> np.ndarray:
    # B(s) with P0 = 0, P3 = 1: 3(1-s)^2 s p1 + 3(1-s) s^2 p2 + s^3
    return ((1 - 3 * p2 + 3 * p1) * s + (3 * p2 - 6 * p1)) * s * s + 3 * p1 * s


def _dpoly(s: np.ndarray, p1: float, p2: float) -> np.ndarray:
    return (3 * (1 - 3 * p2 + 3 * p1) * s + 2 * (3 * p2 - 6 * p1)) * s + 3 * p1


def bezier_solve_s(t, params: Bezier) -> np.ndarray:
    """Bezier parameter s with x(s) = t (x is monotone because x1, x2 in [0, 1])."""
    x1, _, x2, _ = _check_bezier(params)
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    s = t.copy()
    # Newton from s = t; keep only steps that stay in range and have usable slope.
    for _ in range(8):
        dx = _dpoly(s, x1, x2)
        err = _poly(s, x1, x2) - t
        ok = np.abs(dx) > 1e-6
        s = np.where(ok, np.clip(s - err / np.where(ok, dx, 1.0), 0.0, 1.0), s)
    # Bisection safeguard wherever Newton did not converge (flat slopes, etc.).
    bad = np.abs(_poly(s, x1, x2) - t) > 1e-12
    if np.any(bad):
        lo = np.zeros_like(t)
        hi = np.ones_like(t)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = _poly(mid, x1, x2) < t
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        s = np.where(bad, 0.5 * (lo + hi), s)
    return s


def bezier(t, params: Bezier):
    """CSS cubic-bezier(x1, y1, x2, y2) evaluated at time fraction t."""
    x1, y1, x2, y2 = _check_bezier(params)
    s = bezier_solve_s(t, (x1, y1, x2, y2))
    return _out(_poly(s, y1, y2), t)


# ---------------------------------------------------------------------- spring


def _check_spring(stiffness: float, damping: float) -> None:
    if stiffness <= 0:
        raise ValueError(f"stiffness must be > 0, got {stiffness}")
    if damping < 0:
        raise ValueError(f"damping must be >= 0, got {damping}")


def spring(t, stiffness: float, damping: float):
    """Damped spring (mass 1) from 0 to 1 with zero initial velocity; t in seconds."""
    _check_spring(stiffness, damping)
    t_in = t
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    w0 = np.sqrt(stiffness)
    zeta = damping / (2.0 * w0)
    if abs(zeta - 1.0) < 1e-6:
        x = 1.0 - np.exp(-w0 * t) * (1.0 + w0 * t)
    elif zeta < 1.0:
        wd = w0 * np.sqrt(1.0 - zeta * zeta)
        a = zeta * w0
        x = 1.0 - np.exp(-a * t) * (np.cos(wd * t) + (a / wd) * np.sin(wd * t))
    else:
        root = np.sqrt(zeta * zeta - 1.0)
        r1 = -w0 * (zeta - root)  # slow root
        r2 = -w0 * (zeta + root)  # fast root
        x = 1.0 - (r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / (r2 - r1)
    return _out(x, t_in)


def spring_settle_time(stiffness: float, damping: float, tol: float = 1e-3) -> float:
    """Time after which |spring - 1| stays below tol."""
    _check_spring(stiffness, damping)
    w0 = np.sqrt(stiffness)
    zeta = damping / (2.0 * w0)
    if zeta == 0:
        raise ValueError("undamped spring never settles")
    # Upper bound from the decay envelope of the slowest mode.
    if zeta < 1.0:
        a = zeta * w0
        amp = 1.0 / np.sqrt(1.0 - zeta * zeta)  # envelope amplitude of cos + (a/wd) sin
        t_hi = np.log(amp / tol) / a
    else:
        slow = w0 * (zeta - np.sqrt(max(zeta * zeta - 1.0, 0.0)))
        # Critical/overdamped response is bounded by (1 + w0 t) e^{-slow t}; iterate.
        t_hi = 1.0 / slow
        while (1.0 + w0 * t_hi) * np.exp(-slow * t_hi) > tol:
            t_hi *= 1.5
    t_hi = max(float(t_hi), 1e-6)
    ts = np.linspace(0.0, t_hi, 20001)
    over = np.abs(np.asarray(spring(ts, stiffness, damping)) - 1.0) >= tol
    if not np.any(over):
        return 0.0
    last = int(np.nonzero(over)[0][-1])
    return float(ts[min(last + 1, len(ts) - 1)])


def spring_curve(stiffness: float, damping: float, tol: float = 1e-3) -> CurveFn:
    """Spring as a curve on t in [0, 1], where t = 1 is the settle time."""
    T = spring_settle_time(stiffness, damping, tol)

    def f(t):
        return spring(np.clip(np.asarray(t, dtype=float), 0.0, 1.0) * T, stiffness, damping)

    return f


# --------------------------------------------------------------------- presets

# Cubic-bezier approximations. CSS keywords are exact; Penner entries follow
# easings.net. TODO(M4): elastic and bounce need their own model (springs cover
# elastic-like motion).
PRESETS: dict[str, Bezier] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease-in": (0.42, 0.0, 1.0, 1.0),
    "ease-out": (0.0, 0.0, 0.58, 1.0),
    "ease-in-out": (0.42, 0.0, 0.58, 1.0),
    "easeInSine": (0.12, 0.0, 0.39, 0.0),
    "easeOutSine": (0.61, 1.0, 0.88, 1.0),
    "easeInOutSine": (0.37, 0.0, 0.63, 1.0),
    "easeInQuad": (0.11, 0.0, 0.5, 0.0),
    "easeOutQuad": (0.5, 1.0, 0.89, 1.0),
    "easeInOutQuad": (0.45, 0.0, 0.55, 1.0),
    "easeInCubic": (0.32, 0.0, 0.67, 0.0),
    "easeOutCubic": (0.33, 1.0, 0.68, 1.0),
    "easeInOutCubic": (0.65, 0.0, 0.35, 1.0),
    "easeInQuart": (0.5, 0.0, 0.75, 0.0),
    "easeOutQuart": (0.25, 1.0, 0.5, 1.0),
    "easeInOutQuart": (0.76, 0.0, 0.24, 1.0),
    "easeInQuint": (0.64, 0.0, 0.78, 0.0),
    "easeOutQuint": (0.22, 1.0, 0.36, 1.0),
    "easeInOutQuint": (0.83, 0.0, 0.17, 1.0),
    "easeInExpo": (0.7, 0.0, 0.84, 0.0),
    "easeOutExpo": (0.16, 1.0, 0.3, 1.0),
    "easeInOutExpo": (0.87, 0.0, 0.13, 1.0),
    "easeInCirc": (0.55, 0.0, 1.0, 0.45),
    "easeOutCirc": (0.0, 0.55, 0.45, 1.0),
    "easeInOutCirc": (0.85, 0.0, 0.15, 1.0),
    "easeInBack": (0.36, 0.0, 0.66, -0.56),
    "easeOutBack": (0.34, 1.56, 0.64, 1.0),
    "easeInOutBack": (0.68, -0.6, 0.32, 1.6),
}

# Exact Penner functions (formulas as on easings.net).
_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1.0


def _in_out(f_in: CurveFn) -> tuple[CurveFn, CurveFn, CurveFn]:
    """Build (in, out, in-out) from an ease-in function."""

    def f_out(t):
        return 1.0 - f_in(1.0 - t)

    def f_io(t):
        return np.where(t < 0.5, 0.5 * f_in(2.0 * t), 1.0 - 0.5 * f_in(2.0 - 2.0 * t))

    return f_in, f_out, f_io


def _expo_in(t):
    with np.errstate(over="ignore", under="ignore"):
        return np.where(t <= 0.0, 0.0, np.power(2.0, 10.0 * t - 10.0))


def _back_in_out(t):
    return np.where(
        t < 0.5,
        (np.power(2 * t, 2) * ((_C2 + 1) * 2 * t - _C2)) / 2,
        (np.power(2 * t - 2, 2) * ((_C2 + 1) * (t * 2 - 2) + _C2) + 2) / 2,
    )


_FAMILIES: dict[str, CurveFn] = {
    "Sine": lambda t: 1.0 - np.cos(t * np.pi / 2),
    "Quad": lambda t: t**2,
    "Cubic": lambda t: t**3,
    "Quart": lambda t: t**4,
    "Quint": lambda t: t**5,
    "Expo": _expo_in,
    "Circ": lambda t: 1.0 - np.sqrt(np.clip(1.0 - t * t, 0.0, None)),
    "Back": lambda t: _C3 * t**3 - _C1 * t**2,
}

EXACT: dict[str, CurveFn] = {}
for _fam, _fin in _FAMILIES.items():
    _i, _o, _io = _in_out(_fin)
    EXACT[f"easeIn{_fam}"] = _i
    EXACT[f"easeOut{_fam}"] = _o
    EXACT[f"easeInOut{_fam}"] = _io
EXACT["easeInOutBack"] = _back_in_out  # Penner's in-out back uses c1 * 1.525
EXACT["linear"] = lambda t: t


def _clamped(f: CurveFn) -> CurveFn:
    def g(t):
        t_in = t
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        x = np.asarray(f(t), dtype=float)
        x = np.where(t <= 0.0, 0.0, np.where(t >= 1.0, 1.0, x))  # exact endpoints
        return _out(x, t_in)

    return g


EXACT = {k: _clamped(v) for k, v in EXACT.items()}

_GSAP_FAMILIES = {
    "power1": "Quad", "quad": "Quad",
    "power2": "Cubic", "cubic": "Cubic",
    "power3": "Quart", "quart": "Quart",
    "power4": "Quint", "quint": "Quint", "strong": "Quint",
    "sine": "Sine", "expo": "Expo", "circ": "Circ", "back": "Back",
}
GSAP_ALIASES: dict[str, str] = {"none": "linear", "power0.none": "linear"}
for _g, _fam in _GSAP_FAMILIES.items():
    GSAP_ALIASES[f"{_g}.in"] = f"easeIn{_fam}"
    GSAP_ALIASES[f"{_g}.out"] = f"easeOut{_fam}"
    GSAP_ALIASES[f"{_g}.inOut"] = f"easeInOut{_fam}"


# ---------------------------------------------------------------- comparisons


def grid(n: int = N_SAMPLES) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def resolve(curve: Curve) -> CurveFn | Bezier:
    """Turn a name into an exact function (preferred) or bezier params."""
    if isinstance(curve, str):
        name = GSAP_ALIASES.get(curve, curve)
        if name in EXACT:
            return EXACT[name]
        if name in PRESETS:
            return PRESETS[name]
        raise KeyError(f"unknown ease {curve!r}")
    return curve


def as_samples(curve: Curve, n: int = N_SAMPLES) -> np.ndarray:
    """Sample a curve on the uniform grid of n points on [0, 1]."""
    curve = resolve(curve)
    if isinstance(curve, np.ndarray):
        if curve.shape != (n,):
            raise ValueError(f"sampled curve must have shape ({n},), got {curve.shape}")
        return curve.astype(float)
    if callable(curve):
        return np.asarray(curve(grid(n)), dtype=float)
    return np.asarray(bezier(grid(n), curve), dtype=float)


def curve_distance(a: Curve, b: Curve, n: int = N_SAMPLES) -> float:
    """Max absolute progress difference between two curves on [0, 1]."""
    return float(np.max(np.abs(as_samples(a, n) - as_samples(b, n))))


class PresetMatch(NamedTuple):
    name: str
    kind: str  # "exact" or "bezier"
    dist: float


_TABLE: list[tuple[str, str, np.ndarray]] = [
    (name, "exact", as_samples(f)) for name, f in EXACT.items()
] + [(name, "bezier", as_samples(p)) for name, p in PRESETS.items()]


def nearest_preset(curve: Curve) -> PresetMatch:
    """Closest named ease, searching exact functions first, then beziers."""
    samples = as_samples(curve)
    best = None
    for name, kind, ref in _TABLE:
        d = float(np.max(np.abs(samples - ref)))
        if best is None or d < best.dist:  # strict: earlier entries win ties
            best = PresetMatch(name, kind, d)
    return best
