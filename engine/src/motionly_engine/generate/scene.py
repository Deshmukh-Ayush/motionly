"""Scene model, seeded sampler and truth evaluation for ground-truth clips.

Coordinates: output pixels, OpenCV pixel-index convention (pixel i's centre is
at i). `x`, `y` are the element centre, `scale` multiplies the base size,
`opacity` is 0..1. Times are in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from motionly_engine import __version__
from motionly_engine.fit import easing

PROPS = ("x", "y", "scale", "opacity")
SHAPES = ("rect", "circle", "text")
WORDS = ("Motion", "Launch", "Hello", "Ship it", "New", "Build", "Fast", "Pro", "Today", "Hey")
STAGGER_P = 0.35
VISUAL_TOL = 0.01  # 1% of the change, per CLAUDE.md
SCALE_RANGE = (0.35, 1.6)
MAX_SPRING_S = 2.0
CSS_NAMED = ("ease", "ease-in", "ease-out", "ease-in-out")
NAMED = tuple(easing.EXACT) + CSS_NAMED


def _r(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


# ------------------------------------------------------------------- model


@dataclass
class Ease:
    type: str  # "named" | "bezier" | "spring"
    name: str | None = None
    params: tuple[float, float, float, float] | None = None
    stiffness: float | None = None
    damping: float | None = None

    @property
    def kind(self) -> str | None:
        if self.type != "named":
            return None
        return "exact" if self.name in easing.EXACT else "bezier"

    def progress(self, u, duration_s: float):
        """Progress at time fraction u in [0, 1] of a tween lasting duration_s."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        if self.type == "named":
            if self.name in easing.EXACT:
                return np.asarray(easing.EXACT[self.name](u), dtype=float)
            return np.asarray(easing.bezier(u, easing.PRESETS[self.name]), dtype=float)
        if self.type == "bezier":
            return np.asarray(easing.bezier(u, self.params), dtype=float)
        return np.asarray(easing.spring(u * duration_s, self.stiffness, self.damping), dtype=float)

    def to_dict(self) -> dict:
        if self.type == "named":
            return {"type": "named", "name": self.name, "kind": self.kind}
        if self.type == "bezier":
            return {"type": "bezier", "params": list(self.params)}
        return {"type": "spring", "stiffness": self.stiffness, "damping": self.damping, "mass": 1.0}

    @classmethod
    def from_dict(cls, d: dict) -> Ease:
        if d["type"] == "named":
            return cls("named", name=d["name"])
        if d["type"] == "bezier":
            return cls("bezier", params=tuple(d["params"]))
        return cls("spring", stiffness=d["stiffness"], damping=d["damping"])


@dataclass
class Tween:
    prop: str
    from_: float
    to: float
    start_s: float
    duration_s: float
    ease: Ease

    def progress_at(self, t):
        return self.ease.progress((np.asarray(t, dtype=float) - self.start_s) / self.duration_s, self.duration_s)

    def value_at(self, t):
        t = np.asarray(t, dtype=float)
        v = self.from_ + (self.to - self.from_) * self.progress_at(t)
        # Exact holds outside the tween (springs end within 1e-3 of target).
        v = np.where(t <= self.start_s, self.from_, v)
        return np.where(t >= self.start_s + self.duration_s, self.to, v)

    def visual_times(self, step: float = 1e-4) -> tuple[float, float]:
        """When progress first passes 1% and when it settles within 1% of the end."""
        ts = np.arange(self.start_s, self.start_s + self.duration_s + step, step)
        p = self.progress_at(ts)
        moved = np.nonzero(np.abs(p) >= VISUAL_TOL)[0]
        unsettled = np.nonzero(np.abs(p - 1.0) > VISUAL_TOL)[0]
        vs = ts[moved[0]] if len(moved) else self.start_s
        ve = ts[min(unsettled[-1] + 1, len(ts) - 1)] if len(unsettled) else self.start_s
        return float(vs), float(ve)

    def to_dict(self) -> dict:
        vs, ve = self.visual_times()
        return {
            "prop": self.prop,
            "from": self.from_,
            "to": self.to,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "visual_start_s": _r(vs),
            "visual_end_s": _r(ve),
            "ease": self.ease.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tween:
        return cls(d["prop"], d["from"], d["to"], d["start_s"], d["duration_s"], Ease.from_dict(d["ease"]))


@dataclass
class Element:
    id: str
    shape: str
    color: tuple[int, int, int]
    size: tuple[float, float]  # base (w, h) in output px at scale 1
    slot: tuple[float, float, float, float]  # x0, y0, x1, y1 the element must stay inside
    rest: dict[str, float]  # values for props without a tween
    tweens: list[Tween] = field(default_factory=list)
    text: str | None = None
    font_px: float | None = None
    corner: float = 0.0  # rounded-rect corner radius as a fraction of min(w, h)

    @property
    def label(self) -> str:
        return f'text "{self.text}"' if self.shape == "text" else {"rect": "rounded rect", "circle": "circle"}[self.shape]

    def tween_for(self, prop: str) -> Tween | None:
        return next((tw for tw in self.tweens if tw.prop == prop), None)

    def value_at(self, prop: str, t):
        tw = self.tween_for(prop)
        if tw is None:
            return np.full(np.shape(t), self.rest[prop], dtype=float)
        return tw.value_at(t)

    def bbox_at(self, t) -> np.ndarray:
        """[x0, y0, x1, y1] per time (shape (..., 4))."""
        x, y, s = (self.value_at(p, t) for p in ("x", "y", "scale"))
        hw, hh = self.size[0] / 2 * s, self.size[1] / 2 * s
        return np.stack([x - hw, y - hh, x + hw, y + hh], axis=-1)

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox_at(0.0)
        d = {
            "id": self.id,
            "label": self.label,
            "shape": self.shape,
            "color": list(self.color),
            "size": [_r(self.size[0]), _r(self.size[1])],
            "slot": [_r(v) for v in self.slot],
            "rest": dict(self.rest),
            "bbox": [_r(x0), _r(y0), _r(x1 - x0), _r(y1 - y0)],
            "tweens": [tw.to_dict() for tw in self.tweens],
        }
        if self.shape == "text":
            d["text"], d["font_px"] = self.text, self.font_px
        if self.shape == "rect":
            d["corner"] = self.corner
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Element:
        return cls(
            id=d["id"], shape=d["shape"], color=tuple(d["color"]), size=tuple(d["size"]),
            slot=tuple(d["slot"]), rest=dict(d["rest"]),
            tweens=[Tween.from_dict(t) for t in d["tweens"]],
            text=d.get("text"), font_px=d.get("font_px"), corner=d.get("corner", 0.0),
        )


@dataclass
class Scene:
    seed: int
    index: int
    width: int
    height: int
    fps: int
    duration_s: float
    background: tuple[int, int, int]
    elements: list[Element]
    groups: list[dict] = field(default_factory=list)
    crf: int | None = None

    @property
    def id(self) -> str:
        return f"clip_{self.index:03d}"

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps))

    def frame_times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    def to_dict(self) -> dict:
        return {
            "engine_version": __version__,
            "video": {
                "id": self.id, "fps": self.fps, "duration_s": self.duration_s,
                "width": self.width, "height": self.height, "crf": self.crf,
                "n_frames": self.n_frames,
            },
            "seed": self.seed,
            "index": self.index,
            "background": list(self.background),
            "elements": [e.to_dict() for e in self.elements],
            "groups": self.groups,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        v = d["video"]
        return cls(
            seed=d["seed"], index=d["index"], width=v["width"], height=v["height"], fps=v["fps"],
            duration_s=v["duration_s"], background=tuple(d["background"]),
            elements=[Element.from_dict(e) for e in d["elements"]], groups=d["groups"], crf=v.get("crf"),
        )


# ----------------------------------------------------------------- sampler


def _luma(c) -> float:
    r, g, b = (v / 255 for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _colors(rng: np.random.Generator, n: int) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
    bg = tuple(int(v) for v in rng.integers(0, 256, 3))
    out = []
    while len(out) < n:
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        if abs(_luma(c) - _luma(bg)) >= 0.3:
            out.append(c)
    return bg, out


def _slots(rng: np.random.Generator, n: int, w: int, h: int) -> list[tuple[float, float, float, float]]:
    m = 0.03 * min(w, h)
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = (2, 1) if rng.random() < 0.7 else (1, 2)
    elif n == 3:
        cols, rows = (3, 1) if rng.random() < 0.7 else (1, 3)
    else:
        cols, rows = 2, 2
    cw, rh = (w - m) / cols, (h - m) / rows
    return [
        (m + c * cw, m + r * rh, (c + 1) * cw, (r + 1) * rh)
        for r in range(rows) for c in range(cols)
    ][:n]


def _sample_ease(rng: np.random.Generator, opacity: bool) -> Ease:
    u = rng.random()
    if opacity:
        if u < 0.6:
            pool = [n for n in NAMED if "Back" not in n]
            return Ease("named", name=str(rng.choice(pool)))
        kind = "bezier"
    else:
        kind = "named" if u < 0.5 else ("bezier" if u < 0.75 else "spring")
    if kind == "named":
        return Ease("named", name=str(rng.choice(NAMED)))
    if kind == "bezier":
        ylo, yhi = (0.0, 1.0) if opacity else (-0.4, 1.4)
        while True:
            p = (_r(rng.uniform(0, 1), 3), _r(rng.uniform(ylo, yhi), 3),
                 _r(rng.uniform(0, 1), 3), _r(rng.uniform(ylo, yhi), 3))
            if easing.nearest_preset(p).dist >= 0.03:
                return Ease("bezier", params=p)
    while True:
        k = _r(rng.uniform(80, 400), 2)
        zeta = rng.uniform(0.3, 1.1)
        c = _r(2 * zeta * math.sqrt(k), 3)
        if easing.spring_settle_time(k, c) <= MAX_SPRING_S:
            return Ease("spring", stiffness=k, damping=c)


def _duration(rng: np.random.Generator, ease: Ease) -> float:
    if ease.type == "spring":
        return _r(math.ceil(easing.spring_settle_time(ease.stiffness, ease.damping) * 1e4) / 1e4)
    return _r(rng.uniform(0.3, 1.5))


def _progress_range(ease: Ease, duration_s: float) -> tuple[float, float]:
    p = ease.progress(np.linspace(0, 1, 4001), duration_s)
    return float(min(p.min(), 0.0)), float(max(p.max(), 1.0))


def _place(rng, lo: float, hi: float, delta: float, pr: tuple[float, float]) -> float:
    """`from` such that from + delta * [pmin, pmax] stays inside [lo, hi]."""
    a, b = sorted((delta * pr[0], delta * pr[1]))
    return float(rng.uniform(lo - a, hi - b))


def _shape(rng, slot) -> dict:
    sw, sh = slot[2] - slot[0], slot[3] - slot[1]
    smin = min(sw, sh)
    shape = str(rng.choice(SHAPES))
    if shape == "circle":
        d = rng.uniform(0.12, 0.26) * smin
        return {"shape": "circle", "size": (_r(d), _r(d))}
    if shape == "rect":
        w = min(rng.uniform(0.12, 0.3) * smin * rng.uniform(1.0, 2.0), 0.28 * sw)
        h = rng.uniform(0.12, 0.26) * smin
        return {"shape": "rect", "size": (_r(w), _r(h)), "corner": _r(rng.uniform(0.1, 0.5), 3)}
    from motionly_engine.generate.render import text_size  # lazy: keeps sampler import light

    word = str(rng.choice(WORDS))
    font_px = _r(rng.uniform(0.14, 0.24) * smin, 2)
    w, h = text_size(word, font_px)
    if w > 0.28 * sw:
        font_px = _r(font_px * 0.28 * sw / w, 2)
        w, h = text_size(word, font_px)
    return {"shape": "text", "size": (_r(w), _r(h)), "text": word, "font_px": font_px}


def _fit_scale(rng, delta_mag: float, pr) -> tuple[float, float] | None:
    lo, hi = SCALE_RANGE
    for _ in range(50):
        delta = delta_mag * (1 if rng.random() < 0.5 else -1)
        a, b = sorted((delta * pr[0], delta * pr[1]))
        if (hi - b) - (lo - a) > 0:
            f = _r(_place(rng, lo, hi, delta, pr), 3)
            return f, _r(f + delta, 3)
        delta_mag *= 0.85
    return None


def _build_element(rng, eid, slot, color, props, eases, starts, durations, scale_vals=None,
                   pos_delta=None, opacity_vals=None) -> Element:
    """Sample values so the swept bbox stays inside `slot`."""
    el = Element(id=eid, color=color, slot=slot, rest={"scale": 1.0, "opacity": 1.0}, **_shape(rng, slot))
    tweens: dict[str, Tween] = {}
    # Scale first: it sets how much room position needs.
    smax = 1.0
    if "scale" in props:
        e, s0, d = eases["scale"], starts["scale"], durations["scale"]
        pr = _progress_range(e, d)
        vals = scale_vals or _fit_scale(rng, rng.uniform(0.25, 0.6), pr)
        tw = Tween("scale", vals[0], vals[1], s0, d, e)
        tweens["scale"] = tw
        sv = tw.value_at(np.linspace(s0, s0 + d, 2001))
        smax = float(max(sv.max(), vals[0], vals[1]))
        el.rest["scale"] = vals[0]
    for axis, (lo_s, hi_s), half in (("x", (slot[0], slot[2]), el.size[0] / 2), ("y", (slot[1], slot[3]), el.size[1] / 2)):
        lo, hi = lo_s + half * smax + 1.0, hi_s - half * smax - 1.0
        if lo > hi:
            raise RuntimeError("element too large for its slot")
        if axis in props:
            e, s0, d = eases[axis], starts[axis], durations[axis]
            pr = _progress_range(e, d)
            room = (hi - lo) / (pr[1] - pr[0])
            if pos_delta is not None:
                delta = pos_delta
            else:
                delta = rng.uniform(0.35, 0.9) * room * (1 if rng.random() < 0.5 else -1)
            f = _r(_place(rng, lo, hi, delta, pr), 2)
            tweens[axis] = Tween(axis, f, _r(f + delta, 2), s0, d, e)
            el.rest[axis] = f
        else:
            el.rest[axis] = _r(rng.uniform(lo, hi), 2)
    if "opacity" in props:
        vals = opacity_vals or [(0.0, 1.0), (1.0, 0.0), (0.2, 1.0), (1.0, 0.3)][int(rng.integers(4))]
        tweens["opacity"] = Tween("opacity", vals[0], vals[1], starts["opacity"], durations["opacity"], eases["opacity"])
        el.rest["opacity"] = vals[0]
    el.tweens = [tweens[p] for p in props]
    return el


def _clip_bbox(el: Element, times: np.ndarray) -> tuple[float, float, float, float]:
    b = el.bbox_at(times)
    return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())


def sample_scene(seed: int, index: int, width: int = 960, height: int = 540, fps: int = 30) -> Scene:
    rng = np.random.default_rng([seed, index])
    stagger = rng.random() < STAGGER_P
    n = int(rng.integers(2, 5)) if stagger else int(rng.integers(1, 5))
    bg, colors = _colors(rng, n)
    slots = _slots(rng, n, width, height)
    elements: list[Element] = []
    groups: list[dict] = []

    if stagger:
        prop = str(rng.choice(PROPS))
        ease = _sample_ease(rng, opacity=prop == "opacity")
        dur = _duration(rng, ease)
        start0 = _r(rng.uniform(0.3, 1.0))
        step = _r(rng.uniform(0.05, 0.2), 3)
        kw: dict = {}
        if prop == "scale":
            kw["scale_vals"] = _fit_scale(rng, rng.uniform(0.25, 0.6), _progress_range(ease, dur))
        elif prop == "opacity":
            kw["opacity_vals"] = [(0.0, 1.0), (1.0, 0.0), (0.2, 1.0)][int(rng.integers(3))]
        else:
            # Shared delta must fit the smallest slot along that axis (assume elements up to 30% of slot).
            pr = _progress_range(ease, dur)
            i0, i1 = (0, 2) if prop == "x" else (1, 3)
            span = min(s[i1] - s[i0] for s in slots) * 0.45
            kw["pos_delta"] = _r(rng.uniform(0.5, 1.0) * span / (pr[1] - pr[0]) * (1 if rng.random() < 0.5 else -1), 2)
        for i in range(n):
            start = _r(start0 + i * step)
            elements.append(_build_element(rng, f"el{i}", slots[i], colors[i], [prop], {prop: ease},
                                           {prop: start}, {prop: dur}, **kw))
        groups.append({"type": "stagger", "elements": [e.id for e in elements], "step_s": step})
    else:
        for i in range(n):
            k = 1 if rng.random() < 0.5 else 2
            props = [str(p) for p in rng.choice(PROPS, size=k, replace=False)]
            eases = {p: _sample_ease(rng, opacity=p == "opacity") for p in props}
            durations = {p: _duration(rng, eases[p]) for p in props}
            starts = {props[0]: _r(rng.uniform(0.3, 1.2))}
            if k == 2:
                p0 = props[0]
                starts[props[1]] = _r(starts[p0] + rng.uniform(0.0, durations[p0] + 0.4))
            elements.append(_build_element(rng, f"el{i}", slots[i], colors[i], props, eases, starts, durations))

    end = max(tw.start_s + tw.duration_s for e in elements for tw in e.tweens)
    duration = math.ceil((end + rng.uniform(0.5, 1.0)) * fps) / fps
    scene = Scene(seed, index, width, height, fps, _r(duration, 6), bg, elements, groups)
    check_scene(scene)
    return scene


def check_scene(scene: Scene) -> None:
    """Raise if any element leaves its slot / the frame or opacity leaves [0, 1]."""
    times = np.arange(0.0, scene.duration_s + 1e-9, 1 / 600)
    for el in scene.elements:
        x0, y0, x1, y1 = _clip_bbox(el, times)
        sx0, sy0, sx1, sy1 = el.slot
        eps = 1e-6
        if x0 < sx0 - eps or y0 < sy0 - eps or x1 > sx1 + eps or y1 > sy1 + eps:
            raise RuntimeError(f"{scene.id}/{el.id} leaves its slot: {(x0, y0, x1, y1)} vs {el.slot}")
        if x0 < 0 or y0 < 0 or x1 > scene.width or y1 > scene.height:
            raise RuntimeError(f"{scene.id}/{el.id} leaves the frame")
        o = el.value_at("opacity", times)
        if o.min() < -1e-9 or o.max() > 1 + 1e-9:
            raise RuntimeError(f"{scene.id}/{el.id} opacity outside [0, 1]")
