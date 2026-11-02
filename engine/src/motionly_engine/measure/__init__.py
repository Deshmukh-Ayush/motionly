"""Video -> per-element signals over time (simple flat backgrounds)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motionly_engine import __version__
from motionly_engine.measure.background import estimate_flat
from motionly_engine.measure.decode import iter_frames, probe
from motionly_engine.measure.elements import UnionAccumulator, find_components
from motionly_engine.measure.signals import MomentAccumulator, merge_components, signals_from_moments

PROPS = ("x", "y", "scale", "opacity")


def _nan_to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else round(float(v), 5) for v in a]


@dataclass
class ElementSignals:
    id: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 of the region the element sweeps
    color: list[float]
    signals: dict[str, np.ndarray]
    mass: np.ndarray
    opacity_method: str = "core-or-mass"  # "template" when a full-opacity template was used

    def to_dict(self) -> dict:
        return {
            "id": self.id, "region_bbox": list(self.bbox), "color": [round(c, 2) for c in self.color],
            "signals": {p: _nan_to_none(self.signals[p]) for p in PROPS},
            "size_px": _nan_to_none(self.signals["size_px"]),
            "width_px": _nan_to_none(self.signals["width_px"]),
            "height_px": _nan_to_none(self.signals["height_px"]),
            "mass": _nan_to_none(self.mass),
            "opacity_method": self.opacity_method,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ElementSignals:
        def arr(v):
            return np.array([np.nan if x is None else x for x in v], dtype=float)

        sig = {p: arr(d["signals"][p]) for p in PROPS}
        for k in ("size_px", "width_px", "height_px"):
            sig[k] = arr(d[k]) if k in d else np.full(len(d["mass"]), np.nan)
        return cls(d["id"], tuple(d["region_bbox"]), d["color"], sig, arr(d["mass"]),
                   d.get("opacity_method", "core-or-mass"))


@dataclass
class Signals:
    clip: str
    width: int
    height: int
    fps: float
    timestamps: np.ndarray
    background: list[float]
    elements: list[ElementSignals]

    def to_dict(self) -> dict:
        return {
            "engine_version": __version__,
            "clip": self.clip,
            "video": {"width": self.width, "height": self.height, "fps": self.fps,
                      "n_frames": len(self.timestamps)},
            "timestamps": [round(float(t), 6) for t in self.timestamps],
            "background": [round(c, 2) for c in self.background],
            "elements": [e.to_dict() for e in self.elements],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Signals:
        v = d["video"]
        return cls(d["clip"], v["width"], v["height"], v["fps"], np.array(d["timestamps"]),
                   d["background"], [ElementSignals.from_dict(e) for e in d["elements"]])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> Signals:
        return cls.from_dict(json.loads(Path(path).read_text()))


def measure(clip: Path) -> Signals:
    clip = Path(clip)
    info = probe(clip)
    n = info.n_frames
    min_dim = min(info.width, info.height)

    # Background from the border strip of the first frames (flat backgrounds only).
    samples = []
    for f in iter_frames(clip, info.width, info.height):
        samples.append(f.copy())
        if len(samples) >= 8:
            break
    bg = estimate_flat(samples)

    # Pass 1: union of pixels that ever differ from background.
    acc = UnionAccumulator((info.height, info.width), bg.color, thr=max(12.0, 4 * bg.sigma))
    for f in iter_frames(clip, info.width, info.height):
        acc.add(f)
    comps = find_components(acc, min_dim)

    # Pass 2: per-component weighted moments per frame.
    mom = MomentAccumulator(comps, bg.color, noise_l1=bg.sigma)
    count = 0
    for f in iter_frames(clip, info.width, info.height):
        mom.add(f)
        count += 1
    if count != n:
        raise RuntimeError(f"decoded {count} frames but probe reported {n}")
    moms = mom.moments()

    elements = []
    merged = merge_components(moms, comps, bg.color, min_dim, max(info.width, info.height))
    for gi, (group, m, color) in enumerate(merged):
        sig = signals_from_moments(m)
        x0 = min(comps[i].bbox[0] for i in group)
        y0 = min(comps[i].bbox[1] for i in group)
        x1 = max(comps[i].bbox[2] for i in group)
        y1 = max(comps[i].bbox[3] for i in group)
        elements.append(ElementSignals(f"el{gi}", (x0, y0, x1, y1), color.tolist(), sig, m[:, 0]))

    # Pass 3: template opacity (two more decodes: grab the full-opacity frames, then solve alpha).
    from motionly_engine.measure.template import capture_templates, plan, solve_alpha

    plans = plan(elements, bg.color)
    capture_templates(iter_frames(clip, info.width, info.height), plans, info.width, info.height)
    solve_alpha(iter_frames(clip, info.width, info.height), plans, info.width, info.height, n)
    for p in plans:
        if p.alpha is not None:
            p.el.signals["opacity"] = p.alpha
            p.el.opacity_method = "template"
    return Signals(clip.stem, info.width, info.height, info.fps, info.timestamps, bg.color.tolist(), elements)
