"""Supersampled frame renderer for ground-truth scenes.

Each element is rasterised once as an alpha sprite (analytic coverage for
shapes, Pillow for text) plus a 2x mip pyramid. Per frame the sprite is placed
with a subpixel affine warp on a 2x supersampled canvas, only inside the
element's region, then box-downsampled to the output size.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motionly_engine.generate.scene import Element, Scene

SS = 2  # supersampling factor
_PAD = 3


@lru_cache(maxsize=64)
def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=px)


def text_size(text: str, font_px: float) -> tuple[float, float]:
    """Ink size (w, h) in output px, measured at 8x for stability."""
    x0, y0, x1, y1 = _font(max(1, round(font_px * 8))).getbbox(text)
    return (x1 - x0) / 8, (y1 - y0) / 8


def _text_alpha(text: str, font_px: float, res: float) -> tuple[np.ndarray, float, float]:
    font = _font(max(1, round(font_px * res)))
    x0, y0, x1, y1 = font.getbbox(text)
    img = Image.new("L", (x1 - x0 + 2 * _PAD, y1 - y0 + 2 * _PAD), 0)
    ImageDraw.Draw(img).text((_PAD - x0, _PAD - y0), text, fill=255, font=font)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a, _PAD + (x1 - x0 - 1) / 2, _PAD + (y1 - y0 - 1) / 2


def _shape_alpha(el: Element, res: float) -> tuple[np.ndarray, float, float]:
    w, h = el.size[0] * res, el.size[1] * res
    W, H = math.ceil(w) + 2 * _PAD, math.ceil(h) + 2 * _PAD
    cx, cy = (W - 1) / 2, (H - 1) / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if el.shape == "circle":
        sdf = np.hypot(xx - cx, yy - cy) - w / 2
    else:
        r = el.corner * min(w, h)
        qx = np.abs(xx - cx) - (w / 2 - r)
        qy = np.abs(yy - cy) - (h / 2 - r)
        sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return np.clip(0.5 - sdf, 0.0, 1.0).astype(np.float32), cx, cy


@dataclass
class _Level:
    alpha: np.ndarray
    cx: float
    cy: float
    res: float  # sprite px per output px at element scale 1


def build_sprite(el: Element, max_scale: float) -> list[_Level]:
    res = SS * max(1.0, max_scale)
    if el.shape == "text":
        a, cx, cy = _text_alpha(el.text, el.font_px, res)
    else:
        a, cx, cy = _shape_alpha(el, res)
    levels = [_Level(a, cx, cy, res)]
    while min(a.shape) > 8 and levels[-1].res > SS * 0.3:
        h, w = a.shape
        a = np.pad(a, ((0, h % 2), (0, w % 2)))
        a = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
        last = levels[-1]
        levels.append(_Level(a, (last.cx - 0.5) / 2, (last.cy - 0.5) / 2, last.res / 2))
    return levels


def _draw(out: np.ndarray, bg: np.ndarray, levels: list[_Level], color: np.ndarray,
          x: float, y: float, s: float, o: float) -> None:
    if o <= 0 or s <= 0:
        return
    need = SS * s
    lvl = next((lv for lv in reversed(levels) if lv.res >= need), levels[0])
    k = need / lvl.res
    tx, ty = SS * x + 0.5, SS * y + 0.5  # output pixel-index coords -> supersampled coords
    h, w = lvl.alpha.shape
    H, W = out.shape[0] * SS, out.shape[1] * SS
    x0 = max(0, (math.floor(tx - k * (lvl.cx + 1)) - 2) // 2 * 2)
    y0 = max(0, (math.floor(ty - k * (lvl.cy + 1)) - 2) // 2 * 2)
    x1 = min(W, -(-(math.ceil(tx + k * (w - lvl.cx)) + 2) // 2) * 2)
    y1 = min(H, -(-(math.ceil(ty + k * (h - lvl.cy)) + 2) // 2) * 2)
    if x1 <= x0 or y1 <= y0:
        return
    M = np.array([[k, 0, tx - k * lvl.cx - x0], [0, k, ty - k * lvl.cy - y0]], dtype=np.float64)
    a = cv2.warpAffine(lvl.alpha, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = (a * o)[..., None]
    roi = bg * (1 - a) + color * a
    small = cv2.resize(roi, ((x1 - x0) // SS, (y1 - y0) // SS), interpolation=cv2.INTER_AREA)
    out[y0 // SS:y1 // SS, x0 // SS:x1 // SS] = np.clip(np.rint(small), 0, 255).astype(np.uint8)


def render_frames(scene: Scene) -> Iterator[np.ndarray]:
    """Yield uint8 RGB frames (H, W, 3) at t = i / fps. Element regions never overlap."""
    bg = np.array(scene.background, dtype=np.float32)
    times = scene.frame_times()
    dense = np.linspace(0, scene.duration_s, 4001)
    sprites = []
    for el in scene.elements:
        smax = float(max(el.value_at("scale", dense).max(), el.value_at("scale", times).max()))
        sprites.append(build_sprite(el, smax))
    colors = [np.array(el.color, dtype=np.float32) for el in scene.elements]
    tracks = [{p: el.value_at(p, times) for p in ("x", "y", "scale", "opacity")} for el in scene.elements]
    base = np.empty((scene.height, scene.width, 3), dtype=np.uint8)
    base[:] = np.array(scene.background, dtype=np.uint8)
    for i in range(len(times)):
        out = base.copy()
        for levels, color, tr in zip(sprites, colors, tracks):
            _draw(out, bg, levels, color, tr["x"][i], tr["y"][i], tr["scale"][i], tr["opacity"][i])
        yield out
