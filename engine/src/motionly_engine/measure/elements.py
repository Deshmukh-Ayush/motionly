"""Find elements as connected regions of the union of pixels that ever differ from background."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Component:
    label: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    mask: np.ndarray  # bool, cropped to bbox
    color: np.ndarray  # float RGB


class UnionAccumulator:
    """Streams frames; keeps the union foreground mask and the farthest-from-background colour per pixel."""

    def __init__(self, shape: tuple[int, int], bg_color: np.ndarray, thr: float):
        self.bg = np.empty((*shape, 3), dtype=np.uint8)
        self.bg[:] = np.clip(np.rint(bg_color), 0, 255).astype(np.uint8)
        self.thr = thr
        self.union = np.zeros(shape, dtype=bool)
        self.max_dist = np.zeros(shape, dtype=np.float32)
        self.max_color = np.zeros((*shape, 3), dtype=np.uint8)

    def add(self, frame: np.ndarray) -> None:
        # L1 colour distance (>= L2), computed with OpenCV for speed.
        d = cv2.transform(cv2.absdiff(frame, self.bg).astype(np.float32), np.ones((1, 3), np.float32))
        self.union |= d > self.thr
        upd = d > self.max_dist
        np.maximum(self.max_dist, d, out=self.max_dist)
        np.copyto(self.max_color, frame, where=upd[..., None])


def find_components(acc: UnionAccumulator, min_dim: int) -> list[Component]:
    mask = acc.union.astype(np.uint8)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)  # drop isolated noise pixels
    r = max(2, int(round(0.006 * min_dim)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grouped = cv2.dilate(mask, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    # Measurement mask: the union plus a small rim (anti-aliased edges, compression halo).
    rim = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    min_area = max(12, int((min_dim / 540) ** 2 * 12))
    comps = []
    for lab in range(1, n):
        x, y, w, h, _ = stats[lab]
        sel = (labels[y:y + h, x:x + w] == lab) & rim[y:y + h, x:x + w]
        core = sel & acc.union[y:y + h, x:x + w]
        if core.sum() < min_area:
            continue
        dist = acc.max_dist[y:y + h, x:x + w][core]
        cols = acc.max_color[y:y + h, x:x + w][core].astype(np.float64)
        top = dist >= np.quantile(dist, 0.95)
        comps.append(Component(lab, (int(x), int(y), int(x + w), int(y + h)), sel, cols[top].mean(axis=0)))
    return comps
