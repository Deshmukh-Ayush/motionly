"""Background estimate for flat-colour backgrounds (per-pixel models come in M6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Background:
    color: np.ndarray  # float RGB
    sigma: float  # robust noise of L1 colour distance to background


def border_pixels(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    b = max(2, int(round(0.01 * min(w, h))))
    return np.concatenate([
        frame[:b].reshape(-1, 3), frame[-b:].reshape(-1, 3),
        frame[:, :b].reshape(-1, 3), frame[:, -b:].reshape(-1, 3),
    ]).astype(np.float64)


def estimate_flat(frames: list[np.ndarray]) -> Background:
    """Median colour of a thin border strip over sample frames."""
    px = np.concatenate([border_pixels(f) for f in frames])
    color = np.median(px, axis=0)
    dist = np.abs(px - color).sum(axis=1)  # L1, same metric as the foreground test
    sigma = float(1.4826 * np.median(np.abs(dist - np.median(dist))) + np.median(dist))
    return Background(color, max(sigma, 0.5))
