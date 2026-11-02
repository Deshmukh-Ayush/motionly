"""Split a signal into moves: holds, then transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MERGE_GAP = 3  # frames: spring oscillations pass through zero velocity briefly
MAX_HOLD_S = 0.6  # hold context kept on each side of a move for fitting
SCALES = (1, 3, 8)  # frame offsets for multi-scale change detection


@dataclass
class Move:
    i0: int  # first moving sample (index into the valid samples)
    i1: int  # last moving sample
    w0: int  # fit window start (inclusive)
    w1: int  # fit window end (inclusive)


def noise_sigma(v: np.ndarray) -> float:
    """Per-sample noise from the median absolute first difference (holds dominate)."""
    d = np.diff(v)
    if len(d) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def find_moves(t: np.ndarray, v: np.ndarray, floor: float, sigma_floor: float = 0.0) -> list[Move]:
    """t, v: valid samples only. floor: minimum change for a real move (prop units)."""
    n = len(v)
    if n < 4:
        return []
    sigma = max(noise_sigma(v), sigma_floor)
    moving = np.zeros(n, dtype=bool)
    for k in SCALES:
        if n <= k:
            continue
        d = np.abs(v[k:] - v[:-k])
        hit = np.nonzero(d > 4 * np.sqrt(2) * sigma + 1e-12)[0]
        for i in hit:
            moving[i:i + k + 1] = True
    runs: list[list[int]] = []
    i = 0
    while i < n:
        if moving[i]:
            j = i
            while j + 1 < n and moving[j + 1]:
                j += 1
            if runs and i - runs[-1][1] - 1 <= MERGE_GAP:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    # Multi-scale detection marks a few hold frames around each move; trim them back.
    moves = []
    for a, b in runs:
        seg = v[max(0, a - 1):b + 2]
        change = max(abs(float(v[min(b + 1, n - 1)] - v[max(a - 1, 0)])), float(np.ptp(seg)))
        if change < max(floor, 6 * sigma):
            continue
        moves.append([a, b])
    out = []
    for k, (a, b) in enumerate(moves):
        lo = moves[k - 1][1] + 1 if k > 0 else 0
        hi = moves[k + 1][0] - 1 if k + 1 < len(moves) else n - 1
        # Leave the neighbour's hold to it: split the gap in the middle.
        if k > 0:
            lo = (moves[k - 1][1] + a) // 2 + 1
        if k + 1 < len(moves):
            hi = (b + moves[k + 1][0]) // 2
        lo = max(lo, int(np.searchsorted(t, t[a] - MAX_HOLD_S)))
        hi = min(hi, int(np.searchsorted(t, t[b] + MAX_HOLD_S, side="right")) - 1)
        out.append(Move(a, b, lo, hi))
    return out
