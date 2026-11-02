"""Encode RGB frames to H.264 MP4 through an FFmpeg pipe."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def encode(frames: Iterable[np.ndarray], path: Path, width: int, height: int, fps: int, crf: int = 26) -> int:
    """Write frames (uint8 RGB) to `path`; returns the number of frames written."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", "-fflags", "+bitexact", "-map_metadata", "-1",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        for f in frames:
            assert f.shape == (height, width, 3) and f.dtype == np.uint8
            proc.stdin.write(f.tobytes())
            n += 1
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err}")
    return n
