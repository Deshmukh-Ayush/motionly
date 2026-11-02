"""Decode video frames with their real timestamps (never assume constant frame rate)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    timestamps: np.ndarray  # seconds, one per decoded frame, presentation order

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    st = d["streams"][0]
    ts = np.array([float(f["best_effort_timestamp_time"]) for f in d["frames"]], dtype=float)
    ts = np.sort(ts)
    return VideoInfo(int(st["width"]), int(st["height"]), float(Fraction(st["avg_frame_rate"])), ts)


def iter_frames(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Yield uint8 RGB frames (H, W, 3), streamed from FFmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()
