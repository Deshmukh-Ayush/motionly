"""Ground-truth test clips with known timing and easing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from motionly_engine import __version__
from motionly_engine.generate.encode import encode
from motionly_engine.generate.render import render_frames
from motionly_engine.generate.scene import Scene, sample_scene

SIZES = {"960x540": (960, 540), "1920x1080": (1920, 1080)}
FPS_CHOICES = (24, 30, 60)


def size_for(size: str, index: int) -> tuple[int, int]:
    if size == "mixed":
        return SIZES["960x540"] if index % 2 == 0 else SIZES["1920x1080"]
    if size in SIZES:
        return SIZES[size]
    w, h = (int(v) for v in size.lower().split("x"))
    return w, h


def generate_clip(seed: int, index: int, out_dir: Path, width: int, height: int, fps: int = 30, crf: int = 26) -> Scene:
    scene = sample_scene(seed, index, width, height, fps)
    scene.crf = crf
    out_dir.mkdir(parents=True, exist_ok=True)
    n = encode(render_frames(scene), out_dir / f"{scene.id}.mp4", width, height, fps, crf)
    assert n == scene.n_frames
    (out_dir / f"{scene.id}.json").write_text(json.dumps(scene.to_dict(), indent=2))
    return scene


def generate_clips(count: int, seed: int, out_dir: Path, size: str = "mixed", fps: int = 30, crf: int = 26,
                   on_clip=None) -> dict:
    if fps not in FPS_CHOICES:
        raise ValueError(f"fps must be one of {FPS_CHOICES}")
    out_dir = Path(out_dir)
    stats = {"clips": 0, "tweens": 0, "stagger": 0, "eases": Counter(), "seconds": 0.0}
    for i in range(count):
        w, h = size_for(size, i)
        scene = generate_clip(seed, i, out_dir, w, h, fps, crf)
        stats["clips"] += 1
        stats["stagger"] += bool(scene.groups)
        stats["seconds"] += scene.duration_s
        for el in scene.elements:
            for tw in el.tweens:
                stats["tweens"] += 1
                stats["eases"][tw.ease.type] += 1
        if on_clip:
            on_clip(scene)
    manifest = {"engine_version": __version__, "seed": seed, "count": count, "size": size, "fps": fps, "crf": crf,
                "clips": [f"clip_{i:03d}" for i in range(count)]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    stats["eases"] = dict(stats["eases"])
    return stats
