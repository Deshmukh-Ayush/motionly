import json

import numpy as np
import pytest

from motionly_engine.generate import generate_clip
from motionly_engine.measure import Signals, measure
from motionly_engine.measure.plot import plot_element, truth_vs_measured

# seed 21 at 960x540 (the smaller benchmark size): together these cover circle / rect / text, x / y / scale / opacity,
# named / bezier / spring eases and a stagger scene.
CASES = [0, 6, 8, 12, 15]


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("gt")
    out = []
    for i in CASES:
        scene = generate_clip(21, i, d, 960, 540)
        out.append((scene, measure(d / f"{scene.id}.mp4")))
    return out


def _match(scene, sig):
    pairs = []
    for te in scene.elements:
        for me in sig.elements:
            cx, cy = (me.bbox[0] + me.bbox[2]) / 2, (me.bbox[1] + me.bbox[3]) / 2
            if te.slot[0] <= cx <= te.slot[2] and te.slot[1] <= cy <= te.slot[3]:
                pairs.append((te, me))
    return pairs


def test_elements_and_timestamps(clips):
    for scene, sig in clips:
        assert len(sig.elements) == len(scene.elements), scene.id
        assert len(_match(scene, sig)) == len(scene.elements)
        np.testing.assert_allclose(sig.timestamps, np.arange(scene.n_frames) / scene.fps, atol=1e-4)
        assert np.max(np.abs(np.array(sig.background) - scene.background)) <= 3


def test_signals_follow_truth(clips):
    for scene, sig in clips:
        t = sig.timestamps
        for te, me in _match(scene, sig):
            op = te.value_at("opacity", t)
            vis = np.isfinite(me.signals["x"]) & (op >= 0.5)  # centroid is noisy when nearly invisible
            assert vis.sum() > 5
            i0, i1 = np.argmax(vis), np.nonzero(vis)[0][-1]
            for p in ("x", "y"):
                tv, mv = te.value_at(p, t), me.signals[p]
                if te.shape == "text":  # truth is the ink bbox centre: compare change over time
                    tv, mv = tv - tv[i0], mv - mv[i0]
                    tol = 1.5 if te.tween_for("scale") else 1.0  # centroid offset scales with the text
                else:
                    tol = 1.0
                assert np.max(np.abs(tv[vis] - mv[vis])) < tol, (scene.id, te.id, p)
            # Scale / opacity on frames where the element is at least 12 px in its smaller dimension
            # (tiny text loses contrast to compression, which biases mass-based measures).
            ts = te.value_at("scale", t)
            big = np.isfinite(me.signals["x"]) & (min(te.size) * ts >= 12) & (op >= 0.5)
            ref = np.nonzero(big)[0][-1]
            np.testing.assert_array_less(np.abs(ts[big] / ts[ref] - me.signals["scale"][big] / me.signals["scale"][ref]), 0.03)
            sized = min(te.size) * ts >= 12
            # Template opacity: thin moving text strokes are smeared by the codec, up to ~0.065 error.
            op_tol = 0.07 if te.shape == "text" else 0.05
            np.testing.assert_array_less(np.abs(op[sized] - me.signals["opacity"][sized]), op_tol)


def test_roundtrip_and_plot(clips, tmp_path):
    scene, sig = clips[0]
    sig.save(tmp_path / "signals.json")
    back = Signals.load(tmp_path / "signals.json")
    assert len(back.elements) == len(sig.elements)
    np.testing.assert_allclose(back.elements[0].signals["x"], sig.elements[0].signals["x"], atol=1e-4)
    t, series, title, _ = truth_vs_measured(sig, scene, 0)
    png = plot_element(t, series, title, tmp_path / "plot.png")
    assert png.stat().st_size > 1000
    json.dumps(sig.to_dict())


@pytest.mark.parametrize("index", [29, 31])
def test_one_element_per_truth_element_1080p(tmp_path, index):
    """Seed 1 scenes that split before: 29 = scaling text "Pro" (letters drift apart as it scales),
    31 = a fast mover whose frame-to-frame handoff happens at opacity 0.2."""
    scene = generate_clip(1, index, tmp_path, 1920, 1080)
    sig = measure(tmp_path / f"{scene.id}.mp4")
    assert len(sig.elements) == len(scene.elements)
    assert len(_match(scene, sig)) == len(scene.elements)
