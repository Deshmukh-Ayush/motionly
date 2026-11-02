import json
import subprocess

import numpy as np
import pytest

from motionly_engine.generate import generate_clip
from motionly_engine.generate.render import render_frames
from motionly_engine.generate.scene import (
    Ease, Element, Scene, Tween, check_scene, sample_scene,
)


def _boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_deterministic():
    a = sample_scene(7, 3).to_dict()
    assert a == sample_scene(7, 3).to_dict()
    assert a != sample_scene(7, 4).to_dict()


def test_roundtrip():
    d = sample_scene(7, 5).to_dict()
    assert Scene.from_dict(json.loads(json.dumps(d))).to_dict() == d


@pytest.mark.parametrize("size", [(960, 540), (1920, 1080)])
def test_constraints(size):
    for i in range(150):
        s = sample_scene(11, i, *size)
        check_scene(s)  # slot/frame containment + opacity range on a dense grid
        assert 1 <= len(s.elements) <= 4
        for e in s.elements:
            assert 1 <= len(e.tweens) <= 2
            assert len({t.prop for t in e.tweens}) == len(e.tweens)
            assert all(t.start_s >= 0.3 for t in e.tweens)
        for a in range(len(s.elements)):
            for b in range(a + 1, len(s.elements)):
                assert not _boxes_overlap(s.elements[a].slot, s.elements[b].slot)


def test_stagger_fraction_and_shape():
    scenes = [sample_scene(3, i) for i in range(400)]
    stag = [s for s in scenes if s.groups]
    assert 0.28 <= len(stag) / len(scenes) <= 0.42
    for s in stag:
        g = s.groups[0]
        tws = [s.elements[int(eid[2:])].tweens[0] for eid in g["elements"]]
        assert len(tws) >= 2
        assert len({json.dumps(t.ease.to_dict()) for t in tws}) == 1
        assert len({t.duration_s for t in tws}) == 1
        assert len({t.prop for t in tws}) == 1
        np.testing.assert_allclose(np.diff([t.start_s for t in tws]), g["step_s"], atol=2e-4)


def test_custom_bezier_is_not_a_preset():
    from motionly_engine.fit.easing import nearest_preset

    for i in range(200):
        for e in sample_scene(5, i).elements:
            for t in e.tweens:
                if t.ease.type == "bezier":
                    assert nearest_preset(t.ease.params).dist >= 0.03


def test_tween_values_and_visual_times():
    tw = Tween("x", 100.0, 200.0, 1.0, 2.0, Ease("named", name="linear"))
    assert tw.value_at(0.5) == 100.0 and tw.value_at(1.0) == 100.0
    assert tw.value_at(2.0) == pytest.approx(150.0)
    assert tw.value_at(3.0) == 200.0 and tw.value_at(9.0) == 200.0
    vs, ve = tw.visual_times()
    assert vs == pytest.approx(1.0 + 0.01 * 2.0, abs=2e-4)
    assert ve == pytest.approx(1.0 + 0.99 * 2.0, abs=2e-4)


def test_spring_visual_end_after_overshoot():
    tw = Tween("y", 0.0, 100.0, 0.5, 1.5, Ease("spring", stiffness=200.0, damping=8.0))
    vs, ve = tw.visual_times()
    assert 0.5 < vs < ve <= 2.0
    assert np.all(np.abs(tw.progress_at(np.linspace(ve, 2.0, 500)) - 1) <= 0.0101)


def _centroid(frame, bg):
    d = np.abs(frame.astype(np.float64) - bg).sum(axis=2)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    return (d * xx).sum() / d.sum(), (d * yy).sum() / d.sum()


def test_render_subpixel_centroid():
    el = Element(id="el0", shape="circle", color=(250, 250, 250), size=(30.0, 30.0),
                 slot=(0, 0, 320, 180), rest={"y": 90.3, "scale": 1.0, "opacity": 1.0},
                 tweens=[Tween("x", 60.0, 250.0, 0.1, 1.0, Ease("named", name="easeInOutCubic"))])
    scene = Scene(1, 0, 320, 180, 30, 1.5, (10, 20, 30), [el])
    bg = np.array(scene.background, dtype=np.float64)
    for i, frame in enumerate(render_frames(scene)):
        t = i / 30
        cx, cy = _centroid(frame, bg)
        assert abs(cx - el.value_at("x", t)) < 0.25, (i, cx)
        assert abs(cy - 90.3) < 0.25


def test_render_scale_and_opacity():
    el = Element(id="el0", shape="rect", color=(255, 255, 255), size=(40.0, 20.0), corner=0.0,
                 slot=(0, 0, 320, 180), rest={"x": 160.0, "y": 90.0, "scale": 1.0, "opacity": 1.0},
                 tweens=[Tween("opacity", 1.0, 0.0, 0.0, 1.0, Ease("named", name="linear"))])
    scene = Scene(1, 0, 320, 180, 30, 1.0, (0, 0, 0), [el])
    frames = list(render_frames(scene))
    mass = [f.astype(float).sum() / (3 * 255) for f in frames]
    assert mass[0] == pytest.approx(800, rel=0.01)  # 40 x 20 px fully opaque
    assert mass[15] == pytest.approx(400, rel=0.02)  # half opacity


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,r_frame_rate,nb_read_frames,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


@pytest.mark.parametrize("fps", [30, 24])
def test_encode_clip(tmp_path, fps):
    scene = generate_clip(2, 0, tmp_path, 320, 180, fps=fps, crf=26)
    info = _probe(tmp_path / "clip_000.mp4")
    assert info["codec_name"] == "h264"
    assert info["r_frame_rate"] == f"{fps}/1"
    assert int(info["nb_read_frames"]) == scene.n_frames
    truth = json.loads((tmp_path / "clip_000.json").read_text())
    assert truth["video"]["fps"] == fps and truth["video"]["crf"] == 26
