import json

import numpy as np
import pytest

from motionly_engine.fit import easing
from motionly_engine.fit.group import find_groups
from motionly_engine.fit.segment import find_moves
from motionly_engine.fit.timeline import fit_signal
from motionly_engine.fit.visual import visual_times
from motionly_engine.generate.scene import Ease, Tween

FPS = 30
FRAME = 1 / FPS
T = np.arange(0, 3.0, FRAME)


def _signal(tween, sigma=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return tween.value_at(T) + rng.normal(0, sigma, len(T))


def _fit_one(tween, sigma=0.2, seed=0):
    v = _signal(tween, sigma, seed)
    out = fit_signal(T, v, np.ones_like(T), tween.prop, 1.5, FRAME)
    assert len(out) == 1, out
    return out[0]


def _check_timing(fit, tween, tol_frames=1 / 3):
    vs, ve = tween.visual_times()
    assert abs(fit["visual_start_s"] - vs) < tol_frames * FRAME, (fit["visual_start_s"], vs)
    assert abs(fit["visual_end_s"] - ve) < tol_frames * FRAME, (fit["visual_end_s"], ve)


def _curve_of(fit):
    e = fit["ease"]
    if e["type"] == "named":
        from motionly_engine.fit.models import named_curve
        return named_curve(e["name"])
    if e["type"] == "bezier":
        return tuple(e["params"])
    return easing.spring_curve(e["stiffness"], e["damping"])


def test_exact_ease_out_expo():
    tw = Tween("x", 100.0, 300.0, 0.6, 1.1, Ease("named", name="easeOutExpo"))
    fit = _fit_one(tw)
    assert fit["ease"]["type"] == "named"
    assert fit["ease"]["name"] == "easeOutExpo" and fit["ease"]["kind"] == "exact"
    _check_timing(fit, tw)
    assert abs(fit["from"] - 100) < 0.5 and abs(fit["to"] - 300) < 0.5


def test_exact_ease_in_out_expo_is_named_not_bezier():
    tw = Tween("x", 100.0, 300.0, 0.6, 1.0, Ease("named", name="easeInOutExpo"))
    fit = _fit_one(tw)
    assert fit["ease"]["type"] == "named", fit["ease"]
    assert fit["ease"]["name"] == "easeInOutExpo" and fit["ease"]["kind"] == "exact"
    _check_timing(fit, tw)


def test_css_ease_is_named():
    tw = Tween("y", 50.0, 250.0, 0.5, 0.9, Ease("named", name="ease"))
    fit = _fit_one(tw)
    assert fit["ease"]["type"] == "named" and fit["ease"]["name"] == "ease", fit["ease"]
    _check_timing(fit, tw)


def test_custom_bezier_recovered():
    params = (0.2, 0.75, 0.45, 1.15)
    assert easing.nearest_preset(params).dist >= 0.03
    tw = Tween("x", 400.0, 150.0, 0.7, 1.2, Ease("bezier", params=params))
    fit = _fit_one(tw)
    assert easing.curve_distance(_curve_of(fit), params) < 0.03, fit["ease"]
    _check_timing(fit, tw)


def test_underdamped_spring_chosen():
    k, c = 200.0, 8.0
    tw = Tween("y", 100.0, 300.0, 0.5, easing.spring_settle_time(k, c), Ease("spring", stiffness=k, damping=c))
    fit = _fit_one(tw)
    assert fit["ease"]["type"] == "spring", fit["ease"]
    assert abs(fit["ease"]["stiffness"] - k) / k < 0.03
    assert abs(fit["ease"]["damping"] - c) / c < 0.05
    _check_timing(fit, tw)


def test_segments_two_moves_and_flat():
    a = Tween("x", 0.0, 100.0, 0.3, 0.5, Ease("named", name="easeInOutCubic"))
    b = Tween("x", 100.0, 40.0, 1.6, 0.6, Ease("named", name="easeOutCubic"))
    rng = np.random.default_rng(1)
    v = np.where(T < 1.2, a.value_at(T), b.value_at(T)) + rng.normal(0, 0.2, len(T))
    moves = find_moves(T, v, 1.5)
    assert len(moves) == 2
    fits = fit_signal(T, v, np.ones_like(T), "x", 1.5, FRAME)
    assert len(fits) == 2
    _check_timing(fits[0], a, 0.5)
    _check_timing(fits[1], b, 0.5)
    flat = 50 + rng.normal(0, 0.3, len(T))
    assert find_moves(T, flat, 1.5) == []


@pytest.mark.parametrize("ease,dur", [
    (Ease("named", name="easeOutBack"), 0.8),
    (Ease("bezier", params=(0.3, -0.2, 0.6, 1.3)), 1.0),
    (Ease("spring", stiffness=150.0, damping=6.0), 2.0),
])
def test_visual_times_match_generator(ease, dur):
    tw = Tween("x", 0.0, 10.0, 0.4, dur, ease)
    assert visual_times(tw.progress_at, tw.start_s, tw.duration_s) == tw.visual_times()


def test_groups():
    ease = {"type": "named", "name": "easeOutCubic"}
    mk = lambda s: {"prop": "y", "start_s": s, "duration_s": 0.6, "ease": ease}
    g = find_groups([("el0", mk(0.5)), ("el1", mk(0.6)), ("el2", mk(0.7))], FRAME)
    assert g == [{"type": "stagger", "prop": "y", "elements": ["el0", "el1", "el2"], "step_s": 0.1}]
    other = {"prop": "y", "start_s": 0.55, "duration_s": 0.6, "ease": {"type": "named", "name": "linear"}}
    g = find_groups([("el0", mk(0.5)), ("el1", other)], FRAME)
    assert g == []
    g = find_groups([("el0", mk(0.5)), ("el1", mk(0.51))], FRAME)
    assert g[0]["type"] == "shared"


def test_fit_clip_end_to_end(tmp_path):
    from motionly_engine.fit.timeline import fit_clip
    from motionly_engine.generate import generate_clip

    scene = generate_clip(21, 6, tmp_path, 960, 540)
    tl = fit_clip(tmp_path / f"{scene.id}.mp4", tmp_path / "run")
    assert (tmp_path / "run" / "timeline.json").exists() and (tmp_path / "run" / "signals.json").exists()
    assert tl["video"]["width"] == 960 and abs(tl["video"]["fps"] - 30) < 0.01
    assert len(tl["elements"]) == len(scene.elements)
    n_truth = sum(len(e.tweens) for e in scene.elements)
    n_meas = sum(len(e["tweens"]) for e in tl["elements"])
    assert n_meas == n_truth
    for e in tl["elements"]:
        for t in e["tweens"]:
            assert {"prop", "from", "to", "start_s", "duration_s", "visual_start_s", "visual_end_s",
                    "ease", "confidence", "rmse"} <= set(t)
            assert t["confidence"] == "measured"
            assert {"type", "nearest", "nearest_dist"} <= set(t["ease"])
    json.dumps(tl)


def test_select_prefers_named_within_noise():
    import numpy as np

    from motionly_engine.fit.models import Candidate
    from motionly_engine.fit.select import select

    n, sigma = 60, 0.5
    bez = Candidate("bezier", None, np.zeros(8), rss=10.0, n=n, sigma=sigma)
    # Named costs 0.2 more mean squared residual per sample than the bezier: within noise (0.25).
    named = Candidate("named", "easeOutCubic", np.zeros(4), rss=10.0 + 0.2 * n, n=n, sigma=sigma)
    assert select([bez, named]).name == "easeOutCubic"
    worse = Candidate("named", "linear", np.zeros(4), rss=10.0 + 2.0 * n, n=n, sigma=sigma)
    assert select([bez, worse]).kind == "bezier"
