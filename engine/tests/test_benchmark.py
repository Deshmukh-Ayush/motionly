import copy
import json

import pytest

from motionly_engine.benchmark.match import match_clip
from motionly_engine.benchmark.metrics import summarize, target_rows, tween_metrics
from motionly_engine.generate.scene import Ease, Tween

FPS = 30


def _truth_tween(prop="x", start=0.5, dur=0.8, ease=None, f=100.0, to=300.0):
    tw = Tween(prop, f, to, start, dur, ease or Ease("named", name="easeOutCubic"))
    return tw.to_dict()


def _truth(tweens_by_el):
    els = []
    for i, tws in enumerate(tweens_by_el):
        els.append({"id": f"el{i}", "shape": "circle", "slot": [i * 500, 0, i * 500 + 480, 500], "tweens": tws})
    return {"video": {"id": "clip_x", "fps": FPS, "width": 960, "height": 540}, "elements": els, "groups": []}


def _timeline(tweens_by_el):
    els = []
    for i, tws in enumerate(tweens_by_el):
        els.append({"id": f"m{i}", "bbox": [i * 500 + 100, 100, 50, 50], "tweens": tws})
    return {"elements": els, "groups": []}


def _as_measured(tt, shift_s=0.0):
    mt = copy.deepcopy(tt)
    for k in ("start_s", "visual_start_s", "visual_end_s"):
        mt[k] += shift_s
    return mt


def test_perfect_match_passes():
    tt = _truth_tween()
    m = match_clip(_truth([[tt]]), _timeline([[_as_measured(tt)]]))
    assert len(m.pairs) == 1 and not m.missed and not m.spurious
    r = tween_metrics(tt, m.pairs[0][3], FPS, "circle")
    assert r["timing_ok"] and r["trajectory_ok"] and r["name_ok"]
    assert r["trajectory_err"] < 1e-9 and r["value_err"] == 0


def test_two_frame_shift_fails_timing():
    tt = _truth_tween()
    r = tween_metrics(tt, _as_measured(tt, 2 / FPS), FPS, "circle")
    assert not r["timing_ok"] and abs(r["start_err_frames"] - 2) < 1e-6
    assert not r["trajectory_ok"]


def test_extra_tween_is_spurious_and_missing_is_missed():
    tt = _truth_tween()
    extra = _truth_tween(prop="opacity", f=1.0, to=0.9)
    m = match_clip(_truth([[tt], [_truth_tween(prop="y")]]), _timeline([[_as_measured(tt), extra]]))
    assert len(m.pairs) == 1 and len(m.spurious) == 1 and len(m.missed) == 1
    s = summarize([tween_metrics(tt, m.pairs[0][3], FPS, "circle")], len(m.spurious), 2, len(m.missed))
    assert s["spurious_rate"] == 0.5 and s["missed_rate"] == 0.5
    assert [ok for *_, ok in target_rows(s)] == [True, True, True, False]


def test_name_equivalence_and_curve_distance():
    tt = _truth_tween(ease=Ease("named", name="easeOutCubic"))
    alias = _as_measured(tt)
    alias["ease"] = {"type": "named", "name": "power2.out"}
    assert tween_metrics(tt, alias, FPS, "circle")["name_ok"]
    close = _as_measured(tt)
    close["ease"] = {"type": "bezier", "params": [0.33, 1.0, 0.68, 1.0]}  # easings.net approximation
    assert tween_metrics(tt, close, FPS, "circle")["name_ok"]
    wrong = _as_measured(tt)
    wrong["ease"] = {"type": "named", "name": "linear"}
    assert not tween_metrics(tt, wrong, FPS, "circle")["name_ok"]


def test_text_compares_change_not_position():
    tt = _truth_tween()
    mt = _as_measured(tt)
    mt["from"] += 3.0
    mt["to"] += 3.0  # constant centroid offset
    assert tween_metrics(tt, mt, FPS, "text")["value_err"] == pytest.approx(0.0)
    assert tween_metrics(tt, mt, FPS, "circle")["value_err"] == pytest.approx(3.0)


def test_end_to_end_small(tmp_path):
    from motionly_engine.benchmark.run import run_benchmark
    from motionly_engine.generate import generate_clips

    gt = tmp_path / "gt"
    generate_clips(2, 21, gt, size="960x540")
    res = run_benchmark(gt, tmp_path / "run")
    assert (tmp_path / "run" / "report.md").read_text().startswith("# Benchmark")
    s = res["summary"]
    assert s["matched"] + s["missed"] == sum(
        len(e["tweens"]) for i in range(2) for e in json.loads((gt / f"clip_{i:03d}.json").read_text())["elements"])
    assert s["matched"] > 0
