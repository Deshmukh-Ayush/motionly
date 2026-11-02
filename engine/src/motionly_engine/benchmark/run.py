"""Run measure + fit over a ground-truth set and score it."""

from __future__ import annotations

import json
import time
from pathlib import Path

from motionly_engine import __version__
from motionly_engine.benchmark.match import match_clip
from motionly_engine.benchmark.metrics import duration_bucket, summarize, tween_metrics, unobservable
from motionly_engine.benchmark.report import write_report
from motionly_engine.fit.timeline import fit_clip
from motionly_engine.generate.scene import Scene  # read-only: truth evaluation


def _stagger_rows(truth: dict, tl: dict, emap: dict[str, str], fps: float) -> list[dict]:
    rows = []
    inv = {}
    for m, t in emap.items():
        inv.setdefault(t, m)
    for g in truth["groups"]:
        if g["type"] != "stagger":
            continue
        want = {inv.get(e) for e in g["elements"]}
        hit = None
        for mg in tl["groups"]:
            if mg["type"] == "stagger" and set(mg["elements"]) == want:
                hit = mg
        ok = hit is not None and abs(hit["step_s"] - g["step_s"]) <= 1 / fps
        rows.append({"clip": truth["video"]["id"], "n": len(g["elements"]), "step_s": g["step_s"],
                     "measured_step_s": hit["step_s"] if hit else None, "ok": ok})
    return rows


def run_benchmark(gt_dir: Path, run_dir: Path, on_clip=None, reuse: bool = True) -> dict:
    gt_dir, run_dir = Path(gt_dir), Path(run_dir)
    manifest = json.loads((gt_dir / "manifest.json").read_text())
    rows, spurious, missed, stagger, unobs = [], [], [], [], []
    n_measured = 0
    t0 = time.time()
    for clip_id in manifest["clips"]:
        truth = json.loads((gt_dir / f"{clip_id}.json").read_text())
        out = run_dir / "clips" / clip_id
        out.mkdir(parents=True, exist_ok=True)
        tc = time.time()
        tl = fit_clip(gt_dir / f"{clip_id}.mp4", out, reuse=reuse)
        m = match_clip(truth, tl)
        fps = truth["video"]["fps"]
        shapes = {e["id"]: e["shape"] for e in truth["elements"]}
        scene_els = {e.id: e for e in Scene.from_dict(truth).elements}
        n_unobs_meas = 0
        for te, me, tt, mt in m.pairs:
            if unobservable(scene_els[te], tt):
                unobs.append({"clip": clip_id, "truth_el": te, "shape": shapes[te], "tween": tt, "measured": mt})
                n_unobs_meas += 1
                continue
            r = tween_metrics(tt, mt, fps, shapes[te])
            r.update({"clip": clip_id, "truth_el": te, "meas_el": me, "prop": tt["prop"], "shape": shapes[te],
                      "resolution": f"{truth['video']['width']}x{truth['video']['height']}",
                      "duration_bucket": duration_bucket(tt["duration_s"]),
                      "truth": tt, "measured": mt})
            rows.append(r)
        spurious += [{"clip": clip_id, "meas_el": e, "tween": t} for e, t in m.spurious]
        for e, t in m.missed:
            item = {"clip": clip_id, "truth_el": e, "shape": shapes[e], "tween": t}
            if unobservable(scene_els[e], t):
                unobs.append({**item, "measured": None})
            else:
                missed.append(item)
        n_measured += sum(len(e["tweens"]) for e in tl["elements"]) - n_unobs_meas
        stagger += _stagger_rows(truth, tl, m.element_map, fps)
        if on_clip:
            on_clip(clip_id, len(m.pairs), len(m.spurious), len(m.missed), time.time() - tc)
    summary = summarize(rows, len(spurious), n_measured, len(missed))
    summary["stagger_detected"] = sum(r["ok"] for r in stagger)
    summary["stagger_total"] = len(stagger)
    summary["unobservable"] = len(unobs)
    result = {
        "engine_version": __version__,
        "gt_dir": str(gt_dir),
        "manifest": manifest,
        "summary": summary,
        "elapsed_s": round(time.time() - t0, 1),
        "tweens": rows,
        "spurious": spurious,
        "missed": missed,
        "stagger": stagger,
        "unobservable": unobs,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(result, indent=1))
    write_report(result, run_dir / "report.md")
    return result
