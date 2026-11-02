"""report.md for a benchmark run."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from motionly_engine.benchmark.metrics import summarize, target_rows


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _ease_str(e: dict) -> str:
    if e["type"] == "named":
        return e["name"]
    if e["type"] == "bezier":
        return "bezier(" + ", ".join(f"{v:.2f}" for v in e["params"]) + ")"
    return f"spring(k={e['stiffness']:.0f}, c={e['damping']:.1f})"


def breakdown(rows: list[dict], key: str, missed: list[dict], spurious_by: dict | None = None) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    miss = defaultdict(int)
    for m in missed:
        k = m.get(key)
        if k is not None:
            miss[k] += 1
    lines = [f"| {key.replace('_', ' ')} | n | timing ok | trajectory ok | named ok | median |Δstart| (fr) | median |Δend| (fr) | missed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(groups, key=str):
        g = groups[k]
        named = [r for r in g if r["named"]]
        nm = f"{_pct(sum(r['name_ok'] for r in named) / len(named))} ({len(named)})" if named else "–"
        lines.append(
            f"| {k} | {len(g)} | {_pct(np.mean([r['timing_ok'] for r in g]))} | "
            f"{_pct(np.mean([r['trajectory_ok'] for r in g]))} | {nm} | "
            f"{np.median([abs(r['start_err_frames']) for r in g]):.2f} | "
            f"{np.median([abs(r['end_err_frames']) for r in g]):.2f} | {miss.get(k, 0)} |")
    return lines


def write_report(res: dict, path: Path) -> None:
    s, rows, man = res["summary"], res["tweens"], res["manifest"]
    L = [f"# Benchmark: {Path(res['gt_dir']).name}", "",
         f"- engine {res['engine_version']}, ground truth `{res['gt_dir']}` (seed {man['seed']}, {man['count']} clips, "
         f"size {man['size']}, {man['fps']} fps, CRF {man['crf']})",
         f"- tweens: {s['matched']} matched, {s['missed']} missed ({_pct(s['missed_rate'])} of truth), "
         f"{s['spurious']} spurious of {s['measured']} measured",
         f"- stagger groups detected (same elements, step within 1 frame): {s['stagger_detected']} / {s['stagger_total']}",
         f"- run time {res['elapsed_s']:.0f} s", "",
         "## Targets (CLAUDE.md)", "", "| Metric | Value | Target | |", "|---|---:|---:|---|"]
    for label, v, tgt, ok in target_rows(s):
        L.append(f"| {label} | {_pct(v)} | {tgt} | {'PASS' if ok else 'FAIL'} |")
    L += ["", f"Named-ease metric is over the {s['named_n']} truth tweens with a named ease. Timing and trajectory "
          "are over matched tweens; missed tweens are listed separately.", ""]
    for key, title in (("family", "Ease family"), ("prop", "Property"), ("duration_bucket", "Duration"),
                       ("resolution", "Resolution"), ("shape", "Shape")):
        missed = []
        for m in res["missed"]:
            mm = dict(m)
            mm["prop"] = m["tween"]["prop"]
            missed.append(mm)
        L += [f"## By {title.lower()}", ""] + breakdown(rows, key, missed) + [""]

    L += ["## Worst 15 tweens (by trajectory error)", "",
          "| clip | el | prop | truth ease | measured ease | Δstart (fr) | Δend (fr) | traj err | ease dist |",
          "|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["trajectory_err"])[:15]:
        L.append(f"| {r['clip']} | {r['truth_el']} ({r['shape']}) | {r['prop']} | {_ease_str(r['truth']['ease'])} | "
                 f"{_ease_str(r['measured']['ease'])} | {r['start_err_frames']:+.2f} | {r['end_err_frames']:+.2f} | "
                 f"{r['trajectory_err']:.3f} | {r['ease_dist']:.3f} |")
    L += ["", "## Spurious tweens", ""]
    if res["spurious"]:
        L += ["| clip | el | prop | from → to | start | ease |", "|---|---|---|---|---:|---|"]
        for sp in res["spurious"]:
            t = sp["tween"]
            L.append(f"| {sp['clip']} | {sp['meas_el']} | {t['prop']} | {t['from']:.3f} → {t['to']:.3f} | "
                     f"{t['start_s']:.2f} | {_ease_str(t['ease'])} |")
    else:
        L.append("None.")
    L += ["", "## Missed tweens", ""]
    if res["missed"]:
        L += ["| clip | el | prop | from → to | start | ease |", "|---|---|---|---|---:|---|"]
        for ms in res["missed"]:
            t = ms["tween"]
            L.append(f"| {ms['clip']} | {ms['truth_el']} ({ms['shape']}) | {t['prop']} | {t['from']:.3f} → {t['to']:.3f} | "
                     f"{t['start_s']:.2f} | {_ease_str(t['ease'])} |")
    else:
        L.append("None.")
    path.write_text("\n".join(L) + "\n")


__all__ = ["write_report", "summarize"]
