"""Match measured elements and tweens to the ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClipMatch:
    pairs: list[tuple[str, str, dict, dict]] = field(default_factory=list)  # (truth el, meas el, truth tw, meas tw)
    missed: list[tuple[str, dict]] = field(default_factory=list)  # (truth el, truth tween)
    spurious: list[tuple[str, dict]] = field(default_factory=list)  # (meas el, meas tween)
    element_map: dict[str, str] = field(default_factory=dict)  # meas el -> truth el


def _centre(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def match_elements(truth: dict, timeline: dict) -> dict[str, str | None]:
    """Measured element -> truth element whose slot contains the measured region centre."""
    out = {}
    for me in timeline["elements"]:
        cx, cy = _centre(me["bbox"])
        hit = None
        for te in truth["elements"]:
            x0, y0, x1, y1 = te["slot"]
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                hit = te["id"]
        out[me["id"]] = hit
    return out


def _score(a: dict, b: dict) -> float:
    ov = min(a["visual_end_s"], b["visual_end_s"]) - max(a["visual_start_s"], b["visual_start_s"])
    if ov > 0:
        return ov
    mid_a = (a["visual_start_s"] + a["visual_end_s"]) / 2
    mid_b = (b["visual_start_s"] + b["visual_end_s"]) / 2
    return -abs(mid_a - mid_b)


def match_clip(truth: dict, timeline: dict, max_gap_s: float = 0.5) -> ClipMatch:
    res = ClipMatch()
    emap = match_elements(truth, timeline)
    res.element_map = {k: v for k, v in emap.items() if v is not None}
    t_by_id = {e["id"]: e for e in truth["elements"]}
    claimed: dict[str, list[dict]] = {e["id"]: [] for e in truth["elements"]}
    for me in timeline["elements"]:
        te_id = emap[me["id"]]
        if te_id is None:
            res.spurious += [(me["id"], t) for t in me["tweens"]]
            continue
        # Greedy per prop, best overlap first.
        truth_tw = [t for t in t_by_id[te_id]["tweens"] if t not in claimed[te_id]]
        cand = []
        for mt in me["tweens"]:
            for tt in truth_tw:
                if tt["prop"] == mt["prop"]:
                    s = _score(tt, mt)
                    if s > -max_gap_s:
                        cand.append((s, id(mt), mt, tt))
        cand.sort(key=lambda c: -c[0])
        used_m, used_t = set(), set()
        for s, _, mt, tt in cand:
            if id(mt) in used_m or id(tt) in used_t:
                continue
            used_m.add(id(mt))
            used_t.add(id(tt))
            claimed[te_id].append(tt)
            res.pairs.append((te_id, me["id"], tt, mt))
        res.spurious += [(me["id"], mt) for mt in me["tweens"] if id(mt) not in used_m]
    for te in truth["elements"]:
        res.missed += [(te["id"], t) for t in te["tweens"] if t not in claimed[te["id"]]]
    return res
