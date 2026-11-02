"""Detect stagger (same ease and duration, offset starts) and shared timing across elements."""

from __future__ import annotations

import numpy as np

from motionly_engine.fit import easing


def _curve(e: dict):
    if e["type"] == "named":
        from motionly_engine.fit.models import named_curve

        return named_curve(e["name"])
    if e["type"] == "bezier":
        return tuple(e["params"])
    return easing.spring_curve(e["stiffness"], e["damping"])


def find_groups(tweens: list[tuple[str, dict]], frame: float) -> list[dict]:
    """tweens: (element_id, tween dict). Returns CLAUDE.md-style group entries."""
    used = set()
    groups = []
    order = sorted(range(len(tweens)), key=lambda i: tweens[i][1]["start_s"])
    for a in order:
        if a in used:
            continue
        ea, ta = tweens[a]
        members = [a]
        for b in order:
            if b == a or b in used or b in members:
                continue
            eb, tb = tweens[b]
            if eb in {tweens[m][0] for m in members} or tb["prop"] != ta["prop"]:
                continue
            if abs(tb["duration_s"] - ta["duration_s"]) > frame:
                continue
            if easing.curve_distance(_curve(ta["ease"]), _curve(tb["ease"])) > 0.03:
                continue
            members.append(b)
        if len(members) < 2:
            continue
        starts = np.array([tweens[m][1]["start_s"] for m in members])
        steps = np.diff(starts)
        ids = [tweens[m][0] for m in members]
        if np.all(np.abs(steps) <= frame):
            groups.append({"type": "shared", "prop": ta["prop"], "elements": ids,
                           "start_s": round(float(np.mean(starts)), 4)})
        else:
            step = float(np.median(steps))
            if step <= frame or np.any(np.abs(steps - step) > 1.5 * frame):
                continue
            groups.append({"type": "stagger", "prop": ta["prop"], "elements": ids, "step_s": round(step, 4)})
        used.update(members)
    return groups
