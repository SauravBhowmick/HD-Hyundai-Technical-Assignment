"""Rule-based likely-component inference.

Justification (also recorded in the README): the dataset has only four
components, and the strongest signals for which one is about to fail are
already in our feature set:

1. ``hours_since_maint_comp{k}`` - if a component has not been serviced for a
   long time, it is the more likely culprit.
2. Recent error counts - errors are mapped to suspected components via the
   ``features.error_to_component`` configuration constant.

We score each component as

    score(k) = w1 * (hours_since_maint_comp_k / max_hours_overall)
             + w2 * sum_of_error_counts_mapped_to_k

and return the argmax plus the evidence that justified it. The rule is
deterministic, fully explainable, and consistent with the ``main_evidence``
list emitted by the digital twin.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

COMPONENTS = ("comp1", "comp2", "comp3", "comp4")


def score_components(feature_row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    """Return a dict of component -> score, normalised to sum to 1.0."""
    hours = {c: float(feature_row.get(f"hours_since_maint_{c}", 0.0)) for c in COMPONENTS}
    max_h = max(hours.values()) or 1.0

    error_window = max(cfg["features"]["error_window_h"])
    err_map: dict[str, str] = cfg["features"]["error_to_component"]
    err_counts: dict[str, float] = {c: 0.0 for c in COMPONENTS}
    for err_id, comp in err_map.items():
        key = f"count_{err_id}_{error_window}h"
        v = feature_row.get(key, 0.0)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = 0.0
        err_counts[comp] += float(v)

    w_maint, w_err = 0.6, 0.4
    raw = {
        c: w_maint * (hours[c] / max_h) + w_err * err_counts[c]
        for c in COMPONENTS
    }
    total = sum(raw.values())
    if total <= 0:
        return {c: 1 / len(COMPONENTS) for c in COMPONENTS}
    return {c: v / total for c, v in raw.items()}


def pick_component(feature_row: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, float]:
    scores = score_components(feature_row, cfg)
    best = max(scores.items(), key=lambda kv: kv[1])
    return best
