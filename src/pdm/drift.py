"""Lightweight drift report.

Computes the Population Stability Index (PSI) per feature between the train
and test slices. Outputs a markdown table to ``artifacts/drift_report.md``
ranked by PSI. PSI bands are the standard ones used in credit risk
literature: < 0.1 stable, 0.1-0.25 minor shift, > 0.25 major shift.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .features import build_features, feature_columns
from .io import load_raw
from .split import time_split


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if qs.size < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    e_hist, _ = np.histogram(expected, bins=qs)
    a_hist, _ = np.histogram(actual, bins=qs)
    e = e_hist / max(e_hist.sum(), 1)
    a = a_hist / max(a_hist.sum(), 1)
    eps = 1e-6
    return float(np.sum((a - e) * np.log((a + eps) / (e + eps))))


def _band(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "minor shift"
    return "major shift"


def build_drift_report(cfg: dict[str, Any]) -> Path:
    frames = load_raw(cfg)
    feats = build_features(frames, cfg)
    feats["label"] = 0
    X_train, _, X_test, _ = time_split(feats, feats["label"], cfg)
    cols = feature_columns(feats)

    rows: list[dict[str, Any]] = []
    for c in cols:
        psi = _psi(X_train[c].to_numpy(dtype=float), X_test[c].to_numpy(dtype=float))
        rows.append({"feature": c, "psi": psi, "band": _band(psi)})
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    out = resolve_path(cfg, "drift_report")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Drift report",
        "",
        f"Cutoff: `{cfg['split']['cutoff']}` (train: rows before, test: rows on/after)",
        "",
        "PSI bands: < 0.1 stable, 0.1-0.25 minor shift, >= 0.25 major shift.",
        "",
        "| Feature | PSI | Band |",
        "|---|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| `{r['feature']}` | {r['psi']:.4f} | {r['band']} |")
    summary = df["band"].value_counts().to_dict()
    lines += ["", "## Summary", "", f"```\n{summary}\n```", ""]
    out.write_text("\n".join(lines))
    return out
