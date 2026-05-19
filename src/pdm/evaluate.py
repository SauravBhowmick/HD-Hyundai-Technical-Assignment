"""Model evaluation.

Computes the metrics required by the brief:
* PR-AUC (primary), ROC-AUC.
* Precision/recall and confusion matrix at the chosen threshold.
* False alarms per machine-month (an "alarm" = positive prediction; a
  false alarm = positive prediction with no real failure in the next 24h).

Also produces three plots: PR curve, ROC curve, and a probability histogram.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision * recall) / np.where((precision + recall) == 0, 1, precision + recall)
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def _alarms_per_machine_month(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    machine_ids: pd.Series,
    timestamps: pd.Series,
) -> dict[str, float]:
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "machineID": machine_ids.values,
        "datetime": pd.to_datetime(timestamps).values,
    })
    df["false_alarm"] = (df["y_pred"] == 1) & (df["y_true"] == 0)
    df["alarm"] = df["y_pred"] == 1
    span_h = (df["datetime"].max() - df["datetime"].min()).total_seconds() / 3600.0
    months = max(span_h / (24 * 30.0), 1e-9)
    n_machines = max(df["machineID"].nunique(), 1)
    return {
        "alarms_per_machine_month": float(df["alarm"].sum() / (n_machines * months)),
        "false_alarms_per_machine_month": float(df["false_alarm"].sum() / (n_machines * months)),
        "total_alarms": int(df["alarm"].sum()),
        "total_false_alarms": int(df["false_alarm"].sum()),
        "test_span_months": float(months),
        "n_machines": int(n_machines),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    machine_ids: pd.Series,
    timestamps: pd.Series,
    threshold: float | None = None,
    plots_dir: Path | None = None,
    label: str = "model",
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    if threshold is None:
        threshold = _f1_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    alarms = _alarms_per_machine_month(y_true, y_pred, machine_ids, timestamps)

    metrics: dict[str, Any] = {
        "label": label,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        **alarms,
    }

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_pr(y_true, y_prob, pr_auc, plots_dir / f"pr_curve_{label}.png", label)
        _plot_roc(y_true, y_prob, roc_auc, plots_dir / f"roc_curve_{label}.png", label)
        _plot_hist(y_prob, threshold, plots_dir / f"prob_hist_{label}.png", label)

    return metrics


def _plot_pr(y_true, y_prob, pr_auc, path: Path, label: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{label}  PR curve (AP={pr_auc:.3f})  -- test set")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_roc(y_true, y_prob, roc_auc, path: Path, label: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{label}  ROC curve (AUC={roc_auc:.3f})  -- test set")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_hist(y_prob, threshold, path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=50)
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count")
    ax.set_title(f"{label}  probability distribution (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curve_comparison(
    runs: list[dict[str, Any]],
    plots_dir: Path,
    suffix: str = "comparison",
    title_prefix: str = "",
) -> dict[str, Path]:
    """Overlay PR and ROC curves for ``len(runs)`` models on a single chart each.

    ``runs`` is a list of dicts with keys ``label``, ``y_true``, ``y_prob``.
    Each curve's legend label is the model name plus its AUC, so the chart
    is self-describing. Also draws a baseline:

    * PR: positive-class prevalence (a no-skill classifier saturates here).
    * ROC: the y=x diagonal.

    Returns ``{"pr": Path, "roc": Path}`` to the produced PNGs.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- PR ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    prevalence_seen: float | None = None
    for run in runs:
        y_true = np.asarray(run["y_true"]).astype(int)
        y_prob = np.asarray(run["y_prob"]).astype(float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))
        ax.plot(recall, precision, lw=2, label=f"{run['label']}  AP={ap:.3f}")
        if prevalence_seen is None and len(y_true):
            prevalence_seen = float(y_true.mean())
    if prevalence_seen is not None:
        ax.axhline(prevalence_seen, color="grey", lw=1, linestyle=":",
                   label=f"no-skill (prevalence={prevalence_seen:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    base_pr = "PR curves (test set)"
    ax.set_title(f"{title_prefix}{base_pr}" if title_prefix else base_pr)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    pr_path = plots_dir / f"pr_curve_{suffix}.png"
    fig.savefig(pr_path, dpi=130)
    plt.close(fig)

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for run in runs:
        y_true = np.asarray(run["y_true"]).astype(int)
        y_prob = np.asarray(run["y_prob"]).astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))
        ax.plot(fpr, tpr, lw=2, label=f"{run['label']}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":", label="no-skill (AUC=0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    base_roc = "ROC curves (test set)"
    ax.set_title(f"{title_prefix}{base_roc}" if title_prefix else base_roc)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    roc_path = plots_dir / f"roc_curve_{suffix}.png"
    fig.savefig(roc_path, dpi=130)
    plt.close(fig)

    return {"pr": pr_path, "roc": roc_path}


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
