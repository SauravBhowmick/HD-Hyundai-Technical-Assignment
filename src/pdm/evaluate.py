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
    ax.set_title(f"{label}  PR curve (AP={pr_auc:.3f})")
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
    ax.set_title(f"{label}  ROC curve (AUC={roc_auc:.3f})")
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
    ax.set_title(f"{label}  probability distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
