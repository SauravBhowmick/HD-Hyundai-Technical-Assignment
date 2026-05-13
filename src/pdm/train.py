"""Training entry-point.

Trains a baseline (logistic regression on standardized features) and a final
model (LightGBM), logs both as MLflow runs in the same experiment, and saves
the best model (by test PR-AUC) plus a metrics JSON for later inspection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import resolve_path
from .evaluate import evaluate_predictions, write_metrics
from .features import build_features, feature_columns
from .io import load_raw
from .labels import label_in_window
from .split import time_split
from .validate import validate


def _feature_hash(features: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    h.update(",".join(columns).encode("utf-8"))
    h.update(np.ascontiguousarray(features[columns].fillna(0).values).tobytes())
    return h.hexdigest()


def _build_baseline(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(**params)),
    ])


def _build_lgbm(params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(**params, verbosity=-1)


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    plots_dir = resolve_path(cfg, "plots_dir")
    mlruns_dir = resolve_path(cfg, "mlruns_dir")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment("pdm-digital-twin")

    print("[train] loading raw data...")
    frames = load_raw(cfg)
    report = validate(frames)
    if not report.ok:
        raise RuntimeError(f"Validation failed: {report.issues}")
    print(f"[train] validation OK. {report.info}")

    print("[train] building features...")
    features = build_features(frames, cfg)
    labels = label_in_window(features, frames["failures"], cfg["label"]["horizon_h"])
    features["label"] = labels

    feat_cols = feature_columns(features)
    fhash = _feature_hash(features, feat_cols)
    (artifacts_dir / "feature_hash.txt").write_text(fhash + "\n")
    print(f"[train] features: {features.shape}, feature_hash={fhash[:12]}")

    print("[train] time-aware split...")
    X_train, y_train, X_test, y_test = time_split(features, features["label"], cfg)

    train_mask = X_train[feat_cols].notna().all(axis=1)
    X_train_fit = X_train.loc[train_mask, feat_cols]
    y_train_fit = y_train.loc[train_mask]
    test_mask = X_test[feat_cols].notna().all(axis=1)
    X_test_fit = X_test.loc[test_mask, feat_cols]
    y_test_fit = y_test.loc[test_mask]
    machines_test = X_test.loc[test_mask, "machineID"]
    dt_test = X_test.loc[test_mask, "datetime"]

    print(f"[train] train rows={len(X_train_fit)} (pos={int(y_train_fit.sum())}), "
          f"test rows={len(X_test_fit)} (pos={int(y_test_fit.sum())})")

    parent_metrics: dict[str, Any] = {"runs": {}, "feature_hash": fhash,
                                      "feature_columns": feat_cols,
                                      "train_rows": int(len(X_train_fit)),
                                      "test_rows": int(len(X_test_fit)),
                                      "train_positives": int(y_train_fit.sum()),
                                      "test_positives": int(y_test_fit.sum()),
                                      "cutoff": cfg["split"]["cutoff"]}

    best_run: dict[str, Any] = {"name": None, "pr_auc": -1.0, "model": None,
                                "threshold": None, "metrics": None}

    baseline_params = dict(cfg["models"]["baseline"]["params"])
    lgbm_params = dict(cfg["models"]["final"]["params"])

    runs = [
        ("baseline_lr", _build_baseline(baseline_params), baseline_params),
        ("lightgbm_v1", _build_lgbm(lgbm_params), lgbm_params),
    ]

    for name, model, params in runs:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**{f"p_{k}": v for k, v in params.items()},
                               "model": name,
                               "cutoff": cfg["split"]["cutoff"],
                               "horizon_h": cfg["label"]["horizon_h"],
                               "feature_hash": fhash[:12]})
            print(f"[train] fitting {name}...")
            model.fit(X_train_fit, y_train_fit.values)
            y_prob = model.predict_proba(X_test_fit)[:, 1]
            metrics = evaluate_predictions(
                y_test_fit.values, y_prob, machines_test, dt_test,
                threshold=None, plots_dir=plots_dir, label=name,
            )
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            print(f"[train]   {name}: PR-AUC={metrics['pr_auc']:.4f}  "
                  f"ROC-AUC={metrics['roc_auc']:.4f}  "
                  f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
                  f"thr={metrics['threshold']:.3f}")
            parent_metrics["runs"][name] = metrics
            if metrics["pr_auc"] > best_run["pr_auc"]:
                best_run = {"name": name, "pr_auc": metrics["pr_auc"],
                            "model": model, "threshold": metrics["threshold"],
                            "metrics": metrics}

    model_file = resolve_path(cfg, "model_file")
    threshold_file = resolve_path(cfg, "threshold_file")
    metrics_file = resolve_path(cfg, "metrics_file")

    joblib.dump({
        "model": best_run["model"],
        "feature_columns": feat_cols,
        "model_name": best_run["name"],
        "feature_hash": fhash,
    }, model_file)
    threshold_file.write_text(json.dumps({
        "threshold": best_run["threshold"],
        "model_name": best_run["name"],
    }, indent=2))
    parent_metrics["best_run"] = best_run["name"]
    parent_metrics["best_threshold"] = best_run["threshold"]
    write_metrics(parent_metrics, metrics_file)

    print(f"[train] best model: {best_run['name']} (PR-AUC={best_run['pr_auc']:.4f}) "
          f"-> {model_file}")
    return parent_metrics
