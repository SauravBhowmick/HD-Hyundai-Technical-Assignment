"""Inference: produce a probability + feature row for a (machine, timestamp).

We always rebuild features from the raw CSVs to make the path deterministic
and to keep one source of truth (the same feature code is used in train and
inference). For efficiency, ``predict_batch`` builds features once and reuses
them across all requested rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .features import build_features
from .io import load_raw


@dataclass
class Predictor:
    model: Any
    feature_columns: list[str]
    model_name: str
    threshold: float
    features: pd.DataFrame
    cfg: dict[str, Any]

    @classmethod
    def from_artifacts(cls, cfg: dict[str, Any] | None = None) -> "Predictor":
        cfg = cfg or load_config()
        bundle = joblib.load(resolve_path(cfg, "model_file"))
        threshold = json.loads(resolve_path(cfg, "threshold_file").read_text())["threshold"]
        frames = load_raw(cfg)
        feats = build_features(frames, cfg)
        return cls(
            model=bundle["model"],
            feature_columns=bundle["feature_columns"],
            model_name=bundle["model_name"],
            threshold=float(threshold),
            features=feats,
            cfg=cfg,
        )

    def predict_row(self, machineID: int, timestamp: pd.Timestamp) -> dict[str, Any]:
        ts = pd.Timestamp(timestamp)
        sub = self.features[
            (self.features["machineID"] == machineID)
            & (self.features["datetime"] == ts)
        ]
        if sub.empty:
            sub = self.features[
                (self.features["machineID"] == machineID)
                & (self.features["datetime"] <= ts)
            ].tail(1)
        if sub.empty:
            raise ValueError(f"No feature row for machineID={machineID} at {ts}")

        row = sub.iloc[0]
        X = sub[self.feature_columns].fillna(0)
        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "machineID": int(row["machineID"]),
            "timestamp": row["datetime"].to_pydatetime(),
            "probability": prob,
            "threshold": self.threshold,
            "feature_row": row.to_dict(),
        }

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """``rows`` must have columns ``machineID`` and ``datetime``."""
        join = rows.merge(self.features, on=["machineID", "datetime"], how="left")
        X = join[self.feature_columns].fillna(0)
        probs = self.model.predict_proba(X)[:, 1]
        join["probability"] = probs
        return join
