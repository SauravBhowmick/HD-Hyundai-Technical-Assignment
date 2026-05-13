"""Time-aware train/test split.

A single cutoff timestamp is used for **all** machines. Rows with
``datetime < cutoff`` go to train; ``datetime >= cutoff`` to test. We never
shuffle - the cutoff and the rationale are documented in the README.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def time_split(
    features: pd.DataFrame,
    labels: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    cutoff = pd.Timestamp(cfg["split"]["cutoff"])
    train_mask = features["datetime"] < cutoff
    test_mask = ~train_mask
    return (
        features.loc[train_mask].reset_index(drop=True),
        labels.loc[train_mask].reset_index(drop=True),
        features.loc[test_mask].reset_index(drop=True),
        labels.loc[test_mask].reset_index(drop=True),
    )
