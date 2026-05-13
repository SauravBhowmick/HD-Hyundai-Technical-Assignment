"""Feature build sanity tests.

Two important guarantees:
1. **No leakage**: the rolling-window features at the *first* timestamp of a
   machine must be NaN, because by definition no prior data exists.
2. **Hours-since-maint** must reflect the latest maintenance event strictly
   before the current timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.config import load_config
from pdm.features import build_features


def _toy_cfg() -> dict:
    return {
        "age_anchor": "2015-01-01 00:00:00",
        "features": {
            "telemetry_cols": ["volt", "rotate", "pressure", "vibration"],
            "rolling_windows_h": [3, 24],
            "error_window_h": [24],
            "error_to_component": {
                "error1": "comp1", "error2": "comp2", "error3": "comp3",
                "error4": "comp4", "error5": "comp1",
            },
        },
    }


def _toy_frames() -> dict[str, pd.DataFrame]:
    hours = pd.date_range("2015-01-01 06:00", periods=12, freq="1h")
    telemetry = pd.DataFrame({
        "datetime": hours,
        "machineID": 1,
        "volt": np.arange(12, dtype=float),
        "rotate": np.arange(12, dtype=float) + 100,
        "pressure": np.arange(12, dtype=float) + 50,
        "vibration": np.arange(12, dtype=float) + 10,
    })
    machines = pd.DataFrame({"machineID": [1], "model": ["model3"], "age": [10]})
    errors = pd.DataFrame({
        "datetime": [pd.Timestamp("2015-01-01 07:00"), pd.Timestamp("2015-01-01 09:00")],
        "machineID": [1, 1],
        "errorID": ["error1", "error2"],
    })
    failures = pd.DataFrame(columns=["datetime", "machineID", "failure"])
    maint = pd.DataFrame({
        "datetime": [pd.Timestamp("2014-12-30 00:00"),
                     pd.Timestamp("2015-01-01 08:00")],
        "machineID": [1, 1],
        "comp": ["comp1", "comp2"],
    })
    for df in (telemetry, errors, failures, maint):
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
    return {"machines": machines, "telemetry": telemetry,
            "errors": errors, "failures": failures, "maint": maint}


def test_no_leakage_first_row_is_nan() -> None:
    feats = build_features(_toy_frames(), _toy_cfg())
    first = feats.iloc[0]
    for c in ("volt_mean_3h", "rotate_mean_3h", "pressure_std_24h"):
        assert pd.isna(first[c]), f"{c} leaked: {first[c]!r}"


def test_hours_since_maint_is_strictly_before() -> None:
    feats = build_features(_toy_frames(), _toy_cfg())
    at_8 = feats[feats["datetime"] == pd.Timestamp("2015-01-01 08:00")].iloc[0]
    at_9 = feats[feats["datetime"] == pd.Timestamp("2015-01-01 09:00")].iloc[0]
    assert at_8["hours_since_maint_comp2"] > 24, \
        "maint at t=8h must not be visible to feature at t=8h"
    assert abs(at_9["hours_since_maint_comp2"] - 1.0) < 1e-6


def test_age_at_t_grows_with_time() -> None:
    feats = build_features(_toy_frames(), _toy_cfg())
    first = feats.iloc[0]
    last = feats.iloc[-1]
    assert last["age_at_t_years"] > first["age_at_t_years"]
    delta = last["age_at_t_years"] - first["age_at_t_years"]
    expected = (last["datetime"] - first["datetime"]).total_seconds() / (86400 * 365.25)
    assert delta == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_feature_build_is_deterministic() -> None:
    cfg = _toy_cfg()
    a = build_features(_toy_frames(), cfg)
    b = build_features(_toy_frames(), cfg)
    pd.testing.assert_frame_equal(a, b)


def test_real_config_loads() -> None:
    """Smoke check: real config has the keys features.py depends on."""
    cfg = load_config()
    assert "telemetry_cols" in cfg["features"]
    assert "rolling_windows_h" in cfg["features"]
