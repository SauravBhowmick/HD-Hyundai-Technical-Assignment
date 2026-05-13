"""Feature engineering.

For each ``(machineID, datetime)`` row in the telemetry grid we compute:

* Rolling mean/std of the four sensor channels over 3h/24h/72h windows.
* Counts of each error type over the last 24h and 72h.
* ``hours_since_maint_comp{1..4}`` - time since the last maintenance event
  for that component (human-interpretable).
* ``age_at_t_years`` - the static machine age uplifted to the timestamp.
* ``model`` one-hot.

Leakage guard: every rolling window is computed with ``closed='left'`` so a
feature at time *t* only uses observations strictly before *t*. The age uplift
is also computed from the timestamp itself, so no peeking into the future.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rolling_stats(
    telemetry: pd.DataFrame,
    cols: list[str],
    windows_h: list[int],
) -> pd.DataFrame:
    out = telemetry[["datetime", "machineID"]].copy()
    t = telemetry.set_index("datetime")
    grouped = t.groupby("machineID", sort=False, group_keys=False)
    for w in windows_h:
        win = f"{w}h"
        rolled_mean = grouped[cols].rolling(win, closed="left").mean()
        rolled_std = grouped[cols].rolling(win, closed="left").std()
        rolled_mean.columns = [f"{c}_mean_{w}h" for c in cols]
        rolled_std.columns = [f"{c}_std_{w}h" for c in cols]
        rolled = pd.concat([rolled_mean, rolled_std], axis=1).reset_index()
        out = out.merge(rolled, on=["machineID", "datetime"], how="left")
    return out


def _error_counts(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    windows_h: list[int],
) -> pd.DataFrame:
    out = telemetry[["datetime", "machineID"]].copy()
    if errors.empty:
        for w in windows_h:
            out[f"errors_total_{w}h"] = 0
        return out

    err_types = sorted(errors["errorID"].dropna().unique().tolist())
    wide = (
        errors.assign(_one=1)
        .pivot_table(
            index=["datetime", "machineID"],
            columns="errorID",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for et in err_types:
        if et not in wide.columns:
            wide[et] = 0

    wide = wide.set_index("datetime").sort_index()
    parts: list[pd.DataFrame] = []
    for mid, g in wide.groupby("machineID", sort=False):
        g = g.drop(columns=["machineID"]).sort_index()
        for w in windows_h:
            rolled = g[err_types].rolling(f"{w}h", closed="left").sum()
            rolled.columns = [f"count_{c}_{w}h" for c in err_types]
            g = g.join(rolled)
        g["machineID"] = mid
        parts.append(g.reset_index())
    counts = pd.concat(parts, ignore_index=True)

    feat_cols = [c for c in counts.columns if c.startswith("count_")]
    counts = counts[["datetime", "machineID", *feat_cols]]
    merged = out.merge(counts, on=["datetime", "machineID"], how="left")
    merged[feat_cols] = merged[feat_cols].fillna(0)
    for w in windows_h:
        merged[f"errors_total_{w}h"] = merged[
            [f"count_{c}_{w}h" for c in err_types]
        ].sum(axis=1)
    return merged


def _hours_since_maint(
    telemetry: pd.DataFrame,
    maint: pd.DataFrame,
) -> pd.DataFrame:
    comps = ["comp1", "comp2", "comp3", "comp4"]
    out = telemetry[["datetime", "machineID"]].copy()
    base = out.assign(_orig=np.arange(len(out)))
    base.sort_values(["datetime", "machineID"], inplace=True, kind="mergesort")

    for comp in comps:
        evt = (
            maint[maint["comp"] == comp][["machineID", "datetime"]]
            .rename(columns={"datetime": "maint_dt"})
            .sort_values(["maint_dt", "machineID"], kind="mergesort")
            .reset_index(drop=True)
        )
        merged = pd.merge_asof(
            base,
            evt,
            left_on="datetime",
            right_on="maint_dt",
            by="machineID",
            direction="backward",
            allow_exact_matches=False,
        )
        delta_h = (merged["datetime"] - merged["maint_dt"]).dt.total_seconds() / 3600.0
        base[f"hours_since_maint_{comp}"] = delta_h.values

    base = base.sort_values("_orig").drop(columns=["_orig"]).reset_index(drop=True)
    feat_cols = [f"hours_since_maint_{c}" for c in comps]
    base[feat_cols] = base[feat_cols].fillna(1e6)
    return base


def _machine_static(
    telemetry: pd.DataFrame,
    machines: pd.DataFrame,
    age_anchor: pd.Timestamp,
) -> pd.DataFrame:
    out = telemetry[["datetime", "machineID"]].merge(machines, on="machineID", how="left")
    delta_days = (out["datetime"] - age_anchor).dt.total_seconds() / 86400.0
    out["age_at_t_years"] = out["age"].astype("float64") + delta_days / 365.25
    one_hot = pd.get_dummies(out["model"], prefix="model").astype("int8")
    out = pd.concat([out, one_hot], axis=1)
    out = out.rename(columns={"age": "age_years"})
    return out[["datetime", "machineID", "model", "age_years", "age_at_t_years",
                *one_hot.columns]]


def build_features(
    frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build the one-row-per-(datetime, machineID) feature frame."""
    telemetry: pd.DataFrame = frames["telemetry"]
    errors: pd.DataFrame = frames["errors"]
    maint: pd.DataFrame = frames["maint"]
    machines: pd.DataFrame = frames["machines"]

    cols = cfg["features"]["telemetry_cols"]
    windows_h = cfg["features"]["rolling_windows_h"]
    err_windows = cfg["features"]["error_window_h"]
    anchor = pd.Timestamp(cfg["age_anchor"])

    base = telemetry[["datetime", "machineID", *cols]].copy()

    roll = _rolling_stats(base, cols, windows_h)
    errs = _error_counts(base, errors, err_windows)
    msnc = _hours_since_maint(base, maint)
    static = _machine_static(base, machines, anchor)

    feats = (
        base.merge(roll, on=["datetime", "machineID"], how="left")
            .merge(errs, on=["datetime", "machineID"], how="left")
            .merge(msnc, on=["datetime", "machineID"], how="left")
            .merge(static, on=["datetime", "machineID"], how="left")
    )

    feats = feats.sort_values(["machineID", "datetime"], kind="mergesort").reset_index(drop=True)
    return feats


FEATURE_EXCLUDE = {"datetime", "machineID", "model", "label",
                   "volt", "rotate", "pressure", "vibration"}


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the ordered list of model-input columns."""
    return [c for c in features.columns if c not in FEATURE_EXCLUDE]
