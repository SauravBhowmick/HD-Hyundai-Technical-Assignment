"""Label generation.

Label = 1 if any failure occurs in the window ``(t, t + horizon]`` for the
same machine, else 0. The window is intentionally **strictly open at t**
(events at exactly t are not future events) and **closed at t + horizon**.

This is encapsulated as a pure function so the test suite can lock in the
boundary semantics.
"""
from __future__ import annotations

import pandas as pd


def label_in_window(
    timestamps: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_h: int = 24,
) -> pd.Series:
    """Return the binary label aligned to ``timestamps``.

    Parameters
    ----------
    timestamps : DataFrame with columns ``datetime`` and ``machineID``.
    failures   : DataFrame with columns ``datetime`` and ``machineID``.
    horizon_h  : prediction horizon in hours (default 24).

    Returns a Series of int8 (0/1) aligned to ``timestamps.index``.
    """
    if timestamps.empty:
        return pd.Series([], dtype="int8")

    horizon = pd.Timedelta(hours=horizon_h)
    fail = failures[["datetime", "machineID"]].copy()
    fail = fail.rename(columns={"datetime": "fail_dt"})
    fail.sort_values(["fail_dt", "machineID"], inplace=True, kind="mergesort")
    fail = fail.reset_index(drop=True)

    base = timestamps[["datetime", "machineID"]].copy()
    base["_orig_idx"] = base.index
    base.sort_values(["datetime", "machineID"], inplace=True, kind="mergesort")

    merged = pd.merge_asof(
        base,
        fail,
        left_on="datetime",
        right_on="fail_dt",
        by="machineID",
        direction="forward",
        allow_exact_matches=False,
    )
    delta = merged["fail_dt"] - merged["datetime"]
    label = ((delta > pd.Timedelta(0)) & (delta <= horizon)).astype("int8")
    label.index = merged["_orig_idx"].values
    return label.sort_index()
