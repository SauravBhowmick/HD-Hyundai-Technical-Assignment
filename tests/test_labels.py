"""Tests for the labelling function (required by the assignment specification)."""
from __future__ import annotations

import pandas as pd

from pdm.labels import label_in_window


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_label_within_horizon_is_one() -> None:
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-01-01 00:00"), _ts("2015-01-01 01:00")],
        "machineID": [1, 1],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-01-01 20:00")],
        "machineID": [1],
    })
    y = label_in_window(timestamps, failures, horizon_h=24)
    assert y.tolist() == [1, 1]


def test_failure_strictly_at_t_does_not_label_current_row() -> None:
    """Failure at exactly the same timestamp is not a *future* failure."""
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-01-01 06:00")],
        "machineID": [1],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-01-01 06:00")],
        "machineID": [1],
    })
    y = label_in_window(timestamps, failures, horizon_h=24)
    assert y.tolist() == [0]


def test_failure_exactly_at_horizon_is_included() -> None:
    """Window is closed on the right: failure at t + horizon counts as 1."""
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-01-01 06:00")],
        "machineID": [1],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-01-02 06:00")],
        "machineID": [1],
    })
    y = label_in_window(timestamps, failures, horizon_h=24)
    assert y.tolist() == [1]


def test_failure_just_after_horizon_is_excluded() -> None:
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-01-01 06:00")],
        "machineID": [1],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-01-02 06:01")],
        "machineID": [1],
    })
    y = label_in_window(timestamps, failures, horizon_h=24)
    assert y.tolist() == [0]


def test_failure_on_different_machine_does_not_label() -> None:
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-01-01 06:00"), _ts("2015-01-01 06:00")],
        "machineID": [1, 2],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-01-01 12:00")],
        "machineID": [2],
    })
    y = label_in_window(timestamps, failures, horizon_h=24)
    assert y.tolist() == [0, 1]


def test_multiple_failures_same_timestamp_counts_once_per_row() -> None:
    """Real data has multiple failures at the same (datetime, machineID).
    The label is still 0/1 - never duplicates rows."""
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-03-18 06:00")],
        "machineID": [2],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-03-19 06:00"), _ts("2015-03-19 06:00")],
        "machineID": [2, 2],
    })
    y = label_in_window(timestamps, failures, horizon_h=24)
    assert y.tolist() == [1]
    assert len(y) == 1


def test_determinism() -> None:
    timestamps = pd.DataFrame({
        "datetime": [_ts("2015-01-01 06:00"), _ts("2015-01-01 12:00"),
                     _ts("2015-01-02 00:00")],
        "machineID": [1, 1, 1],
    })
    failures = pd.DataFrame({
        "datetime": [_ts("2015-01-01 18:00"), _ts("2015-01-02 22:00")],
        "machineID": [1, 1],
    })
    a = label_in_window(timestamps, failures, horizon_h=24).tolist()
    b = label_in_window(timestamps, failures, horizon_h=24).tolist()
    assert a == b == [1, 1, 1]
