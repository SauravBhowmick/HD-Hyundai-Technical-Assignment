"""Deterministic raw-data loading.

We force explicit dtypes, parse timestamps, and apply a canonical sort order
so that the downstream feature build is byte-stable across machines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path

TELEMETRY_DTYPES = {
    "machineID": "int32",
    "volt": "float64",
    "rotate": "float64",
    "pressure": "float64",
    "vibration": "float64",
}
MACHINES_DTYPES = {"machineID": "int32", "model": "string", "age": "int16"}
ERRORS_DTYPES = {"machineID": "int32", "errorID": "string"}
FAILURES_DTYPES = {"machineID": "int32", "failure": "string"}
MAINT_DTYPES = {"machineID": "int32", "comp": "string"}


def _read_csv(path: Path, dtype: dict[str, Any], parse_dt: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=dtype)
    if parse_dt:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_raw(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load all five CSVs from ``paths.raw_dir`` and return them in a dict."""
    raw = resolve_path(cfg, "raw_dir")
    machines = _read_csv(raw / "PdM_machines.csv", MACHINES_DTYPES, parse_dt=False)
    telemetry = _read_csv(raw / "PdM_telemetry.csv", TELEMETRY_DTYPES, parse_dt=True)
    errors = _read_csv(raw / "PdM_errors.csv", ERRORS_DTYPES, parse_dt=True)
    failures = _read_csv(raw / "PdM_failures.csv", FAILURES_DTYPES, parse_dt=True)
    maint = _read_csv(raw / "PdM_maint.csv", MAINT_DTYPES, parse_dt=True)

    machines = machines.sort_values("machineID").reset_index(drop=True)
    for name, df in (("telemetry", telemetry), ("errors", errors),
                     ("failures", failures), ("maint", maint)):
        df.sort_values(["machineID", "datetime"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

    return {
        "machines": machines,
        "telemetry": telemetry,
        "errors": errors,
        "failures": failures,
        "maint": maint,
    }
