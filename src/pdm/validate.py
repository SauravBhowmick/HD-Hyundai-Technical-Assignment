"""Lightweight data-quality checks.

Validation is deliberately simple: schema (columns + dtypes), value ranges
where applicable, no nulls in keys, and a duplicate-key audit. The audit is
informational - the event tables legitimately have duplicate
``(datetime, machineID)`` rows (multiple events at the same hour) - but we
record the count so the README's data-quality note has real numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationReport:
    issues: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": self.issues, "info": self.info}


def _expect_cols(df: pd.DataFrame, name: str, cols: list[str], report: ValidationReport) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        report.issues.append(f"{name}: missing columns {missing}")


def validate(frames: dict[str, pd.DataFrame]) -> ValidationReport:
    r = ValidationReport()

    _expect_cols(frames["machines"], "machines", ["machineID", "model", "age"], r)
    _expect_cols(frames["telemetry"], "telemetry",
                 ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"], r)
    _expect_cols(frames["errors"], "errors", ["datetime", "machineID", "errorID"], r)
    _expect_cols(frames["failures"], "failures", ["datetime", "machineID", "failure"], r)
    _expect_cols(frames["maint"], "maint", ["datetime", "machineID", "comp"], r)
    if r.issues:
        return r

    m = frames["machines"]
    if m["machineID"].is_unique is False:
        r.issues.append("machines.machineID is not unique")
    if (m["age"] < 0).any():
        r.issues.append("machines.age has negative values")

    t = frames["telemetry"]
    for col in ("volt", "rotate", "pressure", "vibration"):
        if t[col].isna().any():
            r.issues.append(f"telemetry.{col} has nulls")
        if (t[col] < 0).any():
            r.issues.append(f"telemetry.{col} has negative values")

    for name in ("telemetry", "errors", "failures", "maint"):
        df = frames[name]
        if df["datetime"].isna().any():
            r.issues.append(f"{name}.datetime has nulls")
        if df["machineID"].isna().any():
            r.issues.append(f"{name}.machineID has nulls")

    info: dict[str, Any] = {}
    for name, df in frames.items():
        info[f"{name}_rows"] = int(len(df))
    for name in ("telemetry", "errors", "failures", "maint"):
        df = frames[name]
        info[f"{name}_dup_keys"] = int(df.duplicated(subset=["datetime", "machineID"]).sum())
    info["telemetry_machines_covered"] = int(frames["telemetry"]["machineID"].nunique())
    info["telemetry_min_dt"] = str(frames["telemetry"]["datetime"].min())
    info["telemetry_max_dt"] = str(frames["telemetry"]["datetime"].max())
    r.info = info
    return r
