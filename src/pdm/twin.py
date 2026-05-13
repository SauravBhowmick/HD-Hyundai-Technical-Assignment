"""Digital-twin JSON assembly.

Pulls the model probability, health state, likely component and a small set
of evidence strings into a single Pydantic schema that the FastAPI endpoint
returns verbatim. The TS dashboard mirrors this exact shape.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from .health import classify_health, confidence_from_prob, prescribe
from .likely_component import pick_component


class DigitalTwin(BaseModel):
    machineID: int
    timestamp: datetime
    failure_risk_24h: float = Field(..., ge=0.0, le=1.0)
    health_state: Literal["healthy", "watch", "degraded", "critical"]
    likely_component: Literal["comp1", "comp2", "comp3", "comp4"] | None
    confidence: Literal["low", "medium", "high"]
    main_evidence: list[str] = Field(default_factory=list, max_length=3)
    prescription: Literal[
        "continue", "monitor", "inspect", "schedule_maintenance", "urgent_maintenance"
    ]


def _format_hours(h: float) -> str:
    if h >= 24:
        return f"{h / 24:.1f} d"
    return f"{h:.1f} h"


def build_evidence(feature_row: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    items: list[tuple[float, str]] = []
    error_window = max(cfg["features"]["error_window_h"])
    for err_id in cfg["features"]["error_to_component"]:
        key = f"count_{err_id}_{error_window}h"
        v = feature_row.get(key, 0)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = 0
        v = float(v)
        if v > 0:
            items.append((v, f"{int(v)}x {err_id} in last {error_window}h"))

    for comp in ("comp1", "comp2", "comp3", "comp4"):
        h = float(feature_row.get(f"hours_since_maint_{comp}", 0.0) or 0.0)
        if h >= 24 * 60:
            items.append((h / 24, f"no maintenance on {comp} for {_format_hours(h)}"))

    for sensor in ("vibration", "pressure", "rotate", "volt"):
        for window in (3, 24):
            key = f"{sensor}_std_{window}h"
            v = feature_row.get(key)
            if v is None or pd.isna(v):
                continue
            if (sensor == "vibration" and v > 8) or (sensor == "pressure" and v > 15) \
               or (sensor == "rotate" and v > 80) or (sensor == "volt" and v > 25):
                items.append((float(v), f"high {sensor} variability ({window}h std={v:.1f})"))

    items.sort(key=lambda kv: kv[0], reverse=True)
    return [s for _, s in items[:3]]


def build_twin(
    machineID: int,
    timestamp: pd.Timestamp,
    probability: float,
    feature_row: dict[str, Any],
    cfg: dict[str, Any],
) -> DigitalTwin:
    state = classify_health(probability, cfg)
    conf = confidence_from_prob(probability)
    if state == "healthy":
        component, _ = (None, 0.0)
    else:
        component, _ = pick_component(feature_row, cfg)
    return DigitalTwin(
        machineID=int(machineID),
        timestamp=pd.Timestamp(timestamp).to_pydatetime(),
        failure_risk_24h=float(probability),
        health_state=state,
        likely_component=component,  # type: ignore[arg-type]
        confidence=conf,
        main_evidence=build_evidence(feature_row, cfg),
        prescription=prescribe(state, conf),
    )
