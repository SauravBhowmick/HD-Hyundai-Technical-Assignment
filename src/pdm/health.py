"""Health state and prescription mapping.

Pure functions so they're trivial to unit-test and re-use from the API.
"""
from __future__ import annotations

from typing import Any, Literal

HealthState = Literal["healthy", "watch", "degraded", "critical"]
Confidence = Literal["low", "medium", "high"]
Prescription = Literal[
    "continue", "monitor", "inspect", "schedule_maintenance", "urgent_maintenance"
]


def classify_health(prob: float, cfg: dict[str, Any]) -> HealthState:
    t = cfg["health_state"]["thresholds"]
    if prob < t["healthy_max"]:
        return "healthy"
    if prob < t["watch_max"]:
        return "watch"
    if prob < t["degraded_max"]:
        return "degraded"
    return "critical"


def confidence_from_prob(prob: float) -> Confidence:
    """Bucket confidence on the distance from the 0.5 decision boundary."""
    margin = abs(prob - 0.5)
    if margin >= 0.30:
        return "high"
    if margin >= 0.15:
        return "medium"
    return "low"


def prescribe(state: HealthState, confidence: Confidence) -> Prescription:
    if state == "healthy":
        return "continue"
    if state == "watch":
        return "monitor"
    if state == "degraded":
        return "inspect"
    return "urgent_maintenance" if confidence == "high" else "schedule_maintenance"
