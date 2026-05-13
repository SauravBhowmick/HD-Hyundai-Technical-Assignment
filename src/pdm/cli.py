"""Typer CLI: train / evaluate / predict / drift / validate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich import print as rprint

from .config import load_config, resolve_path
from .drift import build_drift_report
from .io import load_raw
from .predict import Predictor
from .twin import build_twin
from .validate import validate as validate_fn

app = typer.Typer(add_completion=False, help="PdM Digital Twin CLI")


def _cfg(config: Path | None) -> dict:
    return load_config(config) if config else load_config()


@app.command()
def validate(config: Path = typer.Option(None, exists=True, dir_okay=False)) -> None:
    """Run schema + value-range checks on the raw CSVs."""
    cfg = _cfg(config)
    frames = load_raw(cfg)
    report = validate_fn(frames)
    rprint(report.to_dict())
    raise typer.Exit(code=0 if report.ok else 1)


@app.command()
def train(config: Path = typer.Option(None, exists=True, dir_okay=False)) -> None:
    """Train baseline + LightGBM, log MLflow runs, save best model."""
    from .train import train as train_fn

    cfg = _cfg(config)
    metrics = train_fn(cfg)
    rprint({"best_run": metrics["best_run"],
            "best_pr_auc": metrics["runs"][metrics["best_run"]]["pr_auc"]})


@app.command()
def evaluate(config: Path = typer.Option(None, exists=True, dir_okay=False)) -> None:
    """Reprint the metrics JSON from the last training run."""
    cfg = _cfg(config)
    path = resolve_path(cfg, "metrics_file")
    if not path.exists():
        rprint(f"[red]no metrics at {path}; run `train` first.[/red]")
        raise typer.Exit(1)
    rprint(json.loads(path.read_text()))


@app.command()
def predict(
    machine_id: int = typer.Option(..., "--machine-id"),
    timestamp: str = typer.Option(..., "--timestamp", help="ISO 8601 timestamp"),
    config: Path = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Predict a single (machineID, timestamp) and print the digital-twin JSON."""
    cfg = _cfg(config)
    pred = Predictor.from_artifacts(cfg)
    result = pred.predict_row(machine_id, pd.Timestamp(timestamp))
    twin = build_twin(
        machineID=result["machineID"],
        timestamp=result["timestamp"],
        probability=result["probability"],
        feature_row=result["feature_row"],
        cfg=cfg,
    )
    rprint(json.loads(twin.model_dump_json()))


@app.command()
def drift(config: Path = typer.Option(None, exists=True, dir_okay=False)) -> None:
    """Compute PSI per feature train-vs-test and write the markdown report."""
    cfg = _cfg(config)
    out = build_drift_report(cfg)
    rprint(f"[green]wrote {out}[/green]")


if __name__ == "__main__":
    app()
