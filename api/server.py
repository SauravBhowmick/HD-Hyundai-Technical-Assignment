"""FastAPI app: serves the digital-twin JSON plus dataset upload/training.

Session model
-------------
Until the user uploads their five CSVs and the pipeline succeeds, the analysis
endpoints (`/info`, `/machines`, `/predict`, `/history`, `/metrics`) return HTTP
409 so the web UI keeps the user on the upload screen. Successful uploads write
`artifacts/.session.json` and reset in-memory caches; calling `/session/reset`
removes the sentinel and forces the upload flow again.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdm.config import load_config, resolve_path
from pdm.drift import build_drift_report
from pdm.io import load_raw
from pdm.predict import Predictor
from pdm.train import train as run_training
from pdm.twin import DigitalTwin, build_twin
from pdm.validate import validate

REQUIRED_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors":    "PdM_errors.csv",
    "failures":  "PdM_failures.csv",
    "machines":  "PdM_machines.csv",
    "maint":     "PdM_maint.csv",
}


class PredictRequest(BaseModel):
    machineID: int
    timestamp: datetime


class MachineInfo(BaseModel):
    machineID: int
    model: str
    age_years: int


class HistoryPoint(BaseModel):
    datetime: datetime
    volt: float | None = None
    rotate: float | None = None
    pressure: float | None = None
    vibration: float | None = None
    errors: list[str] = []
    maint: list[str] = []
    failures: list[str] = []


class DatasetInfo(BaseModel):
    min_datetime: datetime
    max_datetime: datetime
    n_machines: int
    model_name: str
    threshold: float


class SessionStatus(BaseModel):
    loaded: bool
    uploaded_at: datetime | None = None
    files: dict[str, str] | None = None
    pipeline: dict[str, Any] | None = None


class StageResult(BaseModel):
    name: str
    ok: bool
    seconds: float
    info: str | None = None


class PipelineResult(BaseModel):
    ok: bool
    stages: list[StageResult]
    metrics: dict[str, Any] | None = None
    error: str | None = None


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(title="PdM Digital Twin", version="0.1.0")
    extra = os.environ.get("PDM_EXTRA_CORS_ORIGINS", "")
    extra_origins = [o.strip() for o in extra.split(",") if o.strip()]
    origins = list(cfg["api"]["cors_origins"]) + extra_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {"cfg": cfg, "predictor": None, "frames": None}

    def _session_file() -> Path:
        return resolve_path(cfg, "artifacts_dir") / ".session.json"

    def _read_session() -> SessionStatus:
        sf = _session_file()
        if not sf.exists():
            return SessionStatus(loaded=False)
        try:
            data = json.loads(sf.read_text())
            return SessionStatus(**data)
        except Exception:
            return SessionStatus(loaded=False)

    def _require_session() -> None:
        if not _read_session().loaded:
            raise HTTPException(
                status_code=409,
                detail="no dataset loaded; POST 5 CSVs to /upload-and-run first.",
            )

    def _predictor() -> Predictor:
        if state["predictor"] is None:
            state["predictor"] = Predictor.from_artifacts(cfg)
        return state["predictor"]

    def _frames() -> dict[str, pd.DataFrame]:
        if state["frames"] is None:
            state["frames"] = load_raw(cfg)
        return state["frames"]

    def _reset_caches() -> None:
        state["predictor"] = None
        state["frames"] = None

    # -- health & session ----------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/session", response_model=SessionStatus)
    def session() -> SessionStatus:
        return _read_session()

    @app.post("/session/reset", response_model=SessionStatus)
    def session_reset() -> SessionStatus:
        sf = _session_file()
        if sf.exists():
            sf.unlink()
        _reset_caches()
        return SessionStatus(loaded=False)

    # -- upload + run --------------------------------------------------------

    @app.post("/upload-and-run", response_model=PipelineResult)
    async def upload_and_run(
        telemetry: UploadFile = File(...),
        errors:    UploadFile = File(...),
        failures:  UploadFile = File(...),
        machines:  UploadFile = File(...),
        maint:     UploadFile = File(...),
    ) -> PipelineResult:
        uploads = {
            "telemetry": telemetry,
            "errors":    errors,
            "failures":  failures,
            "machines":  machines,
            "maint":     maint,
        }
        raw_dir = resolve_path(cfg, "raw_dir")
        raw_dir.mkdir(parents=True, exist_ok=True)
        stages: list[StageResult] = []
        original_names: dict[str, str] = {}

        # stage 1: save files atomically into data/raw with canonical names
        t0 = time.perf_counter()
        try:
            for slot, fobj in uploads.items():
                if not fobj.filename:
                    raise HTTPException(400, f"missing file for slot '{slot}'")
                target = raw_dir / REQUIRED_FILES[slot]
                with target.open("wb") as out:
                    while True:
                        chunk = await fobj.read(1 << 20)  # 1 MiB
                        if not chunk:
                            break
                        out.write(chunk)
                original_names[slot] = fobj.filename
        except HTTPException:
            raise
        except Exception as exc:
            return PipelineResult(
                ok=False,
                stages=stages + [StageResult(name="save", ok=False,
                                             seconds=time.perf_counter() - t0,
                                             info=str(exc))],
                error=f"failed to save uploads: {exc}",
            )
        stages.append(StageResult(name="save", ok=True,
                                  seconds=time.perf_counter() - t0,
                                  info=f"5 files -> {raw_dir}"))

        # stage 2: load + validate
        t0 = time.perf_counter()
        try:
            frames = load_raw(cfg)
            report = validate(frames)
            if not report.ok:
                return PipelineResult(
                    ok=False,
                    stages=stages + [StageResult(name="validate", ok=False,
                                                 seconds=time.perf_counter() - t0,
                                                 info="; ".join(report.issues))],
                    error=f"validation failed: {report.issues}",
                )
            stages.append(StageResult(name="validate", ok=True,
                                      seconds=time.perf_counter() - t0,
                                      info=f"{report.info}"))
        except Exception as exc:
            return PipelineResult(
                ok=False,
                stages=stages + [StageResult(name="validate", ok=False,
                                             seconds=time.perf_counter() - t0,
                                             info=str(exc))],
                error=f"failed to load CSVs: {exc}",
            )

        # stage 3: train (features + LR + LightGBM + evaluation/plots)
        t0 = time.perf_counter()
        try:
            metrics = run_training(cfg)
        except Exception as exc:
            return PipelineResult(
                ok=False,
                stages=stages + [StageResult(name="train", ok=False,
                                             seconds=time.perf_counter() - t0,
                                             info=str(exc))],
                error=f"training failed: {exc}\n{traceback.format_exc(limit=2)}",
            )
        stages.append(StageResult(name="train", ok=True,
                                  seconds=time.perf_counter() - t0,
                                  info=f"best={metrics['best_run']} "
                                       f"PR-AUC={metrics['runs'][metrics['best_run']]['pr_auc']:.3f}"))

        # stage 4: drift report (non-fatal if it errors)
        t0 = time.perf_counter()
        try:
            build_drift_report(cfg)
            stages.append(StageResult(name="drift", ok=True,
                                      seconds=time.perf_counter() - t0,
                                      info="drift_report.md written"))
        except Exception as exc:
            stages.append(StageResult(name="drift", ok=False,
                                      seconds=time.perf_counter() - t0,
                                      info=str(exc)))

        # stage 5: write sentinel + reset caches
        _reset_caches()
        sf = _session_file()
        sf.parent.mkdir(parents=True, exist_ok=True)
        sentinel = {
            "loaded": True,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "files": original_names,
            "pipeline": {
                "best_run": metrics["best_run"],
                "best_threshold": metrics["best_threshold"],
                "feature_hash": metrics["feature_hash"],
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
                "runs": {
                    name: {k: v for k, v in run.items()
                           if k in {"pr_auc", "roc_auc", "precision",
                                    "recall", "f1", "threshold",
                                    "false_alarms_per_machine_month"}}
                    for name, run in metrics["runs"].items()
                },
            },
        }
        sf.write_text(json.dumps(sentinel, indent=2))

        return PipelineResult(ok=True, stages=stages, metrics=sentinel["pipeline"])

    # -- analysis endpoints (gated) -----------------------------------------

    @app.get("/info", response_model=DatasetInfo)
    def info() -> DatasetInfo:
        _require_session()
        f = _frames()
        p = _predictor()
        return DatasetInfo(
            min_datetime=f["telemetry"]["datetime"].min().to_pydatetime(),
            max_datetime=f["telemetry"]["datetime"].max().to_pydatetime(),
            n_machines=int(f["machines"]["machineID"].nunique()),
            model_name=p.model_name,
            threshold=p.threshold,
        )

    @app.get("/machines", response_model=list[MachineInfo])
    def list_machines() -> list[MachineInfo]:
        _require_session()
        m = _frames()["machines"]
        return [
            MachineInfo(machineID=int(r.machineID), model=str(r.model), age_years=int(r.age))
            for r in m.itertuples()
        ]

    @app.post("/predict", response_model=DigitalTwin)
    def predict(req: PredictRequest) -> DigitalTwin:
        _require_session()
        try:
            result = _predictor().predict_row(req.machineID, pd.Timestamp(req.timestamp))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return build_twin(
            machineID=result["machineID"],
            timestamp=result["timestamp"],
            probability=result["probability"],
            feature_row=result["feature_row"],
            cfg=cfg,
        )

    @app.get("/history/{machine_id}", response_model=list[HistoryPoint])
    def history(
        machine_id: int,
        start: datetime = Query(...),
        end: datetime = Query(...),
    ) -> list[HistoryPoint]:
        _require_session()
        f = _frames()
        t = f["telemetry"]
        slice_t = t[(t["machineID"] == machine_id)
                    & (t["datetime"] >= pd.Timestamp(start))
                    & (t["datetime"] <= pd.Timestamp(end))]
        if slice_t.empty:
            return []
        out: dict[pd.Timestamp, HistoryPoint] = {}
        for r in slice_t.itertuples():
            out[r.datetime] = HistoryPoint(
                datetime=r.datetime.to_pydatetime(),
                volt=r.volt, rotate=r.rotate,
                pressure=r.pressure, vibration=r.vibration,
            )
        for name, col in (("errors", "errorID"), ("failures", "failure"), ("maint", "comp")):
            df = f[name]
            sub = df[(df["machineID"] == machine_id)
                     & (df["datetime"] >= pd.Timestamp(start))
                     & (df["datetime"] <= pd.Timestamp(end))]
            for r in sub.itertuples():
                if r.datetime not in out:
                    out[r.datetime] = HistoryPoint(datetime=r.datetime.to_pydatetime())
                getattr(out[r.datetime], name).append(getattr(r, col))
        return [out[ts] for ts in sorted(out)]

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        _require_session()
        p = resolve_path(cfg, "metrics_file")
        if not p.exists():
            raise HTTPException(404, detail="no metrics; train first")
        return json.loads(p.read_text())

    return app


app = create_app()
