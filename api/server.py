"""FastAPI app: serves the digital-twin JSON."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdm.config import load_config, resolve_path
from pdm.io import load_raw
from pdm.predict import Predictor
from pdm.twin import DigitalTwin, build_twin


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

    def _predictor() -> Predictor:
        if state["predictor"] is None:
            state["predictor"] = Predictor.from_artifacts(cfg)
        return state["predictor"]

    def _frames() -> dict[str, pd.DataFrame]:
        if state["frames"] is None:
            state["frames"] = load_raw(cfg)
        return state["frames"]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info", response_model=DatasetInfo)
    def info() -> DatasetInfo:
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
        m = _frames()["machines"]
        return [
            MachineInfo(machineID=int(r.machineID), model=str(r.model), age_years=int(r.age))
            for r in m.itertuples()
        ]

    @app.post("/predict", response_model=DigitalTwin)
    def predict(req: PredictRequest) -> DigitalTwin:
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
        p = resolve_path(cfg, "metrics_file")
        if not p.exists():
            raise HTTPException(404, detail="no metrics; train first")
        return json.loads(p.read_text())

    return app


app = create_app()
