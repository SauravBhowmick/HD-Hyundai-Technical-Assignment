"""FastAPI app: serves the digital-twin JSON plus dataset upload/training.

Session model
-------------
Until the user uploads their five CSVs and the pipeline succeeds, the analysis
endpoints (`/info`, `/machines`, `/predict`, `/history`, `/metrics`) return HTTP
409 so the web UI keeps the user on the upload screen. Successful uploads write
`artifacts/.session.json` atomically and reset in-memory caches; calling
`/session/reset` removes the sentinel and forces the upload flow again.

Atomicity & safety
------------------
``/upload-and-run`` is the only mutator. It stages uploads into
``data/raw/.incoming/<uuid>`` and runs the whole pipeline against a cfg whose
``paths.raw_dir`` and ``paths.artifacts_dir`` point at the staging tree. Only on
full success do we ``os.replace`` each staged file into its canonical location
(raw CSVs + model.joblib + metrics/threshold/drift_report/feature_hash) and
then write ``.session.json`` atomically. Any failure cleans up the staging
tree without touching the live dataset or session. A process-wide lock
serialises uploads (concurrent attempts get HTTP 429), each upload has a
per-file size cap (HTTP 413 on overrun), and full exceptions are logged
server-side while clients only see ``error_id=<short>`` for correlation.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import secrets
import shutil
import threading
import time
import uuid
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

logger = logging.getLogger("pdm.api")

REQUIRED_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors":    "PdM_errors.csv",
    "failures":  "PdM_failures.csv",
    "machines":  "PdM_machines.csv",
    "maint":     "PdM_maint.csv",
}

# Per-file upload size cap (HTTP 413 on overrun). 256 MiB is comfortably larger
# than the Azure PdM telemetry CSV (~74 MiB) yet still bounds a malicious client.
MAX_UPLOAD_BYTES_PER_FILE = int(os.environ.get(
    "PDM_MAX_UPLOAD_BYTES_PER_FILE", str(256 * 1024 * 1024)
))

# Atomic-swap artifact files that the analysis endpoints depend on. Plots are
# swapped as a directory in `_commit_staged`. `mlruns_dir` is deliberately NOT
# rebased so MLflow accumulates runs across uploads.
_ARTIFACT_FILE_KEYS = (
    "model_file",
    "metrics_file",
    "threshold_file",
    "drift_report",
    "feature_hash_file",
)

# Only one upload runs at a time. Concurrent callers get HTTP 429.
_pipeline_lock = threading.Lock()


def _short_id() -> str:
    return secrets.token_hex(4)


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
    error_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers used by /upload-and-run
# ---------------------------------------------------------------------------

def _staged_cfg(cfg: dict[str, Any], staging_root: Path) -> dict[str, Any]:
    """Deep-copy cfg with raw_dir + every artifact path rebased into staging_root.

    `mlruns_dir` is intentionally left alone so MLflow keeps a single history.
    """
    scfg = copy.deepcopy(cfg)
    scfg["paths"] = dict(scfg["paths"])
    art = staging_root / "artifacts"
    scfg["paths"]["raw_dir"]            = str(staging_root)
    scfg["paths"]["artifacts_dir"]      = str(art)
    scfg["paths"]["model_file"]         = str(art / "model.joblib")
    scfg["paths"]["metrics_file"]       = str(art / "metrics.json")
    scfg["paths"]["threshold_file"]     = str(art / "threshold.json")
    scfg["paths"]["drift_report"]       = str(art / "drift_report.md")
    scfg["paths"]["plots_dir"]          = str(art / "plots")
    scfg["paths"]["feature_hash_file"]  = str(art / "feature_hash.txt")
    return scfg


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write `obj` to `path` atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _commit_staged(staged_cfg: dict[str, Any], canonical_cfg: dict[str, Any]) -> None:
    """Atomically swap staged raw CSVs + artifacts into their canonical paths.

    All paths are on the same filesystem, so `os.replace` is atomic per file.
    """
    canon_raw = resolve_path(canonical_cfg, "raw_dir")
    canon_art = resolve_path(canonical_cfg, "artifacts_dir")
    canon_plots = resolve_path(canonical_cfg, "plots_dir")
    canon_raw.mkdir(parents=True, exist_ok=True)
    canon_art.mkdir(parents=True, exist_ok=True)
    canon_plots.mkdir(parents=True, exist_ok=True)

    staged_raw = resolve_path(staged_cfg, "raw_dir")
    for fname in REQUIRED_FILES.values():
        os.replace(staged_raw / fname, canon_raw / fname)

    for key in _ARTIFACT_FILE_KEYS:
        src = resolve_path(staged_cfg, key)
        dst = resolve_path(canonical_cfg, key)
        if src.exists():
            os.replace(src, dst)

    staged_plots = resolve_path(staged_cfg, "plots_dir")
    if staged_plots.is_dir():
        for p in staged_plots.iterdir():
            os.replace(p, canon_plots / p.name)


async def _stage_one_upload(slot: str, fobj: UploadFile, target: Path,
                             max_bytes: int, error_id: str) -> int:
    """Stream `fobj` into `target` capped at `max_bytes`. Raise HTTPException(413)
    on overrun (after deleting the partial write).
    """
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        while True:
            chunk = await fobj.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out.close()
                target.unlink(missing_ok=True)
                logger.warning(
                    "[%s] upload slot=%s exceeded size cap (%d > %d bytes)",
                    error_id, slot, written, max_bytes,
                )
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"file '{slot}' exceeds the {max_bytes // (1 << 20)} MiB "
                        f"per-file upload limit"
                    ),
                )
            out.write(chunk)
    return written


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

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
            logger.exception("failed to parse session sentinel %s", sf)
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
        if not _pipeline_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="another upload is already running; please retry shortly.",
            )

        error_id = _short_id()
        raw_dir = resolve_path(cfg, "raw_dir")
        staging_root = raw_dir / ".incoming" / uuid.uuid4().hex
        stages: list[StageResult] = []

        def _fail(name: str, secs: float,
                  client_msg: str) -> PipelineResult:
            """Build a sanitised PipelineResult and clean up staging."""
            shutil.rmtree(staging_root, ignore_errors=True)
            stages.append(StageResult(
                name=name, ok=False, seconds=secs,
                info=f"failed; error_id={error_id}",
            ))
            return PipelineResult(
                ok=False,
                stages=stages,
                error=client_msg,
                error_id=error_id,
            )

        try:
            staging_root.mkdir(parents=True, exist_ok=False)
            (staging_root / "artifacts").mkdir(parents=True, exist_ok=True)

            uploads = {
                "telemetry": telemetry,
                "errors":    errors,
                "failures":  failures,
                "machines":  machines,
                "maint":     maint,
            }
            original_names: dict[str, str] = {}

            # stage 1: stream + size-cap into staging_root
            t0 = time.perf_counter()
            try:
                for slot, fobj in uploads.items():
                    if not fobj.filename:
                        raise HTTPException(400, f"missing file for slot '{slot}'")
                    target = staging_root / REQUIRED_FILES[slot]
                    await _stage_one_upload(
                        slot, fobj, target, MAX_UPLOAD_BYTES_PER_FILE, error_id
                    )
                    original_names[slot] = fobj.filename
            except HTTPException:
                shutil.rmtree(staging_root, ignore_errors=True)
                raise
            except Exception:
                logger.exception("[%s] failed to stage uploads", error_id)
                return _fail("save", time.perf_counter() - t0,
                             f"failed to save upload; error_id={error_id}")
            stages.append(StageResult(name="save", ok=True,
                                      seconds=time.perf_counter() - t0,
                                      info="5 files staged"))

            scfg = _staged_cfg(cfg, staging_root)

            # stage 2: load + validate against staged dataset
            # (CPU/IO-bound -> threadpool so the event loop stays responsive and
            #  a concurrent /upload-and-run hits the 429 guard cleanly)
            t0 = time.perf_counter()
            try:
                frames = await asyncio.to_thread(load_raw, scfg)
                report = await asyncio.to_thread(validate, frames)
            except Exception:
                logger.exception("[%s] load/validate raised", error_id)
                return _fail("validate", time.perf_counter() - t0,
                             f"failed to load CSVs; error_id={error_id}")
            if not report.ok:
                logger.error("[%s] validation issues: %s", error_id, report.issues)
                return _fail("validate", time.perf_counter() - t0,
                             f"validation failed; error_id={error_id}")
            stages.append(StageResult(name="validate", ok=True,
                                      seconds=time.perf_counter() - t0,
                                      info=f"{report.info}"))

            # stage 3: train (writes into staged artifacts only). Off-loaded
            # to a thread so the event loop keeps serving concurrent requests
            # (which will see the lock held and immediately get 429).
            t0 = time.perf_counter()
            try:
                metrics = await asyncio.to_thread(run_training, scfg)
            except Exception:
                logger.exception("[%s] training raised", error_id)
                return _fail("train", time.perf_counter() - t0,
                             f"training failed; error_id={error_id}")
            stages.append(StageResult(
                name="train", ok=True,
                seconds=time.perf_counter() - t0,
                info=f"best={metrics['best_run']} "
                     f"PR-AUC={metrics['runs'][metrics['best_run']]['pr_auc']:.3f}",
            ))

            # stage 4: drift report (non-fatal -- never blocks a commit)
            t0 = time.perf_counter()
            try:
                await asyncio.to_thread(build_drift_report, scfg)
                stages.append(StageResult(name="drift", ok=True,
                                          seconds=time.perf_counter() - t0,
                                          info="drift_report.md written"))
            except Exception:
                logger.exception("[%s] drift report raised", error_id)
                stages.append(StageResult(name="drift", ok=False,
                                          seconds=time.perf_counter() - t0,
                                          info=f"failed; error_id={error_id}"))

            # stage 5: atomic commit -> live raw_dir + artifacts
            t0 = time.perf_counter()
            try:
                _commit_staged(scfg, cfg)
            except Exception:
                logger.exception("[%s] commit raised", error_id)
                return _fail("commit", time.perf_counter() - t0,
                             f"commit failed; error_id={error_id}")

            _reset_caches()
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
            try:
                _atomic_write_json(_session_file(), sentinel)
            except Exception:
                logger.exception("[%s] sentinel write raised", error_id)
                return _fail("commit", time.perf_counter() - t0,
                             f"failed to write session; error_id={error_id}")
            stages.append(StageResult(name="commit", ok=True,
                                      seconds=time.perf_counter() - t0,
                                      info="atomic swap + sentinel"))

            shutil.rmtree(staging_root, ignore_errors=True)
            return PipelineResult(ok=True, stages=stages,
                                  metrics=sentinel["pipeline"])
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
            _pipeline_lock.release()

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
