# syntax=docker/dockerfile:1.6
#
# Reproducible 3-stage build for the PdM Digital Twin
# ===================================================
#
# Reviewers should be able to run `make docker && make docker-run` on any host
# and get bit-for-bit identical layers (modulo timestamps). To make that true:
#
# 1. Base images are pinned by **multi-arch manifest-list digest** (not floating
#    tags like `node:20-alpine`). Tags are mutable; digests are content
#    addresses. The form `image:tag@sha256:...` resolves to the exact same
#    bytes forever, and BuildKit will still pick the right platform variant.
#
#    To refresh manually after a CVE / base-image upgrade:
#        docker buildx imagetools inspect node:20-alpine     --format '{{.Manifest.Digest}}'
#        docker buildx imagetools inspect python:3.11-slim   --format '{{.Manifest.Digest}}'
#    Or wire up Renovate / Dependabot to bump them via PR.
#
# 2. The image is split into 3 stages so test-only tooling (pytest / httpx /
#    ruff) is built once but **never ships** in the runtime layer:
#
#       web-build       Node 20 alpine    -> /web/dist (React + Vite bundle)
#       train-and-test  Python slim       -> runs the ML pipeline + pytest,
#                                            produces artifacts/, mlruns/, data/
#       runtime         Python slim       -> only runtime deps installed
#                                            (pip install . without .[dev]),
#                                            consumes COPY --from=... outputs.
#
#    Net effect: the runtime image has no pytest / httpx / ruff and is about
#    a couple hundred MB smaller than a single-stage build that left dev deps
#    behind.


# =============================================================================
# Stage 1 -- build the React + Vite frontend.
# =============================================================================
FROM node:20-alpine@sha256:fb4cd12c85ee03686f6af5362a0b0d56d50c58a04632e6c0fb8363f609372293 AS web-build
WORKDIR /web

# Lockfile-first so npm ci's cache is hit on dependency-only changes.
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./

# Empty API base = same-origin fetches. In dev (Vite) the default is "/api"
# (handled by the dev-server proxy); in the bundled image the SPA is served
# by FastAPI itself, so /healthz, /predict, ... live at the document root.
ENV VITE_API_BASE=""
RUN npm run build


# =============================================================================
# Stage 2 -- run the ML pipeline + tests with the *dev* extras installed.
# Outputs we care about (and only those) are pulled into stage 3 below.
# =============================================================================
FROM python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834 AS train-and-test

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY tests ./tests
COPY Makefile ./

# Dev extras = pytest + httpx + ruff. Lives in this stage only; stage 3
# starts fresh with the plain `pip install .` (no [dev]).
RUN pip install --upgrade pip && pip install ".[dev]"

COPY data ./data

# Design choice -- pipeline runs at build time, not container startup.
#
# Trade-off considered (and chosen deliberately):
#
#   * Pro: `make docker-run` is instant -- the image ships a trained model,
#     drift report, plots, and MLflow runs. The build doubles as a CI gate
#     (pytest must pass for the image to publish at all).
#   * Con: build takes ~3 min cold, ~20 s warm.
#
# The "lazy train on first request" alternative was rejected because:
#   (a) it would move the 3-minute cost onto every reviewer's first browser
#       hit, not just the image build;
#   (b) we already expose a user-facing retrain path (POST /upload-and-run)
#       that stages uploads + retrains atomically, so volume-mounted models
#       would duplicate that concern;
#   (c) training is deterministic here -- LightGBM is seeded via
#       configs/default.yaml (random_state: 42) and the global random_seed
#       is 42, so a build against the same source + pinned base-image
#       digests produces the same artifacts on the same platform.
RUN python -m pdm.cli validate \
 && python -m pdm.cli train \
 && python -m pdm.cli drift \
 && pytest


# =============================================================================
# Stage 3 -- runtime image. No test tooling, no [dev] extras.
# =============================================================================
FROM python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Source needed at runtime to import `pdm.*` and `api.server`.
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY configs ./configs

# Runtime deps only. `pyproject.toml`'s [project.optional-dependencies].dev
# (pytest / httpx / ruff) is NOT pulled in -- that's the whole point of the
# stage split.
RUN pip install --upgrade pip && pip install .

# Pre-baked artifacts and source CSVs from the train-and-test stage. Reviewers
# can re-upload via /upload-and-run, but the image works out of the box.
COPY --from=train-and-test /app/artifacts ./artifacts
COPY --from=train-and-test /app/data      ./data
COPY --from=train-and-test /app/mlruns    ./mlruns

# Pre-built SPA. `api/server.py` mounts /app/web/dist at "/" when present.
COPY --from=web-build /web/dist ./web/dist

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
