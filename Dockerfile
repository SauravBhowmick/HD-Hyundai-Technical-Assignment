# syntax=docker/dockerfile:1.6

# =============================================================================
# Stage 1 -- build the React + Vite frontend.
#
# Output: /web/dist (static HTML/JS/CSS bundle).
# Why a separate stage: lets us produce a tiny set of static files using Node,
# then drop Node entirely from the runtime image. The runtime image only needs
# Python + libgomp.
# =============================================================================
FROM node:20-alpine AS web-build
WORKDIR /web

# Lockfile-first so `npm ci` cache is hit on dependency-only changes.
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./

# Empty API base = same-origin fetches. In dev (Vite) the default is "/api"
# (handled by the dev-server proxy); in the bundled image the SPA is served
# by FastAPI itself, so /healthz, /predict, ... live at the document root.
ENV VITE_API_BASE=""
RUN npm run build


# =============================================================================
# Stage 2 -- Python backend image. Also ships the static SPA built above so
# `docker run -p 8000:8000 pdm-digital-twin` brings up the full stack.
# =============================================================================
FROM python:3.11-slim AS base

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

RUN pip install --upgrade pip && pip install ".[dev]"

COPY data ./data

# Train + drift + tests at build time so the image ships with a known-good
# model and known-passing tests. Reviewers get a working API on first run.
RUN python -m pdm.cli validate \
 && python -m pdm.cli train \
 && python -m pdm.cli drift \
 && pytest

# Pull the pre-built SPA from the Node stage. `api/server.py` mounts this
# directory at `/` via StaticFiles when it exists.
COPY --from=web-build /web/dist ./web/dist

ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
