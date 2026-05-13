# syntax=docker/dockerfile:1.6
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

# Run the pipeline once at build time so the image ships with a trained model.
RUN python -m pdm.cli validate \
 && python -m pdm.cli train \
 && python -m pdm.cli drift \
 && pytest

ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
