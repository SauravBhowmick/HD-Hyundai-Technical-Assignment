#!/usr/bin/env bash
# start.sh -- bring up the full PdM Digital Twin stack via Docker Compose.
#
# A thin convenience wrapper around `docker compose up`. It:
#   - builds the pdm-digital-twin image (first run only, unless --rebuild)
#     and pulls ghcr.io/mlflow/mlflow:latest on first run
#   - starts both services (pdm + mlflow) side-by-side via docker-compose.yml
#   - waits for /healthz on the API and the MLflow UI's root to respond
#   - prints a friendly banner with the URLs
#   - on Ctrl+C, tears the stack down cleanly via `docker compose down`
#
# Maps onto the make targets like this:
#   ./start.sh             ==  make docker-run    + health checks + banner + cleanup
#   ./start.sh --rebuild   ==  make compose-up    + health checks + banner + cleanup
#
# Usage
#   ./start.sh                  bring up the stack (default; reuses cached image)
#   ./start.sh --rebuild        force `docker compose up --build` (slow, ~3 min)
#   ./start.sh -h / --help      show this help

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

IMAGE="pdm-digital-twin"
API_URL="http://localhost:8000/healthz"
MLFLOW_URL="http://localhost:5050/"
REBUILD=0

log()  { printf "\033[1;34m[start]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[start]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[start]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'USAGE'
start.sh -- bring up the PdM Digital Twin stack via Docker Compose.

This script is the one-command entrypoint to the full demo: the FastAPI
backend, the bundled React/Vite frontend (served by the same FastAPI
process), and the MLflow tracking UI. All three run as containers defined
in docker-compose.yml; nothing runs on the host.

Usage:
  ./start.sh             bring up the stack (reuses cached image)
                         -- same image-handling as `make docker-run`
  ./start.sh --rebuild   force `docker compose up --build` (slow)
                         -- same image-handling as `make compose-up`

URLs:
  Dashboard + API    -> http://localhost:8000
  MLflow tracking UI -> http://localhost:5050

Press Ctrl+C while this script is in the foreground to stop and remove
the containers cleanly. Leave it running to keep the stack up.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)  REBUILD=1 ;;
    -h|--help)  usage; exit 0 ;;
    *)          err "unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done

# -------- prerequisites --------

if ! command -v docker >/dev/null 2>&1; then
  err "docker not found on PATH. Install Docker Desktop and retry."
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  err "Docker daemon not reachable. Start Docker Desktop and retry."
  exit 1
fi

# Create the host mlruns/ before compose bind-mounts it so the directory
# is owned by the host user (Linux only -- macOS handles this fine either way).
mkdir -p mlruns

# -------- bring up the stack --------

# Build only when explicitly asked, or when the pdm image isn't cached yet.
# `--build` triggers a full pipeline run (validate + train + drift + pytest)
# inside the image (~3 min on a cold cache), so we avoid it when the image
# is already present.
BUILD_FLAG=()
if [[ $REBUILD -eq 1 ]]; then
  log "rebuilding the pdm image (--rebuild)..."
  BUILD_FLAG=(--build)
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "no cached $IMAGE image found; building (~3 min on first run)..."
  BUILD_FLAG=(--build)
else
  log "reusing cached $IMAGE image (use --rebuild to force a fresh build)."
fi

cleanup() {
  local exit_code=$?
  echo
  log "stopping and removing containers..."
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  log "bye"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

log "starting compose stack (pdm + mlflow)..."
if ! docker compose up -d --remove-orphans "${BUILD_FLAG[@]}"; then
  err "docker compose up failed. Common cause: another process or orphan"
  err "container is holding host port :8000 or :5050. Diagnose with:"
  err "  docker ps -a"
  err "  lsof -nP -iTCP:8000 -sTCP:LISTEN"
  err "  lsof -nP -iTCP:5050 -sTCP:LISTEN"
  exit 1
fi

# -------- health checks --------

wait_for_http() {
  local label=$1 url=$2 timeout=${3:-90}
  for ((i=0; i<timeout; i++)); do
    if curl -sf -o /dev/null "$url"; then
      log "$label ready"
      return 0
    fi
    sleep 1
  done
  warn "$label did not respond within ${timeout}s -- check 'docker compose logs $3'"
  return 1
}

wait_for_http "API    :8000" "$API_URL"    90 || true
wait_for_http "MLflow :5050" "$MLFLOW_URL" 60 || true

# -------- banner --------

cat <<EOF

==========================================================
  PdM Digital Twin is up

  Dashboard + API   :  http://localhost:8000
  MLflow tracking UI:  http://localhost:5050

  Tail logs (anothertab):  docker compose logs -f
  Stop the stack         :  Ctrl+C here (this script tears it down)
==========================================================

EOF

# -------- foreground log stream --------
# Streaming logs in the foreground keeps this script alive so Ctrl+C can
# fire the cleanup trap above. If you'd rather leave the stack running and
# exit this terminal, use `make docker-run` directly instead.
docker compose logs -f
