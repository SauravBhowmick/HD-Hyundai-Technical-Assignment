#!/usr/bin/env bash
# start.sh -- bring up the whole PdM Digital Twin stack at once.
#
#   Backend  : FastAPI (in Docker by default, or locally with --local)
#   Frontend : Vite dev server (React + TypeScript)
#   MLflow   : MLflow UI against ./mlruns
#
# Ctrl+C stops everything cleanly.
#
# Examples:
#   ./start.sh                  # docker API + web + mlflow
#   ./start.sh --local          # local uvicorn instead of docker
#   ./start.sh --no-mlflow      # skip the MLflow UI
#   ./start.sh --no-web         # API + mlflow only
#   ./start.sh --rebuild        # force a docker image rebuild

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

# -------- configuration --------
MODE="docker"            # docker | local
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
IMAGE="pdm-digital-twin"
CONTAINER="pdm-twin-api"
LOG_DIR="$SCRIPT_DIR/logs"
WITH_MLFLOW=1
WITH_WEB=1
REBUILD=0

usage() {
  cat <<'USAGE'
start.sh -- bring up the whole PdM Digital Twin stack at once.

  Backend  : FastAPI (in Docker by default, or locally with --local)
  Frontend : Vite dev server (React + TypeScript)
  MLflow   : MLflow UI against ./mlruns

Ctrl+C stops everything cleanly.

Examples:
  ./start.sh                  docker API + web + mlflow (default)
  ./start.sh --local          local uvicorn instead of docker
  ./start.sh --no-mlflow      skip the MLflow UI
  ./start.sh --no-web         API + mlflow only
  ./start.sh --rebuild        force a docker image rebuild

Env vars (defaults shown):
  API_PORT=8000  WEB_PORT=5173  MLFLOW_PORT=5000
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)      MODE=local ;;
    --docker)     MODE=docker ;;
    --no-mlflow)  WITH_MLFLOW=0 ;;
    --no-web)     WITH_WEB=0 ;;
    --rebuild)    REBUILD=1 ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "[start] unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"

log()  { printf "\033[1;34m[start]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[start]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[start]\033[0m %s\n" "$*" >&2; }

port_in_use() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for_http() {
  local url=$1 timeout=${2:-60}
  for ((i=0; i<timeout; i++)); do
    if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

PIDS=()

cleanup() {
  local exit_code=$?
  echo
  log "shutting down..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$MODE" == docker ]]; then
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
    docker rm   "$CONTAINER" >/dev/null 2>&1 || true
  fi
  log "bye"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

# -------- preflight --------
if [[ "$MODE" == local && ! -x .venv/bin/uvicorn ]]; then
  log "creating .venv and installing python deps (one-off)..."
  make install
fi

if [[ "$MODE" == docker ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    err "docker not found on PATH. Install Docker Desktop or run with --local."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon not reachable. Start Docker Desktop or run with --local."
    exit 1
  fi
  if [[ "$REBUILD" -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "building Docker image '$IMAGE' (this trains the model inside the image, ~2 min)..."
    docker build -t "$IMAGE" . | tail -5
  fi
fi

if [[ $WITH_WEB -eq 1 && ! -d web/node_modules ]]; then
  log "installing web deps (one-off)..."
  (cd web && npm install) >/dev/null
fi

if [[ $WITH_MLFLOW -eq 1 && ! -x .venv/bin/mlflow ]]; then
  warn "MLflow CLI not in .venv. Run 'make install' once to enable the MLflow UI."
  WITH_MLFLOW=0
fi

# Pass any non-default web origin to the API so CORS works without a rebuild.
EXTRA_CORS="http://localhost:$WEB_PORT,http://127.0.0.1:$WEB_PORT"

# -------- API --------
if port_in_use "$API_PORT"; then
  warn "port $API_PORT already in use; skipping API startup (using whatever is there)."
else
  if [[ "$MODE" == docker ]]; then
    log "starting Docker API on :$API_PORT (container=$CONTAINER, image=$IMAGE)..."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER" \
      -p "$API_PORT:8000" \
      -e "PDM_EXTRA_CORS_ORIGINS=$EXTRA_CORS" \
      "$IMAGE" \
      > "$LOG_DIR/api.cid" 2>&1
    # stream container logs into logs/api.log
    docker logs -f "$CONTAINER" > "$LOG_DIR/api.log" 2>&1 &
    PIDS+=($!)
  else
    log "starting uvicorn (local) on :$API_PORT..."
    PDM_EXTRA_CORS_ORIGINS="$EXTRA_CORS" \
      .venv/bin/uvicorn api.server:app --host 0.0.0.0 --port "$API_PORT" \
      > "$LOG_DIR/api.log" 2>&1 &
    PIDS+=($!)
  fi
fi

log "waiting for API /healthz on :$API_PORT ..."
if ! wait_for_http "http://localhost:$API_PORT/healthz" 90; then
  err "API did not become healthy in 90s. Tail of $LOG_DIR/api.log:"
  tail -30 "$LOG_DIR/api.log" >&2 || true
  exit 1
fi
log "API healthy ✓"

# -------- MLflow UI --------
if [[ $WITH_MLFLOW -eq 1 ]]; then
  if port_in_use "$MLFLOW_PORT"; then
    warn "port $MLFLOW_PORT busy (macOS AirPlay uses 5000 by default). Skipping MLflow."
    warn "  To enable: System Settings -> General -> AirDrop & Handoff -> turn off 'AirPlay Receiver',"
    warn "  or pass MLFLOW_PORT=5050 ./start.sh"
    WITH_MLFLOW=0
  else
    log "starting MLflow UI on :$MLFLOW_PORT (file store at ./mlruns)..."
    .venv/bin/mlflow ui \
      --backend-store-uri "file://$PWD/mlruns" \
      --host 0.0.0.0 --port "$MLFLOW_PORT" \
      > "$LOG_DIR/mlflow.log" 2>&1 &
    PIDS+=($!)
  fi
fi

# -------- Web --------
if [[ $WITH_WEB -eq 1 ]]; then
  if port_in_use "$WEB_PORT"; then
    warn "port $WEB_PORT busy; skipping Vite (open it manually)."
  else
    log "starting Vite dev server on :$WEB_PORT (api base = http://localhost:$API_PORT)..."
    (
      cd web
      VITE_API_BASE="http://localhost:$API_PORT" \
        npm run dev -- --port "$WEB_PORT" --strictPort
    ) > "$LOG_DIR/web.log" 2>&1 &
    PIDS+=($!)
    # Wait briefly for Vite to bind so the URL is reachable from the summary.
    wait_for_http "http://localhost:$WEB_PORT" 30 || true
  fi
fi

cat <<EOF

==========================================================
  PdM Digital Twin is up ($MODE mode)

  API     :  http://localhost:$API_PORT     (POST /predict, GET /info, ...)
  Web     :  http://localhost:$WEB_PORT
  MLflow  :  http://localhost:$MLFLOW_PORT  $( [[ $WITH_MLFLOW -eq 0 ]] && echo "(disabled)" )

  Logs    :  $LOG_DIR/{api,web,mlflow}.log
  Press Ctrl+C to stop everything.
==========================================================

EOF

# Wait on whatever is running. In pure-docker mode (no web, no mlflow) we still
# block here so Ctrl+C reaches the trap.
if [[ ${#PIDS[@]} -gt 0 ]]; then
  wait "${PIDS[@]}" 2>/dev/null || true
fi
while true; do sleep 60; done
