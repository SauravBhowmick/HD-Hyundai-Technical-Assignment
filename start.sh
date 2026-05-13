#!/usr/bin/env bash
# start.sh -- bring up the whole PdM Digital Twin stack at once.
#
#   Backend  : FastAPI -- locally with uvicorn (default), or in Docker, or both.
#   Frontend : Vite dev server (React + TypeScript)
#   MLflow   : MLflow UI against ./mlruns
#
# Ctrl+C stops everything cleanly.
#
# Modes
#   ./start.sh                     local uvicorn + web + mlflow (default)
#   ./start.sh --with-docker       local uvicorn AND a side-by-side Docker
#                                  container (on $DOCKER_API_PORT, default 8002).
#   ./start.sh --docker            Docker only, no local uvicorn.
#
# Toggles
#   --no-mlflow    skip the MLflow UI
#   --no-web       skip the Vite dev server (API only)
#   --rebuild      force a docker image rebuild before starting
#
# Env vars (defaults shown)
#   API_PORT=8000         port for the local uvicorn (and for the web's API base)
#   DOCKER_API_PORT=8002  port the Docker container is published on
#   WEB_PORT=5173         Vite dev server port
#   MLFLOW_PORT=5000      MLflow UI port

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

# -------- configuration --------
MODE="local"              # local | docker
WITH_DOCKER=0             # in local mode, also run docker side-by-side
API_PORT="${API_PORT:-8000}"
DOCKER_API_PORT="${DOCKER_API_PORT:-8002}"
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

Default behaviour: run uvicorn locally, the Vite dev server, and the MLflow UI.
Optionally, bring up the Docker container too (side-by-side or instead of local).

Modes:
  ./start.sh                  local uvicorn + web + mlflow (default)
  ./start.sh --with-docker    local AND side-by-side Docker on DOCKER_API_PORT
  ./start.sh --docker         Docker only, no local uvicorn

Toggles:
  --no-mlflow      skip the MLflow UI
  --no-web         skip the Vite dev server
  --rebuild        force a docker image rebuild before starting

Env vars (defaults shown):
  API_PORT=8000          local uvicorn port; the web uses this for /api
  DOCKER_API_PORT=8002   port for the Docker container
  WEB_PORT=5173          Vite dev server port
  MLFLOW_PORT=5000       MLflow UI port

Press Ctrl+C to stop everything cleanly.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)        MODE=local ;;
    --docker)       MODE=docker ;;
    --with-docker)  MODE=local; WITH_DOCKER=1 ;;
    --no-mlflow)    WITH_MLFLOW=0 ;;
    --no-web)       WITH_WEB=0 ;;
    --rebuild)      REBUILD=1 ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "[start] unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done

# In docker-only mode the web should hit the docker container.
if [[ "$MODE" == docker ]]; then
  WEB_API_PORT="$DOCKER_API_PORT"
else
  WEB_API_PORT="$API_PORT"
fi

mkdir -p "$LOG_DIR"

VENV_PY=".venv/bin/python"

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
DOCKER_STARTED=0

cleanup() {
  local exit_code=$?
  echo
  log "shutting down..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  if [[ $DOCKER_STARTED -eq 1 ]]; then
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
    docker rm   "$CONTAINER" >/dev/null 2>&1 || true
  fi
  log "bye"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

require_local_env() {
  if [[ ! -x "$VENV_PY" ]]; then
    log "creating .venv and installing python deps (one-off)..."
    make install
  fi
  if ! "$VENV_PY" -c "import pdm.cli" >/dev/null 2>&1; then
    err ".venv is present but the pdm package isn't importable; rebuild with:"
    err "  rm -rf .venv && make install"
    exit 1
  fi
  if [[ ! -f artifacts/model.joblib ]]; then
    log "no trained model in artifacts/; running 'make train' (~30s)..."
    make train
  fi
}

require_docker_env() {
  if ! command -v docker >/dev/null 2>&1; then
    err "docker not found on PATH. Install Docker Desktop or skip with --local."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon not reachable. Start Docker Desktop or use --local."
    exit 1
  fi
  if [[ "$REBUILD" -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "building Docker image '$IMAGE' (trains model inside the image, ~2 min)..."
    docker build -t "$IMAGE" . | tail -5
  fi
}

# -------- preflight --------
NEED_LOCAL=0
NEED_DOCKER=0
[[ "$MODE" == local  ]] && NEED_LOCAL=1
[[ "$MODE" == docker ]] && NEED_DOCKER=1
[[ $WITH_DOCKER -eq 1 ]] && NEED_DOCKER=1

[[ $NEED_LOCAL  -eq 1 ]] && require_local_env
[[ $NEED_DOCKER -eq 1 ]] && require_docker_env

if [[ $WITH_WEB -eq 1 && ! -d web/node_modules ]]; then
  log "installing web deps (one-off)..."
  (cd web && npm install) >/dev/null
fi

if [[ $WITH_MLFLOW -eq 1 ]]; then
  if [[ ! -x "$VENV_PY" ]]; then
    warn "$VENV_PY not found. Run 'make install' once to enable the MLflow UI."
    WITH_MLFLOW=0
  elif ! "$VENV_PY" -c "import mlflow" >/dev/null 2>&1; then
    warn "mlflow not importable from $VENV_PY (broken venv?)."
    warn "  Run 'rm -rf .venv && make install' to rebuild the venv."
    WITH_MLFLOW=0
  fi
fi

# Web origins to whitelist in the API's CORS (so a custom WEB_PORT works).
EXTRA_CORS="http://localhost:$WEB_PORT,http://127.0.0.1:$WEB_PORT"

# -------- local API (uvicorn) --------
if [[ $NEED_LOCAL -eq 1 ]]; then
  if port_in_use "$API_PORT"; then
    if curl -sf "http://localhost:$API_PORT/healthz" >/dev/null 2>&1; then
      log "port $API_PORT already serving a healthy PdM API; reusing it."
    else
      err "port $API_PORT is in use by something else (not the PdM API)."
      err "what's listening:"
      lsof -nP -iTCP:"$API_PORT" -sTCP:LISTEN 2>&1 | sed -n '1,4p' >&2 || true
      err ""
      err "fix it by either:"
      err "  - killing that process    (e.g.  kill <pid>)"
      err "  - using a different port  (e.g.  API_PORT=8001 ./start.sh)"
      exit 1
    fi
  else
    log "starting uvicorn (local) on :$API_PORT..."
    PDM_EXTRA_CORS_ORIGINS="$EXTRA_CORS" \
      "$VENV_PY" -m uvicorn api.server:app \
      --host 0.0.0.0 --port "$API_PORT" \
      > "$LOG_DIR/api.log" 2>&1 &
    PIDS+=($!)
  fi

  log "waiting for local API /healthz on :$API_PORT ..."
  if ! wait_for_http "http://localhost:$API_PORT/healthz" 90; then
    err "local API did not become healthy in 90s. Tail of $LOG_DIR/api.log:"
    tail -30 "$LOG_DIR/api.log" >&2 || true
    exit 1
  fi
  log "local API healthy ✓"
fi

# -------- Docker API --------
if [[ $NEED_DOCKER -eq 1 ]]; then
  if port_in_use "$DOCKER_API_PORT"; then
    if curl -sf "http://localhost:$DOCKER_API_PORT/healthz" >/dev/null 2>&1; then
      log "port $DOCKER_API_PORT already serving a healthy PdM API; reusing it."
    else
      err "port $DOCKER_API_PORT is in use by something else."
      lsof -nP -iTCP:"$DOCKER_API_PORT" -sTCP:LISTEN 2>&1 | sed -n '1,4p' >&2 || true
      err "use a different DOCKER_API_PORT=... or free the port."
      exit 1
    fi
  else
    log "starting Docker API on :$DOCKER_API_PORT (container=$CONTAINER, image=$IMAGE)..."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER" \
      -p "$DOCKER_API_PORT:8000" \
      -e "PDM_EXTRA_CORS_ORIGINS=$EXTRA_CORS" \
      "$IMAGE" \
      > "$LOG_DIR/api.cid" 2>&1
    DOCKER_STARTED=1
    docker logs -f "$CONTAINER" > "$LOG_DIR/api.docker.log" 2>&1 &
    PIDS+=($!)
  fi

  log "waiting for Docker API /healthz on :$DOCKER_API_PORT ..."
  if ! wait_for_http "http://localhost:$DOCKER_API_PORT/healthz" 90; then
    err "Docker API did not become healthy in 90s. Tail of $LOG_DIR/api.docker.log:"
    tail -30 "$LOG_DIR/api.docker.log" >&2 || true
    exit 1
  fi
  log "Docker API healthy ✓"
fi

# -------- MLflow UI --------
if [[ $WITH_MLFLOW -eq 1 ]]; then
  if port_in_use "$MLFLOW_PORT"; then
    warn "port $MLFLOW_PORT busy (macOS AirPlay uses 5000 by default). Skipping MLflow."
    warn "  Disable 'AirPlay Receiver' in System Settings, or pass MLFLOW_PORT=5050 ./start.sh"
    WITH_MLFLOW=0
  else
    log "starting MLflow UI on :$MLFLOW_PORT (file store at ./mlruns)..."
    "$VENV_PY" -m mlflow ui \
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
    log "starting Vite dev server on :$WEB_PORT (api base = http://localhost:$WEB_API_PORT)..."
    (
      cd web
      VITE_API_BASE="http://localhost:$WEB_API_PORT" \
        npm run dev -- --port "$WEB_PORT" --strictPort
    ) > "$LOG_DIR/web.log" 2>&1 &
    PIDS+=($!)
    wait_for_http "http://localhost:$WEB_PORT" 30 || true
  fi
fi

cat <<EOF

==========================================================
  PdM Digital Twin is up

  Web        :  http://localhost:$WEB_PORT
  API (local):  $( [[ $NEED_LOCAL  -eq 1 ]] && echo "http://localhost:$API_PORT"        || echo "-" )
  API (dock) :  $( [[ $NEED_DOCKER -eq 1 ]] && echo "http://localhost:$DOCKER_API_PORT" || echo "-" )
  MLflow     :  $( [[ $WITH_MLFLOW -eq 1 ]] && echo "http://localhost:$MLFLOW_PORT"     || echo "(disabled)" )

  Web -> API :  http://localhost:$WEB_API_PORT
  Logs       :  $LOG_DIR/{api,api.docker,web,mlflow}.log
  Press Ctrl+C to stop everything.
==========================================================

EOF

if [[ ${#PIDS[@]} -gt 0 ]]; then
  wait "${PIDS[@]}" 2>/dev/null || true
fi
while true; do sleep 60; done
