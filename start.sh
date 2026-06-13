#!/usr/bin/env bash
# start.sh -- interactive launcher for the PdM Digital Twin stack.
#
# Prompts for Docker vs local development, then:
#   Docker  -> compose up (pdm :8000 + mlflow :5050), with image checks and
#              self-healing if containers are stale or unhealthy
#   Local   -> uvicorn :8000 + Vite :5173 (+ optional MLflow :5050 on host)
#
# Non-interactive flags:
#   ./start.sh --docker          skip prompt, use Docker
#   ./start.sh --local           skip prompt, use local dev servers
#   ./start.sh --rebuild         Docker only: force image rebuild
#   ./start.sh -h / --help       show help

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

IMAGE="pdm-digital-twin"
MLFLOW_IMAGE="ghcr.io/mlflow/mlflow:v3.12.0"
API_URL="http://localhost:8000/healthz"
MLFLOW_URL="http://localhost:5050/"
WEB_URL="http://localhost:5173/"
VENV="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"

MODE=""       # docker | local
REBUILD=0
API_PID=""
WEB_PID=""
MLFLOW_PID=""

log()  { printf "\033[1;34m[start]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[start]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[start]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'USAGE'
start.sh -- interactive launcher for the PdM Digital Twin.

At startup you choose how to run the stack (or pass --docker / --local).

Docker mode (recommended for reviewers):
  - API + bundled dashboard on http://localhost:8000
  - MLflow tracking UI on http://localhost:5050
  - Reuses cached images when healthy; rebuilds or recycles on failure

Local dev mode:
  - FastAPI on http://localhost:8000 (reload)
  - Vite dashboard on http://localhost:5173
  - Optional MLflow UI on http://localhost:5050

Usage:
  ./start.sh                 interactive prompt
  ./start.sh --docker        Docker Compose (cached image when possible)
  ./start.sh --docker --rebuild
                             force `docker compose up --build`
  ./start.sh --local         host uvicorn + Vite (+ optional mlflow ui)

Press Ctrl+C to stop background processes started by this script.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)   MODE=docker ;;
    --local)    MODE=local ;;
    --rebuild)  REBUILD=1 ;;
    -h|--help)  usage; exit 0 ;;
    *)          err "unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done

# -------- mode prompt --------

prompt_mode() {
  if [[ -n "$MODE" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    log "non-interactive shell with no --docker/--local; defaulting to Docker."
    MODE=docker
    return 0
  fi

  echo
  echo "How do you want to run the PdM Digital Twin?"
  echo "  1) Docker  — API + dashboard :8000, MLflow :5050 (recommended)"
  echo "  2) Local   — API :8000, Vite :5173, optional MLflow :5050"
  echo
  printf "Choice [1/2] (default 1): "
  local choice=""
  read -r choice
  case "${choice:-1}" in
    1|docker|Docker|d|D) MODE=docker ;;
    2|local|Local|l|L)   MODE=local ;;
    *)
      err "invalid choice: $choice"
      exit 2
      ;;
  esac
}

# -------- shared helpers --------

wait_for_http() {
  local label=$1 url=$2 timeout=${3:-90}
  for ((i=0; i<timeout; i++)); do
    if curl -sf -o /dev/null "$url"; then
      log "$label ready"
      return 0
    fi
    sleep 1
  done
  warn "$label did not respond within ${timeout}s"
  return 1
}

http_healthy() {
  curl -sf -o /dev/null "$API_URL" && curl -sf -o /dev/null "$MLFLOW_URL"
}

mkdir -p mlruns "$LOG_DIR"

# -------- Docker helpers --------

docker_prereqs() {
  if ! command -v docker >/dev/null 2>&1; then
    err "docker not found on PATH. Install Docker Desktop and retry,"
    err "or run: ./start.sh --local"
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon not reachable. Start Docker Desktop and retry."
    exit 1
  fi
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

image_smoke_test() {
  # Quick sanity check: can the runtime image import the API module?
  docker run --rm --entrypoint python "$IMAGE" \
    -c "import api.server" >/dev/null 2>&1
}

compose_running() {
  docker compose ps --status running --services 2>/dev/null \
    | grep -qx 'pdm' \
    && docker compose ps --status running --services 2>/dev/null \
    | grep -qx 'mlflow'
}

ensure_mlflow_image() {
  if image_exists "$MLFLOW_IMAGE"; then
    return 0
  fi
  log "pulling $MLFLOW_IMAGE ..."
  docker pull "$MLFLOW_IMAGE"
}

print_docker_banner() {
  cat <<EOF

==========================================================
  PdM Digital Twin is up (Docker)

  Dashboard + API   :  http://localhost:8000
  MLflow tracking UI:  http://localhost:5050

  Tail logs (another tab):  docker compose logs -f
  Stop the stack         :  Ctrl+C here (this script tears it down)
==========================================================

EOF
}

heal_docker_stack() {
  local step=1

  log "recovery step $step/4: soft restart of compose services..."
  docker compose restart >/dev/null 2>&1 || true
  if wait_for_http "pdm" "$API_URL" 30 && wait_for_http "mlflow" "$MLFLOW_URL" 30; then
    log "stack recovered after restart"
    return 0
  fi
  step=$((step + 1))

  log "recovery step $step/4: recycle containers (compose down + up)..."
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  if ! docker compose up -d --remove-orphans; then
    warn "compose up failed during recycle"
  elif wait_for_http "pdm" "$API_URL" 90 && wait_for_http "mlflow" "$MLFLOW_URL" 60; then
    log "stack recovered after container recycle"
    return 0
  fi
  step=$((step + 1))

  log "recovery step $step/4: rebuild pdm image and recreate stack..."
  if ! docker compose up -d --build --remove-orphans; then
    warn "compose up --build failed"
  elif wait_for_http "pdm" "$API_URL" 120 && wait_for_http "mlflow" "$MLFLOW_URL" 60; then
    log "stack recovered after rebuild"
    return 0
  fi
  step=$((step + 1))

  log "recovery step $step/4: remove local pdm image and force a clean rebuild..."
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  docker rmi -f "$IMAGE" >/dev/null 2>&1 || true
  if ! docker compose up -d --build --remove-orphans; then
    err "clean rebuild failed."
    err "Diagnose with:"
    err "  docker compose logs"
    err "  lsof -nP -iTCP:8000 -sTCP:LISTEN"
    err "  lsof -nP -iTCP:5050 -sTCP:LISTEN"
    return 1
  fi
  if wait_for_http "pdm" "$API_URL" 120 && wait_for_http "mlflow" "$MLFLOW_URL" 60; then
    log "stack recovered after clean rebuild"
    return 0
  fi

  err "stack still unhealthy after all recovery steps."
  err "check: docker compose logs"
  return 1
}

docker_cleanup() {
  local exit_code=$?
  echo
  log "stopping Docker Compose stack..."
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  log "bye"
  exit "$exit_code"
}

run_docker() {
  docker_prereqs
  ensure_mlflow_image

  local need_up=1
  local build_flag=()

  if [[ $REBUILD -eq 1 ]]; then
    log "forced rebuild requested (--rebuild)"
    build_flag=(--build)
  elif ! image_exists "$IMAGE"; then
    log "no cached $IMAGE image; building (~3 min on first run)..."
    build_flag=(--build)
  elif ! image_smoke_test; then
    warn "cached $IMAGE image failed smoke test; scheduling rebuild"
    build_flag=(--build)
  else
    log "cached $IMAGE image looks OK"
  fi

  if [[ ${#build_flag[@]} -eq 0 ]] && compose_running && http_healthy; then
    log "compose stack already running and healthy — reusing it"
    need_up=0
  elif [[ ${#build_flag[@]} -eq 0 ]] && compose_running; then
    warn "containers are up but health checks failed — attempting recovery"
    if heal_docker_stack; then
      need_up=0
    fi
  fi

  trap docker_cleanup EXIT INT TERM

  if [[ $need_up -eq 1 ]]; then
    log "starting compose stack (pdm + mlflow)..."
    if ! docker compose up -d --remove-orphans "${build_flag[@]}"; then
      err "docker compose up failed; attempting recovery..."
      heal_docker_stack || exit 1
    elif ! wait_for_http "pdm" "$API_URL" 90 || ! wait_for_http "mlflow" "$MLFLOW_URL" 60; then
      warn "services started but health checks failed; attempting recovery..."
      heal_docker_stack || exit 1
    fi
  else
    wait_for_http "pdm" "$API_URL" 5 || true
    wait_for_http "mlflow" "$MLFLOW_URL" 5 || true
  fi

  print_docker_banner
  docker compose logs -f
}

# -------- local helpers --------

local_cleanup() {
  local exit_code=$?
  echo
  log "stopping local processes..."
  [[ -n "$API_PID" ]]    && kill "$API_PID"    2>/dev/null || true
  [[ -n "$WEB_PID" ]]    && kill "$WEB_PID"    2>/dev/null || true
  [[ -n "$MLFLOW_PID" ]] && kill "$MLFLOW_PID" 2>/dev/null || true
  rm -f "$LOG_DIR"/api.pid "$LOG_DIR"/web.pid "$LOG_DIR"/mlflow.pid 2>/dev/null || true
  log "bye"
  exit "$exit_code"
}

ensure_venv() {
  if [[ -x "$VENV/bin/uvicorn" && -x "$VENV/bin/python" ]]; then
    return 0
  fi
  warn "Python venv not ready at $VENV"
  if [[ -t 0 ]]; then
    printf "Run 'make install' now? [Y/n]: "
    local ans=""
    read -r ans
    case "${ans:-Y}" in
      n|N|no|No) err "aborting — run 'make install' first."; exit 1 ;;
    esac
  else
    log "running make install (non-interactive)..."
  fi
  make install
}

ensure_web_deps() {
  if [[ -d web/node_modules ]]; then
    return 0
  fi
  warn "web/node_modules missing"
  if [[ -t 0 ]]; then
    printf "Run 'make web-install' now? [Y/n]: "
    local ans=""
    read -r ans
    case "${ans:-Y}" in
      n|N|no|No) err "aborting — run 'make web-install' first."; exit 1 ;;
    esac
  else
    log "running make web-install..."
  fi
  make web-install
}

maybe_start_mlflow_local() {
  local start_mlflow=0
  if [[ -t 0 ]]; then
    printf "Start MLflow UI on :5050? [y/N]: "
    local ans=""
    read -r ans
    case "${ans:-N}" in
      y|Y|yes|Yes) start_mlflow=1 ;;
    esac
  fi
  if [[ $start_mlflow -eq 0 ]]; then
    return 0
  fi
  if curl -sf -o /dev/null "$MLFLOW_URL"; then
    log "MLflow already responding on :5050 — skipping"
    return 0
  fi
  log "starting MLflow UI on :5050..."
  "$VENV/bin/mlflow" ui --backend-store-uri "$PWD/mlruns" --port 5050 \
    >"$LOG_DIR/mlflow.log" 2>&1 &
  MLFLOW_PID=$!
  echo "$MLFLOW_PID" >"$LOG_DIR/mlflow.pid"
  wait_for_http "mlflow" "$MLFLOW_URL" 30 || true
}

print_local_banner() {
  cat <<EOF

==========================================================
  PdM Digital Twin is up (local dev)

  API (reload)      :  http://localhost:8000
  Vite dashboard    :  http://localhost:5173
  MLflow (optional) :  http://localhost:5050

  Logs: $LOG_DIR/{api,web,mlflow}.log
  Stop: Ctrl+C here
==========================================================

EOF
}

run_local() {
  ensure_venv
  ensure_web_deps
  trap local_cleanup EXIT INT TERM

  log "starting FastAPI on :8000..."
  "$VENV/bin/uvicorn" api.server:app --host 0.0.0.0 --port 8000 --reload \
    >"$LOG_DIR/api.log" 2>&1 &
  API_PID=$!
  echo "$API_PID" >"$LOG_DIR/api.pid"

  log "starting Vite on :5173..."
  (cd web && npm run dev) >"$LOG_DIR/web.log" 2>&1 &
  WEB_PID=$!
  echo "$WEB_PID" >"$LOG_DIR/web.pid"

  maybe_start_mlflow_local

  wait_for_http "api" "$API_URL" 60 || warn "API slow to start — see $LOG_DIR/api.log"
  wait_for_http "web" "$WEB_URL" 60 || warn "Vite slow to start — see $LOG_DIR/web.log"

  print_local_banner

  # Keep script alive until Ctrl+C; surface API logs in foreground.
  tail -f "$LOG_DIR/api.log" &
  local tail_pid=$!
  wait "$API_PID" 2>/dev/null || true
  kill "$tail_pid" 2>/dev/null || true
}

# -------- main --------

prompt_mode

case "$MODE" in
  docker) run_docker ;;
  local)  run_local ;;
  *)
    err "internal error: unknown mode '$MODE'"
    exit 2
    ;;
esac
