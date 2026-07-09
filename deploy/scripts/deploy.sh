#!/usr/bin/env bash
# ============================================================
# AIOS Production Deployment Script (Linux / macOS)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$PROJECT_DIR/deploy"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Config ----
COMPOSE_FILES="-f $DEPLOY_DIR/docker-compose.yml"
PROFILES=""
ENV_FILE="$DEPLOY_DIR/.env"

# ---- Parse arguments ----
WITH_GPU=false
WITH_MONITORING=false
COMMAND="up"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)       WITH_GPU=true; shift ;;
    --monitor)   WITH_MONITORING=true; shift ;;
    --build)     BUILD="--build"; shift ;;
    --down)      COMMAND="down"; shift ;;
    --logs)      COMMAND="logs"; shift ;;
    --restart)   COMMAND="restart"; shift ;;
    --env)       ENV_FILE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "  --gpu        Enable GPU acceleration"
      echo "  --monitor    Enable Prometheus/Grafana monitoring"
      echo "  --build      Rebuild images"
      echo "  --down       Stop all services"
      echo "  --logs       Tail logs"
      echo "  --restart    Restart all services"
      echo "  --env FILE   Path to .env file (default: $ENV_FILE)"
      exit 0 ;;
    *) error "Unknown argument: $1" ;;
  esac
done

# ---- Validate environment ----
if [ ! -f "$ENV_FILE" ]; then
  warn ".env file not found at $ENV_FILE — copying from .env.example"
  cp "$DEPLOY_DIR/.env.example" "$ENV_FILE"
fi

# ---- Select compose files ----
if [ "$WITH_GPU" = true ]; then
  COMPOSE_FILES="$COMPOSE_FILES -f $DEPLOY_DIR/docker-compose.gpu.yml"
  info "GPU acceleration enabled"
fi

if [ "$WITH_MONITORING" = true ]; then
  COMPOSE_FILES="$COMPOSE_FILES -f $DEPLOY_DIR/docker-compose.monitoring.yml"
  info "Monitoring stack enabled"
fi

# ---- Docker compose commands ----
cd "$PROJECT_DIR"

case "$COMMAND" in
  up)
    info "Starting AIOS production stack..."
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES --env-file "$ENV_FILE" up -d ${BUILD:-}
    info "AIOS is running:"
    info "  API:      http://localhost:8000"
    info "  Docs:     http://localhost:8000/docs"
    if [ "$WITH_MONITORING" = true ]; then
      info "  Grafana:  http://localhost:3000 (admin:aiosadmin)"
      info "  Prometheus: http://localhost:9090"
    fi
    info "  Logs:     docker compose $COMPOSE_FILES logs -f"
    ;;
  down)
    info "Stopping AIOS production stack..."
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES down
    info "All services stopped."
    ;;
  restart)
    info "Restarting AIOS production stack..."
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES restart
    info "All services restarted."
    ;;
  logs)
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES logs -f
    ;;
esac
