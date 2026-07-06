#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

docker_cmd() {
  ${SUDO} docker "$@"
}

compose() {
  docker_cmd compose "$@"
}

ensure_swap() {
  if grep -q '^/swapfile ' /proc/swaps; then
    return
  fi
  if [ ! -f /swapfile ]; then
    ${SUDO} fallocate -l 2G /swapfile || ${SUDO} dd if=/dev/zero of=/swapfile bs=1M count=2048
    ${SUDO} chmod 600 /swapfile
    ${SUDO} mkswap /swapfile
  fi
  ${SUDO} swapon /swapfile
  if ! grep -q '^/swapfile ' /etc/fstab; then
    echo '/swapfile swap swap defaults 0 0' | ${SUDO} tee -a /etc/fstab >/dev/null
  fi
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    if command -v dnf >/dev/null 2>&1; then
      ${SUDO} dnf install -y docker curl
    elif command -v apt-get >/dev/null 2>&1; then
      ${SUDO} apt-get update
      ${SUDO} apt-get install -y docker.io curl
    else
      echo "Unsupported Linux distro: install Docker manually" >&2
      exit 1
    fi
  fi

  ${SUDO} systemctl enable --now docker

  if docker_cmd compose version >/dev/null 2>&1; then
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    if command -v dnf >/dev/null 2>&1; then
      ${SUDO} dnf install -y curl
    elif command -v apt-get >/dev/null 2>&1; then
      ${SUDO} apt-get update
      ${SUDO} apt-get install -y curl
    fi
  fi

  arch="$(uname -m)"
  case "${arch}" in
    x86_64) compose_arch="x86_64" ;;
    aarch64|arm64) compose_arch="aarch64" ;;
    *) echo "Unsupported Docker Compose architecture: ${arch}" >&2; exit 1 ;;
  esac

  ${SUDO} mkdir -p /usr/local/lib/docker/cli-plugins
  ${SUDO} curl -fsSL \
    "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-${compose_arch}" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  ${SUDO} chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
}

wait_for_db() {
  for _ in $(seq 1 60); do
    if compose exec -T db pg_isready -U "${POSTGRES_USER:-hyper_diligence}" -d "${POSTGRES_DB:-hyper_diligence}" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  echo "Postgres did not become ready" >&2
  compose logs db >&2 || true
  exit 1
}

chunk_count() {
  compose run --rm app python - <<'PY'
from app.db import count_chunks

try:
    print(count_chunks())
except Exception:
    print(0)
PY
}

ensure_swap
ensure_docker

compose build app
compose up -d db
wait_for_db

compose run --rm app python -m app.db --init

chunks="$(chunk_count | tail -n 1 | tr -dc '0-9')"
if [ -z "${chunks}" ] || [ "${chunks}" = "0" ]; then
  compose run --rm app python -m app.ingest.pipeline --tickers AAPL MSFT NVDA JPM TSLA
fi

compose up -d app
curl --fail --silent --show-error http://127.0.0.1/health
