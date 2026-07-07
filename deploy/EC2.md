# EC2 deploy note

The GitHub Actions `deploy` job runs these server-side commands over SSH after `build-test` passes and the required deploy secrets exist. Values shown in angle brackets are placeholders; do not commit real secrets.

```bash
set -euo pipefail

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

if ! command -v git >/dev/null 2>&1; then
  if command -v dnf >/dev/null 2>&1; then
    ${SUDO} dnf install -y git
  elif command -v apt-get >/dev/null 2>&1; then
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y git
  else
    echo "Unsupported Linux distro: install git manually" >&2
    exit 1
  fi
fi

REPO_URL="https://github.com/iamirrf/Hyper-Diligence.git"
APP_DIR="${HOME}/hyper-diligence"

if [ -d "${APP_DIR}/.git" ]; then
  cd "${APP_DIR}"
  git fetch origin main
  git checkout main
  git pull --ff-only origin main
else
  git clone --branch main "${REPO_URL}" "${APP_DIR}"
  cd "${APP_DIR}"
fi

cat > .env <<'ENV'
POSTGRES_USER=hyper_diligence
POSTGRES_PASSWORD=<choose-a-db-password>
POSTGRES_DB=hyper_diligence
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
CHAT_PROVIDER=extractive
AWS_REGION=us-east-1
S3_BUCKET=<your-filings-bucket>
S3_ENABLED=true
EDGAR_USER_AGENT="<your-sec-user-agent>"
ENV

bash scripts/deploy_ec2.sh
```

The script builds the FastAPI image, starts Postgres with pgvector, initializes the schema, ingests the five-ticker corpus if no chunks are loaded, starts the app on port 80, and verifies `http://127.0.0.1/health` from inside the instance.
