#!/usr/bin/env bash
set -Eeuo pipefail
trap 'code=$?; echo "[ENTRYPOINT] exit code=${code}";' EXIT

echo "[ENTRYPOINT] start"
echo "APP_MODE=${APP_MODE:-<unset>}"

cd /app
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
$PY -V || true

if [ "${APP_MODE:-serverless}" = "serverless" ]; then
  echo "[ENTRYPOINT] launching handler.py"
  exec $PY handler.py
else
  echo "[ENTRYPOINT] launching uvicorn"
  exec $PY -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
