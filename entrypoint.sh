#!/usr/bin/env bash
set -e

echo "APP_MODE=${APP_MODE}"

# python3 우선, 없으면 python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
$PY -V || true

cd /app

if [ "${APP_MODE}" = "serverless" ]; then
  # RunPod Serverless 핸들러
  exec $PY handler.py
else
  # 로컬/HTTP 모드 (FastAPI)
  exec $PY -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
