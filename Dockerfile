# syntax=docker/dockerfile:1.6

# ✔ 존재하는 유효 태그 사용 (CUDA 12.4.1 / Python 3.11)
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 네트워크/SSL/대용량 파일 대비: ca-certificates, git, git-lfs, ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs ffmpeg \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 소스
COPY . /app

# 전송/캐시 최적화 + 로깅 강화
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# ▶ 베이크 토글: 기본 true (필요시 RunPod Build Args/Env에서 BAKE_MODE=false로 건너뛰기)
ARG BAKE_MODE=true
ENV BAKE_MODE=${BAKE_MODE}

# 네트워크/디스크 빠른 프리플라이트 + 모델 베이크
RUN --mount=type=cache,target=/root/.cache/huggingface \
    bash -lc 'set -euo pipefail; \
      echo "[NET] DNS test:"; \
      getent hosts huggingface.co || true; \
      echo "[NET] TLS test:"; \
      curl -I https://huggingface.co -m 10 || true; \
      echo "[DISK] before:"; df -h; \
      if [ "${BAKE_MODE}" = "true" ]; then \
        echo "[BAKE] starting download_models.py"; \
        python -u scripts/download_models.py; \
        echo "[DISK] after:"; df -h; \
      else \
        echo "[BAKE] skipped (BAKE_MODE=false)"; \
      fi'

# 런타임 (서버리스/로컬 전환)
ENV APP_MODE=fastapi
EXPOSE 8000
CMD ["/bin/bash","-lc",'if [ "$APP_MODE" = "serverless" ]; then python handler.py; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi' ]
