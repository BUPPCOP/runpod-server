# syntax=docker/dockerfile:1.6

# ✔ 존재하는 유효 태그 (CUDA 12.4.1 / Python 3.11)
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 필수 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs ffmpeg \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 의존성
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 소스
COPY . /app

# 모델 전송/캐시 최적화
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# 빌드 시 모델 베이크 (필요시 BAKE_MODE=false로 스킵)
ARG BAKE_MODE=true
ENV BAKE_MODE=${BAKE_MODE}

RUN --mount=type=cache,target=/root/.cache/huggingface \
    bash -lc 'set -euo pipefail; \
      echo "[NET] DNS test:"; getent hosts huggingface.co || true; \
      echo "[NET] TLS test:"; curl -I https://huggingface.co -m 10 || true; \
      echo "[DISK] before:"; df -h; \
      if [ "${BAKE_MODE}" = "true" ]; then \
        echo "[BAKE] starting download_models.py"; \
        python -u scripts/download_models.py; \
        echo "[DISK] after:"; df -h; \
      else \
        echo "[BAKE] skipped (BAKE_MODE=false)"; \
      fi'

# ▶ 엔트리포인트 스크립트 복사 및 권한
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 런타임 설정
ENV APP_MODE=fastapi
EXPOSE 8000

# ✅ 시작 커맨드 고정 (RunPod UI Start Command/Args 비워두기 권장)
ENTRYPOINT ["/app/entrypoint.sh"]
