# syntax=docker/dockerfile:1.6

# ✔ 유효한 베이스 태그 (권장)
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 최소 도구
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 소스
COPY . /app

# 전송/캐시 최적화
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# ✅ 빌드 시 모델을 베이크(캐시 레이어 활용)
# BuildKit 필요. RunPod는 기본 활성화.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -u scripts/download_models.py

# 런타임
ENV APP_MODE=fastapi
EXPOSE 8000
CMD ["/bin/bash","-lc",'if [ "$APP_MODE" = "serverless" ]; then python handler.py; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi' ]
