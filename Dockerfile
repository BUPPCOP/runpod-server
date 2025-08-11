# syntax=docker/dockerfile:1.6

# ✔ 존재하는 유효 태그 사용 (CUDA 12.4.1 / Python 3.11)
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# 필요한 도구 최소 설치 (git-lfs, ffmpeg 포함)
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 소스
COPY . /app

# 전송/캐시 최적화
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# 빌드 시 모델 베이킹(캐시 레이어로 가속)
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -u scripts/download_models.py

# 런타임 설정 (서버리스/로컬 전환)
ENV APP_MODE=fastapi
EXPOSE 8000
CMD ["/bin/bash","-lc",'if [ "$APP_MODE" = "serverless" ]; then python handler.py; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi' ]
