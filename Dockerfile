# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# 전송/캐시 최적화 + 자세한 파이썬 실행(-u)
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# 캐시 레이어 활용하여 재빌드 가속
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -u scripts/download_models.py

ENV APP_MODE=fastapi
EXPOSE 8000
CMD ["/bin/bash","-lc",'if [ "$APP_MODE" = "serverless" ]; then python handler.py; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi' ]
