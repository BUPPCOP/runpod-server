# syntax=docker/dockerfile:1.6

# --- 권장: RunPod PyTorch 베이스(토치/쿠다 호환 안정)
# 사용할 수 있는 최신/안정 태그로 교체하세요 (예시는 개념용)
# ARG BASE_IMAGE=runpod/pytorch:2.4.0-cuda12.1
# FROM ${BASE_IMAGE}

# --- 대안: 경량 Python 베이스 (CPU 토치가 깔릴 수 있으니 주의)
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# ---- 모델 베이크 (빌드 시 1회) ----
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
# 선택: 레포 오버라이드
ARG AD_LIGHTNING_REPO=ByteDance/AnimateDiff-Lightning
ARG BASE_REPO=runwayml/stable-diffusion-v1-5
ENV AD_LIGHTNING_REPO=${AD_LIGHTNING_REPO}
ENV BASE_REPO=${BASE_REPO}

RUN python scripts/download_models.py

# ---- 실행 모드 전환 ----
# 기본: Serverless 핸들러 (RunPod /run 용)
ENV APP_MODE=serverless

# Serverless 모드
CMD ["/bin/sh", "-lc", "if [ \"$APP_MODE\" = \"serverless\" ]; then python -u handler.py; else python -m uvicorn app.main:app --host 0.0.0.0 --port 8000; fi"]
