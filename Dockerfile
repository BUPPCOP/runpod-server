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

# 🔧 윈도우 CRLF 방지 + 실행 권한 보장
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 모델 전송/캐시 최적화
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 베이크 모드 스위치 (기본 true)
ARG BAKE_MODE=true
ENV BAKE_MODE=${BAKE_MODE}

# 베이크 실행 (BAKE_MODE=true일 때만)
RUN /bin/bash -lc '\
      echo "[DISK] before:"; df -h; \
      if [ "${BAKE_MODE}" = "true" ]; then \
        echo "[BAKE] starting download_models.py"; \
        python -u scripts/download_models.py; \
        echo "[DISK] after:"; df -h; \
      else \
        echo "[BAKE] skipped (BAKE_MODE=false)"; \
      fi'

# ▶ ENTRYPOINT: bash -lc 로 확실하게 진입
ENTRYPOINT ["/bin/bash","-lc","/app/entrypoint.sh"]
