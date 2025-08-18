# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs ffmpeg \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App
COPY . /app
# CRLF 방지 + 실행권한
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# ---- 베이크 고정 ----
# BAKE_MODE 기본값 true (RunPod Build Args에서 덮어쓰지 않도록)
ARG BAKE_MODE=true
ENV BAKE_MODE=${BAKE_MODE}

# 모델 캐시 최적화
ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 베이크: 실패 시 즉시 빌드 중단
RUN /bin/bash -lc '\
  echo "[DISK] before:"; df -h; \
  if [ "${BAKE_MODE}" = "true" ]; then \
    echo "[BAKE] downloading models..."; \
    python -u scripts/download_models.py || { echo "[BAKE] FAILED"; exit 1; }; \
    echo "[BAKE] OK"; \
  else \
    echo "[BAKE] is false -> build will fail (we require baked models)"; \
    exit 2; \
  fi; \
  echo "[DISK] after:"; df -h'

ENTRYPOINT ["/bin/bash","-lc","/app/entrypoint.sh"]
