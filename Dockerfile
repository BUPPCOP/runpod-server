# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs ffmpeg \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# App
COPY . /app
# CRLF 방지 + 실행권한
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# ---- Bake settings (안정 우선) ----
ARG BAKE_MODE=true
ENV BAKE_MODE=${BAKE_MODE}

# 주석은 별도 줄에 두고, ENV는 한 줄에 하나로!
# (고속전송 끄는 안전 모드)
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Bake & cleanup
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
  echo "[CLEAN] purge caches"; \
  rm -rf /root/.cache/pip/* /root/.cache/huggingface/* /var/lib/apt/lists/*; \
  find /app -name \"__pycache__\" -type d -exec rm -rf {} +; \
  du -sh /app/models || true; \
  echo "[DISK] after:"; df -h \
'

ENTRYPOINT ["/bin/bash","-lc","/app/entrypoint.sh"]
