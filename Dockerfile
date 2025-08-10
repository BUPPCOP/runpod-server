# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

# ---- 필수 패키지 ----
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사하여 캐시 활용
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 소스 복사
COPY . /app

# ---- 빌드 시 모델 다운로드 ----
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python - << 'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os, pathlib

BASE = pathlib.Path("app/models")
(BASE / "animatediff").mkdir(parents=True, exist_ok=True)
(BASE / "base").mkdir(parents=True, exist_ok=True)

print("📦 Downloading base model (emilianJR/epiCRealism)...")
snapshot_download(
    repo_id="emilianJR/epiCRealism",
    local_dir=str(BASE / "base"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None,
)

print("📦 Downloading AnimateDiff-Lightning (4-step) adapter...")
hf_hub_download(
    repo_id="ByteDance/AnimateDiff-Lightning",
    filename="animatediff_lightning_4step_diffusers.safetensors",
    local_dir=str(BASE / "animatediff"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None,
)
print("✅ Models baked into the image.")
PY

# ---- 런타임 ----
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
