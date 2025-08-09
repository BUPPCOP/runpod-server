# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

# ---- 필수 패키지 ----
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# ---- 파이썬 의존성 ----
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- 빌드 시 모델 다운로드 (이미지에 포함) ----
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python - << 'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os, sys, pathlib

BASE_DIR = pathlib.Path("app/models")
BASE_DIR.mkdir(parents=True, exist_ok=True)

print("📦 Downloading base model (emilianJR/epiCRealism)...")
snapshot_download(
    repo_id="emilianJR/epiCRealism",
    local_dir=str(BASE_DIR / "base"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN", None)
)

print("📦 Downloading AnimateDiff-Lightning (4-step) adapter...")
hf_hub_download(
    repo_id="ByteDance/AnimateDiff-Lightning",
    filename="animatediff_lightning_4step_diffusers.safetensors",
    local_dir=str(BASE_DIR / "animatediff"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN", None)
)
print("✅ Models baked into the image.")
PY

# ---- 앱 실행 ----
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
