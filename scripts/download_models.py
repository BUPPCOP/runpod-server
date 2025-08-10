import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN") or None

BASE = Path("app/models")
BASE.mkdir(parents=True, exist_ok=True)
(BASE / "animatediff").mkdir(parents=True, exist_ok=True)
(BASE / "base").mkdir(parents=True, exist_ok=True)

print("📦 Downloading base model: emilianJR/epiCRealism -> app/models/base")
snapshot_download(
    repo_id="emilianJR/epiCRealism",
    local_dir=str(BASE / "base"),
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

print("📦 Downloading AnimateDiff-Lightning 4-step -> app/models/animatediff")
hf_hub_download(
    repo_id="ByteDance/AnimateDiff-Lightning",
    filename="animatediff_lightning_4step_diffusers.safetensors",
    local_dir=str(BASE / "animatediff"),
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

print("✅ Model assets baked into the image.")
