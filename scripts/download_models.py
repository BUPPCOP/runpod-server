from huggingface_hub import snapshot_download, hf_hub_download
import os
from pathlib import Path

BASE_DIR = Path("app/models")
BASE_DIR.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set")

print("📦 Downloading base model: emilianJR/epiCRealism")
snapshot_download(
    repo_id="emilianJR/epiCRealism",
    local_dir=str(BASE_DIR / "base"),
    local_dir_use_symlinks=False,
    token=token,
)

print("📦 Downloading AnimateDiff-Lightning 4-step adapter")
hf_hub_download(
    repo_id="ByteDance/AnimateDiff-Lightning",
    filename="animatediff_lightning_4step_diffusers.safetensors",
    local_dir=str(BASE_DIR / "animatediff"),
    local_dir_use_symlinks=False,
    token=token,
)

print("✅ Models baked into the image.")
