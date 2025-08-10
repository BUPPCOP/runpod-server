from huggingface_hub import snapshot_download, hf_hub_download
import os, pathlib

HF_TOKEN = os.getenv("HF_TOKEN")

BASE = pathlib.Path("app/models")
(BASE / "animatediff").mkdir(parents=True, exist_ok=True)
(BASE / "base").mkdir(parents=True, exist_ok=True)

print("📦 base model...")
snapshot_download(
    repo_id="emilianJR/epiCRealism",
    local_dir=str(BASE / "base"),
    local_dir_use_symlinks=False,
    token=HF_TOKEN or None,
)

print("📦 lightning 4-step...")
hf_hub_download(
    repo_id="ByteDance/AnimateDiff-Lightning",
    filename="animatediff_lightning_4step_diffusers.safetensors",
    local_dir=str(BASE / "animatediff"),
    local_dir_use_symlinks=False,
    token=HF_TOKEN or None,
)

print("✅ done")
