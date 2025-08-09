from huggingface_hub import snapshot_download, hf_hub_download
import os

# 다운로드 경로
BASE_DIR = "./app/models"
os.makedirs(BASE_DIR, exist_ok=True)

# ✅ 1. Base 모델 다운로드
print("📦 Downloading base model (epiCRealism)...")
snapshot_download(
    repo_id="emilianJR/epiCRealism",
    local_dir=os.path.join(BASE_DIR, "base"),
    local_dir_use_symlinks=False
)

# ✅ 2. AnimateDiff Lightning 4-step adapter 다운로드
print("📦 Downloading AnimateDiff Lightning checkpoint (4-step)...")
adapter_path = hf_hub_download(
    repo_id="ByteDance/AnimateDiff-Lightning",  # ✅ 여기가 핵심
    filename="animatediff_lightning_4step_diffusers.safetensors",
    local_dir=os.path.join(BASE_DIR, "animatediff"),
    local_dir_use_symlinks=False
)

print("✅ All models downloaded successfully.")
print("➡ Motion adapter saved at:", adapter_path)
