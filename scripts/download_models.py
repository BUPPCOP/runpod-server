# scripts/download_models.py
import os
from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN", None)

# 빌드 인자로 대체 가능
AD_LIGHTNING_REPO = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")  # 예시 레포
BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")                  # SD1.5 예시
TARGET_DIR = os.getenv("TARGET_DIR", "/app/models")


def pull(repo_id: str, subdir: str):
    local_dir = os.path.join(TARGET_DIR, subdir)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[DL] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
        tqdm_class=None,
    )
    print(f"[OK] {repo_id}")


if __name__ == "__main__":
    pull(BASE_REPO, "sd_base")
    pull(AD_LIGHTNING_REPO, "ad_lightning")
    print("[DONE] All models cached in image.")
