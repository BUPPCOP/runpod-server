import os, sys, json, traceback
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path("/app/models")
SD_DIR = MODELS_DIR / "sd_base"
AD_DIR = MODELS_DIR / "ad_lightning"

BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")
AD_REPO   = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _pick_ad_safetensors(ad_dir: Path) -> str:
    """레포에서 받은 .safetensors 중 4‑step/diffusers 포함 파일을 우선 선택"""
    cands = list(ad_dir.glob("*.safetensors"))
    if not cands:
        raise RuntimeError("AD Lightning *.safetensors not found")
    # 우선순위: 4step, diffusers 키워드
    def score(p: Path):
        name = p.name.lower()
        return int("4step" in name) + int("diffusers" in name)
    cands.sort(key=score, reverse=True)
    return cands[0].name  # 파일명만 반환

def write_ad_config(ad_dir: Path, weight_name: str):
    cfg = ad_dir / "config.json"
    template = {
        "_class_name": "MotionAdapter",
        "sample_size": 512,
        "motion_modules": [weight_name]
    }
    cfg.write_text(json.dumps(template, indent=2))
    print(f"[AD] Wrote config -> {cfg} (motion_modules={weight_name})")

def main():
    print("[ENV] HF_HOME:", os.getenv("HF_HOME"))
    ensure_dir(MODELS_DIR); ensure_dir(SD_DIR); ensure_dir(AD_DIR)

    print("[OK] downloading SD base:", BASE_REPO)
    snapshot_download(repo_id=BASE_REPO, local_dir=SD_DIR.as_posix(), local_dir_use_symlinks=False)

    print("[OK] downloading AD Lightning:", AD_REPO)
    snapshot_download(repo_id=AD_REPO, local_dir=AD_DIR.as_posix(), local_dir_use_symlinks=False)

    weight = _pick_ad_safetensors(AD_DIR)
    write_ad_config(AD_DIR, weight)

    print("[DONE] Models baked at", MODELS_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
