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

def write_ad_config_if_missing(ad_dir: Path):
    cfg = ad_dir / "config.json"
    if cfg.exists():
        print("[AD] config.json already exists")
        return
    template = {
        "_class_name": "MotionAdapter",
        "sample_size": 512,
        "motion_modules": ["animatediff_lightning_4step_diffusers.safetensors"]
    }
    cfg.write_text(json.dumps(template, indent=2))
    print(f"[AD] Wrote config template -> {cfg}")

def main():
    print("[ENV] HF_HOME:", os.getenv("HF_HOME"))
    ensure_dir(MODELS_DIR); ensure_dir(SD_DIR); ensure_dir(AD_DIR)

    print("[OK] downloading SD base:", BASE_REPO)
    snapshot_download(repo_id=BASE_REPO, local_dir=SD_DIR.as_posix(), local_dir_use_symlinks=False)

    print("[OK] downloading AD Lightning:", AD_REPO)
    snapshot_download(repo_id=AD_REPO, local_dir=AD_DIR.as_posix(), local_dir_use_symlinks=False)

    write_ad_config_if_missing(AD_DIR)

    safes = list(AD_DIR.glob("*.safetensors"))
    if not safes:
        raise RuntimeError("AD Lightning *.safetensors not found")
    if not (AD_DIR / "config.json").exists():
        raise RuntimeError("AD Lightning config.json not found")

    print("[DONE] Models baked at", MODELS_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
