import os, sys, json, traceback
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path("/app/models")
SD_DIR = MODELS_DIR / "sd_base"
AD_DIR = MODELS_DIR / "ad_lightning"

# 필요 시 @commit 으로 리비전 고정 가능
BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")
AD_REPO   = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")
HF_TOKEN  = os.getenv("HF_TOKEN")  # 비공개/레이트리밋 회피용(선택)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _pick_ad_safetensors(ad_dir: Path) -> str:
    cands = list(ad_dir.glob("*.safetensors"))
    if not cands:
        raise RuntimeError(f"AD Lightning *.safetensors not found in {ad_dir}")
    # 우선순위: 4step / diffusers 포함 파일 가점
    def score(p: Path):
        n = p.name.lower()
        return int("4" in n and "step" in n) + int("diffusers" in n)
    cands.sort(key=score, reverse=True)
    return cands[0].name

def write_ad_config(ad_dir: Path, weight_name: str):
    cfg = ad_dir / "config.json"
    payload = {
        "_class_name": "MotionAdapter",
        "sample_size": 512,
        "motion_modules": [weight_name],
    }
    cfg.write_text(json.dumps(payload, indent=2))
    print(f"[AD] Wrote config -> {cfg} (motion_modules={weight_name})", flush=True)

def main():
    print("[BAKE] HF_HOME:", os.getenv("HF_HOME"), flush=True)
    ensure_dir(MODELS_DIR); ensure_dir(SD_DIR); ensure_dir(AD_DIR)

    print(f"[BAKE] SD base: {BASE_REPO}", flush=True)
    snapshot_download(
        repo_id=BASE_REPO,
        local_dir=SD_DIR.as_posix(),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

    print(f"[BAKE] AD Lightning: {AD_REPO}", flush=True)
    snapshot_download(
        repo_id=AD_REPO,
        local_dir=AD_DIR.as_posix(),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    print("[BAKE] AD files:", sorted(os.listdir(AD_DIR)), flush=True)

    # config.json 생성 (파일명 자동 감지)
    weight = _pick_ad_safetensors(AD_DIR)
    write_ad_config(AD_DIR, weight)

    # Sanity check (강제)
    if not (SD_DIR / "model_index.json").exists():
        raise RuntimeError(f"SD base model_index.json missing in {SD_DIR}")
    if not (AD_DIR / "config.json").exists():
        raise RuntimeError(f"AD config.json missing in {AD_DIR}")

    print("[BAKE] DONE at", MODELS_DIR, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
