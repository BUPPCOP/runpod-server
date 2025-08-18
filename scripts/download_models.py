import os, sys, json, time, traceback
from pathlib import Path
# ⛔️ HfHubHTTPError 불필요 — 일부 버전에서 공개되지 않음
from huggingface_hub import snapshot_download

MODELS_DIR = Path("/app/models")
SD_DIR = MODELS_DIR / "sd_base"
AD_DIR = MODELS_DIR / "ad_lightning"

BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")
AD_REPO   = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")
HF_TOKEN  = os.getenv("HF_TOKEN")

MAX_RETRY = int(os.getenv("HF_DOWNLOAD_RETRY", "3"))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def retry_snapshot(repo_id: str, local_dir: str):
    """간단 재시도 (예외 타입에 의존 X)"""
    last_err = None
    for i in range(1, MAX_RETRY + 1):
        try:
            print(f"[DL] {repo_id} -> {local_dir} (try {i}/{MAX_RETRY})", flush=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
            return
        except Exception as e:  # ← 여기만 잡으면 됩니다
            last_err = e
            wait = min(30, 2 ** i)
            print(f"[DL] failed try {i}: {e} ; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise last_err  # 최종 실패 시 그대로 올려 중단

def pick_safetensors(d: Path) -> str:
    cands = list(d.glob("*.safetensors"))
    if not cands:
        raise RuntimeError(f"AD Lightning *.safetensors not found in {d}")
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
    print(f"[AD] config -> {cfg} (motion_modules={weight_name})", flush=True)

def sanity_sd(sd_dir: Path):
    for p in [sd_dir / "model_index.json", sd_dir / "scheduler",
              sd_dir / "unet", sd_dir / "vae", sd_dir / "text_encoder", sd_dir / "tokenizer"]:
        if not p.exists():
            raise RuntimeError(f"SD base missing: {p}")
    def has_weights(sub: str):
        sdir = sd_dir / sub
        return any(sdir.glob("diffusion_pytorch_model*.safetensors")) or \
               any(sdir.glob("diffusion_pytorch_model*.bin"))
    if not has_weights("unet"):
        raise RuntimeError("SD base UNet weights missing (unet/diffusion_pytorch_model*)")
    if not has_weights("vae"):
        raise RuntimeError("SD base VAE  weights missing (vae/diffusion_pytorch_model*)")

def main():
    print("[BAKE] HF_HOME:", os.getenv("HF_HOME"), flush=True)
    ensure_dir(MODELS_DIR); ensure_dir(SD_DIR); ensure_dir(AD_DIR)

    print(f"[BAKE] SD base: {BASE_REPO}", flush=True)
    retry_snapshot(BASE_REPO, SD_DIR.as_posix())
    sanity_sd(SD_DIR)

    print(f"[BAKE] AD Lightning: {AD_REPO}", flush=True)
    retry_snapshot(AD_REPO, AD_DIR.as_posix())
    print("[BAKE] AD files:", sorted(os.listdir(AD_DIR)), flush=True)

    weight = pick_safetensors(AD_DIR)
    write_ad_config(AD_DIR, weight)

    if not (AD_DIR / "config.json").exists():
        raise RuntimeError(f"AD config.json missing in {AD_DIR}")

    print("[BAKE] DONE at", MODELS_DIR, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
