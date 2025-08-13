import os, sys, time, traceback, subprocess, json
from pathlib import Path
from typing import Iterable
from huggingface_hub import snapshot_download, hf_hub_download
import huggingface_hub as hf

HF_TOKEN = os.getenv("HF_TOKEN")

BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")
AD_LIGHTNING_REPO = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")

TARGET_DIR = Path(os.getenv("TARGET_DIR", "/app/models"))

BASE_PATTERNS = [
    "feature_extractor/**","scheduler/**","vae/**","text_encoder/**","tokenizer/**","unet/**",
    "model_index.json","*.json","*.txt","*.safetensors","*.bin",
]

# AD-Lightning: 가중치 + (레포에 있으면) json류도 받아본다.
AD_PATTERNS = [
    "animatediff_lightning_4step_diffusers.safetensors",
    "*.json",
    "README.md",
    "comfyui/**",   # 레포 구조상 같이 올 때가 있어 무시해도 무해
]

MAX_RETRIES = 3
RETRY_WAIT = 10

def _print_env():
    print("[ENV] HF_TOKEN set:", bool(HF_TOKEN))
    print("[ENV] HF_HOME:", os.getenv("HF_HOME"))
    print("[ENV] HF_TRANSFER:", os.getenv("HF_HUB_ENABLE_HF_TRANSFER"))
    print("[ENV] PYTHON:", sys.version)
    print("[ENV] huggingface_hub:", getattr(hf, "__version__", "unknown"))
    try:
        out = subprocess.check_output(["git-lfs","--version"]).decode().strip()
        print("[ENV] git-lfs:", out)
    except Exception as e:
        print("[WARN] git-lfs not available:", e)

def _du_h(p: Path) -> str:
    try:
        out = subprocess.check_output(["du","-sh", str(p)]).decode().split()[0]
        return out
    except Exception:
        return "-"

def preflight(repo: str, file_candidates: list[str]):
    last_err = None
    for fname in file_candidates:
        try:
            tmp = hf_hub_download(repo_id=repo, filename=fname, token=HF_TOKEN)
            print(f"[PREFLIGHT OK] {repo}:{fname} -> {tmp}")
            return
        except Exception as e:
            print(f"[PREFLIGHT TRY FAIL] {repo}:{fname} -> {e}", file=sys.stderr)
            last_err = e
    print(f"[PREFLIGHT ERR] {repo} -> {last_err}", file=sys.stderr)
    raise last_err

def pull(repo_id: str, subdir: str, patterns: Iterable[str]):
    local_dir = TARGET_DIR / subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[DL] {repo_id} → {local_dir} (try {attempt}/{MAX_RETRIES})")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                allow_patterns=list(patterns),
                resume_download=True,
                max_workers=8,
            )
            print(f"[OK] {repo_id} done. dir={local_dir} size={_du_h(local_dir)}")
            return
        except Exception as e:
            last_err = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            etype = type(e).__name__
            print(f"[ERR] repo={repo_id} type={etype} code={code} msg={e}", file=sys.stderr)
            traceback.print_exc()
            if code in (401,403,404):
                break
        time.sleep(RETRY_WAIT)
    raise SystemExit(f"[FATAL] download failed: {repo_id} -> {last_err}")

def ensure_ad_config(ad_dir: Path):
    """AD-Lightning 폴더에 Diffusers 로더가 요구하는 config.json을 생성(없을 때만)."""
    cfg = ad_dir / "config.json"
    midx = ad_dir / "model_index.json"
    wt  = ad_dir / "animatediff_lightning_4step_diffusers.safetensors"

    # 기본 가중치는 반드시 있어야 함
    if not wt.exists():
        raise SystemExit(f"[FATAL] Missing AD weight: {wt}")

    # 이미 config/model_index가 있으면 그대로 사용
    if cfg.exists() or midx.exists():
        print("[AD] Found existing config:", cfg.exists(), "model_index:", midx.exists())
        return

    # ---- 템플릿 생성 ----
    # 주의: 아래 값들은 SD1.5 + AnimateDiff(MotionAdapter) 기본 구조에 맞춘 보편 템플릿.
    # Lightning 4-step용 가중치는 모듈 구조가 동일하고, 하이퍼파라미터(스텝 경량화)는 가중치에 내포됨.
    template = {
        "_class_name": "MotionAdapter",
        "_diffusers_version": "0.29.0",
        "motion_config": {
            "motion_module_type": "Vanilla",
            "use_motion_module": True,
            "num_attention_blocks": 2,        # commonly used
            "num_transformer_blocks": 1
        },
        # UNet 블록 폭(SD1.5 기준)
        "block_out_channels": [320, 640, 1280, 1280],
        "cross_attention_dim": 768,
        "infer_steps_default": 4,            # Lightning 4-step 힌트(정보성)
        "dtype": "fp16"
    }
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"[AD] Wrote config template -> {cfg}")

if __name__ == "__main__":
    try:
        print("[INFO] download_models.py start")
        _print_env()

        preflight(BASE_REPO, ["model_index.json", "README.md"])
        # AD 레포에는 config가 없을 수 있지만, 시도는 한번 해본다(없으면 TRY FAIL로만 남음)
        try:
            preflight(AD_LIGHTNING_REPO, ["config.json", "model_index.json", "README.md"])
        except Exception:
            pass

        pull(BASE_REPO, "sd_base", BASE_PATTERNS)
        pull(AD_LIGHTNING_REPO, "ad_lightning", AD_PATTERNS)

        # AD 구성 보장
        ad_dir = TARGET_DIR / "ad_lightning"
        try:
            listing = [p.name for p in ad_dir.glob("*")]
            print(f"[AD] dir listing -> {listing}")
        except Exception:
            print("[AD] cannot list:", ad_dir)
        ensure_ad_config(ad_dir)

        total = _du_h(TARGET_DIR)
        print(f"[DONE] Models baked at {TARGET_DIR} (total {total})")

    except SystemExit as e:
        print(str(e), file=sys.stderr); sys.exit(1)
    except Exception as e:
        print("[UNCAUGHT]", e, file=sys.stderr); traceback.print_exc(); sys.exit(1)
