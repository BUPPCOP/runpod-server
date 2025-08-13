import os, sys, time, traceback, subprocess
from pathlib import Path
from typing import Iterable
from huggingface_hub import snapshot_download, hf_hub_download
import huggingface_hub as hf

HF_TOKEN = os.getenv("HF_TOKEN")

BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")
AD_LIGHTNING_REPO = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")

TARGET_DIR = Path(os.getenv("TARGET_DIR", "/app/models"))

# SD1.5: 파이프라인 구성요소 + model_index.json 필요
BASE_PATTERNS = [
    "feature_extractor/**","scheduler/**","vae/**","text_encoder/**","tokenizer/**","unet/**",
    "model_index.json","*.json","*.txt","*.safetensors","*.bin",
]

# AD-Lightning: 가중치 + 구성 json 포함(레포 구조 차이를 커버하기 위해 *.json 허용)
AD_PATTERNS = [
    "animatediff_lightning_4step_diffusers.safetensors",
    "config.json",
    "model_index.json",
    "*.json",
    "README.md",
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
    """작고 확실한 파일로 권한/존재/토큰 문제를 빌드 초기에 확인"""
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
            if code in (401,403,404):  # 권한/존재 문제는 재시도 무의미
                break
        time.sleep(RETRY_WAIT)
    raise SystemExit(f"[FATAL] download failed: {repo_id} -> {last_err}")

if __name__ == "__main__":
    try:
        print("[INFO] download_models.py start")
        _print_env()

        # SD: model_index.json/README, AD: config/model_index/README 중 하나라도 잡히는지 확인
        preflight(BASE_REPO, ["model_index.json", "README.md"])
        preflight(AD_LIGHTNING_REPO, ["config.json", "model_index.json", "README.md"])

        pull(BASE_REPO, "sd_base", BASE_PATTERNS)
        pull(AD_LIGHTNING_REPO, "ad_lightning", AD_PATTERNS)

        # 🔒 sanity check: AD에 weights + (config.json or model_index.json) 둘 다 있어야 함
        ad_dir = TARGET_DIR / "ad_lightning"
        needs = [ad_dir / "animatediff_lightning_4step_diffusers.safetensors"]
        cfg_ok = (ad_dir / "config.json").exists() or (ad_dir / "model_index.json").exists()
        missing = [str(p) for p in needs if not p.exists()]
        if missing or not cfg_ok:
            try:
                listing = [p.name for p in ad_dir.glob("*")]
            except Exception:
                listing = ["<cannot list>"]
            print(f"[SANITY] AD dir -> {ad_dir} : {listing}", file=sys.stderr)
            raise SystemExit(f"[FATAL] AD-Lightning files incomplete. missing={missing}, cfg_ok={cfg_ok}")

        total = _du_h(TARGET_DIR)
        print(f"[DONE] Models baked at {TARGET_DIR} (total {total})")

    except SystemExit as e:
        print(str(e), file=sys.stderr); sys.exit(1)
    except Exception as e:
        print("[UNCAUGHT]", e, file=sys.stderr); traceback.print_exc(); sys.exit(1)
