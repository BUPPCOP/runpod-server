# scripts/download_models.py
# 이미지 빌드 시 필요한 모델만 '부분 다운로드'하여 /app/models 에 베이킹
# - Hugging Face 토큰/권한/404/네트워크 문제를 명확히 로깅
# - hf_transfer + resume_download 로 속도/안정성 강화
# - 특정 버전에서 HfHubHTTPError 경로가 달라지는 문제를 우회(범용 예외 처리)

import os
import sys
import time
import traceback
import subprocess
from pathlib import Path
from typing import Iterable
from huggingface_hub import snapshot_download, hf_hub_download
import huggingface_hub as hf

HF_TOKEN = os.getenv("HF_TOKEN")

# 필요 시 환경변수로 교체 가능
BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")
AD_LIGHTNING_REPO = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")

TARGET_DIR = Path(os.getenv("TARGET_DIR", "/app/models"))

# 용량/시간 절감을 위해 꼭 쓰는 파일만 허용
BASE_PATTERNS = [
    "feature_extractor/**",
    "scheduler/**",
    "vae/**",
    "text_encoder/**",
    "tokenizer/**",
    "unet/**",
    "model_index.json",
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.bin",
]
AD_PATTERNS = [
    "**/animatediff_lightning_4step_diffusers*/**",
    "model_index.json",
    "*.json",
    "*.safetensors",
    "*.bin",
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
        out = subprocess.check_output(["git-lfs", "--version"]).decode().strip()
        print("[ENV] git-lfs:", out)
    except Exception as e:
        print("[WARN] git-lfs not available:", e)


def _du_h(path: Path) -> str:
    try:
        out = subprocess.check_output(["du", "-sh", str(path)]).decode().split()[0]
        return out
    except Exception:
        return "-"


def preflight(repo: str, file: str = "model_index.json"):
    """권한/토큰/404 문제를 빌드 초기에 바로 드러내기"""
    try:
        tmp = hf_hub_download(repo_id=repo, filename=file, token=HF_TOKEN)
        print(f"[PREFLIGHT OK] {repo}:{file} -> {tmp}")
    except Exception as e:
        print(f"[PREFLIGHT ERR] {repo}:{file} -> {e}", file=sys.stderr)
        raise


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
            # 권한/존재 문제는 재시도 무의미
            if code in (401, 403, 404):
                break
        time.sleep(RETRY_WAIT)
    raise SystemExit(f"[FATAL] download failed: {repo_id} -> {last_err}")


if __name__ == "__main__":
    try:
        print("[INFO] download_models.py start")
        _print_env()

        # 권한/존재/토큰 문제를 초기에 확인
        preflight(BASE_REPO)
        preflight(AD_LIGHTNING_REPO)

        pull(BASE_REPO, "sd_base", BASE_PATTERNS)
        pull(AD_LIGHTNING_REPO, "ad_lightning", AD_PATTERNS)

        total = _du_h(TARGET_DIR)
        print(f"[DONE] Models baked at {TARGET_DIR} (total {total})")

    except SystemExit as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("[UNCAUGHT]", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
