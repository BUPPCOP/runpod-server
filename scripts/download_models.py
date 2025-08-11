# scripts/download_models.py
import os
import sys
import time
from typing import Iterable
from huggingface_hub import snapshot_download, hf_hub_download, HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN", None)

# ▼ 필요한 경우에만 교체 가능 (Build Args/Env로 주입 가능)
AD_LIGHTNING_REPO = os.getenv("AD_LIGHTNING_REPO", "ByteDance/AnimateDiff-Lightning")
# diffusers 포맷 SD1.5 (전체 스냅샷은 큼 → allow_patterns로 필요한 것만)
BASE_REPO = os.getenv("BASE_REPO", "runwayml/stable-diffusion-v1-5")

TARGET_DIR = os.getenv("TARGET_DIR", "/app/models")

# 필요한 파일만 받기 (용량/시간 절감)
BASE_PATTERNS: list[str] = [
    "feature_extractor/*",
    "scheduler/*",
    "vae/*",
    "text_encoder/*",
    "tokenizer/*",
    "unet/*",
    "model_index.json",
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.bin",
]
AD_PATTERNS: list[str] = [
    # lightning 4-step diffusers 가중치/구성만
    "**/animatediff_lightning_4step_diffusers*/**",
    "model_index.json",
    "*.json",
    "*.safetensors",
    "*.bin",
]

MAX_RETRIES = 3
RETRY_WAIT = 10

def pull(repo_id: str, subdir: str, patterns: Iterable[str]):
    local_dir = os.path.join(TARGET_DIR, subdir)
    os.makedirs(local_dir, exist_ok=True)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[DL] repo={repo_id} → {local_dir} (try {attempt}/{MAX_RETRIES})")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                allow_patterns=list(patterns),
                max_workers=8,  # 병렬
                tqdm_class=None,
                resume_download=True,
            )
            print(f"[OK] {repo_id}")
            return
        except HfHubHTTPError as e:
            last_err = e
            # 권한/토큰/쿼터 오류는 바로 실패 메시지 명확히
            print(f"[ERR] HfHubHTTPError for {repo_id}: {e}", file=sys.stderr)
            if e.response is not None and e.response.status_code in (401, 403, 404):
                break  # 재시도 의미 없음
        except Exception as e:
            last_err = e
            print(f"[ERR] Generic error for {repo_id}: {e}", file=sys.stderr)
        time.sleep(RETRY_WAIT)
    raise SystemExit(f"[FATAL] Failed to download {repo_id}: {last_err}")

if __name__ == "__main__":
    # 토큰 체크(선택): 공개 레포면 없어도 되지만, 쿼터/속도 이슈 완화 위해 권장
    if HF_TOKEN is None:
        print("[WARN] HF_TOKEN not set. Public repos are fine, but you may hit rate limits.")

    pull(BASE_REPO, "sd_base", BASE_PATTERNS)
    pull(AD_LIGHTNING_REPO, "ad_lightning", AD_PATTERNS)
    print("[DONE] All models cached in image.")
