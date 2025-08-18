import os, json
from typing import Optional, List, Tuple
import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter
from app.utils import save_mp4

MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
SD_DIR = os.path.join(MODELS_DIR, "sd_base")
AD_DIR = os.path.join(MODELS_DIR, "ad_lightning")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_image(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((768, 768))  # OOM 방지
    return im

def _ensure_models():
    print(f"[INFER] DEVICE={DEVICE}, cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[INFER] SD_DIR={SD_DIR}", flush=True)
    print(f"[INFER] AD_DIR={AD_DIR}", flush=True)
    try:
        print("[AD_PATH]", AD_DIR, "->", os.listdir(AD_DIR), flush=True)
    except Exception as e:
        print("[AD_PATH] cannot list:", AD_DIR, e, flush=True)

def _ensure_ad_config():
    """실수로 config.json이 누락됐을 때만 복구(가중치가 있을 때)"""
    cfg_path = os.path.join(AD_DIR, "config.json")
    if os.path.exists(cfg_path):
        return
    safes = [f for f in os.listdir(AD_DIR) if f.endswith(".safetensors")]
    if not safes:
        raise RuntimeError(f"AD Lightning *.safetensors not found in {AD_DIR}")
    def score(name: str):
        n = name.lower()
        return int("4" in n and "step" in n) + int("diffusers" in n)
    safes.sort(key=score, reverse=True)
    weight = safes[0]
    payload = {
        "_class_name": "MotionAdapter",
        "sample_size": 512,
        "motion_modules": [weight]
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[AD] runtime wrote config.json -> {cfg_path} (motion_modules={weight})", flush=True)

def run_inference_animatediff(
    image_path: str,
    seed: int = 1234,
    num_frames: int = 16,
    fps: int = 8,
    guidance_scale: float = 1.0,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns: (ok, out_video_path, reason)
    """
    _ensure_models()
    try:
        # 베이크가 원칙이지만, 혹시 모를 config 누락 복구
        _ensure_ad_config()

        torch.manual_seed(seed)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        adapter = MotionAdapter.from_pretrained(AD_DIR, torch_dtype=dtype)
        pipe = AnimateDiffPipeline.from_pretrained(
            SD_DIR,
            motion_adapter=adapter,
            torch_dtype=dtype,
        ).to(DEVICE)

        if DEVICE == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass

        init_image = _load_image(image_path)
        result = pipe(
            image=init_image,       # 위치 인자 금지(버전 시그니처 차이 회피)
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            output_type="pil",
        )
        frames: List[Image.Image] = result.frames if hasattr(result, "frames") else result[0]

        out_path = f"/tmp/out_{os.path.basename(image_path)}.mp4"
        save_mp4(frames, out_path, fps=fps)
        print(f"[INFER] wrote {out_path}", flush=True)
        return True, out_path, None

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] AnimateDiff inference failed: {e}", flush=True)
        print(tb, flush=True)
        return False, None, f"{e}\n{tb}"
