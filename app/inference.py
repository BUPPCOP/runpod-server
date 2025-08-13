import os
from typing import Optional

import torch
from PIL import Image

from diffusers import (
    DDIMScheduler,
    MotionAdapter,
    AnimateDiffPipeline,   # diffusers >= 0.29.0
)

from app.utils import save_mp4

MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 디렉토리 (빌드 시 download_models.py가 내려줌)
SD_DIR = os.path.join(MODELS_DIR, "sd_base")
AD_DIR = os.path.join(MODELS_DIR, "ad_lightning")

_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    # 디버그: AD 폴더 내용 찍어두기
    try:
        print("[AD_PATH]", AD_DIR, "->", os.listdir(AD_DIR))
    except Exception:
        print("[AD_PATH] cannot list:", AD_DIR)

    # AnimateDiff-Lightning 모션 어댑터
    adapter = MotionAdapter.from_pretrained(
        AD_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )

    # SD1.5(Base) + MotionAdapter 결합
    pipe = AnimateDiffPipeline.from_pretrained(
        SD_DIR,
        motion_adapter=adapter,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,  # 필요시 활성화
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_vae_slicing()
    else:
        pipe.enable_vae_slicing()

    _pipe = pipe
    return _pipe

def run_inference_animatediff(
    image_path: str,
    output_path: str,
    num_frames: int = 16,
    fps: int = 8,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
) -> bool:
    """정적 이미지 → 짧은 모션 영상(mp4) 생성"""
    try:
        pipe = get_pipeline()
        generator = torch.Generator(device=DEVICE)
        if seed is not None:
            generator = generator.manual_seed(int(seed))

        img = Image.open(image_path).convert("RGB")

        result = pipe(
            prompt="",
            image=img,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        frames = result.frames  # List[PIL.Image.Image]

        save_mp4(frames, output_path, fps=fps)
        return True
    except Exception as e:
        print(f"[ERROR] AnimateDiff inference failed: {e}")
        return False
