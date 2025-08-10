import os
import pathlib
import random
from typing import Optional, List

import torch
from PIL import Image
import imageio.v2 as imageio

from diffusers import (
    MotionAdapter,
    AnimateDiffPipeline,
    DPMSolverMultistepScheduler,
)

# ===== 디바이스/정밀도 선택 =====
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

# ===== 모델 경로 =====
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_ROOT = BASE_DIR / "models"
BASE_MODEL_DIR = MODELS_ROOT / "base"          # emilianJR/epiCRealism 이 구워질 경로
ADAPTER_DIR    = MODELS_ROOT / "animatediff"   # Lightning safetensors 저장 경로

# 파일명(도커 빌드에서 내려받음)
LIGHTNING_FILENAME = "animatediff_lightning_4step_diffusers.safetensors"


_pipe: Optional[AnimateDiffPipeline] = None


def _load_pipeline() -> AnimateDiffPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    # 1) 모션 어댑터: 기본(일반) 어댑터를 허브에서 바로 가져오거나, 필요 시 로컬 커스텀 로딩
    #    Lightning은 LoRA 형식 가중치로 제공되며, 어댑터에 추가로 로드합니다.
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5", torch_dtype=DTYPE
    )

    # 2) 베이스 파이프라인 로드
    #    (도커 이미지에 구워진 로컬 경로가 있으면 그걸 우선 사용)
    base_model_path = str(BASE_MODEL_DIR) if BASE_MODEL_DIR.exists() else "emilianJR/epiCRealism"

    pipe = AnimateDiffPipeline.from_pretrained(
        base_model_path,
        motion_adapter=adapter,
        torch_dtype=DTYPE
    )

    # 스케줄러는 DPM-Solver 멀티스텝으로 설정(라이트닝 4스텝과 궁합 좋음)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 3) Lightning 4스텝 가중치 로드 (있으면)
    lightning_path = ADAPTER_DIR / LIGHTNING_FILENAME
    if lightning_path.exists():
        try:
            # LoRA 형식으로 생성기(UNet)에 가중치 로드
            pipe.load_lora_weights(
                str(ADAPTER_DIR),
                weight_name=LIGHTNING_FILENAME
            )
            # LoRA 스케일은 기본 1.0, 필요 시 조정
            pipe.fuse_lora()
            print("[AnimateDiff] Lightning (4-step) weights loaded and fused.")
        except Exception as e:
            print(f"[WARN] Failed to load Lightning weights: {e}")

    # 성능 최적화
    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_model_cpu_offload") and DEVICE.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

    _pipe = pipe
    return pipe


def generate_video(
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    num_frames: int = 16,
    num_inference_steps: int = 4,        # Lightning에 맞춰 4스텝 기본
    guidance_scale: float = 1.0,         # Lightning은 보통 낮은 CFG
    seed: Optional[int] = None,
    fps: int = 8,
    out_path: Optional[str] = None,
) -> str:
    """
    텍스트 프롬프트로 짧은 GIF 생성 후 파일 경로 반환
    """
    pipe = _load_pipeline()

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # AnimateDiffPipeline은 frames(list[PIL.Image])를 반환
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    frames: List[Image.Image] = result.frames

    # 저장 경로
    if out_path is None:
        out_dir = BASE_DIR.parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"out_{seed}.gif")

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
