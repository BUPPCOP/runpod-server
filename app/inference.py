import os
import torch
from PIL import Image
from typing import List
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from safetensors.torch import load_file

BASE_DIR = os.path.dirname(__file__)
BASE_MODEL_DIR = os.path.join(BASE_DIR, "models", "base")
ADAPTER_PATH   = os.path.join(BASE_DIR, "models", "animatediff", "animatediff_lightning_4step_diffusers.safetensors")

# 디바이스/정밀도 자동 선택 (RunPod GPU에서 자동 'cuda')
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"🖥️ device={device}, dtype={dtype}")

# 전역 로딩 (프로세스 부팅 시 1회)
print("📦 Loading motion adapter...")
adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(ADAPTER_PATH, device=device))

print("📦 Loading base model pipeline...")
pipe = AnimateDiffPipeline.from_pretrained(
    BASE_MODEL_DIR,
    motion_adapter=adapter,
    torch_dtype=dtype
).to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
    beta_schedule="linear"
)

def _save_gif(frames: List[Image.Image], out_path: str, fps: int = 8):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    duration = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=2
    )

def generate_video(
    prompt: str,
    steps: int = 4,
    guidance_scale: float = 1.0,
    fps: int = 8,
    out_path: str = "output.gif"
) -> str:
    print(f"🚀 prompt={prompt}, steps={steps}, guidance={guidance_scale}, fps={fps}")
    with torch.inference_mode():
        out = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=steps)
    frames = out.frames[0]  # 배치 1개 가정
    _save_gif(frames, out_path, fps=fps)
    print(f"✅ saved: {out_path}")
    return out_path
