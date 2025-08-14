import os
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

def run_inference_animatediff(
    image_path: str,
    seed: int = 1234,
    num_frames: int = 16,
    fps: int = 8,
    guidance_scale: float = 1.0,
) -> Tuple[bool, Optional[str], Optional[str]]:
    _ensure_models()
    try:
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

        # ★ 가장 중요한 수정: image= 로 명시
        result = pipe(
            image=init_image,
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
