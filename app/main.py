from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from app.inference import generate_video

app = FastAPI(title="AnimateDiff-Lightning Server", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="텍스트 프롬프트")
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_frames: int = 16
    steps: int = Field(4, description="num_inference_steps (Lightning은 4 권장)")
    guidance_scale: float = 1.0
    seed: Optional[int] = None
    fps: int = 8
    out_path: Optional[str] = None


class GenerateResponse(BaseModel):
    path: str
    seed: int


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    path = generate_video(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_frames=req.num_frames,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        fps=req.fps,
        out_path=req.out_path,
    )
    # seed는 파일명에서 복구할 수 없으니 요청값 그대로 반환(없으면 None)
    return GenerateResponse(path=path, seed=req.seed if req.seed is not None else -1)
