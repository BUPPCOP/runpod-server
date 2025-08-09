from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.inference import generate_video

app = FastAPI(title="AnimateDiff Lightning API")

class PromptRequest(BaseModel):
    prompt: str = Field(..., description="텍스트 프롬프트")
    steps: int = Field(4, ge=1, le=8, description="추론 스텝(1/2/4/8)")
    guidance_scale: float = Field(1.0, ge=0.0, le=10.0)
    fps: int = Field(8, ge=1, le=60)
    out_path: str = Field("output.gif")

@app.get("/")
def root():
    return {"status": "ok", "message": "AnimateDiff Lightning API"}

@app.post("/generate")
def generate(req: PromptRequest):
    path = generate_video(
        prompt=req.prompt,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        fps=req.fps,
        out_path=req.out_path
    )
    return {"status": "success", "output": path}
