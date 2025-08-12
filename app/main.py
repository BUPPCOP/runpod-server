# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid

from app.inference import run_inference_animatediff

OUTPUT_DIR = "/app/outputs"
INPUT_DIR = "/app/inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

app = FastAPI(title="AnimateDiff-Lightning API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"available": ["AnimateDiff-Lightning"]}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    frames: int = Form(16),
    fps: int = Form(8),
    guidance: float = Form(1.0),
    seed: int | None = Form(None),
):
    uid = str(uuid.uuid4())
    in_path = os.path.join(INPUT_DIR, f"{uid}_{image.filename}")
    out_path = os.path.join(OUTPUT_DIR, f"{uid}.mp4")

    with open(in_path, "wb") as f:
        f.write(await image.read())

    ok = run_inference_animatediff(
        in_path,
        out_path,
        num_frames=int(frames),
        fps=int(fps),
        guidance_scale=float(guidance),
        seed=seed,
    )
    if not ok:
        return JSONResponse(status_code=500, content={"error": "inference_failed"})

    # 데모용: 파일 직접 다운로드 라우트 제공
    return {"id": uid, "video_path": f"/download/{uid}.mp4"}


@app.get("/download/{file}")
def download(file: str):
    path = os.path.join(OUTPUT_DIR, file)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return FileResponse(path, media_type="video/mp4", filename=file)