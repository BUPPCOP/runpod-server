from fastapi import FastAPI
from pathlib import Path

app = FastAPI(title="RunPod Serverless Demo")

@app.get("/health")
def health():
    # 간단한 존재 확인: 빌드 타임에 구워넣은 모델 파일들이 있는지 체크
    base_ok = Path("app/models/base").exists()
    ad_ok = Path("app/models/animatediff/animatediff_lightning_4step_diffusers.safetensors").exists()
    return {"status": "ok", "base_model": base_ok, "animatediff": ad_ok}

# 실제 inference 엔드포인트는 여기에 추가
# @app.post("/generate")
# def generate(...): ...
