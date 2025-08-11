# RunPod Serverless + AnimateDiff-Lightning

## 1) 로컬 실행 (FastAPI)
```bash
export HF_TOKEN=hf_xxx
export APP_MODE=fastapi
uvicorn app.main:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs
