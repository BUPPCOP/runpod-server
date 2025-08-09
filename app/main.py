from fastapi import FastAPI
from app.inference import generate_video
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RunPod Serverless API"}

@app.post("/run")
def run_inference():
    result = generate_video()
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
