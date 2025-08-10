# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

# ---- OS deps ----
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (레이어 캐시 극대화) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- 앱 소스 ----
COPY . /app

# ---- (선택) 빌드 시 모델 다운로드 ----
# RunPod 템플릿의 Build Args에 HF_TOKEN 값을 넣으면 private 모델도 가능
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 모델을 이미지에 구워넣기(빌드 시 1회 다운로드)
RUN python scripts/download_models.py

# ---- 런타임 ----
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
