FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치(캐시 최적화)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 소스 복사
COPY . /app

# HF 토큰
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN test -n "$HF_TOKEN" || (echo "HF_TOKEN not set" && exit 1)

# 빌드 시 모델 다운로드 (이미지에 구워넣기)
RUN python scripts/download_models.py

# 앱 실행
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
