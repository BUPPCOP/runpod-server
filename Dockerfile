# syntax=docker/dockerfile:1.6

# RunPod의 CUDA+PyTorch가 사전 설치된 이미지 권장
ARG BASE_IMAGE=runpod/pytorch:2.4.0-cuda12.1
FROM ${BASE_IMAGE}

# 필수 도구만 설치 (추천 패키지 제외 + 캐시 정리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# 의존성 설치
COPY requirements.txt /app/requirements.txt
# torch/torchaudio/torchvision은 베이스 이미지에 있으니 requirements.txt에서 제거 권장
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 복사
COPY . /app

# ----- 모델 베이크 (빌드 시 1회) -----
ARG HF_TOKEN
ARG AD_LIGHTNING_REPO=ByteDance/AnimateDiff-Lightning
ARG BASE_REPO=runwayml/stable-diffusion-v1-5
ENV HF_TOKEN=${HF_TOKEN} \
    AD_LIGHTNING_REPO=${AD_LIGHTNING_REPO} \
    BASE_REPO=${BASE_REPO}

RUN python scripts/download_models.py

# ----- 실행 모드 전환 -----
ENV APP_MODE=serverless
EXPOSE 8000

# Serverless(/run) 기본, FastAPI는 APP_MODE=fastapi로
CMD ["/bin/sh", "-lc", "if [ \"$APP_MODE\" = \"serverless\" ]; then python -u handler.py; else python -m uvicorn app.main:app --host 0.0.0.0 --port 8000; fi"]
