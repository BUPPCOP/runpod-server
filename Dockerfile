FROM python:3.10-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 전체 소스 복사 (models 포함)
COPY . /app

# 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt

# 실행 (handler 역할 수행)
CMD ["python", "-u", "app/main.py"]
