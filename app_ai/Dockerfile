FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Seoul

# Hugging Face 모델 캐시 경로 지정
ENV HF_HOME=/data/model_cache

# 시스템 패키지 설치 (오디오 처리에 필요)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    git \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /caring-voice-ai

# 모델 캐시 디렉토리 생성
RUN mkdir -p /data/model_cache && chmod 777 /data/model_cache

# requirements 설치
COPY app_ai/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# app_ai 코드 복사 (프로젝트 루트 구조 유지)
COPY app_ai/ ./app_ai/

# credentials 디렉토리 생성 (런타임에 볼륨 마운트로 제공)
# 빌드 시에는 빈 디렉토리만 생성하고, 실제 파일은 docker run 시 -v 옵션으로 마운트
RUN mkdir -p ./credentials

# Python 경로 설정 (app_ai를 모듈로 인식)
ENV PYTHONPATH=/caring-voice-ai

# 포트 노출 (AI 서버는 8001 포트 사용)
EXPOSE 8001

# AI 서버 실행 (프로젝트 루트에서 실행)
CMD ["uvicorn", "app_ai.main:app", "--host", "0.0.0.0", "--port", "8001"]
