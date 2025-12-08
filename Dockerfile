FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Seoul

# 2. Hugging Face 모델 캐시 경로 지정
ENV HF_HOME=/data/model_cache

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    git \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /caring-voice

RUN mkdir -p /data/model_cache && chmod 777 /data/model_cache

COPY requirements-heavy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-heavy.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
