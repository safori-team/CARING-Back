# Caring AI Service

AI 분석 전용 서버 - 독립 실행 가능

## 기능

- 음성 분석 (STT + 감정 분석)
- 텍스트 분석 (감정 + 엔티티)
- 요약문 생성 (OpenAI)
- 분석 결과 DB 저장 (VoiceAnalyze, VoiceContent, VoiceJobProcess)

## 실행 방법

### 로컬 개발 환경

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일)
# app_ai/.env.example 파일을 참고하여 .env 파일 생성
# 필수 변수:
#   - DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
#   - GOOGLE_APPLICATION_CREDENTIALS
#   - OPENAI_API_KEY

# 서버 실행
uvicorn app_ai.main:app --host 0.0.0.0 --port 8001 --reload
```

### Docker로 실행

```bash
# Docker 이미지 빌드
docker build -f app_ai/Dockerfile -t caring-ai:latest .

# Docker 컨테이너 실행
docker run -d \
  --name caring-ai \
  -p 8001:8001 \
  --env-file .env \
  -v $(pwd)/credentials:/caring-voice-ai/credentials \
  -v $(pwd)/.cache/huggingface:/data/model_cache \
  caring-ai:latest
```

## API 문서

서버 실행 후 다음 URL에서 Swagger UI 확인:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## 엔드포인트

- `POST /ai/analyze-voice` - 음성 파일 분석 (STT + 감정 분석)
- `POST /ai/analyze-text` - 텍스트 분석 (감정 + 엔티티)
- `POST /ai/generate-summary` - 요약문 생성
- `GET /health` - 헬스 체크

## 환경 변수

필수 환경 변수:
- `DB_HOST` - 데이터베이스 호스트
- `DB_PORT` - 데이터베이스 포트
- `DB_USER` - 데이터베이스 사용자
- `DB_PASSWORD` - 데이터베이스 비밀번호
- `DB_NAME` - 데이터베이스 이름
- `GOOGLE_APPLICATION_CREDENTIALS` - Google Cloud 인증 파일 경로
- `OPENAI_API_KEY` - OpenAI API 키

선택적 환경 변수:
- `DB_CA_LOCATION` - SSL CA 인증서 경로
- `DB_CERT_LOCATION` - SSL 인증서 경로
- `DB_KEY_LOCATION` - SSL 키 경로
- `OPENAI_MODEL` - OpenAI 모델명 (기본값: gpt-4o-mini)
