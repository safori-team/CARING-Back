"""
AI 서버 전용 데이터베이스 연결
- VoiceAnalyze, VoiceContent, VoiceJobProcess 테이블에만 접근
- 독립적인 Connection Pool 사용
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# 데이터베이스 연결 정보 (환경변수에서 로드)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "springproject")
DB_NAME = os.getenv("DB_NAME", "caring_voice")

# SSL 인증서 경로 (환경변수에서만 로드, 기본값 없음)
DB_CA_LOCATION = os.getenv("DB_CA_LOCATION")
DB_CERT_LOCATION = os.getenv("DB_CERT_LOCATION")
DB_KEY_LOCATION = os.getenv("DB_KEY_LOCATION")

# 패스워드에 특수문자가 있을 경우 URL 인코딩
ENCODED_PASSWORD = quote_plus(DB_PASSWORD) if DB_PASSWORD else ""

# 데이터베이스 URL 구성
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

connect_args = {}
if DB_CA_LOCATION and DB_CERT_LOCATION and DB_KEY_LOCATION:
    # PyMySQL은 ssl_ca, ssl_cert, ssl_key를 직접 키로 사용
    connect_args = {
        "ssl_ca": DB_CA_LOCATION,
        "ssl_cert": DB_CERT_LOCATION,
        "ssl_key": DB_KEY_LOCATION,
    }

# SQLAlchemy 엔진 생성 (AI 서버 전용 Connection Pool)
# pool_size와 max_overflow를 명시적으로 설정하여 두 서버 간 Connection Pool 분리
engine = create_engine(
    DATABASE_URL,
    echo=False,  # SQL 쿼리 로깅 (개발 시 True로 설정)
    pool_pre_ping=True,  # 연결 상태 확인
    pool_recycle=3600,   # 연결 재사용 시간 (1시간)
    pool_size=5,         # 기본 연결 풀 크기 (API 서버와 독립)
    max_overflow=10,     # 최대 추가 연결 수 (API 서버와 독립)
    connect_args=connect_args,
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 생성 (모든 모델이 상속받을 클래스)
Base = declarative_base()


def get_db():
    """데이터베이스 세션 의존성 함수 (FastAPI Depends용)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
