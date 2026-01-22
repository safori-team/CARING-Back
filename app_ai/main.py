"""
AI 전용 서버 엔드포인트
- 음성/텍스트 분석 수행 (emotion, STT, NLP)
- 분석 결과를 DB에 직접 저장 (VoiceAnalyze, VoiceContent, VoiceJobProcess)
- 작업 상태 관리
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .routers import ai_router, health_router

app = FastAPI(
    title="Caring AI Service",
    description="AI 분석 전용 서버 - 분석 수행 및 DB 저장",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# 라우터 등록
app.include_router(ai_router.router)
app.include_router(health_router.router)


# ============ 예외 처리 ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """HTTPException 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "statusCode": exc.status_code,
            "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """기타 모든 예외 처리"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "statusCode": 500,
            "message": str(exc)
        }
    )
