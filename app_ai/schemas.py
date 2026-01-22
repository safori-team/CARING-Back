"""
AI 서버용 요청/응답 스키마
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class AnalyzeVoiceRequest(BaseModel):
    """음성 분석 요청 (voice_id 포함)"""
    voice_id: int


class AnalyzeTextRequest(BaseModel):
    """텍스트 분석 요청"""
    text: str
    language_code: str = "ko"
    voice_id: Optional[int] = None  # 제공 시 DB에 저장


class GenerateSummaryRequest(BaseModel):
    """요약문 생성 요청"""
    user_name: str
    data_type: str  # "weekly" or "monthly"
    # 주간 데이터 (data_type="weekly"일 때)
    weekly_data: Optional[Dict[str, List[str]]] = None  # 날짜별 대표 감정 목록
    # 월간 데이터 (data_type="monthly"일 때)
    monthly_data: Optional[Dict[str, int]] = None  # 감정별 빈도수


class AnalyzeVoiceResponse(BaseModel):
    """음성 분석 응답"""
    success: bool
    stt_text: Optional[str] = None
    emotion_scores: Optional[Dict[str, float]] = None
    top_emotion: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class AnalyzeTextResponse(BaseModel):
    """텍스트 분석 응답"""
    success: bool
    sentiment: Optional[Dict[str, float]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class GenerateSummaryResponse(BaseModel):
    """요약문 생성 응답"""
    success: bool
    summary: Optional[str] = None
    error: Optional[str] = None
