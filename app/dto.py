from pydantic import BaseModel
from typing import Optional, List
from datetime import date


# 회원가입 관련 DTO
class SignupRequest(BaseModel):
    name: str
    birthdate: str  # YYYY.MM.DD
    username: str
    password: str
    role: str  # USER or CARE
    connecting_user_code: Optional[str] = None  # CARE 역할일 때 연결할 사용자 username


class SignupResponse(BaseModel):
    message: str
    user_code: str
    username: str
    name: str
    role: str


# 로그인 관련 DTO
class SigninRequest(BaseModel):
    username: str
    password: str


class SigninResponse(BaseModel):
    message: str
    username: str
    name: str
    role: str


# 내정보 조회 관련 DTO
class UserInfoResponse(BaseModel):
    """일반 유저 내정보 조회 응답"""
    name: str
    username: str
    connected_care_name: Optional[str] = None  # 연결된 보호자 이름 (없으면 null)


class CareInfoResponse(BaseModel):
    """보호자 내정보 조회 응답"""
    name: str
    username: str
    connected_user_name: Optional[str] = None  # 연결된 피보호자 이름 (없으면 null)


# Notification 관련 DTO
class NotificationItem(BaseModel):
    """알림 항목"""
    notification_id: int
    voice_id: int
    name: str
    top_emotion: Optional[str] = None
    created_at: str  # ISO 형식


class NotificationListResponse(BaseModel):
    """알림 목록 응답"""
    notifications: List[NotificationItem]


# Top Emotion 관련 DTO
class TopEmotionResponse(BaseModel):
    """그날의 대표 emotion 응답"""
    date: str  # YYYY-MM-DD
    top_emotion: Optional[str] = None


class CareTopEmotionResponse(BaseModel):
    """보호자용 그날의 대표 emotion 응답"""
    date: str  # YYYY-MM-DD
    user_name: str  # 연결된 유저 이름
    top_emotion: Optional[str] = None


# 음성 관련 DTO
class VoiceUploadRequest(BaseModel):
    folder: Optional[str] = None
    language_code: str = "ko-KR"


class VoiceUploadResponse(BaseModel):
    success: bool
    message: str


class UserVoiceUploadRequest(BaseModel):
    language_code: str = "ko-KR"


class UserVoiceUploadResponse(BaseModel):
    success: bool
    message: str
    voice_id: Optional[int] = None


class VoiceQuestionUploadResponse(BaseModel):
    success: bool
    message: str
    voice_id: Optional[int] = None
    question_id: Optional[int] = None


class VoiceListItem(BaseModel):
    voice_id: int
    created_at: str
    emotion: Optional[str] = None
    question_title: Optional[str] = None
    content: str
    s3_url: Optional[str] = None


class UserVoiceListResponse(BaseModel):
    success: bool
    voices: list[VoiceListItem]


class CareVoiceListItem(BaseModel):
    voice_id: int
    created_at: str
    emotion: Optional[str] = None


class CareUserVoiceListResponse(BaseModel):
    success: bool
    voices: list[CareVoiceListItem]


class UserVoiceDetailResponse(BaseModel):
    voice_id: int
    title: Optional[str] = None
    top_emotion: Optional[str] = None
    created_at: str
    voice_content: Optional[str] = None
    s3_url: Optional[str] = None


class VoiceAnalyzePreviewResponse(BaseModel):
    voice_id: Optional[int] = None
    happy_bps: int
    sad_bps: int
    neutral_bps: int
    angry_bps: int
    anxiety_bps: Optional[int] = 0  # fear -> anxiety (출력용)
    surprise_bps: int
    top_emotion: Optional[str] = None
    top_confidence_bps: Optional[int] = None
    model_version: Optional[str] = None


class VoiceDetailResponse(BaseModel):
    voice_id: str
    filename: str
    status: str
    duration_sec: float
    analysis: dict


# 감정 분석 관련 DTO
class EmotionAnalysisResponse(BaseModel):
    voice_key: str
    emotion_analysis: dict


# STT 관련 DTO
class TranscribeRequest(BaseModel):
    language_code: str = "ko-KR"


class TranscribeResponse(BaseModel):
    transcript: str
    confidence: float
    language_code: str
    audio_duration: float
    sample_rate: int


# NLP 관련 DTO
class NLPAnalysisRequest(BaseModel):
    text: str
    language_code: str = "ko"


class SentimentResponse(BaseModel):
    sentiment: dict
    sentences: list[dict]
    language_code: str


class EntitiesResponse(BaseModel):
    entities: list[dict]
    language_code: str


class SyntaxResponse(BaseModel):
    tokens: list[dict]
    language_code: str


class ComprehensiveAnalysisResponse(BaseModel):
    text: str
    language_code: str
    sentiment_analysis: dict
    entity_analysis: dict
    syntax_analysis: dict


# 공통 응답 DTO
class ErrorResponse(BaseModel):
    detail: str


class SuccessResponse(BaseModel):
    message: str
    status: str = "success"


# OpenAI 분석 결과 DTO
class AnalysisResultResponse(BaseModel):
    """OpenAI 종합분석 결과 응답"""
    source: str  # weekly | frequency
    message: str


class WeeklyDayItem(BaseModel):
    date: str
    weekday: str
    top_emotion: Optional[str] = None


class WeeklyAnalysisCombinedResponse(BaseModel):
    """주간 종합분석: OpenAI 메시지 + 기존 주간 요약"""
    message: str
    weekly: List[WeeklyDayItem]


class FrequencyAnalysisCombinedResponse(BaseModel):
    """월간 빈도 종합분석: OpenAI 메시지 + 기존 빈도 결과"""
    message: str
    frequency: dict
