"""
AI 서버 전용 Voice Repository
- VoiceAnalyze, VoiceContent, VoiceJobProcess 저장 로직
"""
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from ..models import VoiceAnalyze, VoiceContent, VoiceJobProcess


def ensure_job_row(session: Session, voice_id: int) -> VoiceJobProcess:
    """작업 상태 행이 없으면 생성"""
    row = session.query(VoiceJobProcess).filter(VoiceJobProcess.voice_id == voice_id).first()
    if not row:
        row = VoiceJobProcess(voice_id=voice_id, text_done=0, audio_done=0, locked=0)
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def mark_text_done(session: Session, voice_id: int) -> None:
    """텍스트 작업 완료 표시"""
    row = ensure_job_row(session, voice_id)
    row.text_done = 1
    session.commit()


def mark_audio_done(session: Session, voice_id: int) -> None:
    """오디오 작업 완료 표시"""
    row = ensure_job_row(session, voice_id)
    row.audio_done = 1
    session.commit()


def create_voice_content(
    session: Session,
    voice_id: int,
    content: str,
    score_bps: Optional[int] = None,
    magnitude_x1000: Optional[int] = None,
    locale: Optional[str] = None,
    provider: Optional[str] = None,
    model_version: Optional[str] = None,
    confidence_bps: Optional[int] = None
) -> VoiceContent:
    """음성 전사 및 텍스트 감정 분석 데이터 생성"""
    # 기존 데이터가 있으면 업데이트, 없으면 생성
    voice_content = session.query(VoiceContent).filter(VoiceContent.voice_id == voice_id).first()
    
    if voice_content:
        voice_content.content = content
        if score_bps is not None:
            voice_content.score_bps = score_bps
        if magnitude_x1000 is not None:
            voice_content.magnitude_x1000 = magnitude_x1000
        if locale is not None:
            voice_content.locale = locale
        if provider is not None:
            voice_content.provider = provider
        if model_version is not None:
            voice_content.model_version = model_version
        if confidence_bps is not None:
            voice_content.confidence_bps = confidence_bps
    else:
        voice_content = VoiceContent(
            voice_id=voice_id,
            content=content,
            score_bps=score_bps,
            magnitude_x1000=magnitude_x1000,
            locale=locale,
            provider=provider,
            model_version=model_version,
            confidence_bps=confidence_bps
        )
        session.add(voice_content)
    
    session.commit()
    session.refresh(voice_content)
    return voice_content


def create_voice_analyze(
    session: Session,
    voice_id: int,
    happy_bps: int,
    sad_bps: int,
    neutral_bps: int,
    angry_bps: int,
    fear_bps: int,
    surprise_bps: int,
    top_emotion: Optional[str] = None,
    top_confidence_bps: Optional[int] = None,
    model_version: Optional[str] = None
) -> VoiceAnalyze:
    """음성 감정 분석 데이터 생성"""
    # 기존 데이터가 있으면 업데이트, 없으면 생성
    voice_analyze = session.query(VoiceAnalyze).filter(VoiceAnalyze.voice_id == voice_id).first()
    
    if voice_analyze:
        voice_analyze.happy_bps = happy_bps
        voice_analyze.sad_bps = sad_bps
        voice_analyze.neutral_bps = neutral_bps
        voice_analyze.angry_bps = angry_bps
        voice_analyze.fear_bps = fear_bps
        voice_analyze.surprise_bps = surprise_bps
        if top_emotion is not None:
            voice_analyze.top_emotion = top_emotion
        if top_confidence_bps is not None:
            voice_analyze.top_confidence_bps = top_confidence_bps
        if model_version is not None:
            voice_analyze.model_version = model_version
    else:
        voice_analyze = VoiceAnalyze(
            voice_id=voice_id,
            happy_bps=happy_bps,
            sad_bps=sad_bps,
            neutral_bps=neutral_bps,
            angry_bps=angry_bps,
            fear_bps=fear_bps,
            surprise_bps=surprise_bps,
            top_emotion=top_emotion,
            top_confidence_bps=top_confidence_bps,
            model_version=model_version
        )
        session.add(voice_analyze)
    
    session.commit()
    session.refresh(voice_analyze)
    return voice_analyze
