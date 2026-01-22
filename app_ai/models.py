"""
AI 서버 전용 모델
- VoiceAnalyze, VoiceContent, VoiceJobProcess만 정의
- 최소한의 의존성으로 DB 접근
"""
from sqlalchemy import Column, BigInteger, String, DateTime, Integer, SmallInteger, Text, ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base


class VoiceContent(Base):
    """음성 전사 및 텍스트 감정 분석 테이블"""
    __tablename__ = "voice_content"
    
    voice_content_id = Column(BigInteger, primary_key=True, autoincrement=True)
    voice_id = Column(BigInteger, ForeignKey("voice.voice_id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)  # 전사 내용
    score_bps = Column(SmallInteger, nullable=True)  # -10000~10000 (감정 점수 * 10000)
    magnitude_x1000 = Column(Integer, nullable=True)  # 0~? (감정 강도 * 1000)
    locale = Column(String(10), nullable=True)  # 'ko-KR' 등
    provider = Column(String(32), nullable=True)  # 'google', 'aws' 등
    model_version = Column(String(32), nullable=True)
    confidence_bps = Column(SmallInteger, nullable=True)  # 0~10000 (신뢰도 * 10000)
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    
    # 관계 설정 (AI 서버에서는 조회만 하므로 관계 없이도 가능)
    # voice = relationship("Voice", back_populates="voice_content")
    
    # 제약 조건
    __table_args__ = (
        UniqueConstraint('voice_id', name='uq_vc_voice'),  # 1:1 관계
    )


class VoiceAnalyze(Base):
    """음성 감정 분석 테이블"""
    __tablename__ = "voice_analyze"
    
    voice_analyze_id = Column(BigInteger, primary_key=True, autoincrement=True)
    voice_id = Column(BigInteger, ForeignKey("voice.voice_id", ondelete="CASCADE"), nullable=False)
    happy_bps = Column(SmallInteger, nullable=False)  # 0~10000
    sad_bps = Column(SmallInteger, nullable=False)  # 0~10000
    neutral_bps = Column(SmallInteger, nullable=False)  # 0~10000
    angry_bps = Column(SmallInteger, nullable=False)  # 0~10000
    fear_bps = Column(SmallInteger, nullable=False)  # 0~10000
    surprise_bps = Column(SmallInteger, nullable=False, default=0)  # 0~10000
    top_emotion = Column(String(16), nullable=True)  # 'neutral' 등
    top_confidence_bps = Column(SmallInteger, nullable=True)  # 0~10000
    model_version = Column(String(32), nullable=True)
    analyzed_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    
    # 관계 설정 (AI 서버에서는 조회만 하므로 관계 없이도 가능)
    # voice = relationship("Voice", back_populates="voice_analyze")
    
    # 제약 조건
    __table_args__ = (
        UniqueConstraint('voice_id', name='uq_va_voice'),
        CheckConstraint("happy_bps <= 10000 AND sad_bps <= 10000 AND neutral_bps <= 10000 AND angry_bps <= 10000 AND fear_bps <= 10000 AND surprise_bps <= 10000", name='check_emotion_bps_range'),
        CheckConstraint("happy_bps + sad_bps + neutral_bps + angry_bps + fear_bps + surprise_bps = 10000", name='check_emotion_bps_sum'),
    )


class VoiceJobProcess(Base):
    """비동기 작업 상태 및 집계 락 관리"""
    __tablename__ = "voice_job_process"
    
    voice_id = Column(BigInteger, ForeignKey("voice.voice_id", ondelete="CASCADE"), primary_key=True)
    text_done = Column(SmallInteger, nullable=False, default=0)
    audio_done = Column(SmallInteger, nullable=False, default=0)
    locked = Column(SmallInteger, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # 관계 설정 (AI 서버에서는 조회만 하므로 관계 없이도 가능)
    # voice = relationship("Voice", back_populates="voice_job_process", uselist=False)
