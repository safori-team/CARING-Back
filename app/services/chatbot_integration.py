import os
import json
import boto3
import logging
from fastapi import Depends
from datetime import datetime
from typing import Tuple, Optional
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from botocore.config import Config
from ..database import get_db
from ..models import Voice, VoiceContent, VoiceComposite, User
from ..exceptions import InternalServerException, DatabaseException

logger = logging.getLogger(__name__)

def send_analysis_to_chatbot(voice_id: int, db: Session = Depends(get_db)):
    """
    분석이 완료된 음성 데이터를 조회하여 챗봇 SQS로 이벤트를 전송합니다.

    Args:
        voice_id (int): 조회할 음성 데이터 ID.
        db (Session): FastAPI Depends를 통해 주입된 DB 세션.

    Returns:
        dict: 전송 성공 메시지.
    """

    sqs_url = os.getenv("CHATBOT_SQS_URL")
    if not sqs_url:
        logger.error("Required environment variable 'CHATBOT_SQS_URL' is missing.")
        raise InternalServerException(
            message="Required configuration missing. Cannot initiate SQS event."
        )

    result = None
    try:
        result: Optional[Tuple[Voice, VoiceContent, VoiceComposite, User]] = (
            db.query(Voice, VoiceContent, VoiceComposite, User)
            .join(VoiceContent, Voice.voice_id == VoiceContent.voice_id)
            .join(VoiceComposite, Voice.voice_id == VoiceComposite.voice_id)
            .join(User, Voice.user_id == User.user_id)
            .options(joinedload(Voice.questions))
            .filter(Voice.voice_id == voice_id)
            .first()
        )

    except SQLAlchemyError as e:
        # DB 연결 끊김, 쿼리 구문 오류 등 SQLAlchemy 관련 심각한 오류 발생
        logger.error(f"[ChatbotIntegration] Database query failed for voice_id={voice_id}: {e}", exc_info=True)
        raise DatabaseException(
            message="Failed to retrieve data due to an internal database error."
        ) from e

    if not result:
        logger.warning(f"[ChatbotIntegration] Data not found for voice_id={voice_id}. Skipping SQS send.")
        return {"message": f"Voice ID {voice_id} not found, no event sent."}

    # 데이터 추출 및 페이로드 구성
    try:
        voice, content, composite, user = result

        # 추가 맥락 정보 추출
        question_text = voice.questions[0].content if voice.questions else None

        # 메시지 페이로드 구성
        message_body = _create_message_body(voice, content, composite, user, question_text)

        # 5. SQS 전송
        response = _send_sqs_message(sqs_url, message_body)

        logger.info(f"[ChatbotIntegration] Sent to chatbot SQS. MsgID={response.get('MessageId')}, VoiceID={voice_id}")
        return {"message": f"Analysis event for voice_id {voice_id} sent successfully."}

    except Exception as e:
        # DB 오류나 SQS URL 누락 외의 기타 런타임 오류 (예: SQS 전송 실패, 데이터 구조 오류 등)
        logger.error(f"[ChatbotIntegration] Failed to process or send SQS: {e}", exc_info=True)
        raise InternalServerException(
            message=f"An unexpected error occurred during SQS preparation or transmission: {type(e).__name__}"
        ) from e

# --- 유틸리티 함수 분리 ---

def _create_message_body(voice, content, composite, user, question_text: Optional[str]):
    """메시지 본문을 구성합니다."""

    # 10000bps 단위를 1.0 기준으로 변환
    emotion_payload = {
        "top_emotion": composite.top_emotion,
        "confidence": (composite.top_emotion_confidence_bps or 0) / 10000.0,
        "details": {
            "happy": composite.happy_bps / 10000.0,
            "sad": composite.sad_bps / 10000.0,
            "angry": composite.angry_bps / 10000.0,
            "anxiety": composite.fear_bps / 10000.0,
            "neutral": composite.neutral_bps / 10000.0,
            "surprise": composite.surprise_bps / 10000.0,
        },
        "valence": composite.valence_x1000 / 1000.0,
        "arousal": composite.arousal_x1000 / 1000.0
    }

    return {
        "source": "mind-diary",
        "event": "analysis_completed",

        # [핵심 식별자]
        "user_id": user.username,
        "voice_id": voice.voice_id,

            # [대화 생성을 위한 핵심 맥락]
            "user_name": user.name,           # 예: "홍길동"
            "question": question_text,        # 예: "오늘 감사한 일은 무엇인가요?"
            "content": content.content,       # 사용자 답변 내용
            "recorded_at": voice.created_at.isoformat() if voice.created_at else None, # 실제 녹음 시간

            # [감정 데이터]
            "emotion": emotion_payload,
            "timestamp": datetime.now().isoformat() # 전송 시간
        }

def _send_sqs_message(sqs_url: str, message_body: dict):
    """실제로 SQS 메시지를 전송합니다."""
    boto_config = Config(
        connect_timeout=5,
        read_timeout=10,
        retries={'mode': 'standard', 'total_max_attempts': 3}
    )

    client = boto3.client(
        'sqs',
        region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
        config=boto_config
    )

    return client.send_message(
        QueueUrl=sqs_url,
        MessageBody=json.dumps(message_body, ensure_ascii=False)
    )
