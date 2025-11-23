import os
import json
import boto3
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from ..models import Voice, VoiceContent, VoiceComposite, User

logger = logging.getLogger(__name__)

def send_analysis_to_chatbot(db: Session, voice_id: int):
    """
    분석이 완료된 음성 데이터를 조회하여 챗봇 SQS로 이벤트를 전송합니다.
    """
    sqs_url = os.getenv("CHATBOT_SQS_URL")
    if not sqs_url:
        return

    try:
        # 필요한 데이터 조회 (Voice, Content, User, Composite + Questions)
        result = (
            db.query(Voice, VoiceContent, VoiceComposite, User)
            .join(VoiceContent, Voice.voice_id == VoiceContent.voice_id)
            .join(VoiceComposite, Voice.voice_id == VoiceComposite.voice_id)
            .join(User, Voice.user_id == User.user_id)
            .options(joinedload(Voice.questions))
            .filter(Voice.voice_id == voice_id)
            .first()
        )

        if not result:
            logger.warning(f"[ChatbotIntegration] Data not found for voice_id={voice_id}")
            return

        voice, content, composite, user = result

        # 추가 맥락 정보 추출
        question_text = voice.questions[0].content if voice.questions else None

        # 메시지 페이로드 구성
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
            # 긍/부정 수치도 필요하다면 추가
            "valence": composite.valence_x1000 / 1000.0,
            "arousal": composite.arousal_x1000 / 1000.0
        }

        message_body = {
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

        # 4. SQS 전송
        client = boto3.client(
            'sqs',
            region_name=os.getenv("AWS_REGION", "ap-northeast-2")
        )

        response = client.send_message(
            QueueUrl=sqs_url,
            MessageBody=json.dumps(message_body, ensure_ascii=False)
        )

        logger.info(f"[ChatbotIntegration] Sent to chatbot SQS. MsgID={response.get('MessageId')}")

    except Exception as e:
        logger.error(f"[ChatbotIntegration] Failed to send SQS: {e}", exc_info=True)
