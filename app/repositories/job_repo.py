from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from ..models import VoiceJobProcess, Voice, User
from ..services.composite_service import CompositeService


def _send_composite_completion_notification(session: Session, voice_id: int):
    """voice_composite 생성 완료 시 연결된 CARE 사용자에게 알림 발송 및 알림 기록 저장"""
    # 1. voice 조회
    voice = session.query(Voice).filter(Voice.voice_id == voice_id).first()
    if not voice:
        return
    
    # 2. USER 조회
    user = session.query(User).filter(User.user_id == voice.user_id).first()
    if not user or user.role != 'USER':
        return  # USER만 처리
    
    # 3. 연결된 CARE 사용자 찾기 (connecting_user_code = user.username인 CARE)
    care_user = session.query(User).filter(
        User.role == 'CARE',
        User.connecting_user_code == user.username
    ).first()
    
    if not care_user:
        return  # 연결된 CARE 사용자가 없으면 알림 발송 안 함
    
    # 4. voice_composite에서 top_emotion 조회
    from ..models import VoiceComposite, Notification
    voice_composite = session.query(VoiceComposite).filter(
        VoiceComposite.voice_id == voice_id
    ).first()
    top_emotion = voice_composite.top_emotion if voice_composite else None

    # 5. 알림 기록 생성 (DB 저장) - FCM 실패와 무관하게 반드시 저장
    try:
        notification = Notification(
            voice_id=voice_id,
            name=user.name,
            top_emotion=top_emotion
        )
        session.add(notification)
        session.commit()
        session.refresh(notification)

        import logging
        logging.info(
            f"Notification record created: notification_id={notification.notification_id}, voice_id={voice_id}"
        )
    except Exception as e:
        import logging
        logging.error(f"Failed to create notification record: {str(e)}")
        # Notification 생성 실패는 전체 프로세스 중단
        return

    # FCM 푸시 알림은 비활성화 상태이므로 DB Notification 저장까지만 수행


def ensure_job_row(session: Session, voice_id: int) -> VoiceJobProcess:
    row = session.query(VoiceJobProcess).filter(VoiceJobProcess.voice_id == voice_id).first()
    if not row:
        row = VoiceJobProcess(voice_id=voice_id, text_done=0, audio_done=0, locked=0)
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def mark_text_done(session: Session, voice_id: int) -> None:
    row = ensure_job_row(session, voice_id)
    row.text_done = 1
    session.commit()


def mark_audio_done(session: Session, voice_id: int) -> None:
    row = ensure_job_row(session, voice_id)
    row.audio_done = 1
    session.commit()


def try_aggregate(session: Session, voice_id: int) -> bool:
    """Try to aggregate when both tasks are done; use a simple DB lock flag to prevent race."""
    try:
        from ..performance_logger import get_performance_logger
        logger = get_performance_logger(voice_id)
        
        row = session.query(VoiceJobProcess).with_for_update().filter(VoiceJobProcess.voice_id == voice_id).first()
        if not row:
            return False
        if row.locked:
            return False
        if not (row.text_done and row.audio_done):
            return False

        row.locked = 1
        session.commit()
        try:
            logger.log_step("voice_composite 입력 시작", category="async")
            service = CompositeService(session)
            service.compute_and_save_composite(voice_id)
            logger.log_step("완료", category="async")
        except Exception:
            session.rollback()
            refreshed = session.query(VoiceJobProcess).with_for_update().filter(
                VoiceJobProcess.voice_id == voice_id
            ).first()
            if refreshed and refreshed.locked:
                refreshed.locked = 0
                session.commit()
            raise
        else:
            row.locked = 0
            session.commit()
        
        # voice_composite 생성 완료 → 연결된 CARE 사용자에게 알림 발송
        try:
            _send_composite_completion_notification(session, voice_id)
        except Exception as e:
            # 알림 실패는 로그만 남기고 전체 프로세스는 계속 진행
            import logging
            logging.error(f"Failed to send FCM notification for voice_id={voice_id}: {str(e)}")

        # 로그 파일 저장 및 정리
        logger.save_to_file()
        from ..performance_logger import clear_logger
        clear_logger(voice_id)
        
        return True
    except SQLAlchemyError:
        session.rollback()
        return False
