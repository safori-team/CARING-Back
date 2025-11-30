import os
import asyncio
import uuid
import logging
from typing import Dict, Any
from io import BytesIO
from datetime import datetime
from sqlalchemy.orm import Session
import httpx
from fastapi import UploadFile

from ..services.va_fusion import fuse_VA
from ..nlp_service import analyze_text_sentiment
from ..emotion_service import analyze_voice_emotion
from ..stt_service import transcribe_voice
from ..s3_service import upload_fileobj, get_presigned_url
from ..auth_service import get_auth_service
from ..voice_service import get_voice_service
from ..exceptions import (
    ValidationException,
    InternalServerException,
    RuntimeException,
    NotFoundException,
    ExternalAPIException,
)


logger = logging.getLogger(__name__)

class AnalyzeChatService:
    """음성 파일 분석 및 chatbot API 전송 서비스"""

    def __init__(self, db: Session):
        self.db = db
        self.voice_service = get_voice_service(db)
    
    
    async def analyze_and_send(
        self,
        file: UploadFile,
        session_id: str,
        user_id: str,
        question: str
    ) -> Dict[str, Any]:
        """
        음성 파일을 분석하고 외부 chatbot API로 전송
        
        Args:
            file: 업로드된 음성 파일
            session_id: 세션 ID
            user_id: 사용자 ID
            question: 질문 내용
            
        Returns:
            Dict: 외부 API 응답
        """
        logger.info(f"[AnalyzeChatService][analyze_and_send] START session_id={session_id}, user_id={user_id}, filename={getattr(file, 'filename', 'N/A')}, content_type={getattr(file, 'content_type', 'N/A')}")

        content_type = getattr(file, "content_type", "") or ""

        filename = file.filename or "upload"
        lower_name = filename.lower()
        allowed_ext = (lower_name.endswith('.wav') or lower_name.endswith('.m4a'))
        allowed_ct = any(ct in content_type for ct in ("audio/wav", "audio/x-wav", "audio/m4a", "audio/x-m4a", "audio/mp4"))
        if not (allowed_ext or allowed_ct):
            return {
                "success": False,
                "message": "Only wav/m4a audio is allowed"
            }
        # 확장자 미포함/이상치인 경우 Content-Type 기반으로 보정
        if not '.' in filename or (not lower_name.endswith('.wav') and not lower_name.endswith('.m4a')):
            if "m4a" in content_type or "mp4" in content_type:
                filename = (filename.rsplit('.', 1)[0] if '.' in filename else filename) + ".m4a"
            else:
                filename = (filename.rsplit('.', 1)[0] if '.' in filename else filename) + ".wav"
        
        # 4. 파일 읽기 및 WAV 변환 (비동기로 처리하여 블로킹 방지)
        # 파일 포인터 초기화 후 읽기 (다른 STT 서비스와 동일한 패턴)
        await file.seek(0)
        file_content = await file.read()
        
        logger.info(f"[AnalyzeChatService][analyze_and_send] 파일 읽기 완료 - size={len(file_content)} bytes, filename={filename}")
        
        # 파일 크기 검증
        if len(file_content) < 100:
            logger.error(f"[AnalyzeChatService][analyze_and_send] 파일이 너무 작음 - size={len(file_content)} bytes")
            return {
                "success": False,
                "message": f"File too small: {len(file_content)} bytes"
            }
        
        # CPU 집약적 작업을 스레드 풀에서 실행
        wav_content, wav_filename = await asyncio.to_thread(
            self.voice_service._convert_to_wav, file_content, filename
        )

        s3_key = await asyncio.to_thread(
            self._upload_to_s3,
            wav_content,
            wav_filename,
            session_id,
            user_id,
            content_type
        )

        # S3 Key로 Presigned URL 생성
        bucket = os.getenv("S3_BUCKET_NAME")
        s3_url = None
        if bucket and s3_key:
            s3_url = get_presigned_url(bucket, s3_key, expires_in=3600 * 24 * 7)

        # 2. STT (음성 → 텍스트)
        content = await asyncio.to_thread(
            self._transcribe_audio,
            wav_content,
            wav_filename,
            content_type
        )

        # 3. 음성 감정 분석
        emotion_data = await asyncio.to_thread(
            self._analyze_emotion,
            wav_content,
            wav_filename,
            "audio/wav",
            content
        )

        # 4. 사용자 정보 조회
        user_name = self._get_user_name(user_id)

        # 5. 외부 API 호출 (s3_url 전달)
        return await self._send_to_chatbot(
            content=content,
            emotion=emotion_data,
            question=question,
            user_id=user_id,
            user_name=user_name,
            session_id=session_id,
            s3_url=s3_url
        )
    
    def _upload_to_s3(
        self,
        file_content: bytes,
        filename: str,
        session_id: str,
        user_id: str,
        content_type: str
    ) -> str:
        """S3에 파일 업로드"""
        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise InternalServerException("S3_BUCKET_NAME not configured")
        
        # S3 키 생성: {session_id}/{user_id}/{filename}
        s3_key = f"{session_id}/{user_id}/{filename}" if session_id and user_id else f"chat/{filename}"
        
        # S3 업로드
        file_obj = BytesIO(file_content)
        upload_fileobj(bucket=bucket, key=s3_key, fileobj=file_obj, content_type=content_type)
        
        return s3_key
    
    def _transcribe_audio(
        self,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> str:
        """STT로 음성을 텍스트로 변환"""

        # INFO 레벨은 기본 로거 설정에 따라 출력되지 않을 수 있으므로 WARNING으로 남겨 확실히 보이게 한다.
        logger.warning("AnalyzeChat STT 시작 - filename=%s, content_type=%s", filename, content_type)

        class FileWrapper:
            def __init__(self, content: bytes, filename: str):
                self.file = BytesIO(content)
                self.filename = filename
                self.content_type = "audio/m4a" if filename.endswith('.m4a') else "audio/wav"
        
        wrapped_file = FileWrapper(file_content, filename)
        stt_result = transcribe_voice(wrapped_file)
        
        if stt_result.get("error"):
            raise RuntimeException(f"STT failed: {stt_result.get('error')}")
        
        content = stt_result.get("transcript", "")
        if not content:
            raise ValidationException("STT result is empty")
        
        return content
    
    def _analyze_emotion(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        text_content: str
    ) -> Dict[str, Any]:
        """음성 및 텍스트 감정 분석"""
        
        class FileWrapper:
            def __init__(self, content: bytes, filename: str, content_type: str):
                self.file = BytesIO(content)
                self.filename = filename
                self.content_type = content_type
        
        wrapped_file = FileWrapper(file_content, filename, content_type)

        # 3-1. Audio 감정 분석
        wrapped_file.file.seek(0)
        emotion_result = analyze_voice_emotion(wrapped_file)
        if emotion_result.get("error"):
            raise RuntimeException(f"Emotion analysis failed: {emotion_result.get('error')}")
        
        audio_probs = emotion_result.get("emotion_scores", {})
        
        # 3-2. 텍스트 감정 분석
        text_sentiment = analyze_text_sentiment(text_content, language_code="ko")
        if text_sentiment.get("error"):
            raise RuntimeException(f"Text sentiment analysis failed: {text_sentiment.get('error')}")
        
        sentiment_data = text_sentiment.get("sentiment", {})
        text_score = sentiment_data.get("score", 0.0)  # [-1, 1]
        text_magnitude = sentiment_data.get("magnitude", 0.0)  # [0, +inf)
        
        # 3-3. VA Fusion으로 arousal, valence 계산
        va_result = fuse_VA(audio_probs, text_score, text_magnitude)
        
        # arousal, valence를 [0, 1] 범위로 변환 ([-1, 1] -> [0, 1])
        valence = (va_result.get("V_final", 0.0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        arousal = (va_result.get("A_final", 0.0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        
        # emotion details 구성 (bps를 [0, 1] 범위로 변환)
        per_emotion_bps = va_result.get("per_emotion_bps", {})
        details = {
            "angry": per_emotion_bps.get("angry", 0) / 10000.0,
            "anxiety": per_emotion_bps.get("fear", 0) / 10000.0,  # fear -> anxiety
            "happy": per_emotion_bps.get("happy", 0) / 10000.0,
            "neutral": per_emotion_bps.get("neutral", 0) / 10000.0,
            "sad": per_emotion_bps.get("sad", 0) / 10000.0,
            "surprise": per_emotion_bps.get("surprise", 0) / 10000.0,
        }
        
        top_emotion = va_result.get("top_emotion", "neutral")
        # fear -> anxiety 변환
        if top_emotion == "fear":
            top_emotion = "anxiety"
        
        top_confidence_bps = va_result.get("top_confidence_bps", 0)
        confidence = top_confidence_bps / 10000.0  # [0, 1]
        
        return {
            "arousal": round(arousal, 2),
            "confidence": round(confidence, 2),
            "details": {k: round(v, 2) for k, v in details.items()},
            "top_emotion": top_emotion,
            "valence": round(valence, 2)
        }
    
    def _get_user_name(self, user_id: str) -> str:
        """사용자 이름 조회"""
        if not user_id:
            raise ValidationException("user_id is required")
        
        auth_service = get_auth_service(self.db)
        user = auth_service.get_user_by_username(user_id)
        if not user:
            raise NotFoundException(f"User not found: {user_id}")
        
        return user.name
    
    async def _send_to_chatbot(
        self,
        content: str,
        emotion: Dict[str, Any],
        question: str,
        user_id: str,
        user_name: str,
        session_id: str,
        s3_url: str = None  # [추가]
    ) -> Dict[str, Any]:
        """외부 chatbot API로 전송"""

        # (endpoint: /chatbot/voice-reframing)
        request_payload = {
            "user_id": user_id,
            "session_id": session_id,
            "user_input": content,
            "user_name": user_name,
            "emotion": emotion,
            "s3_url": s3_url  # [추가] Chatbot 요청 바디에 포함
        }
        
        chatbot_url = os.getenv("CHATBOT_API_URL")
        
        if not chatbot_url:
            raise InternalServerException("CHATBOT_API_URL not configured")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    chatbot_url,
                    json=request_payload,
                    headers={"accept": "application/json", "Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            error_message = self._extract_external_error(exc)
            raise ExternalAPIException(status_code=exc.response.status_code, message=error_message)
        except httpx.RequestError as exc:
            raise ExternalAPIException(status_code=500, message=f"External API request failed: {str(exc)}")

    @staticmethod
    def _extract_external_error(exc: httpx.HTTPStatusError) -> str:
        """외부 API 예외 메시지를 문자열로 변환"""
        try:
            json_body = exc.response.json()
            if isinstance(json_body, dict) and "message" in json_body:
                return str(json_body["message"])
            return str(json_body)
        except Exception:
            return f"External API error: {exc.response.text}"


def get_analyze_chat_service(db: Session) -> AnalyzeChatService:
    """AnalyzeChatService 인스턴스 생성"""
    return AnalyzeChatService(db)
