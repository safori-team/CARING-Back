"""
AI 서버 통신 클라이언트
httpx를 사용하여 app_ai/main.py의 엔드포인트들을 호출
"""
import os
from typing import Dict, Any, Optional, List
import httpx
from fastapi import UploadFile
from io import BytesIO
import logging

from ..exceptions import InternalServerException, ExternalAPIException

logger = logging.getLogger(__name__)

# AI 서버 URL (환경 변수에서 읽기, 기본값: http://localhost:8001)
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8001")

# 타임아웃 설정 (초)
TIMEOUT = 300.0  # 5분


class AIClient:
    """AI 서버 통신 클라이언트"""
    
    def __init__(self, base_url: str = AI_SERVER_URL, timeout: float = TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    async def analyze_voice(self, file: UploadFile, voice_id: Optional[int] = None) -> Dict[str, Any]:
        """
        음성 파일을 AI 서버에 전송하여 STT 텍스트와 감정 분석 점수를 받아옴
        voice_id가 제공되면 AI 서버에서 DB에 직접 저장
        
        Args:
            file: 업로드된 음성 파일
            voice_id: voice_id (제공 시 AI 서버에서 DB 저장)
            
        Returns:
            Dict: {
                "success": bool,
                "stt_text": str,
                "emotion_scores": Dict[str, float],
                "top_emotion": str,
                "confidence": float,
                "error": Optional[str]
            }
        """
        try:
            # 파일 읽기
            file_content = await file.read()
            await file.seek(0)  # 파일 포인터 초기화
            
            # multipart/form-data로 전송
            files = {
                'file': (file.filename, BytesIO(file_content), file.content_type or 'audio/m4a')
            }
            
            # Form 데이터에 voice_id 추가
            data = {}
            if voice_id is not None:
                data['voice_id'] = voice_id
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/ai/analyze-voice",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"AI 서버 analyze_voice 오류: {error_msg}")
                    raise ExternalAPIException(
                        status_code=500,
                        message=f"AI 서버 분석 실패: {error_msg}"
                    )
                
                return result
                
        except httpx.TimeoutException as e:
            logger.error(f"AI 서버 analyze_voice 타임아웃: {e}")
            raise ExternalAPIException(
                status_code=504,
                message="AI 서버 응답 시간 초과"
            )
        except httpx.RequestError as e:
            logger.error(f"AI 서버 analyze_voice 연결 오류: {e}")
            raise ExternalAPIException(
                status_code=503,
                message=f"AI 서버 연결 실패: {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"AI 서버 analyze_voice HTTP 오류: {e.response.status_code} - {e.response.text}")
            raise ExternalAPIException(
                status_code=e.response.status_code,
                message=f"AI 서버 HTTP 오류: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"AI 서버 analyze_voice 예외: {e}", exc_info=True)
            raise InternalServerException(f"AI 서버 통신 중 오류 발생: {str(e)}")
    
    async def analyze_text(self, text: str, language_code: str = "ko", voice_id: Optional[int] = None) -> Dict[str, Any]:
        """
        텍스트를 AI 서버에 전송하여 감정 및 엔티티 분석 결과를 받아옴
        voice_id가 제공되면 AI 서버에서 DB에 직접 저장
        
        Args:
            text: 분석할 텍스트
            language_code: 언어 코드 (기본값: ko)
            voice_id: voice_id (제공 시 AI 서버에서 DB 저장)
            
        Returns:
            Dict: {
                "success": bool,
                "sentiment": Dict[str, float],
                "entities": List[Dict[str, Any]],
                "error": Optional[str]
            }
        """
        try:
            payload = {
                "text": text,
                "language_code": language_code
            }
            
            # voice_id가 제공되면 JSON body에 추가
            if voice_id is not None:
                payload['voice_id'] = voice_id
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/ai/analyze-text",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"AI 서버 analyze_text 오류: {error_msg}")
                    raise ExternalAPIException(
                        status_code=500,
                        message=f"AI 서버 분석 실패: {error_msg}"
                    )
                
                return result
                
        except httpx.TimeoutException as e:
            logger.error(f"AI 서버 analyze_text 타임아웃: {e}")
            raise ExternalAPIException(
                status_code=504,
                message="AI 서버 응답 시간 초과"
            )
        except httpx.RequestError as e:
            logger.error(f"AI 서버 analyze_text 연결 오류: {e}")
            raise ExternalAPIException(
                status_code=503,
                message=f"AI 서버 연결 실패: {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"AI 서버 analyze_text HTTP 오류: {e.response.status_code} - {e.response.text}")
            raise ExternalAPIException(
                status_code=e.response.status_code,
                message=f"AI 서버 HTTP 오류: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"AI 서버 analyze_text 예외: {e}", exc_info=True)
            raise InternalServerException(f"AI 서버 통신 중 오류 발생: {str(e)}")
    
    async def generate_summary(
        self,
        user_name: str,
        data_type: str,
        weekly_data: Optional[Dict[str, List[str]]] = None,
        monthly_data: Optional[Dict[str, int]] = None
    ) -> str:
        """
        AI 서버에 요약문 생성 요청
        
        Args:
            user_name: 사용자 이름
            data_type: "weekly" 또는 "monthly"
            weekly_data: 주간 데이터 (data_type="weekly"일 때)
            monthly_data: 월간 데이터 (data_type="monthly"일 때)
            
        Returns:
            str: 생성된 요약문
        """
        try:
            payload = {
                "user_name": user_name,
                "data_type": data_type
            }
            
            if data_type == "weekly":
                if weekly_data is None:
                    raise ValueError("weekly_data is required when data_type is 'weekly'")
                payload["weekly_data"] = weekly_data
            elif data_type == "monthly":
                if monthly_data is None:
                    raise ValueError("monthly_data is required when data_type is 'monthly'")
                payload["monthly_data"] = monthly_data
            else:
                raise ValueError("data_type must be 'weekly' or 'monthly'")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/ai/generate-summary",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"AI 서버 generate_summary 오류: {error_msg}")
                    raise ExternalAPIException(
                        status_code=500,
                        message=f"AI 서버 요약 생성 실패: {error_msg}"
                    )
                
                summary = result.get("summary", "")
                if not summary:
                    raise InternalServerException("AI 서버에서 빈 요약문을 반환했습니다")
                
                return summary
                
        except httpx.TimeoutException as e:
            logger.error(f"AI 서버 generate_summary 타임아웃: {e}")
            raise ExternalAPIException(
                status_code=504,
                message="AI 서버 응답 시간 초과"
            )
        except httpx.RequestError as e:
            logger.error(f"AI 서버 generate_summary 연결 오류: {e}")
            raise ExternalAPIException(
                status_code=503,
                message=f"AI 서버 연결 실패: {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"AI 서버 generate_summary HTTP 오류: {e.response.status_code} - {e.response.text}")
            raise ExternalAPIException(
                status_code=e.response.status_code,
                message=f"AI 서버 HTTP 오류: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"AI 서버 generate_summary 예외: {e}", exc_info=True)
            if isinstance(e, (ValueError, InternalServerException, ExternalAPIException)):
                raise
            raise InternalServerException(f"AI 서버 통신 중 오류 발생: {str(e)}")


# 전역 인스턴스
_ai_client: Optional[AIClient] = None


def get_ai_client() -> AIClient:
    """AI 클라이언트 싱글톤 인스턴스 반환"""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client
