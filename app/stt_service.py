import io
import tempfile
import os
import logging
from typing import Dict, Any, Optional

from google.cloud import speech
from google.oauth2 import service_account
import librosa
import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


class GoogleSTTService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Google Cloud Speech-to-Text 클라이언트 초기화"""
        try:
            # 환경변수에서 서비스 계정 키 파일 경로 가져오기
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if credentials_path and os.path.exists(credentials_path):
                # 서비스 계정 키 파일로 인증
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.client = speech.SpeechClient(credentials=credentials)
            else:
                # 기본 인증 (환경변수 GOOGLE_APPLICATION_CREDENTIALS 설정됨)
                self.client = speech.SpeechClient()
                
        except Exception as e:
            print(f"Google STT 클라이언트 초기화 실패: {e}")
            self.client = None
    
    def transcribe_audio(self, audio_file, language_code: str = "ko-KR", timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        음성 파일을 텍스트로 변환합니다.
        
        Args:
            audio_file: 업로드된 음성 파일 (FastAPI UploadFile)
            language_code: 언어 코드 (기본값: ko-KR)
            
        Returns:
            Dict: STT 결과
        """
        if not self.client:
            return {
                "error": "Google STT 클라이언트가 초기화되지 않았습니다",
                "transcript": "",
                "confidence": 0.0
            }
        
        tmp_file_path: Optional[str] = None
        try:
            # 업로드 확장자에 맞춰 임시 파일로 저장 (기본: .wav)
            orig_name = getattr(audio_file, "filename", "") or ""
            _, ext = os.path.splitext(orig_name)
            suffix = ext if ext.lower() in [".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".caf"] else ".wav"

            logger.info(
                "[GoogleSTT] 입력 파일 준비 시작 - orig_name=%s, suffix=%s, language=%s",
                orig_name,
                suffix,
                language_code,
            )

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                content = audio_file.file.read()
                audio_file.file.seek(0)
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            logger.info(
                "[GoogleSTT] 임시 파일 생성 완료 - path=%s, size=%d bytes",
                tmp_file_path,
                len(content) if content is not None else -1,
            )
            
            # 오디오 파일 로드 및 전처리 (견고한 로더)
            def robust_load(path: str, target_sr: int = 16000):
                """soundfile 우선, 실패 시 librosa로 폴백. 모노, 정규화 반환."""
                try:
                    data, sr = sf.read(path, always_2d=True, dtype="float32")  # (N, C)
                    logger.debug("[GoogleSTT] soundfile 로드 성공 - sr=%d, shape=%s", sr, getattr(data, "shape", None))
                    if data.ndim == 2 and data.shape[1] > 1:
                        data = data.mean(axis=1)  # mono
                    else:
                        data = data.reshape(-1)
                    if sr != target_sr:
                        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                    return data, sr
                except Exception as e:
                    logger.warning(
                        "[GoogleSTT] soundfile 로드 실패, librosa 폴백 시도 - path=%s, error=%s",
                        path,
                        str(e),
                    )
                    # 폴백: librosa가 내부적으로 audioread/ffmpeg 사용
                    y, sr = librosa.load(path, sr=target_sr, mono=True)
                    logger.debug(
                        "[GoogleSTT] librosa 폴백 로드 성공 - sr=%d, len=%d",
                        sr,
                        len(y) if y is not None else -1,
                    )
                    return y.astype("float32"), sr

            logger.info("[GoogleSTT] 오디오 로드/전처리 시작 - path=%s", tmp_file_path)
            audio_data, sample_rate = robust_load(tmp_file_path, 16000)
            logger.info(
                "[GoogleSTT] 오디오 로드/전처리 완료 - sample_rate=%d, samples=%d",
                sample_rate,
                len(audio_data) if audio_data is not None else -1,
            )
            
            # 오디오 데이터를 bytes로 변환
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_bytes = (audio_data * 32767).astype('int16').tobytes()
            logger.info(
                "[GoogleSTT] 오디오 직렬화 완료 - byte_length=%d",
                len(audio_bytes),
            )
            
            # Google Cloud Speech-to-Text 요청 구성
            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                audio_channel_count=1,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                model="latest_long",
                enable_spoken_emojis=False,
            )
            logger.info(
                "[GoogleSTT] STT 요청 구성 완료 - sample_rate=%d, model=%s, enhanced=%s",
                sample_rate,
                "latest_long",
                "enabled",
            )
            
            # STT 요청 실행 (타임아웃 적용)
            logger.info("[GoogleSTT] STT 요청 시작 - timeout=%s, audio_duration=%.2fs", timeout_seconds, len(audio_data) / sample_rate)
            response = self.client.recognize(config=config, audio=audio, timeout=timeout_seconds)
            logger.info(
                "[GoogleSTT] STT 응답 수신 - results=%d",
                len(response.results),
            )
            
            # 결과 처리 - 모든 results를 합쳐서 전체 텍스트 생성
            if response.results:
                # 모든 결과의 transcript를 합침 (전체 오디오 텍스트)
                full_transcript = " ".join(
                    result.alternatives[0].transcript 
                    for result in response.results
                )
                # 평균 confidence 계산
                confidences = [
                    result.alternatives[0].confidence 
                    for result in response.results 
                    if result.alternatives[0].confidence > 0
                ]
                # TODO : 추후 Confidence가 낮으면 인식에서 제외하는 로직 추가
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                logger.info(
                    "[GoogleSTT] STT 결과 파싱 완료 - transcript_length=%d, num_results=%d, avg_confidence=%.3f",
                    len(full_transcript),
                    len(response.results),
                    avg_confidence,
                )
                return {
                    "transcript": full_transcript,
                    "confidence": avg_confidence,
                    "language_code": language_code,
                    "audio_duration": len(audio_data) / sample_rate,
                    "sample_rate": sample_rate
                }
            else:
                logger.warning("[GoogleSTT] STT 결과 없음 - 빈 results")
                return {
                    "error": "음성을 인식할 수 없습니다",
                    "transcript": "",
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error("[GoogleSTT] STT 처리 중 예외 발생 - error=%s", str(e), exc_info=True)
            return {
                "error": f"STT 처리 중 오류 발생: {str(e)}",
                "transcript": "",
                "confidence": 0.0
            }
        finally:
            # 임시 파일 정리
            if tmp_file_path:
                try:
                    os.unlink(tmp_file_path)
                    logger.debug("[GoogleSTT] 임시 파일 삭제 완료 - path=%s", tmp_file_path)
                except OSError as e:
                    logger.warning(
                        "[GoogleSTT] 임시 파일 삭제 실패 - path=%s, error=%s",
                        tmp_file_path,
                        str(e),
                    )


# 전역 인스턴스
stt_service = GoogleSTTService()


def transcribe_voice(audio_file, language_code: str = "ko-KR", timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """음성을 텍스트로 변환하는 함수"""
    return stt_service.transcribe_audio(audio_file, language_code, timeout_seconds)
