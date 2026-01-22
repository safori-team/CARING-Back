"""
AI 분석 라우터
- 음성 분석, 텍스트 분석, 요약문 생성 엔드포인트
- voice_id가 제공되면 DB에 직접 저장 (AI 서버 책임)
"""
import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..schemas import (
    AnalyzeTextRequest,
    AnalyzeTextResponse,
    GenerateSummaryRequest,
    GenerateSummaryResponse,
    AnalyzeVoiceResponse
)
from ..services.stt_service import transcribe_voice
from ..services.emotion_service import analyze_voice_emotion
from ..services.nlp_service import analyze_text_sentiment, analyze_text_entities
from ..database import get_db
from ..repositories.voice_repo import (
    create_voice_content,
    create_voice_analyze,
    ensure_job_row,
    mark_text_done,
    mark_audio_done
)
from openai import OpenAI

router = APIRouter(prefix="/ai", tags=["AI"])


def _get_openai_client():
    """OpenAI 클라이언트 생성 (env에서 키 로드)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured in environment")
    return OpenAI(api_key=api_key)


def _call_openai(messages: List[Dict[str, str]], model: str | None = None) -> str:
    """Chat Completions 호출 래퍼 (analysis_service.py의 로직 참조)"""
    client = _get_openai_client()
    use_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=use_model,
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


@router.post("/analyze-voice", response_model=AnalyzeVoiceResponse)
async def analyze_voice(
    file: UploadFile = File(..., description="음성 파일"),
    voice_id: Optional[int] = Form(None, description="voice_id (제공 시 DB에 저장)"),
    db: Session = Depends(get_db)
):
    """
    음성 파일을 받아 감정 분석 점수와 STT 텍스트를 한꺼번에 반환
    
    - STT: 음성을 텍스트로 변환
    - 감정 분석: 음성 감정 점수 추출
    - voice_id가 제공되면 DB에 직접 저장 (VoiceAnalyze, VoiceJobProcess)
    """
    try:
        # 1. STT 처리
        stt_result = transcribe_voice(file)
        if stt_result.get("error"):
            return AnalyzeVoiceResponse(
                success=False,
                error=f"STT 오류: {stt_result.get('error')}"
            )
        
        stt_text = stt_result.get("transcript", "")
        
        # 파일 포인터를 처음으로 되돌려서 감정 분석에서도 사용 가능하도록
        file.file.seek(0)
        
        # 2. 감정 분석 처리
        emotion_result = analyze_voice_emotion(file)
        if emotion_result.get("error"):
            # STT는 성공했지만 감정 분석이 실패한 경우, STT 텍스트는 반환
            return AnalyzeVoiceResponse(
                success=False,
                stt_text=stt_text,
                error=f"감정 분석 오류: {emotion_result.get('error')}"
            )
        
        emotion_scores = emotion_result.get("emotion_scores", {})
        top_emotion = emotion_result.get("top_emotion") or emotion_result.get("emotion")
        confidence = emotion_result.get("confidence", 0.0)
        
        # 3. voice_id가 제공되면 DB에 저장
        if voice_id is not None:
            try:
                # VoiceAnalyze 저장
                def to_bps(v: float) -> int:
                    try:
                        return max(0, min(10000, int(round(float(v) * 10000))))
                    except Exception:
                        return 0
                
                probs = emotion_scores
                happy = to_bps(probs.get("happy", probs.get("happiness", 0)))
                sad = to_bps(probs.get("sad", probs.get("sadness", 0)))
                neutral = to_bps(probs.get("neutral", 0))
                angry = to_bps(probs.get("angry", probs.get("anger", 0)))
                fear_prob = (
                    float(probs.get("fear", 0) or 0)
                    + float(probs.get("anxiety", 0) or 0)
                    + float(probs.get("fearful", 0) or 0)
                )
                fear = to_bps(fear_prob)
                surprise = to_bps(probs.get("surprise", probs.get("surprised", 0)))
                
                # 합계가 0이면 중립 100%
                total_raw = happy + sad + neutral + angry + fear + surprise
                if total_raw == 0:
                    happy, sad, neutral, angry, fear, surprise = 0, 0, 10000, 0, 0, 0
                else:
                    # 비율 보정(라운딩 후 합 10000로 맞춤)
                    scale = 10000 / float(total_raw)
                    vals = {
                        "happy": int(round(happy * scale)),
                        "sad": int(round(sad * scale)),
                        "neutral": int(round(neutral * scale)),
                        "angry": int(round(angry * scale)),
                        "fear": int(round(fear * scale)),
                        "surprise": int(round(surprise * scale)),
                    }
                    diff = 10000 - sum(vals.values())
                    if diff != 0:
                        key_max = max(vals, key=lambda k: vals[k])
                        vals[key_max] = max(0, min(10000, vals[key_max] + diff))
                    happy, sad, neutral, angry, fear, surprise = (
                        vals["happy"], vals["sad"], vals["neutral"], vals["angry"], vals["fear"], vals["surprise"]
                    )
                
                create_voice_analyze(
                    session=db,
                    voice_id=voice_id,
                    happy_bps=happy,
                    sad_bps=sad,
                    neutral_bps=neutral,
                    angry_bps=angry,
                    fear_bps=fear,
                    surprise_bps=surprise,
                    top_emotion=top_emotion,
                    top_confidence_bps=to_bps(confidence),
                    model_version="ai-server"
                )
                
                # audio_done 표시
                mark_audio_done(db, voice_id)
                
            except Exception as e:
                # DB 저장 실패해도 분석 결과는 반환
                print(f"DB 저장 실패 (voice_id={voice_id}): {e}")
                db.rollback()
        
        return AnalyzeVoiceResponse(
            success=True,
            stt_text=stt_text,
            emotion_scores=emotion_scores,
            top_emotion=top_emotion,
            confidence=confidence
        )
        
    except Exception as e:
        return AnalyzeVoiceResponse(
            success=False,
            error=f"처리 중 오류 발생: {str(e)}"
        )


@router.post("/analyze-text", response_model=AnalyzeTextResponse)
async def analyze_text(
    request: AnalyzeTextRequest,
    db: Session = Depends(get_db)
):
    """
    구글 NLP를 사용하여 텍스트 감정 및 엔티티를 분석
    
    - 감정 분석: 텍스트의 감정 점수 (score, magnitude)
    - 엔티티 분석: 텍스트에서 추출한 엔티티 목록
    - voice_id가 제공되면 DB에 직접 저장 (VoiceContent, VoiceJobProcess)
    """
    try:
        text = request.text
        language_code = request.language_code
        voice_id = request.voice_id
        
        if not text or not text.strip():
            return AnalyzeTextResponse(
                success=False,
                error="텍스트가 비어있습니다"
            )
        
        # 1. 감정 분석
        sentiment_result = analyze_text_sentiment(text, language_code)
        if sentiment_result.get("error"):
            return AnalyzeTextResponse(
                success=False,
                error=f"감정 분석 오류: {sentiment_result.get('error')}"
            )
        
        sentiment = sentiment_result.get("sentiment", {})
        
        # 2. 엔티티 분석
        entities_result = analyze_text_entities(text, language_code)
        entities = []
        if not entities_result.get("error"):
            entities = entities_result.get("entities", [])
        
        # 3. voice_id가 제공되면 DB에 저장
        if voice_id is not None:
            try:
                score_bps = int(sentiment.get("score", 0) * 10000) if sentiment.get("score") else None
                magnitude_x1000 = int(sentiment.get("magnitude", 0) * 1000) if sentiment.get("magnitude") else None
                
                create_voice_content(
                    session=db,
                    voice_id=voice_id,
                    content=text,
                    score_bps=score_bps,
                    magnitude_x1000=magnitude_x1000,
                    locale="ko-KR",
                    provider="google",
                    confidence_bps=0  # STT confidence는 별도로 관리되지 않음
                )
                
                # text_done 표시
                mark_text_done(db, voice_id)
                
            except Exception as e:
                # DB 저장 실패해도 분석 결과는 반환
                print(f"DB 저장 실패 (voice_id={voice_id}): {e}")
                db.rollback()
        
        return AnalyzeTextResponse(
            success=True,
            sentiment={
                "score": sentiment.get("score", 0.0),
                "magnitude": sentiment.get("magnitude", 0.0)
            },
            entities=entities
        )
        
    except Exception as e:
        return AnalyzeTextResponse(
            success=False,
            error=f"처리 중 오류 발생: {str(e)}"
        )


@router.post("/generate-summary", response_model=GenerateSummaryResponse)
async def generate_summary(request: GenerateSummaryRequest):
    """
    analysis_service.py의 _call_openai 로직을 사용하여 주간/월간 요약문을 생성
    
    - 주간 요약: 날짜별 대표 감정 목록을 바탕으로 주간 감정 추세 요약
    - 월간 요약: 감정별 빈도수를 바탕으로 월간 감정 경향 요약
    - DB 접근 없이 순수 계산만 수행
    """
    try:
        user_name = request.user_name
        data_type = request.data_type
        
        if data_type not in ["weekly", "monthly"]:
            return GenerateSummaryResponse(
                success=False,
                error="data_type은 'weekly' 또는 'monthly'여야 합니다"
            )
        
        # 프롬프트 구성 (analysis_service.py의 로직 참조)
        messages: List[Dict[str, str]]
        
        if data_type == "weekly":
            if not request.weekly_data:
                return GenerateSummaryResponse(
                    success=False,
                    error="주간 데이터(weekly_data)가 필요합니다"
                )
            
            # 주간 프롬프트 구성
            lines = [f"대상 사용자: {user_name}"]
            by_day = request.weekly_data
            if not by_day:
                lines.append("최근 7일 동안 감정 분석 데이터가 없습니다.")
            else:
                lines.append("최근 7일 간 날짜별 대표 감정 목록입니다.")
                for day in sorted(by_day.keys()):
                    vals = ", ".join(by_day[day]) if by_day[day] else "(없음)"
                    lines.append(f"- {day}: {vals}")
            
            system = {
                "role": "system",
                "content": (
                    "너는 노년층 혹은 장애인 케어 서비스의 감정 코치다. 한국어로 공감적이고 자연스럽게, 1~3문장으로 "
                    "주간 감정 추세를 반드시 요약해라. 데이터가 적어도 관찰 가능한 내용을 바탕으로 요약을 제공해야 한다. "
                    "추측하지 말고 관찰적인 표현만 사용하고, 과장 없이 사실 중심으로 서술해라. "
                    "조언은 최소화하고 관찰 결과에 집중해라. 또한 early/mid/late(초반/중반/후반) 시기를 구분하여 감정 흐름이 바뀌는 지점을 틀림없이 분석하여라. "
                    "감정 라벨은 반드시 {happy, sad, neutral, angry, anxiety, surprise} 집합만 사용한다. fear는 anxiety로 매핑할것.\n\n"
                    "좋은 예시:\n"
                    "- '주 초반에는 즐겁고 안정적인 날들이 많았지만, 목요일부터 감정상태가 급격히 나빠지고 있습니다.'\n"
                    "- '이번 주는 전체적으로 즐거운 감정 혹은 안정된 상태를 유지하고 있어요.'\n"
                    "- '최근 7일 동안 감정 분석 데이터가 없었습니다.'"
                ),
            }
            user = {
                "role": "user",
                "content": (
                    "다음 날짜별 감정 목록을 바탕으로 주간 감정 추세를 한 문단(1~3문장)으로 요약해줘. "
                    "초반/중반/후반 흐름을 구분하고, 감정 매핑 오류가 없도록 분노와 불안을 혼동하지 마. 불안은 anxiety로 표기해. "
                    "데이터가 적어도 관찰 가능한 내용을 바탕으로 반드시 요약을 제공해줘.\n\n" + "\n".join(lines)
                ),
            }
            messages = [system, user]
            
        else:  # monthly
            if not request.monthly_data:
                return GenerateSummaryResponse(
                    success=False,
                    error="월간 데이터(monthly_data)가 필요합니다"
                )
            
            # 월간 프롬프트 구성
            from collections import Counter
            ordered_labels = ["happy", "sad", "neutral", "angry", "anxiety", "surprise"]
            counts = request.monthly_data
            norm_counts = {k: int(counts.get(k, 0)) for k in ordered_labels}
            total = sum(norm_counts.values()) or 1
            pct = {k: int(round(v * 100.0 / total)) for k, v in norm_counts.items()}
            ranked = sorted(norm_counts.items(), key=lambda kv: kv[1], reverse=True)
            items = ", ".join([f"{k}:{v}" for k, v in ranked]) if counts else "(데이터 없음)"
            
            system = {
                "role": "system",
                "content": (
                    "너는 노년층 혹은 장애인 케어 서비스의 감정 코치다. 한국어로 공감적이고 자연스럽게, 1~3문장으로 "
                    "월간 감정 빈도 특성을 반드시 요약해라. 데이터가 적어도 관찰 가능한 내용을 바탕으로 요약을 제공해야 한다. "
                    "추측하지 말고 관찰적인 표현만 사용하고, 과장 없이 사실 중심으로 서술해라. "
                    "조언은 최소화하고 관찰 결과에 집중해라. 감정 라벨은 반드시 {happy, sad, neutral, angry, anxiety, surprise}만 사용하고, fear는 anxiety로 해석한다. "
                    "다음 규칙을 반드시 준수하라: (1) 수치(rank)에 맞게 기술하고, 상위 감정들만 강조하라. (2) '상대적으로 높다/많다'라는 표현은 해당 감정의 빈도가 같은 달 내 다른 감정보다 순위가 높거나, 상위권(1~2위)이며 비율 차이가 10%p 이내일 때만 사용하라. "
                    "(3) 슬픔(sad)과 불안(anxiety), 분노(angry)를 혼동하지 말고, 각 감정명은 정확히 표기하라. (4) 데이터가 적을 경우 과장하지 말고 '일부 확인'과 같은 표현을 사용하라.\n\n"
                    "좋은 예시:\n"
                    "- '10월은 평온하고 안정적인 마음으로 시작하셨네요! 다만, 슬픔, 불안과 같은 감정들이 일부 확인되는것으로 보입니다.'\n"
                    "- '이번 달에는 화가 나는 감정이 다소 자주 나타났습니다. 이는 일상에서의 스트레스나 불만이 일부 확인된 것으로 보입니다.'\n"
                    "- '이번 달에는 감정 분석 데이터가 없었습니다.'"
                ),
            }
            user = {
                "role": "user",
                "content": (
                    f"대상 사용자: {user_name}\n"
                    f"총합: {total}건\n"
                    f"정렬(내림차순): {items}\n"
                    f"백분율: happy={pct['happy']}%, sad={pct['sad']}%, neutral={pct['neutral']}%, angry={pct['angry']}%, anxiety={pct['anxiety']}%, surprise={pct['surprise']}%\n"
                    "위의 수치에 정확히 기반하여 월간 감정 경향을 1~3문장으로 요약해줘. 순위/비율과 모순되는 표현은 사용하지 마."
                ),
            }
            messages = [system, user]
        
        # OpenAI 호출
        summary = _call_openai(messages)
        
        return GenerateSummaryResponse(
            success=True,
            summary=summary
        )
        
    except Exception as e:
        return GenerateSummaryResponse(
            success=False,
            error=f"처리 중 오류 발생: {str(e)}"
        )
