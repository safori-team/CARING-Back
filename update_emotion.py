#!/usr/bin/env python3
"""voice_analyze 테이블의 top_emotion 업데이트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import get_db
from app.models import VoiceAnalyze

def update_emotion(voice_analyze_id: int, old_emotion: str, new_emotion: str):
    """감정 업데이트"""
    db = next(get_db())
    try:
        voice_analyze = db.query(VoiceAnalyze).filter(
            VoiceAnalyze.voice_analyze_id == voice_analyze_id
        ).first()
        
        if not voice_analyze:
            print(f"❌ voice_analyze_id={voice_analyze_id}를 찾을 수 없습니다.")
            return False
        
        if voice_analyze.top_emotion != old_emotion:
            print(f"⚠️  현재 감정이 '{voice_analyze.top_emotion}'입니다. (예상: '{old_emotion}')")
            print(f"   그대로 '{new_emotion}'로 변경하시겠습니까?")
        
        print(f"업데이트 전:")
        print(f"   - voice_analyze_id: {voice_analyze.voice_analyze_id}")
        print(f"   - voice_id: {voice_analyze.voice_id}")
        print(f"   - top_emotion: {voice_analyze.top_emotion}")
        
        voice_analyze.top_emotion = new_emotion
        db.commit()
        db.refresh(voice_analyze)
        
        print(f"\n✅ 업데이트 완료:")
        print(f"   - top_emotion: {voice_analyze.top_emotion}")
        
        return True
        
    except Exception as e:
        db.rollback()
        print(f"❌ 업데이트 실패: {str(e)}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    # voice_analyze_id=1, anxiety -> fear
    update_emotion(voice_analyze_id=1, old_emotion="anxiety", new_emotion="fear")








