#!/usr/bin/env python3
"""CARE 역할 사용자 목록 확인"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import get_db
from app.models import User

def list_care_users():
    """CARE 역할 사용자 목록 출력"""
    db = next(get_db())
    try:
        care_users = db.query(User).filter(User.role == 'CARE').all()
        
        if not care_users:
            print("❌ CARE 역할 사용자가 없습니다.")
            return
        
        print(f"✅ CARE 역할 사용자 목록 ({len(care_users)}명):")
        print("-" * 80)
        for user in care_users:
            print(f"   - username: {user.username}")
            print(f"     name: {user.name}")
            print(f"     connecting_user_code: {user.connecting_user_code}")
            if user.connecting_user_code:
                connected = db.query(User).filter(User.username == user.connecting_user_code).first()
                if connected:
                    print(f"     → 연결된 피보호자: {connected.name} ({connected.username})")
                else:
                    print(f"     → ⚠️ 연결된 피보호자를 찾을 수 없음")
            print()
    finally:
        db.close()

if __name__ == "__main__":
    list_care_users()

