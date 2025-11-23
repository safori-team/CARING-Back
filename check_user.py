#!/usr/bin/env python3
"""사용자 정보 확인 스크립트"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import SessionLocal, get_db
from app.models import User

def check_user(username: str):
    """사용자 정보 확인"""
    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            print(f"❌ 사용자 '{username}'를 찾을 수 없습니다.")
            return
        
        print(f"✅ 사용자 정보:")
        print(f"   - user_id: {user.user_id}")
        print(f"   - username: {user.username}")
        print(f"   - user_code: {user.user_code}")
        print(f"   - name: {user.name}")
        print(f"   - role: {user.role}")
        print(f"   - connecting_user_code: {user.connecting_user_code}")
        
        if user.role != 'CARE':
            print(f"\n⚠️  경고: 사용자 역할이 'CARE'가 아닙니다. (현재: '{user.role}')")
        
        if not user.connecting_user_code:
            print(f"\n⚠️  경고: 연결된 피보호자 코드가 설정되어 있지 않습니다.")
        else:
            # 연결된 사용자 확인
            connected_user = db.query(User).filter(User.username == user.connecting_user_code).first()
            if connected_user:
                print(f"\n✅ 연결된 피보호자:")
                print(f"   - username: {connected_user.username}")
                print(f"   - name: {connected_user.name}")
                print(f"   - role: {connected_user.role}")
            else:
                print(f"\n❌ 연결된 피보호자 '{user.connecting_user_code}'를 찾을 수 없습니다.")
                
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python check_user.py <username>")
        sys.exit(1)
    
    check_user(sys.argv[1])

