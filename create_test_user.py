#!/usr/bin/env python3
"""테스트용 사용자 생성 스크립트"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import SessionLocal
from app.auth_service import get_auth_service
from datetime import datetime
import secrets
import string

def generate_random_string(length: int = 8) -> str:
    """랜덤 문자열 생성"""
    characters = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def create_test_user(
    name: str = None,
    username: str = None,
    password: str = "test1234",
    role: str = "USER",
    birthdate: str = None
):
    """테스트용 사용자 생성"""
    db = SessionLocal()
    try:
        auth_service = get_auth_service(db)
        
        # 기본값 설정
        if not name:
            name = f"테스트사용자_{generate_random_string(4)}"
        if not username:
            username = f"test_user_{generate_random_string(8)}"
        if not birthdate:
            # 기본 생년월일: 1990.01.01
            birthdate = "1990.01.01"
        
        print(f"사용자 생성 중...")
        print(f"  이름: {name}")
        print(f"  아이디: {username}")
        print(f"  비밀번호: {password}")
        print(f"  역할: {role}")
        print(f"  생년월일: {birthdate}")
        
        result = auth_service.signup(
            name=name,
            birthdate=birthdate,
            username=username,
            password=password,
            role=role,
            connecting_user_code=None
        )
        
        if result["success"]:
            print(f"\n✅ 사용자 생성 성공!")
            print(f"  user_code: {result['user_code']}")
            print(f"  username: {result['username']}")
            print(f"  name: {result['name']}")
            print(f"  role: {result['role']}")
            print(f"\n📝 사용 예시:")
            print(f"  user_id (username): {result['username']}")
            return result
        else:
            print(f"\n❌ 사용자 생성 실패: {result.get('error')}")
            return None
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="테스트용 사용자 생성")
    parser.add_argument("--name", type=str, help="사용자 이름")
    parser.add_argument("--username", type=str, help="아이디 (username)")
    parser.add_argument("--password", type=str, default="test1234", help="비밀번호 (기본값: test1234)")
    parser.add_argument("--role", type=str, choices=["USER", "CARE"], default="USER", help="역할 (기본값: USER)")
    parser.add_argument("--birthdate", type=str, help="생년월일 (YYYY.MM.DD 형식)")
    
    args = parser.parse_args()
    
    create_test_user(
        name=args.name,
        username=args.username,
        password=args.password,
        role=args.role,
        birthdate=args.birthdate
    )

