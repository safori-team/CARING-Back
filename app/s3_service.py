import os
from typing import List, Dict
from datetime import timedelta

import boto3  # type: ignore
from botocore.client import Config  # type: ignore


def get_s3_client():
    region = os.getenv("AWS_REGION", "ap-northeast-2")
    kwargs = {
        "region_name": region,
        "config": Config(signature_version="s3v4"),
    }
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
    return boto3.client("s3", **kwargs)


def get_presigned_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    """단일 S3 객체의 presigned URL 생성"""
    s3 = get_s3_client()
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        print(f"Failed to generate presigned URL for {key}: {e}")
        return ""


def list_bucket_objects_with_urls(bucket: str, prefix: str = "", expires_in: int = 3600) -> Dict[str, str]:
    """
    prefix로 시작하는 S3 객체들의 키와 presigned URL을 반환
    
    Args:
        bucket: S3 버킷 이름
        prefix: 객체 키 prefix
        expires_in: URL 만료 시간(초, 기본값 1시간)
        
    Returns:
        Dict[str, str]: {key: presigned_url, ...}
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    result: Dict[str, str] = {}
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            url = get_presigned_url(bucket, key, expires_in)
            result[key] = url
    
    return result


