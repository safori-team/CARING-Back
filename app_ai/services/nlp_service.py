import os
from typing import Dict, Any, List
from google.cloud import language_v1
from google.oauth2 import service_account


class GoogleNLPService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Google Cloud Natural Language API 클라이언트 초기화"""
        try:
            # 환경변수에서 서비스 계정 키 파일 경로 가져오기
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if credentials_path and os.path.exists(credentials_path):
                # 서비스 계정 키 파일로 인증
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.client = language_v1.LanguageServiceClient(credentials=credentials)
            else:
                # 기본 인증 (환경변수 GOOGLE_APPLICATION_CREDENTIALS 설정됨)
                self.client = language_v1.LanguageServiceClient()
                
        except Exception as e:
            print(f"Google NLP 클라이언트 초기화 실패: {e}")
            self.client = None
    
    def analyze_sentiment(self, text: str, language_code: str = "ko") -> Dict[str, Any]:
        """
        텍스트의 감정을 분석합니다.
        
        Args:
            text: 분석할 텍스트
            language_code: 언어 코드 (기본값: ko)
            
        Returns:
            Dict: 감정 분석 결과
        """
        if not self.client:
            return {
                "error": "Google NLP 클라이언트가 초기화되지 않았습니다",
                "sentiment": {"score": 0.0, "magnitude": 0.0},
                "sentences": []
            }
        
        try:
            # 문서 객체 생성
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language=language_code
            )
            
            # 감정 분석 실행
            response = self.client.analyze_sentiment(
                request={'document': document}
            )
            
            # 전체 문서 감정 점수
            document_sentiment = response.document_sentiment
            
            # 문장별 감정 분석
            sentences = []
            for sentence in response.sentences:
                sentences.append({
                    "text": sentence.text.content,
                    "sentiment_score": sentence.sentiment.score,
                    "sentiment_magnitude": sentence.sentiment.magnitude
                })
            
            return {
                "sentiment": {
                    "score": document_sentiment.score,
                    "magnitude": document_sentiment.magnitude
                },
                "sentences": sentences,
                "language_code": language_code
            }
            
        except Exception as e:
            return {
                "error": f"NLP 분석 중 오류 발생: {str(e)}",
                "sentiment": {"score": 0.0, "magnitude": 0.0},
                "sentences": []
            }
    
    def analyze_entities(self, text: str, language_code: str = "ko") -> Dict[str, Any]:
        """
        텍스트에서 엔티티를 추출합니다.
        
        Args:
            text: 분석할 텍스트
            language_code: 언어 코드 (기본값: ko)
            
        Returns:
            Dict: 엔티티 분석 결과
        """
        if not self.client:
            return {
                "error": "Google NLP 클라이언트가 초기화되지 않았습니다",
                "entities": []
            }
        
        try:
            # 문서 객체 생성
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language=language_code
            )
            
            # 엔티티 분석 실행
            response = self.client.analyze_entities(
                request={'document': document}
            )
            
            # 엔티티 정보 추출
            entities = []
            for entity in response.entities:
                entities.append({
                    "name": entity.name,
                    "type": entity.type_.name,
                    "salience": entity.salience,
                    "mentions": [mention.text.content for mention in entity.mentions]
                })
            
            return {
                "entities": entities,
                "language_code": language_code
            }
            
        except Exception as e:
            return {
                "error": f"엔티티 분석 중 오류 발생: {str(e)}",
                "entities": []
            }
    
    def analyze_syntax(self, text: str, language_code: str = "ko") -> Dict[str, Any]:
        """
        텍스트의 구문을 분석합니다.
        
        Args:
            text: 분석할 텍스트
            language_code: 언어 코드 (기본값: ko)
            
        Returns:
            Dict: 구문 분석 결과
        """
        if not self.client:
            return {
                "error": "Google NLP 클라이언트가 초기화되지 않았습니다",
                "tokens": []
            }
        
        try:
            # 문서 객체 생성
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language=language_code
            )
            
            # 구문 분석 실행
            response = self.client.analyze_syntax(
                request={'document': document}
            )
            
            # 토큰 정보 추출
            tokens = []
            for token in response.tokens:
                tokens.append({
                    "text": token.text.content,
                    "part_of_speech": token.part_of_speech.tag.name,
                    "lemma": token.lemma,
                    "dependency_edge": {
                        "head_token_index": token.dependency_edge.head_token_index,
                        "label": token.dependency_edge.label.name
                    }
                })
            
            return {
                "tokens": tokens,
                "language_code": language_code
            }
            
        except Exception as e:
            return {
                "error": f"구문 분석 중 오류 발생: {str(e)}",
                "tokens": []
            }


# 전역 인스턴스
nlp_service = GoogleNLPService()


def analyze_text_sentiment(text: str, language_code: str = "ko") -> Dict[str, Any]:
    """텍스트 감정 분석 함수"""
    return nlp_service.analyze_sentiment(text, language_code)


def analyze_text_entities(text: str, language_code: str = "ko") -> Dict[str, Any]:
    """텍스트 엔티티 분석 함수"""
    return nlp_service.analyze_entities(text, language_code)


def analyze_text_syntax(text: str, language_code: str = "ko") -> Dict[str, Any]:
    """텍스트 구문 분석 함수"""
    return nlp_service.analyze_syntax(text, language_code)
