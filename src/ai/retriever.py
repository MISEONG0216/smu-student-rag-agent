import os
from typing import List

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorRetriever:
    def __init__(self):
        """상명대학교 Qdrant 벡터 검색기 초기화"""

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL 환경변수가 설정되어 있지 않습니다. "
                ".env 파일을 확인해주세요."
            )

        # Qdrant 연결
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        # Qdrant Vector Size 3072에 맞는 임베딩 모델
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )

        # =========================================
        # 상명대학교 학사일정
        # =========================================
        self.academic_store = QdrantVectorStore(
            client=self.client,
            collection_name="SMU_2026_2",
            embedding=self.embeddings
        )

        # =========================================
        # 상명대학교 시간표
        # =========================================
        self.timetable_store = QdrantVectorStore(
            client=self.client,
            collection_name="SMU_2026_2_time",
            embedding=self.embeddings
        )

        print("상명대학교 Qdrant Retriever 초기화 완료")
        print(" - 학사일정: SMU_2026_2")
        print(" - 시간표: SMU_2026_2_time")

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.45,
        categories: List[str] = None
    ) -> List[Document]:
        """
        상명대학교 학사일정 검색

        기본 vector_search 노드에서 사용하는 검색 함수입니다.

        예:
        - 수강신청 기간
        - 개강일
        - 중간고사 기간
        - 기말고사 기간
        - 성적입력 기간
        - 학사일정
        """

        try:
            results = (
                self.academic_store
                .similarity_search_with_relevance_scores(
                    query,
                    k=k
                )
            )

            filtered_results = []

            for doc, score in results:
                print(
                    f"[학사일정 검색] "
                    f"score={score:.4f}, "
                    f"source={doc.metadata.get('source', '알 수 없음')}"
                )

                if score >= score_threshold:
                    filtered_results.append(doc)

            return filtered_results

        except Exception as e:
            print(f"학사일정 벡터 검색 오류: {e}")
            return []

    def search_timetable(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.45
    ) -> List[Document]:
        """
        상명대학교 시간표 Qdrant 검색

        참고:
        과목명, 교수, 학점, 강의시간 등
        정확한 조건 검색은 Supabase SQL을 우선 사용합니다.

        이 함수는 시간표 벡터검색이 필요한 경우
        보조적으로 사용할 수 있습니다.
        """

        try:
            results = (
                self.timetable_store
                .similarity_search_with_relevance_scores(
                    query,
                    k=k
                )
            )

            filtered_results = []

            for doc, score in results:
                print(
                    f"[시간표 검색] "
                    f"score={score:.4f}, "
                    f"source={doc.metadata.get('source', '알 수 없음')}"
                )

                if score >= score_threshold:
                    filtered_results.append(doc)

            return filtered_results

        except Exception as e:
            print(f"시간표 벡터 검색 오류: {e}")
            return []

    def is_relevant(
        self,
        results: List[Document],
        min_count: int = 1
    ) -> bool:
        """
        검색 결과가 충분한지 확인
        """

        return len(results) >= min_count


def get_retriever() -> VectorRetriever:
    """상명대학교 VectorRetriever 반환"""

    return VectorRetriever()