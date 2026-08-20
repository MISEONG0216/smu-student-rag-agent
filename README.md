# smu-student-rag-agent
상명대학교 장학제도, 학생활동, 수강신청 정보를 제공하는 RAG 기반 AI Agent

from langchain_core.documents import Document
import fitz

# TODO: PDF 파일 경로를 입력하세요
# 예시: "../datasets/your_document.pdf"
file_path = "C:/Users/doyoo/smu-ai-service-bootcamp/rag-system/datasets/붙임2_2026학년도_2학기_학과별시간표(2026.08.19.).pdf"

doc = fitz.open(file_path)
docs = []

# 페이지 단위로 Document 생성 (Parent Document)
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text("text", sort=True)

    # 빈 페이지는 스킵
    if len(text.strip()) < 10:
        continue

    docs.append(
        Document(
            page_content=text,
            metadata={
                "source": file_path.split("/")[-1],
                "page": page_num + 1,
                "parent_id": f"page_{page_num + 1}"
            }
        )
    )

doc.close()

print(f"총 {len(docs)}개의 페이지(Parent Document) 로드 완료")
print(f"\n첫 번째 페이지 길이: {len(docs[0].page_content)}자")
print(f"평균 페이지 길이: {sum(len(d.page_content) for d in docs) / len(docs):.0f}자")

# 첫 페이지 내용 미리보기
print(f"\n첫 페이지 내용 미리보기:")
print(docs[0].page_content[:300] + "...")


# **2일차 팀 프로젝트: 문서 기반 RAG 시스템 구축**

## 프로젝트 목표
1. 팀에서 선정한 PDF 문서를 Qdrant Cloud에 저장
2. Parent Document Retriever 패턴 적용
3. 검색 테스트 및 RAG 시스템 구현

## 구현 단계
- 환경 설정 확인
- PDF 문서 로딩
- Child Chunk 생성 및 Qdrant Cloud 저장
- Parent Document 저장
- 검색 테스트
- RAG 시스템 구현 및 테스트


# **3일차 팀 프로젝트: 테이블 데이터 조회 시스템 구축**

## 프로젝트 목표
1. 팀에서 선정한 CSV 테이블 데이터를 Supabase에 적재
2. SQL 쿼리로 데이터 조회 테스트
3. Text2SQL 시스템 구현 및 테스트

## 구현 단계
- 환경 설정 확인
- CSV 파일 확인 및 탐색
- Supabase 연결
- CSV 데이터 업로드
- SQL 쿼리 테스트
- Text2SQL 시스템 구현 및 테스트
