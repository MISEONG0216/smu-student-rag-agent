from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ai.state import AgentState
from ai.retriever import get_retriever
from ai.text2sql import get_text2sql_engine
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

load_dotenv()

llm = init_chat_model("gpt-5.4-mini")

_retriever = None
_text2sql_engine = None


# -----------------------------------------------------------------------------
# Structured Output Models
# -----------------------------------------------------------------------------
class IntentClassification(BaseModel):
    """사용자 질문의 처리 경로"""
    intent: Literal["general", "database", "vector"] = Field(
        description="general, database, vector 중 하나"
    )


class VectorSearchQuery(BaseModel):
    """상명대학교 문서 검색을 위한 쿼리 분석 결과"""
    optimized_query: str = Field(
        description=(
            "상명대학교 문서 검색에 최적화된 쿼리. "
            "제도명, 대상, 학기/연도, 조건, 일정 등 핵심 키워드를 포함한다."
        )
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description=(
            "문서 카테고리. 명확할 때만 1~2개 선택하고 애매하면 null. "
            "가능한 값: 장학제도, 외부활동, 수강신청"
        )
    )


# -----------------------------------------------------------------------------
# Lazy initialization
# -----------------------------------------------------------------------------
def get_cached_retriever():
    """캐시된 retriever 인스턴스 반환"""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def get_cached_text2sql_engine():
    """캐시된 Text2SQL 엔진 반환"""
    global _text2sql_engine
    if _text2sql_engine is None:
        _text2sql_engine = get_text2sql_engine()
    return _text2sql_engine


# -----------------------------------------------------------------------------
# Intent classification
# -----------------------------------------------------------------------------
def classify_intent(state: AgentState) -> AgentState:
    """
    사용자의 질문을 다음 3가지로 분류한다.

    - general: 인사/대화/학교생활 일반 질문
    - database: 강좌·교수·시간표·수강제한·전공인정·부서·위치 등 구조화 DB 조회
    - vector: 장학제도·외부활동·수강신청 규정/절차/공지 등 문서 검색
    """
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided")

    question = (
        messages[-1].content
        if hasattr(messages[-1], "content")
        else str(messages[-1])
    )

    system_prompt = """
당신은 '상명대학교 학생 도우미 AI'의 질문 라우터입니다.
이전 대화 맥락까지 고려하여 현재 질문을 정확히 하나의 처리 방식으로 분류하세요.

[1] general
- 인사, 감사, 간단한 대화
- 검색 없이 답할 수 있는 일반적인 학교생활 안내
예:
  "안녕하세요"
  "고마워"
  "상명대학교 도우미가 뭘 해줄 수 있어?"

[2] database
상명대학교의 구조화된 데이터베이스를 조회해야 정확히 답할 수 있는 질문입니다.
현재 DB에는 다음 종류의 정보가 있습니다.
- 개설 강좌/시간표(course_schedule)
- 수강 제한(course_restrictions)
- 타전공/전공 인정(major_recognition)
- 건물/층별 사무실(office_floors)
- 교내 조직(organizations)
- 학과/부서(departments)

대표적인 database 질문:
- "1학년 3학점 과목 10개 알려줘"
- "AI 관련 과목 중 담당교수와 강의시간 알려줘"
- "수강제한이 있는 과목 알려줘"
- "커뮤니케이션디자인전공으로 인정되는 과목은?"
- "학생복지팀은 어디에 있어?"
- "어떤 학과가 있어?"

중요:
'수강신청'이라는 단어가 있어도 실제 개설 과목, 교수, 시간, 학점, 분반,
수강제한 여부, 전공인정 여부를 묻는다면 database입니다.

[3] vector
학교 공지/PDF/안내문 등 비정형 문서를 찾아야 하는 질문입니다.
주요 문서 분야는 다음과 같습니다.
- 장학제도
- 외부활동
- 수강신청 제도/절차/규정/공지

대표적인 vector 질문:
- "교내 장학금 종류와 신청 조건 알려줘"
- "국가근로장학금 신청 방법은?"
- "학생이 참여할 수 있는 비교과·대외활동 알려줘"
- "수강신청 일정과 절차가 어떻게 돼?"
- "수강신청 정정 기간에는 무엇을 할 수 있어?"
- "수강신청 관련 유의사항 알려줘"

중요:
수강신청의 '제도, 절차, 일정, 규정, 신청방법, 유의사항'을 묻는다면 vector입니다.

판단 기준:
- 정형 데이터의 행/열 조회가 핵심이면 database
- PDF/공지/규정의 내용 검색이 핵심이면 vector
- 둘 다 필요해 보일 경우, 사용자가 직접 요구한 핵심 답변에 더 필요한 쪽을 선택하세요.
"""

    structured_llm = llm.with_structured_output(IntentClassification)
    result = structured_llm.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )

    return {
        "intent": result.intent,
        "question": question,
        "retry_count": 0,
        "error": None,
    }


# -----------------------------------------------------------------------------
# General answer
# -----------------------------------------------------------------------------
def general_answer(state: AgentState) -> AgentState:
    """검색이 필요하지 않은 일반 대화를 처리한다."""
    messages = state.get("messages", [])

    system_prompt = """
당신은 상명대학교 학생들의 학교생활을 돕는 '상명대학교 도우미'입니다.

주요 역할:
- 장학제도 안내
- 외부활동/비교과/학생 참여 프로그램 안내
- 수강신청 및 강의정보 안내
- 교내 조직/학과/사무실 정보 안내

답변 원칙:
- 학생이 이해하기 쉬운 표현을 사용하세요.
- 질문에 먼저 직접 답하고, 필요한 경우 짧은 예시를 덧붙이세요.
- 확인되지 않은 학교 규정, 일정, 금액, 자격조건을 지어내지 마세요.
- 학교 공식 자료나 DB 확인이 필요한 구체적 사실을 일반 지식으로 추측하지 마세요.
"""

    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)

    return {
        "messages": [AIMessage(content=response.content)]
    }


# -----------------------------------------------------------------------------
# Vector search (RAG)
# -----------------------------------------------------------------------------
def vector_search(state: AgentState) -> AgentState:
    """
    Qdrant 벡터 검색을 수행한다.

    1. 대화 맥락을 포함한 완전한 질문 생성
    2. 검색용 쿼리 및 카테고리 추출
    3. Qdrant 검색
    """
    messages = state.get("messages", [])
    original_query = state.get("rewritten_query") or state.get("question", "")

    # 후속 질문이면 이전 맥락을 포함해 독립적인 질문으로 변환
    if len(messages) > 1 and not state.get("rewritten_query"):
        system_prompt_complete = """
당신은 상명대학교 학생 질문 재작성 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 검색 가능한 완전한 질문으로 바꾸세요.

예시:
- 이전: "성적우수 장학금 알려줘" / 현재: "신청기간은?"
  → "상명대학교 성적우수 장학금 신청기간은 언제인가?"
- 이전: "수강신청 정정기간 알려줘" / 현재: "이때 과목 삭제도 가능해?"
  → "상명대학교 수강신청 정정기간에 수강 과목 삭제가 가능한가?"
- 이전: "학생 참여 프로그램 알려줘" / 현재: "AI 관련된 것도 있어?"
  → "상명대학교 학생 참여 프로그램 중 AI 관련 프로그램은 무엇이 있는가?"

완전한 질문만 반환하세요.
현재 질문이 이미 완전하면 그대로 반환하세요.
"""
        response_complete = llm.invoke(
            [SystemMessage(content=system_prompt_complete)] + messages
        )
        original_query = response_complete.content.strip()

    system_prompt = """
당신은 상명대학교 RAG 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 분석하여 학교 PDF/공지 문서 검색에 적합한 쿼리와 카테고리를 만드세요.

사용 가능한 카테고리:
- 장학제도: 교내·교외 장학금, 국가장학금, 국가근로, 선발기준, 지급조건, 신청기간 등
- 외부활동: 비교과, 공모전, 대외활동, 프로그램, 캠프, 현장실습, 취업·진로 활동 등
- 수강신청: 수강신청 일정, 절차, 정정, 취소, 재수강, 학점, 신청 규정과 유의사항 등

카테고리 선택 규칙:
1. 질문과 명확하게 관련된 경우만 1~2개 선택
2. 애매하면 categories=null
3. 억지로 카테고리를 붙이지 않음

optimized_query 작성 규칙:
- '상명대학교'와 핵심 제도/행사/규정 키워드를 포함
- 신청 대상, 신청기간, 자격, 금액, 절차 등 질문의 핵심 조건을 보존
- 너무 긴 문장보다 검색에 유리한 핵심어 중심으로 작성
"""

    structured_llm = llm.with_structured_output(VectorSearchQuery)
    query_analysis = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"다음 질문을 분석하세요:\n\n{original_query}")
    ])

    optimized_query = query_analysis.optimized_query
    categories = query_analysis.categories

    print("[상명대학교 벡터 검색 쿼리 분석]")
    print(f"  원본 쿼리: {original_query}")
    print(f"  최적화된 쿼리: {optimized_query}")
    print(f"  선택된 카테고리: {categories}")

    retriever = get_cached_retriever()
    results = retriever.search(
        optimized_query,
        k=5,
        score_threshold=0.45,
        categories=categories,
    )

    return {
        "vector_results": results,
        "search_query": optimized_query,
    }


# -----------------------------------------------------------------------------
# Query rewrite
# -----------------------------------------------------------------------------
def rewrite_query(state: AgentState) -> AgentState:
    """문서 검색 결과가 부족할 때 검색어를 다시 만든다."""
    messages = state.get("messages", [])
    previous_query = state.get("search_query") or state.get("question", "")

    system_prompt = f"""
당신은 상명대학교 문서 검색 전문가입니다.
첫 번째 검색에서 충분한 결과를 찾지 못했습니다.

이전 검색어:
{previous_query}

검색 개선 방법:
- 제도명/행사명/핵심 명사를 우선 사용
- '상명대학교'를 포함
- 장학금이면 장학명, 대상, 신청, 선발, 지급 등의 동의어 활용
- 외부활동이면 비교과, 프로그램, 공모전, 대외활동, 참가 등의 동의어 활용
- 수강신청이면 수강신청, 정정, 취소, 재수강, 학점, 일정, 유의사항 등의 관련어 활용
- 너무 구체적이어서 검색이 안 되면 조금 일반화
- 너무 일반적이면 질문의 핵심 조건을 추가

재작성된 검색 쿼리만 반환하세요.
"""

    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)

    return {
        "rewritten_query": response.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


# -----------------------------------------------------------------------------
# Text2SQL
# -----------------------------------------------------------------------------
def database_query(state: AgentState) -> AgentState:
    """상명대학교 DB에 대해 Text2SQL을 수행한다."""
    messages = state.get("messages", [])
    question = state.get("question", "")
    previous_error = state.get("error")

    if len(messages) > 1:
        system_prompt = """
당신은 상명대학교 데이터베이스 질문 재작성 전문가입니다.
이전 대화를 참고하여 현재 질문을 단독으로 이해할 수 있는 완전한 질문으로 바꾸세요.

DB에서 다룰 수 있는 정보:
- 개설강좌: 학년, 이수구분, 학수번호, 교과목명, 학점, 이론/실습시간, 분반, 담당교수, 강의시간/강의실
- 수강제한 여부
- 전공 인정 여부
- 교내 조직/부서
- 사무실 위치/층 정보

예시:
- 이전: "AI모빌리티 3학점 과목 알려줘" / 현재: "교수는?"
  → "AI모빌리티 관련 3학점 과목의 담당교수는 누구인가?"
- 이전: "수강제한 과목 알려줘" / 현재: "10개만"
  → "수강제한이 있는 과목을 10개 알려줘"
- 이전: "커뮤니케이션디자인 전공인정 과목" / 현재: "시간도 알려줘"
  → "커뮤니케이션디자인전공 인정 과목의 담당교수와 강의시간을 알려줘"

완전한 질문만 반환하세요.
현재 질문이 이미 완전하면 그대로 반환하세요.
"""
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
        complete_question = response.content.strip()
    else:
        complete_question = question

    text2sql_engine = get_cached_text2sql_engine()
    result = text2sql_engine.query(
        complete_question,
        previous_error=previous_error,
    )

    return {
        "sql_query": result["sql_query"],
        "db_results": result["result"],
        "error": result["error"],
        "retry_count": state.get("retry_count", 0) + 1,
    }


# -----------------------------------------------------------------------------
# Final answer generation
# -----------------------------------------------------------------------------
def generate_answer(state: AgentState) -> AgentState:
    """벡터 검색 또는 DB 검색 결과를 이용해 최종 답변을 생성한다."""
    messages = state.get("messages", [])
    context_parts = []

    if state.get("vector_results"):
        docs = state["vector_results"]
        context_parts.append("[학교 문서 검색 결과]")

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "출처 미상")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "")

            source_info = f"출처={source}, 페이지={page}"
            if category:
                source_info += f", 카테고리={category}"

            context_parts.append(
                f"\n[문서 {i}] {source_info}\n{doc.page_content}"
            )

    if state.get("db_results"):
        context_parts.append(
            f"\n[상명대학교 DB 조회 결과]\n{state['db_results']}"
        )
        if state.get("sql_query"):
            context_parts.append(
                f"\n[실행 SQL]\n{state['sql_query']}"
            )

    if state.get("error"):
        context_parts.append(f"\n[DB 오류]\n{state['error']}")

    context = "\n".join(context_parts)

    system_prompt = f"""
당신은 상명대학교 학생을 위한 공식 정보 탐색형 AI 도우미입니다.
학생이 장학제도, 외부활동, 수강신청, 강의정보, 교내 조직 정보를 쉽게 찾도록 돕습니다.

아래 검색 결과/DB 결과만 근거로 사실 정보를 답하세요.

<context>
{context}
</context>

답변 원칙:
1. 질문에 대한 핵심 답변을 먼저 제시하세요.
2. 여러 결과가 있으면 표 또는 보기 쉬운 목록으로 정리하세요.
3. 과목 질문이면 가능한 경우 과목명, 담당교수, 학점, 강의시간, 강의실, 분반을 명확히 구분하세요.
4. 장학/활동/수강신청 문서 질문이면 가능한 경우 대상, 조건, 신청기간, 신청방법, 주의사항을 구분하세요.
5. 문서 답변에는 근거가 되는 자료명과 페이지를 자연스럽게 표시하세요.
6. DB 결과에 없는 값이나 문서에 없는 조건을 추측해서 만들지 마세요.
7. 자료끼리 내용이 충돌하면 최신 여부를 임의로 판단하지 말고, 각 자료의 차이를 알려주세요.
8. 검색 결과가 부족하면 '현재 등록된 학교 자료에서 확인되지 않습니다'라고 명확히 말하세요.
9. 학생이 바로 행동할 수 있도록 필요한 다음 단계가 자료에 있으면 함께 안내하세요.
10. 내부 구현 용어(Qdrant, 벡터DB, Text2SQL 등)는 사용자가 묻지 않는 한 답변에서 언급하지 마세요.
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )

    return {
        "messages": [AIMessage(content=response.content)]
    }


# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
def route_by_intent(state: AgentState) -> str:
    """의도에 따라 다음 노드를 결정한다."""
    intent = state.get("intent", "general")

    if intent == "database":
        return "database_query"
    if intent == "vector":
        return "vector_search"
    return "general_answer"


def check_vector_results(state: AgentState) -> str:
    """벡터 검색 결과가 충분한지 확인한다."""
    results = state.get("vector_results", [])
    retry_count = state.get("retry_count", 0)

    retriever = get_cached_retriever()

    if retriever.is_relevant(results):
        return "generate_answer"

    if retry_count >= 2:
        return "generate_answer"

    return "rewrite_query"


def check_db_results(state: AgentState) -> str:
    """DB 검색 결과를 확인하고 필요 시 재시도한다."""
    error = state.get("error")
    result = state.get("db_results")
    retry_count = state.get("retry_count", 0)

    text2sql_engine = get_cached_text2sql_engine()

    if not error and result and not text2sql_engine.is_empty_result(result):
        return "generate_answer"

    if retry_count >= 2:
        return "generate_answer"

    return "database_query"
