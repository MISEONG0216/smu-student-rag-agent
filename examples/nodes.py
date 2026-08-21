from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ai.state import AgentState
from ai.retriever import get_retriever
from ai.text2sql import get_text2sql_engine
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv()

llm = init_chat_model("gpt-5.4-mini")

_retriever = None
_text2sql_engine = None


class VectorSearchQuery(BaseModel):
    """벡터 검색을 위한 쿼리 분석 결과"""
    optimized_query: str = Field(
        description="검색에 최적화된 쿼리. 핵심 키워드를 포함하고 명확하게 작성."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="선택된 카테고리 리스트 (1-2개). 명확하게 관련 있는 카테고리만 선택. 애매하거나 불확실한 경우 null 반환. 가능한 값: 새로운_천안_신규_조성_공간, 일상편의_행정_교통, 아이돌봄, 맞춤_복지, 건강_내일_보건, 문화_관광, 즐거운_일상_행사축제, 행복한_변화_달라지는_정책"
    )


def get_cached_retriever():
    """캐시된 retriever 인스턴스 반환 (lazy initialization)"""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def get_cached_text2sql_engine():
    """캐시된 text2sql_engine 인스턴스 반환 (lazy initialization)"""
    global _text2sql_engine
    if _text2sql_engine is None:
        _text2sql_engine = get_text2sql_engine()
    return _text2sql_engine


def classify_intent(state: AgentState) -> AgentState:
    """
    사용자 질문의 의도를 분류하는 노드

    분류 결과:
    - 'general': 일반적인 대화나 인사
    - 'database': 데이터베이스 조회가 필요한 질문
    - 'vector': 문서 검색이 필요한 질문

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # messages에서 질문 추출
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages provided")

    # 마지막 사용자 메시지를 질문으로 사용
    question = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

    system_prompt = """
당신은 상명대학교 학생을 위한 AI의 질문 의도를 분류하는 전문가입니다.

사용자의 질문을 이전 대화 맥락과 함께 분석하여 반드시 다음 3가지 중 하나로 분류하세요.

1. 'general'
- 시간표, 수업, 강의, 교수, 학점, 강의실, 요일, 시간과 관련 없는 일반적인 대화나 질문
- 예:
  "안녕하세요"
  "고마워"
  "너는 누구야"

2. 'database'
- 상명대학교 학생의 수업 및 시간표 데이터베이스에서 조회해야 하는 질문
- 다음과 같은 질문은 반드시 database로 분류하세요.

[수업/학점 관련]
- "2학점 수업 알려줘"
- "3학점 수업 알려줘"
- "학점이 2점인 과목 전부 보여줘"
- "3학점짜리 과목 뭐가 있어?"
- "1학점 수업 알려줘"

[과목 관련]
- "컴퓨터공학과 수업 알려줘"
- "AI모빌리티공학과 수업 알려줘"
- "이번 학기 개설 과목 알려줘"
- "전공 과목 알려줘"
- "교양 과목 알려줘"

[교수 관련]
- "강태구 교수님, 이광재 교수님 수업 알려줘"
- "강태구 교수님, 이광재 교수님이 가르치는 과목 뭐야?"

[시간/요일 관련]
- "월요일 수업 알려줘"
- "월요일에 있는 3학점 수업 알려줘"
- "오전 수업만 알려줘"
- "10시에 시작하는 수업 알려줘"

[강의실 관련]
- "강의실이 어디야?"
- "본관에서 하는 수업 알려줘"

[복합 조건]
- "2학점이고 월요일에 있는 수업 알려줘"
- "3학점 전공 수업 중 오전에 하는 것 알려줘"
- "우리 학과 2학점 수업 전부 알려줘"

또한 상명대학교의 학사/조직/부서/전화번호 등의 데이터베이스에 저장된 정보를 조회해야 하는 질문도 database로 분류하세요.

3. 'vector'
- 상명대학교의 정책, 사업 계획, 공지, 문서, 복지, 문화, 관광 등
문서 내용을 검색해야 하는 질문
- 예:
  "상명대 주요 업무 계획은?"
  "상명대 복지 정책은?"
  "학교 행사 일정은?"
  "학교에서 제공하는 복지 혜택은?"

중요한 규칙:
- 수업, 과목, 강의, 학점, 교수, 강의실, 요일, 시간표와 관련된 질문은 무조건 'database'로 분류하세요.
- 사용자가 "전부", "모두", "다 알려줘", "목록으로 알려줘"라고 요청하면 데이터베이스에서 가능한 모든 조건에 해당하는 결과를 조회해야 합니다.
- 질문에 특정 조건이 있으면 해당 조건을 데이터베이스 조회 조건으로 사용하세요.
- 반드시 'general', 'database', 'vector' 중 하나만 반환하세요.
- 다른 설명은 절대 포함하지 마세요.
"""


    # 시스템 메시지 + 전체 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    intent = response.content.strip().lower()

    # 유효한 의도인지 확인
    if intent not in ['general', 'database', 'vector']:
        intent = 'general'

    return {
        "intent": intent,
        "question": question
    }


def general_answer(state: AgentState) -> AgentState:
    """
    일반적인 질문에 직접 답변하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 자연스럽고 도움이 되는 답변을 제공하세요.
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


def vector_search(state: AgentState) -> AgentState:
    """
    Qdrant 벡터 검색을 수행하는 노드

    1. LLM으로 질문 분석 (최적화된 쿼리 + 카테고리 추출)
    2. 병렬 벡터 검색 수행

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    # 재작성된 쿼리가 있으면 사용, 없으면 원본 질문 사용
    original_query = state.get("rewritten_query") or state.get("question", "")

    # 이전 대화 맥락이 있으면 완전한 질문으로 재구성
    if len(messages) > 1 and not state.get("rewritten_query"):
        # rewritten_query가 없을 때만 (첫 시도) 맥락 고려
        system_prompt_complete = """
당신은 질문 분석 전문가입니다.
이전 대화 맥락을 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

예시:
- 이전: "두정동 공영주차장 언제 완공?" → 현재: "예산은 얼마야?" → 재구성: "두정동 공영주차장 예산은 얼마야?"
- 이전: "상명대 복지 정책" → 현재: "더 자세히 알려줘" → 재구성: "상명대 시간표를 더 자세히 알려줘"

완전한 질문만 반환하세요. 설명은 포함하지 마세요.
만약 현재 질문이 이미 완전하다면 그대로 반환하세요.
"""
        conversation_complete = [SystemMessage(content=system_prompt_complete)] + messages
        response_complete = llm.invoke(conversation_complete)
        original_query = response_complete.content.strip()

    # 1. LLM으로 쿼리 분석 및 카테고리 추출 (Structured Output)
    # 시스템 프롬프트: 역할 정의 및 카테고리 설명
    system_prompt = """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 분석하여 벡터 검색에 최적화된 쿼리를 생성하고, 적절한 카테고리를 선택하는 역할을 수행합니다.

사용 가능한 카테고리:
- 학생_시간표: 상명대학교 학생의 개인 시간표, 수업 시간, 강의 일정, 요일별 수업, 강의실, 교수, 과목 관련
- 수강신청_과목: 수강신청, 과목 선택, 수강 가능 과목, 전공 및 교양 과목 관련
- 학사일정: 개강, 종강, 시험기간, 휴강, 보강, 학사 일정 관련
- 강의_수업정보: 강의계획서, 수업 내용, 교수, 강의실, 수업 방식 등 강의 관련
- 졸업_학점: 졸업 요건, 이수 학점, 전공 학점, 교양 학점 등 졸업 관련
- 장학_등록금: 등록금, 장학금, 학비 지원 관련

카테고리 선택 규칙:
1. 사용자의 질문에서 핵심 정보를 추출합니다.
2. 이전 대화에서 언급된 과목, 학과, 학년, 학기 등의 맥락이 있다면 함께 반영합니다.
3. 시간표를 묻는 질문이라면 요일, 시간, 과목명, 강의실, 교수 등 검색에 필요한 정보를 최대한 포함합니다.
4. 사용자가 "시간표 보여줘", "오늘 수업 뭐야?", "월요일에 뭐 있어?"처럼 간단하게 질문하더라도 이전 대화 맥락을 고려하여 검색 가능한 완전한 쿼리로 만듭니다.

출력 지침:
1. optimized_query: 검색에 효과적인 핵심 키워드를 포함한 쿼리로 최적화
2. categories: 명확하게 관련 있는 카테고리 1-2개를 리스트로 반환. 불확실하면 null"""

    # 유저 프롬프트: 실제 질문
    user_prompt = f"다음 질문을 분석해주세요:\n\n{original_query}"

    # 메시지 객체 생성 (Structured Output용)
    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Structured Output으로 LLM 호출
    structured_llm = llm.with_structured_output(VectorSearchQuery)
    query_analysis = structured_llm.invoke(llm_messages)

    optimized_query = query_analysis.optimized_query
    categories = query_analysis.categories

    print(f"[벡터 검색 쿼리 분석]")
    print(f"  원본 쿼리: {original_query}")
    print(f"  최적화된 쿼리: {optimized_query}")
    print(f"  선택된 카테고리: {categories}")

    # 2. 병렬 벡터 검색 수행 (카테고리 필터 적용)
    retriever = get_cached_retriever()
    results = retriever.search(optimized_query, k=3, score_threshold=0.5, categories=categories)

    return {
        "vector_results": results
    }


def rewrite_query(state: AgentState) -> AgentState:
    """
    검색 결과가 부족할 때 쿼리를 재작성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    system_prompt = """
당신은 검색 쿼리 최적화 전문가입니다.

사용자의 질문이 검색 결과를 얻지 못했습니다.
이전 대화 맥락을 고려하여 질문을 다시 작성하여 더 나은 검색 결과를 얻을 수 있도록 하세요.

최적화 방법:
- 이전 대화에서 언급된 맥락을 포함
- 동의어나 관련 용어 추가
- 질문을 더 구체적이거나 더 일반적으로 변경
- 핵심 키워드 강조

재작성된 쿼리만 반환하세요. 설명은 포함하지 마세요.
"""

    # 시스템 메시지 + 전체 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    rewritten = response.content.strip()

    return {
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1
    }


def database_query(state: AgentState) -> AgentState:
    """
    Text2SQL을 수행하여 데이터베이스를 조회하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])
    question = state.get("question", "")
    previous_error = state.get("error")

    # 이전 대화 맥락이 있으면 완전한 질문으로 재구성
    if len(messages) > 1:
        system_prompt = """
당신은 상명대학교 학생의 수업 및 시간표 질문을 분석하는 전문가입니다.

이전 대화 맥락을 반드시 고려하여 현재 질문을 완전하고 명확한 질문으로 재구성하세요.

특히 다음과 같은 수업 관련 조건을 이전 대화에서 유지하세요:
- 학점
- 과목명
- 교수명
- 학과/전공
- 요일
- 수업 시간
- 강의실
- 과목 구분

예시:
- 이전: "2학점 수업 알려줘" → 현재: "월요일은?"
  → 재구성: "2학점 수업 중 월요일에 있는 수업은?"

- 이전: "3학점 전공 수업 알려줘" → 현재: "오전 수업은?"
  → 재구성: "3학점 전공 수업 중 오전에 있는 수업은?"

- 이전: "김철수 교수님 수업 알려줘" → 현재: "2학점짜리만"
  → 재구성: "김철수 교수님이 담당하는 수업 중 2학점 수업은?"

- 이전: "AI모빌리티공학과 수업 알려줘" → 현재: "2학점은?"
  → 재구성: "AI모빌리티공학과 수업 중 2학점 수업은?"

- 이전: "2학점 수업 알려줘" → 현재: "전부 보여줘"
  → 재구성: "2학점에 해당하는 모든 수업을 보여줘"

현재 질문이 이미 완전한 질문이라면 그대로 반환하세요.

완전한 질문만 반환하세요.
설명은 절대 포함하지 마세요.
"""
        conversation = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(conversation)
        complete_question = response.content.strip()
    else:
        complete_question = question

    # Text2SQL 실행
    text2sql_engine = get_cached_text2sql_engine()
    result = text2sql_engine.query(complete_question, previous_error=previous_error)

    return {
        "sql_query": result["sql_query"],
        "db_results": result["result"],
        "error": result["error"],
        "retry_count": state.get("retry_count", 0) + 1
    }


def generate_answer(state: AgentState) -> AgentState:
    """
    검색 결과를 바탕으로 최종 답변을 생성하는 노드

    Args:
        state: 현재 상태

    Returns:
        업데이트된 상태
    """
    # 대화 히스토리 가져오기
    messages = state.get("messages", [])

    # 컨텍스트 구성
    context_parts = []

    # 벡터 검색 결과가 있으면 추가
    if state.get("vector_results"):
        docs = state["vector_results"]
        context_parts.append("관련 문서:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "알 수 없음")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "")

            # 출처 정보 구성
            source_info = f"출처: {source}, 페이지: {page}"
            if category:
                source_info += f", 카테고리: {category}"

            context_parts.append(f"\n[문서 {i}] {source_info}\n{doc.page_content}")

    # DB 검색 결과가 있으면 추가
    if state.get("db_results"):
        context_parts.append(f"\n\n데이터베이스 조회 결과:\n{state['db_results']}")
        if state.get("sql_query"):
            context_parts.append(f"\n실행된 SQL:\n{state['sql_query']}")

    context = "\n".join(context_parts)

    system_prompt = f"""
당신은 상명대학교 학생을 위한 학사 및 시간표 전문 AI입니다.

사용자의 질문에 답변할 때 반드시 <context> 안에 제공된
데이터베이스 조회 결과와 문서 검색 결과를 기준으로 답변하세요.

<context>
{context}
</context>

[시간표 및 수업 질문 규칙]

1. 사용자가 특정 조건의 수업을 요청하면 해당 조건에 맞는 수업을 모두 보여주세요.

2. 사용자가 "전부", "모두", "다 알려줘", "목록으로 알려줘"라고 요청한 경우
조회 결과에 있는 모든 수업을 빠짐없이 보여주세요.

3. 예를 들어 사용자가
"2학점 수업 알려줘"
라고 질문하면 2학점에 해당하는 모든 수업을 목록으로 보여주세요.

4. 수업을 보여줄 때 가능하다면 다음 정보를 함께 표시하세요.
- 과목명
- 학점
- 교수명
- 요일
- 수업시간
- 강의실
- 학과/전공
- 과목구분

5. 사용자가 특정 조건을 여러 개 제시하면 모든 조건을 만족하는 수업만 보여주세요.

예:
"2학점이고 월요일에 있는 수업 알려줘"
→ 2학점 AND 월요일 조건을 모두 만족하는 수업을 보여줍니다.

6. 결과가 여러 개라면 표 또는 번호 목록으로 보기 쉽게 정리하세요.

7. 데이터베이스 조회 결과에 있는 수업을 임의로 생략하지 마세요.

8. 데이터베이스 결과가 존재한다면 임의로 다른 정보를 만들어내지 마세요.

9. 조회 결과가 없으면
"조건에 맞는 수업을 찾을 수 없습니다."
라고 답변하세요.

10. 사용자가 "몇 개야?"라고 질문하면 조회 결과의 개수를 정확하게 계산해서 알려주세요.

11. 이전 대화에서 언급된 학과, 과목, 교수, 학점 등의 조건이 있다면
현재 질문과 연결하여 답변하세요.

답변은 상명대학교 학생이 실제 시간표를 확인하는 데 도움이 되도록
간결하고 명확하게 작성하세요.
"""

    # 시스템 메시지 + 기존 대화 히스토리
    conversation = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(conversation)
    answer = response.content

    return {
        "messages": [AIMessage(content=answer)]
    }


def route_by_intent(state: AgentState) -> str:
    """
    의도에 따라 다음 노드를 결정하는 라우팅 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    intent = state.get("intent", "general")

    if intent == "general":
        return "general_answer"
    elif intent == "database":
        return "database_query"
    elif intent == "vector":
        return "vector_search"
    else:
        return "general_answer"


def check_vector_results(state: AgentState) -> str:
    """
    벡터 검색 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    results = state.get("vector_results", [])
    retry_count = state.get("retry_count", 0)

    # 결과가 있거나 재시도 횟수가 2회 이상이면 답변 생성
    retriever = get_cached_retriever()
    if retriever.is_relevant(results) or retry_count >= 2:
        return "generate_answer"
    else:
        return "rewrite_query"


def check_db_results(state: AgentState) -> str:
    """
    데이터베이스 검색 결과를 확인하고 다음 노드를 결정하는 함수

    Args:
        state: 현재 상태

    Returns:
        다음 노드 이름
    """
    error = state.get("error")
    result = state.get("db_results")
    retry_count = state.get("retry_count", 0)

    # 오류가 없고 결과가 있으면 답변 생성
    text2sql_engine = get_cached_text2sql_engine()
    if not error and result and not text2sql_engine.is_empty_result(result):
        return "generate_answer"

    # 재시도 횟수가 2회 이상이면 답변 생성 (오류 메시지 포함)
    if retry_count >= 2:
        return "generate_answer"

    # 재시도
    return "database_query"
