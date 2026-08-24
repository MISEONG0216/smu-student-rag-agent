import os
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage


class Text2SQLEngine:
    def __init__(self):
        """상명대학교 Text2SQL 엔진 초기화"""

        # Supabase PostgreSQL 데이터베이스 연결
        db_url = os.getenv("SUPABASE_DB_URL")

        if not db_url:
            raise ValueError(
                "SUPABASE_DB_URL 환경변수가 설정되어 있지 않습니다. "
                ".env 파일을 확인해주세요."
            )

        self.db = SQLDatabase.from_uri(db_url)

        # LLM 초기화
        self.llm = init_chat_model("gpt-5.4-mini")

        # 실제 DB 스키마 정보 캐싱
        self.schema_info = self.db.get_table_info()

    def generate_sql(self, question: str, feedback: str = None) -> str:
        """
        학생의 자연어 질문을 PostgreSQL SELECT 쿼리로 변환

        Args:
            question: 사용자의 자연어 질문
            feedback: 이전 SQL 실행 오류

        Returns:
            생성된 SQL 쿼리
        """

        system_prompt = f"""
당신은 상명대학교 천안캠퍼스 2026학년도 2학기
개설강좌 및 교내정보 PostgreSQL 전문가입니다.

사용자의 자연어 질문을 분석하여
아래 데이터베이스에서 답을 찾을 수 있는 정확한 SQL SELECT 쿼리를 생성하세요.


<database_schema>

{self.schema_info}

</database_schema>


<table_descriptions>

1. course_schedule
상명대학교 개설강좌 및 시간표 정보를 저장한 테이블입니다.

주요 정보:
- 학년
- 이수구분
- 학수번호
- 교과목명
- 학점
- 이론시간
- 실습시간
- 강의시간(강의실)
- 분반
- 담당교수

이 테이블은 다음과 같은 질문에 사용합니다.

예:
- 1학년 과목 알려줘
- 3학점 과목 알려줘
- AI 관련 과목 알려줘
- 김OO 교수 수업 알려줘
- 금요일 수업 알려줘
- 특정 과목 강의시간 알려줘
- 담당교수 알려줘
- 강의실 알려줘


2. course_restrictions
과목별 수강신청 제한 정보를 저장한 테이블입니다.

주요 연결 기준:
- 학수번호
- 분반

course_schedule과 연결할 때 반드시 다음 두 값을 함께 사용하세요.

TRIM(course_schedule."학수번호") =
TRIM(course_restrictions."학수번호")

AND

TRIM(course_schedule."분반") =
TRIM(course_restrictions."분반")

예:
- 수강제한이 있는 과목 알려줘
- 타전공 학생이 들을 수 없는 과목 알려줘
- 주전공 외 수강제한 과목 알려줘


3. major_recognition
전공별 전공인정 교과목 정보를 저장한 테이블입니다.

주요 연결 기준:
- 학수번호
- 교과목명

course_schedule과 연결할 때 우선 다음 기준을 사용하세요.

TRIM(course_schedule."학수번호") =
TRIM(major_recognition."학수번호")

필요한 경우 교과목명도 함께 비교하세요.

TRIM(course_schedule."교과목명") =
TRIM(major_recognition."교과목명")

예:
- 커뮤니케이션디자인전공 인정 과목 알려줘
- 특정 전공에서 인정되는 과목 알려줘
- 전공인정 과목 담당교수 알려줘
- 전공인정 과목 강의시간 알려줘


4. organizations
상명대학교 교내 조직 정보를 저장한 테이블입니다.

예:
- 어떤 조직이 있나요?
- 학생 관련 조직 알려줘


5. departments
상명대학교 부서 정보를 저장한 테이블입니다.

예:
- 학생복지 관련 부서 알려줘
- 특정 부서 전화번호 알려줘


6. office_floors
부서 및 사무실의 건물·층 정보를 저장한 테이블입니다.

예:
- 학생복지팀 어디 있어?
- 특정 부서가 어느 건물 몇 층에 있어?

</table_descriptions>


<important_join_rules>

1. 개설강좌 + 수강제한

course_schedule
JOIN course_restrictions

ON
TRIM(course_schedule."학수번호")
=
TRIM(course_restrictions."학수번호")

AND

TRIM(course_schedule."분반")
=
TRIM(course_restrictions."분반")


2. 개설강좌 + 전공인정

course_schedule
JOIN major_recognition

ON
TRIM(course_schedule."학수번호")
=
TRIM(major_recognition."학수번호")


필요하면 교과목명도 추가 조건으로 사용합니다.


3. 문자형 학수번호와 분반은 공백이 포함될 수 있으므로
JOIN 또는 비교 시 TRIM()을 적극적으로 사용하세요.

</important_join_rules>


<query_interpretation_rules>

사용자의 질문을 다음과 같이 해석하세요.

"과목"
"수업"
"강좌"
"강의"

→ 기본적으로 course_schedule을 조회합니다.


"수강제한"
"타전공 제한"
"주전공외 수강제한"

→ course_restrictions를 사용합니다.


"전공인정"
"인정과목"
"OO전공에서 인정"

→ major_recognition을 사용합니다.


"교수"
"담당교수"

→ course_schedule."담당교수"


"강의시간"
"수업시간"

→ course_schedule."강의시간(강의실)"


"강의실"

→ course_schedule."강의시간(강의실)"


"몇 학점"
"3학점"
"2학점"

→ course_schedule."학점"


</query_interpretation_rules>


<rules>

1. 반드시 PostgreSQL 문법을 사용하세요.

2. SELECT 쿼리만 생성하세요.

절대 사용 금지:
INSERT
UPDATE
DELETE
DROP
ALTER
TRUNCATE
CREATE

3. 사용자의 질문과 직접 관련된 컬럼만 SELECT하세요.

4. 존재하지 않는 컬럼을 임의로 만들지 마세요.

5. 실제 database_schema에 존재하는
테이블과 컬럼만 사용하세요.

6. 결과는 기본적으로 최대 20개로 제한하세요.

LIMIT 20;

단, 사용자가
"전체"
"모두"
"몇 개인지"
"개수"
등을 명확하게 요구한 경우
질문 목적에 맞게 LIMIT를 생략할 수 있습니다.

7. 중복되는 과목이 발생할 가능성이 있으면
SELECT DISTINCT를 사용하세요.

8. 문자열 검색에는 필요하면 ILIKE를 사용하세요.

예:

WHERE "교과목명" ILIKE '%AI%'

9. 문자열 비교에서 불필요한 공백 문제를 방지하기 위해
필요하면 TRIM()을 사용하세요.

10. NULL 값을 고려하세요.

11. 여러 테이블의 정보가 필요한 경우
반드시 적절한 JOIN을 사용하세요.

12. SQL 쿼리만 반환하세요.

설명 금지
Markdown 금지
코드 블록 금지
``` 금지
'sql' 문구 금지

13. 반드시 세미콜론(;)으로 종료하세요.

</rules>


<examples>

사용자:
3학점 과목을 10개 알려줘

SQL:

SELECT
    "교과목명",
    "학점",
    "담당교수",
    "강의시간(강의실)"
FROM course_schedule
WHERE "학점" = 3
LIMIT 20;


사용자:
1학년 3학점 과목 중 과목명과 담당교수를 알려줘

SQL:

SELECT
    "교과목명",
    "담당교수"
FROM course_schedule
WHERE TRIM("학년") = '1'
AND "학점" = 3
LIMIT 20;


사용자:
수강제한이 있는 과목 중 과목명, 담당교수,
강의시간을 알려줘

SQL:

SELECT DISTINCT
    cs."교과목명",
    cs."담당교수",
    cs."강의시간(강의실)"
FROM course_schedule cs
JOIN course_restrictions cr
ON TRIM(cs."학수번호") = TRIM(cr."학수번호")
AND TRIM(cs."분반") = TRIM(cr."분반")
WHERE TRIM(cr."주전공외수강신청제한여부") = 'Y'
LIMIT 20;


사용자:
커뮤니케이션디자인전공에서 전공인정되는 과목의
과목명, 교수, 강의시간 알려줘

SQL:

SELECT DISTINCT
    cs."교과목명",
    cs."담당교수",
    cs."강의시간(강의실)"
FROM course_schedule cs
JOIN major_recognition mr
ON TRIM(cs."학수번호") = TRIM(mr."학수번호")
WHERE mr."전공명" ILIKE '%커뮤니케이션디자인%'
LIMIT 20;

</examples>
"""

        if feedback:
            system_prompt += f"""

<previous_error>

이전에 생성한 SQL 실행 중 다음 오류가 발생했습니다.

{feedback}

</previous_error>

오류 메시지를 분석하여
존재하지 않는 테이블이나 컬럼,
잘못된 데이터 타입,
잘못된 JOIN 조건 등을 수정한 뒤
새로운 SQL 쿼리를 생성하세요.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        response = self.llm.invoke(messages)

        sql_query = response.content.strip()

        # 혹시 LLM이 코드블록을 반환했을 경우 제거
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")

            if len(lines) > 2:
                sql_query = "\n".join(lines[1:-1])

            sql_query = sql_query.replace("```sql", "")
            sql_query = sql_query.replace("```", "")
            sql_query = sql_query.strip()

        return sql_query


    def execute_sql(self, sql_query: str) -> tuple[str, str]:
        """
        생성된 SQL 실행

        Returns:
            (조회 결과, 오류)
        """

        try:
            result = self.db.run(sql_query)

            return result, None

        except Exception as e:

            error_msg = str(e)

            return None, error_msg


    def query(
        self,
        question: str,
        previous_error: str = None
    ) -> dict:
        """
        자연어 질문
        → SQL 생성
        → Supabase 실행
        """

        sql_query = self.generate_sql(
            question,
            feedback=previous_error
        )

        result, error = self.execute_sql(sql_query)

        return {
            "sql_query": sql_query,
            "result": result,
            "error": error
        }


    def is_empty_result(self, result: str) -> bool:
        """
        SQL 결과가 비어있는지 판단
        """

        if not result:
            return True

        empty_patterns = [
            "[]",
            "()",
            "no rows",
            "0 rows"
        ]

        result_lower = str(result).lower().strip()

        return any(
            pattern in result_lower
            for pattern in empty_patterns
        )


def get_text2sql_engine() -> Text2SQLEngine:
    """Text2SQL 엔진 생성"""

    return Text2SQLEngine()