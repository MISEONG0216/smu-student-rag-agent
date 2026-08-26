from langchain.agents import create_agent
# 방금 만든 4개의 도구도 함께 import 해옵니다.
from tools import (
    FILE_TOOLS, 
    get_current_price, 
    get_historical_ohlcv, 
    get_financial_metrics, 
    get_company_disclosures
)

def create_coding_agent():
    system_prompt = """당신은 주식 정보를 알려주는 웹사이트를 만드는 전문 에이전트입니다.

다음과 같은 작업을 수행할 수 있습니다:
- 주식 정보 조회: 현재가, 변동폭, 최근 추이 등을 가져옵니다
- 파일 시스템 작업: 파일 읽기, 쓰기, 삭제, 디렉터리 생성 및 목록 조회
- Python 코드 실행: 웹사이트 구현 및 테스트를 위한 코드 실행

사용자의 요청을 정확히 이해하고, 적절한 도구를 사용하여 작업을 수행하세요.
파일 경로는 상대 경로 또는 절대 경로를 모두 지원합니다.

작업 수행 시 다음 사항을 유의하세요:
1. 파일을 수정하기 전에 먼저 읽어서 내용을 확인하세요
2. 중요한 파일을 삭제하기 전에 사용자에게 확인을 요청하세요
3. 코드 실행 시 보안과 안전성을 고려하세요
4. 에러가 발생하면 명확하게 설명하고 해결 방법을 제시하세요
5. 응답은 한국어로 작성하세요

웹사이트를 만들 때는 사용자가 종목코드를 입력하면 현재 주가와 최근 추이를 볼 수 있는 간단하고 직관적인 UI를 우선 구현하세요."""

    # 1. 새로 만든 주식 도구들을 리스트로 묶어줍니다.
    STOCK_TOOLS = [
        get_current_price,
        get_historical_ohlcv,
        get_financial_metrics,
        get_company_disclosures
    ]
    
    # 2. 기존 FILE_TOOLS 리스트와 STOCK_TOOLS 리스트를 하나로 합칩니다.
    all_tools = FILE_TOOLS + STOCK_TOOLS

    agent_executor = create_agent(
        model="gpt-4o-mini", # (참고) 존재하는 모델명으로 임의 수정했습니다.
        tools=all_tools,     # 합쳐진 전체 도구 리스트를 주입합니다.
        system_prompt=system_prompt
    )

    return agent_executor


agent = create_coding_agent()