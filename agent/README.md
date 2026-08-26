# Agent System

LangChain create_agent 기반 에이전트 시스템 구현 실습 자료입니다.

## 환경 설정

### 1. uv 설치

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### 2. 가상환경 생성 및 패키지 설치

```bash
cd smu-ai-service-bootcamp
cd agent-system

# pyproject.toml을 기반으로 가상환경 생성 및 패키지 설치
uv sync

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate
```

---

### 3. Jupyter Notebook 커널 등록

VS Code에서 Jupyter Notebook을 사용하려면 커널을 등록해야 합니다.

#### Windows

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name=ai-service-agent --display-name="ai service agent"
```

#### macOS/Linux

```bash
.venv/bin/python -m ipykernel install --user --name=ai-service-agent --display-name="ai service agent"
```

커널 등록 후 **VS Code를 리로드**하면 노트북에서 "ai service agent" 커널을 선택할 수 있습니다.


### 4. 환경 변수 설정

루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 작성하세요:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. LangGraph Studio 실행

```bash
# LangGraph Studio 시작
uv run langgraph dev
```

#### Windows (PowerShell)

```powershell
$env:PYTHONUTF8=1; uv run langgraph dev --no-reload --allow-blocking
```

#### Mac/Linux (bash/zsh)

```
PYTHONUTF8=1 uv run langgraph dev --no-reload --allow-blocking
```

브라우저에서 `http://127.0.0.1:2024` 자동 열림

📈 주식 정보 웹사이트 구축 에이전트 (Stock Info Agent)
이 프로젝트는 사용자가 종목 코드를 입력하면 실시간 주가, 과거 추이, 재무 지표, 관련 뉴스를 수집하여 주식 정보 웹사이트를 구축해 주는 LLM 기반 에이전트를 만드는 과정입니다.

📌 1. 프로젝트 핵심 내용
목적: 금융 데이터 수집 도구(Tool)를 만들고, 이를 파일 제어 도구와 결합하여 자율적으로 작동하는 AI 에이전트 구축

주요 기능: 주가 실시간 조회, 과거 차트 데이터 수집, 핵심 재무지표 요약, 관련 뉴스 검색

⚙️ 2. 사전 준비 (라이브러리 설치)
코드를 정상적으로 실행하려면 터미널(CMD, PowerShell, Mac Terminal 등)에서 아래 명령어를 입력하여 필수 패키지를 설치해야 합니다.

⚠️ 주의: 파이썬 쉘(>>> 프롬프트) 내부가 아닌, 일반 터미널 바탕 경로(PS C:\... >)에서 실행해야 합니다!

Bash
# 금융 데이터 수집을 위한 라이브러리 설치
pip install yfinance pandas

# LLM 에이전트 및 도구 구성을 위한 라이브러리 설치
pip install langchain langchain-core
(참고: 설치 중 노란색 글씨로 PATH 관련 Warning이 뜨거나 [notice] 문구가 뜰 수 있으나, 설치 실패가 아니므로 무시하셔도 무방합니다.)

🛠️ 3. 구성된 도구 (Tools) 목록
에이전트가 주식 시장 데이터를 다루기 위해 만들어진 4가지 핵심 함수입니다. tools.py 파일 내에 정의되어 있습니다.

get_current_price: 특정 종목의 실시간 주가, 등락률, 거래량을 조회합니다.

get_historical_ohlcv: 특정 기간 동안의 시가, 고가, 저가, 종가, 거래량(과거 차트 데이터)을 배열 형태로 조회합니다.

get_financial_metrics: 투자의 핵심이 되는 재무 지표(PER, PBR, ROE, 매출액, 영업이익)를 요약해 반환합니다.

get_company_disclosures: 특정 키워드에 해당하는 기업의 최근 주요 공시 및 뉴스 헤드라인을 검색합니다.

🚀 4. 테스트 및 실행 방법
단계 1: 도구(Tool) 단독 테스트
도구들이 정상적으로 작동하는지 확인하려면 테스트 스크립트(test_tools.py)를 실행합니다.

Bash
# 터미널에서 아래 명령어 실행 (경로는 자신의 환경에 맞게 수정)
python test_tools.py
기대 결과: 애플(AAPL)의 현재가, 과거 데이터 배열, 재무 지표 딕셔너리 등이 터미널에 에러 없이 출력되어야 합니다.

단계 2: 에이전트(Agent) 연동
agent.py 파일에서 기존 파일 제어 도구(FILE_TOOLS)와 새로 만든 주식 도구 리스트(STOCK_TOOLS)를 합쳐 에이전트에게 주입합니다.

Python
from langchain.agents import create_agent
from tools import FILE_TOOLS, get_current_price, get_historical_ohlcv, get_financial_metrics, get_company_disclosures

def create_coding_agent():
    # ... (시스템 프롬프트 생략) ...

    # 만든 주식 도구들을 묶습니다.
    STOCK_TOOLS = [
        get_current_price,
        get_historical_ohlcv,
        get_financial_metrics,
        get_company_disclosures
    ]
    
    # 파일 제어 도구와 주식 도구를 하나로 합쳐 에이전트에게 전달합니다.
    all_tools = FILE_TOOLS + STOCK_TOOLS

    agent_executor = create_agent(
        model="gpt-4o-mini",
        tools=all_tools,
        system_prompt=system_prompt
    )

    return agent_executor

agent = create_coding_agent()
💡 5. 팀원들을 위한 트러블슈팅 가이드
ModuleNotFoundError: No module named 'langchain' 에러 발생 시: 해당 패키지가 설치되지 않은 것입니다. pip install langchain을 다시 실행해 주세요.

SyntaxError: invalid syntax (pip install 입력 시): 현재 파이썬 모드(>>>)에 들어가 있는 상태입니다. exit()를 입력하여 빠져나온 뒤 다시 설치 명령어를 입력하세요.

장 마감/주말 데이터 조회: get_current_price 사용 시 change_rate가 None으로 뜰 수 있으나, 이는 API 특성상 장 마감 후 발생할 수 있는 일시적 현상이니 안심하셔도 됩니다.
