from langchain.tools import tool
import subprocess
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Union, Optional
from urllib.parse import quote
from urllib.request import urlopen, Request

import pandas as pd
import yfinance as yf

# ============================================
# 주식 정보 조회 도구
# ============================================

@tool(parse_docstring=True)
def get_current_price(symbol: str) -> str:
    """실시간 주가 조회

    Args:
        symbol: 주식 심볼 (예: AAPL, MSFT, TSLA)

    Returns:
        현재가, 전일 대비 등락률, 당일 거래량
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?interval=1d&range=1d"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        result = data.get("chart", {}).get("result")
        if not result:
            error = data.get("chart", {}).get("error")
            return f"오류: 주식 정보를 찾을 수 없습니다. {error or ''}".strip()

        meta = result[0].get("meta", {})
        current_price = meta.get("regularMarketPrice")
        change_percent = meta.get("regularMarketChangePercent")
        volume = meta.get("regularMarketVolume")
        currency = meta.get("currency", "")
        name = meta.get("shortName") or meta.get("longName") or symbol
        market_time = meta.get("regularMarketTime")

        time_str = ""
        if market_time:
            time_str = datetime.fromtimestamp(market_time).strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"종목: {name} ({symbol.upper()})\n"
            f"현재가: {current_price} {currency}\n"
            f"전일 대비 등락률: {change_percent}%\n"
            f"당일 거래량: {volume}\n"
            f"기준 시각: {time_str if time_str else '정보 없음'}"
        )
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def get_historical_ohlcv(symbol: str, start_date: str, end_date: str, timeframe: str = "day") -> str:
    """과거 주가 조회

    Args:
        symbol: 주식 심볼 (예: AAPL, MSFT, TSLA)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        timeframe: 기준 (day, week, month)

    Returns:
        날짜별 시가(Open), 고가(High), 저가(Low), 종가(Close), 거래량(Volume) 배열
    """
    try:
        if timeframe not in {"day", "week", "month"}:
            return "오류: timeframe은 day, week, month 중 하나여야 합니다."

        interval_map = {"day": "1d", "week": "1wk", "month": "1mo"}
        period1 = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        period2 = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        interval = interval_map[timeframe]

        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
            f"?period1={period1}&period2={period2}&interval={interval}&includePrePost=false&events=div%2Csplits"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        result = data.get("chart", {}).get("result")
        if not result:
            error = data.get("chart", {}).get("error")
            return f"오류: 주식 정보를 찾을 수 없습니다. {error or ''}".strip()

        timestamps = result[0].get("timestamp", [])
        quote_data = result[0].get("indicators", {}).get("quote", [{}])[0]
        opens = quote_data.get("open", [])
        highs = quote_data.get("high", [])
        lows = quote_data.get("low", [])
        closes = quote_data.get("close", [])
        volumes = quote_data.get("volume", [])

        lines = [f"종목: {symbol.upper()} 과거 OHLCV ({timeframe})"]
        for i, ts in enumerate(timestamps):
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            lines.append(
                f"- {date_str}: O={opens[i]}, H={highs[i]}, L={lows[i]}, C={closes[i]}, V={volumes[i]}"
            )

        return "\n".join(lines) if len(lines) > 1 else f"오류: OHLCV 데이터를 가져오지 못했습니다: {symbol}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def get_financial_metrics(ticker: str, period: str = "annual") -> Dict[str, Union[str, float, None]]:
    """투자 판단에 즉시 사용되는 핵심 재무 지표를 요약하여 반환합니다.

    Args:
        ticker: 종목 코드 (예: AAPL, 005930.KS)
        period: annual(연간) 또는 quarterly(분기)

    Returns:
        매출액, 영업이익, PER, PBR, ROE가 포함된 딕셔너리
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        per = info.get("trailingPE")
        pbr = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = round(roe * 100, 2)

        financials = stock.financials if period == "annual" else stock.quarterly_financials
        revenue = None
        operating_income = None
        target_date = None

        if not financials.empty:
            latest_col = financials.columns[0]
            target_date = latest_col.strftime("%Y-%m-%d")

            if "Total Revenue" in financials.index:
                revenue = financials.loc["Total Revenue", latest_col]

            if "Operating Income" in financials.index:
                operating_income = financials.loc["Operating Income", latest_col]

        return {
            "ticker": ticker,
            "report_date": target_date,
            "period": period,
            "revenue": int(revenue) if pd.notna(revenue) else None,
            "operating_income": int(operating_income) if pd.notna(operating_income) else None,
            "PER": round(per, 2) if per else None,
            "PBR": round(pbr, 2) if pbr else None,
            "ROE_percent": roe,
        }
    except Exception as e:
        return {"error": f"재무 데이터를 불러오는 중 오류가 발생했습니다: {str(e)}"}


@tool(parse_docstring=True)
def get_company_disclosures(ticker: str, start_date: str = None, end_date: str = None, keyword: Optional[str] = None) -> List[Dict[str, str]]:
    """해당 기업의 주요 뉴스 헤드라인 및 정보(공시 성격 포함)를 제공합니다.

    Args:
        ticker: 종목 코드
        start_date: 검색 시작일 (YYYY-MM-DD)
        end_date: 검색 종료일 (YYYY-MM-DD)
        keyword: 기사 제목 필터링용 키워드

    Returns:
        날짜, 제목, 출처, 링크가 포함된 딕셔너리 리스트
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        if not news_items:
            return []

        filtered_news = []
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        for item in news_items:
            pub_time = item.get("providerPublishTime")
            if not pub_time:
                continue

            news_date = datetime.fromtimestamp(pub_time)
            title = item.get("title", "")

            if start_dt and news_date < start_dt:
                continue
            if end_dt and news_date > end_dt:
                continue
            if keyword and keyword.lower() not in title.lower():
                continue

            filtered_news.append({
                "date": news_date.strftime("%Y-%m-%d %H:%M:%S"),
                "title": title,
                "publisher": item.get("publisher", "Unknown"),
                "link": item.get("link", "")
            })

        return filtered_news
    except Exception:
        return []


# ============================================
# 파일 시스템 도구
# ============================================

@tool(parse_docstring=True)
def read_file(file_path: str) -> str:
    """파일의 내용을 읽어서 반환합니다.

    Args:
        file_path: 읽을 파일의 경로 (상대 경로 또는 절대 경로)

    Returns:
        파일 내용 또는 오류 메시지
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        line_count = len(content.split("\n"))
        return f"파일: {file_path}\n총 {line_count}줄\n\n{content}"
    except FileNotFoundError:
        return f"오류: 파일을 찾을 수 없습니다: {file_path}"
    except PermissionError:
        return f"오류: 파일에 대한 읽기 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def write_file(file_path: str, content: str) -> str:
    """파일에 내용을 작성합니다. 파일이 없으면 생성하고, 있으면 덮어씁니다.

    Args:
        file_path: 작성할 파일의 경로
        content: 파일에 쓸 내용

    Returns:
        성공 메시지 또는 오류 메시지
    """
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        line_count = len(content.split("\n"))
        return f"성공: 파일이 작성되었습니다: {file_path} (총 {line_count}줄)"
    except PermissionError:
        return f"오류: 파일에 대한 쓰기 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def delete_file(file_path: str) -> str:
    """파일을 삭제합니다.

    Args:
        file_path: 삭제할 파일의 경로

    Returns:
        성공 메시지 또는 오류 메시지
    """
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            return f"성공: 파일이 삭제되었습니다: {file_path}"
        else:
            return f"오류: 파일을 찾을 수 없습니다: {file_path}"
    except PermissionError:
        return f"오류: 파일에 대한 삭제 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def create_directory(dir_path: str) -> str:
    """새로운 디렉터리를 생성합니다.

    Args:
        dir_path: 생성할 디렉터리의 경로

    Returns:
        성공 메시지 또는 오류 메시지
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        return f"성공: 디렉터리가 생성되었습니다: {dir_path}"
    except PermissionError:
        return f"오류: 디렉터리 생성 권한이 없습니다: {dir_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def list_directory(dir_path: str = ".") -> str:
    """디렉터리의 파일과 폴더 목록을 반환합니다.

    Args:
        dir_path: 조회할 디렉터리 경로 (기본값: 현재 디렉터리)

    Returns:
        파일 및 폴더 목록 또는 오류 메시지
    """
    try:
        if not os.path.exists(dir_path):
            return f"오류: 디렉터리를 찾을 수 없습니다: {dir_path}"

        if not os.path.isdir(dir_path):
            return f"오류: {dir_path}는 디렉터리가 아닙니다"

        items = os.listdir(dir_path)

        if not items:
            return f"디렉터리가 비어있습니다: {dir_path}"

        folders = []
        files = []

        for item in sorted(items):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                folders.append(f"[폴더] {item}/")
            else:
                size = os.path.getsize(item_path)
                files.append(f"[파일] {item} ({size} bytes)")

        result = f"디렉터리: {dir_path}\n\n"

        if folders:
            result += "폴더:\n" + "\n".join(folders) + "\n\n"

        if files:
            result += "파일:\n" + "\n".join(files)

        return result

    except PermissionError:
        return f"오류: 디렉터리에 대한 읽기 권한이 없습니다: {dir_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def execute_python_code(code: str) -> str:
    """Python 코드를 실행하고 결과를 반환합니다.

    Args:
        code: 실행할 Python 코드 문자열

    Returns:
        코드 실행 결과 또는 오류 메시지
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.getcwd()
        )

        output_parts = []

        if result.stdout:
            output_parts.append(f"출력:\n{result.stdout.strip()}")

        if result.stderr:
            output_parts.append(f"오류:\n{result.stderr.strip()}")

        if result.returncode == 0:
            if output_parts:
                return "실행 성공\n\n" + "\n\n".join(output_parts)
            else:
                return "실행 성공 (출력 없음)"
        else:
            return f"실행 실패 (종료 코드: {result.returncode})\n\n" + "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return "오류: 코드 실행 시간이 10초를 초과했습니다."
    except Exception as e:
        return f"오류: {str(e)}"


FILE_TOOLS = [
    get_current_price,
    get_historical_ohlcv,
    get_financial_metrics,
    get_company_disclosures,
    read_file,
    write_file,
    delete_file,
    create_directory,
    list_directory,
    execute_python_code
]
