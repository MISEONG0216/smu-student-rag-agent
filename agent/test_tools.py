"""domain-agent tools 테스트 스크립트

실행 예시:
    python test_tools.py
"""

import importlib.util
from pathlib import Path
import pprint


def load_tools_module():
    tools_path = Path(__file__).with_name("tools.py")
    spec = importlib.util.spec_from_file_location("tools_module", tools_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_test(name, func):
    print(f"\n=== {name} ===")
    try:
        result = func()
        if isinstance(result, (dict, list)):
            pprint.pprint(result, width=120)
        else:
            print(result)
    except Exception as e:
        print(f"실패: {e}")


if __name__ == "__main__":
    mod = load_tools_module()

    # 테스트 종목: 미국 주식 기준 AAPL
    ticker = "AAPL"

    run_test("get_current_price", lambda: mod.get_current_price.invoke({"symbol": ticker}) if hasattr(mod.get_current_price, "invoke") else mod.get_current_price(ticker))

    run_test(
        "get_historical_ohlcv",
        lambda: mod.get_historical_ohlcv.invoke({
            "symbol": ticker,
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "timeframe": "day"
        }) if hasattr(mod.get_historical_ohlcv, "invoke") else mod.get_historical_ohlcv(ticker, "2024-01-01", "2024-01-10", "day")
    )

    run_test(
        "get_financial_metrics",
        lambda: mod.get_financial_metrics.invoke({"ticker": ticker, "period": "annual"}) if hasattr(mod.get_financial_metrics, "invoke") else mod.get_financial_metrics(ticker, "annual")
    )

    run_test(
        "get_company_disclosures",
        lambda: mod.get_company_disclosures.invoke({"ticker": ticker, "keyword": "Apple"}) if hasattr(mod.get_company_disclosures, "invoke") else mod.get_company_disclosures(ticker, keyword="Apple")
    )

    # 파일 시스템 도구도 간단히 확인
    run_test("list_directory", lambda: mod.list_directory.invoke({"dir_path": "."}) if hasattr(mod.list_directory, "invoke") else mod.list_directory("."))
