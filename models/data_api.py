from __future__ import annotations
from datetime import datetime, timedelta
import os
import pandas as pd
import requests
import streamlit as st

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "SPY"]
MARKET_INDEX = "^GSPC"
DEFAULT_START_DATE = (datetime.today() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")

def _extract_close(data) -> pd.Series:
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)

    if isinstance(data, pd.Series):
        return pd.to_numeric(data, errors="coerce").dropna()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return pd.Series(dtype=float)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return pd.to_numeric(close, errors="coerce").dropna()

@st.cache_data(ttl=3600)
def get_market_data(tickers=None, start_date: str = DEFAULT_START_DATE, end_date: str | None = None):
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame(), pd.Series(dtype=float)

    if tickers is None:
        tickers = DEFAULT_TICKERS
    if isinstance(tickers, str):
        tickers = [tickers]

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.Timestamp.today().normalize() if end_date is None else pd.to_datetime(end_date)
    if end_ts <= start_ts:
        end_ts = start_ts + pd.Timedelta(days=30)

    prices = pd.DataFrame()
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_ts.strftime("%Y-%m-%d"),
                end=end_ts.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
            close = _extract_close(df)
            if not close.empty:
                prices[ticker.upper()] = close
        except Exception:
            continue

    try:
        market_raw = yf.download(
            MARKET_INDEX,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
        market = _extract_close(market_raw)
    except Exception:
        market = pd.Series(dtype=float)

    if market.empty and "SPY" in prices.columns:
        market = prices["SPY"].dropna().copy()

    return prices.dropna(how="all"), market.dropna()

@st.cache_data(ttl=21600)
def get_macro_data():
    return {
        "risk_free_rate": get_risk_free_rate(),
        "inflation_rate": get_inflation_rate(),
        "usd_cop": get_exchange_rate(),
    }

@st.cache_data(ttl=21600)
def get_risk_free_rate() -> float:
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1]) / 100
    except Exception:
        pass
    return 0.03

@st.cache_data(ttl=21600)
def get_inflation_rate() -> float:
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return 0.025
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "CPIAUCSL",
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 2,
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        obs = response.json().get("observations", [])
        vals = [float(x["value"]) for x in obs if x.get("value", ".") != "."]
        if len(vals) >= 2 and vals[1] != 0:
            monthly = (vals[0] / vals[1]) - 1
            return float(monthly * 12)
    except Exception:
        pass
    return 0.025

@st.cache_data(ttl=21600)
def get_exchange_rate() -> float:
    try:
        import yfinance as yf
        fx = yf.Ticker("USDCOP=X")
        hist = fx.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 4000.0
