"""
LAWI Trading System - Data Fetcher
Pulls OHLCV data from Yahoo Finance for all 4 assets.
No MT5 required — works 24/7 even when laptop is off.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .config import SYMBOLS, LOOKBACK_DAYS, TRAIN_DAYS

logger = logging.getLogger(__name__)


def fetch_ohlcv(ticker: str, days: int = LOOKBACK_DAYS, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker:   Yahoo Finance ticker string  (e.g. "GC=F")
        days:     Number of calendar days to look back
        interval: Yahoo Finance interval ('1h', '1d', etc.)

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex (UTC).
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=days)

    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()

    if raw.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()

    # Flatten multi-level columns if present (yfinance ≥ 0.2 returns MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    logger.info(f"Fetched {len(df)} rows for {ticker} ({interval})")
    return df


def fetch_daily(ticker: str, days: int = TRAIN_DAYS) -> pd.DataFrame:
    """Fetch daily OHLCV — used for training and Daily Bias prediction."""
    return fetch_ohlcv(ticker, days=days, interval="1d")


def fetch_hourly(ticker: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch hourly OHLCV.
    yfinance limits 1h data to the last 730 days max.
    """
    return fetch_ohlcv(ticker, days=min(days, 729), interval="1h")


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H data to 4H bars (yfinance has no native 4H for all assets)."""
    if df_1h.empty:
        return df_1h

    df_4h = df_1h.resample("4H").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()

    return df_4h


def fetch_all_daily() -> dict[str, pd.DataFrame]:
    """Fetch daily data for all LAWI symbols. Returns {symbol: DataFrame}."""
    result = {}
    for symbol, ticker in SYMBOLS.items():
        df = fetch_daily(ticker)
        if not df.empty:
            result[symbol] = df
            logger.info(f"[{symbol}] daily data OK — {len(df)} bars")
        else:
            logger.error(f"[{symbol}] FAILED to fetch daily data")
    return result


def fetch_all_hourly() -> dict[str, pd.DataFrame]:
    """Fetch hourly data for all LAWI symbols. Returns {symbol: DataFrame}."""
    result = {}
    for symbol, ticker in SYMBOLS.items():
        df = fetch_hourly(ticker)
        if not df.empty:
            result[symbol] = df
            logger.info(f"[{symbol}] hourly data OK — {len(df)} bars")
        else:
            logger.error(f"[{symbol}] FAILED to fetch hourly data")
    return result


def get_current_price(symbol: str) -> float | None:
    """Get the latest closing price for a symbol."""
    ticker = SYMBOLS.get(symbol)
    if not ticker:
        return None
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.error(f"get_current_price failed for {symbol}: {e}")
        return None
