"""
LAWI Trading System - ICT 2022 Feature Engineering
Implements Smart Money Concepts from the ICT 2022 Mentorship:
  • Order Blocks (OB)
  • Fair Value Gaps (FVG)
  • Optimal Trade Entry (OTE — 0.618 / 0.705 Fib)
  • Market Structure (HH, HL, LH, LL, MSS, BOS)
  • Liquidity Pools (BSL / SSL sweeps)
  • Kill Zone flags
  • Premium / Discount (above / below 50% of range)
  • Power of 3 (Accumulation → Manipulation → Distribution)
"""

import pandas as pd
import numpy as np
from .config import KILL_ZONES

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _swing_highs(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Return True at bars that are a swing high (local max over left+right window)."""
    result = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left: i + right + 1]
        if series.iloc[i] == window.max():
            result.iloc[i] = True
    return result


def _swing_lows(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Return True at bars that are a swing low (local min over left+right window)."""
    result = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left: i + right + 1]
        if series.iloc[i] == window.min():
            result.iloc[i] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Order Blocks
# ─────────────────────────────────────────────────────────────────────────────

def add_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish OB: last down-close candle before a strong up-move (displacement).
    Bearish OB: last up-close candle before a strong down-move.
    Adds: ob_bullish (0/1), ob_bearish (0/1), ob_strength (float)
    """
    df = df.copy()
    body   = (df["Close"] - df["Open"]).abs()
    avg_body = body.rolling(20).mean()

    # Displacement = candle body > 1.5x average
    displacement = body > (1.5 * avg_body)

    ob_bull = pd.Series(0, index=df.index)
    ob_bear = pd.Series(0, index=df.index)
    ob_str  = pd.Series(0.0, index=df.index)

    for i in range(2, len(df)):
        # Bullish OB: down candle at i-1, strong up displacement at i
        if (df["Close"].iloc[i-1] < df["Open"].iloc[i-1]
                and df["Close"].iloc[i] > df["Open"].iloc[i]
                and displacement.iloc[i]):
            ob_bull.iloc[i-1] = 1
            ob_str.iloc[i-1]  = body.iloc[i] / (avg_body.iloc[i] + 1e-8)

        # Bearish OB: up candle at i-1, strong down displacement at i
        if (df["Close"].iloc[i-1] > df["Open"].iloc[i-1]
                and df["Close"].iloc[i] < df["Open"].iloc[i]
                and displacement.iloc[i]):
            ob_bear.iloc[i-1] = 1
            ob_str.iloc[i-1]  = body.iloc[i] / (avg_body.iloc[i] + 1e-8)

    df["ob_bullish"] = ob_bull
    df["ob_bearish"] = ob_bear
    df["ob_strength"] = ob_str

    # Rolling flags: is there a recent OB within last 5 bars?
    df["ob_bull_nearby"] = df["ob_bullish"].rolling(5).max()
    df["ob_bear_nearby"] = df["ob_bearish"].rolling(5).max()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fair Value Gaps (FVG / Imbalance)
# ─────────────────────────────────────────────────────────────────────────────

def add_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish FVG: Low[i] > High[i-2]  (gap between candle i-2 top and candle i bottom)
    Bearish FVG: High[i] < Low[i-2]
    Adds: fvg_bullish, fvg_bearish, fvg_size (% of price)
    """
    df = df.copy()
    fvg_bull = pd.Series(0, index=df.index)
    fvg_bear = pd.Series(0, index=df.index)
    fvg_size = pd.Series(0.0, index=df.index)

    for i in range(2, len(df)):
        # Bullish FVG
        if df["Low"].iloc[i] > df["High"].iloc[i-2]:
            gap = df["Low"].iloc[i] - df["High"].iloc[i-2]
            fvg_bull.iloc[i] = 1
            fvg_size.iloc[i] = gap / df["Close"].iloc[i] * 100

        # Bearish FVG
        if df["High"].iloc[i] < df["Low"].iloc[i-2]:
            gap = df["Low"].iloc[i-2] - df["High"].iloc[i]
            fvg_bear.iloc[i] = 1
            fvg_size.iloc[i] = gap / df["Close"].iloc[i] * 100

    df["fvg_bullish"]    = fvg_bull
    df["fvg_bearish"]    = fvg_bear
    df["fvg_size_pct"]   = fvg_size
    df["fvg_bull_nearby"] = df["fvg_bullish"].rolling(5).max()
    df["fvg_bear_nearby"] = df["fvg_bearish"].rolling(5).max()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Market Structure (BOS / MSS / HH-HL-LH-LL)
# ─────────────────────────────────────────────────────────────────────────────

def add_market_structure(df: pd.DataFrame, swing_len: int = 3) -> pd.DataFrame:
    """
    Detects:
      - HH (Higher High), HL (Higher Low) → bullish structure
      - LH (Lower High), LL (Lower Low)   → bearish structure
      - BOS (Break of Structure)           → continuation
      - MSS (Market Structure Shift)       → reversal signal
    """
    df = df.copy()
    sh = _swing_highs(df["High"], swing_len, swing_len)
    sl = _swing_lows(df["Low"],  swing_len, swing_len)

    swing_h = df["High"].where(sh).ffill()
    swing_l = df["Low"].where(sl).ffill()

    prev_sh = swing_h.shift(1)
    prev_sl = swing_l.shift(1)

    df["hh"] = ((swing_h > prev_sh) & sh).astype(int)
    df["hl"] = ((swing_l > prev_sl) & sl).astype(int)
    df["lh"] = ((swing_h < prev_sh) & sh).astype(int)
    df["ll"] = ((swing_l < prev_sl) & sl).astype(int)

    # BOS bullish: close breaks above last swing high
    df["bos_bull"] = (df["Close"] > swing_h.shift(1)).astype(int)
    df["bos_bear"] = (df["Close"] < swing_l.shift(1)).astype(int)

    # MSS: bullish BOS after a bearish trend (ll present in last 10 bars)
    df["mss_bull"] = (
        df["bos_bull"] & df["ll"].rolling(10).max().astype(bool)
    ).astype(int)
    df["mss_bear"] = (
        df["bos_bear"] & df["hh"].rolling(10).max().astype(bool)
    ).astype(int)

    # Trend score: +1 per bull signal, -1 per bear signal (rolling 10 bars)
    bull_score = df[["hh","hl","bos_bull"]].rolling(10).sum().sum(axis=1)
    bear_score = df[["lh","ll","bos_bear"]].rolling(10).sum().sum(axis=1)
    df["structure_score"] = bull_score - bear_score   # positive = bullish bias
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Liquidity Pools (BSL / SSL sweeps)
# ─────────────────────────────────────────────────────────────────────────────

def add_liquidity(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    BSL (Buy-side Liquidity) = recent swing highs where stops cluster.
    SSL (Sell-side Liquidity) = recent swing lows.
    Sweep: price momentarily exceeds the level then reverses.
    """
    df = df.copy()
    roll_high = df["High"].rolling(lookback).max()
    roll_low  = df["Low"].rolling(lookback).min()

    # Price swept through BSL but closed below it (bull trap / distribution)
    df["bsl_sweep"] = (
        (df["High"] >= roll_high) & (df["Close"] < roll_high)
    ).astype(int)

    # Price swept through SSL but closed above it (bear trap / accumulation)
    df["ssl_sweep"] = (
        (df["Low"] <= roll_low) & (df["Close"] > roll_low)
    ).astype(int)

    # Relative position of price within recent range (0 = at low, 1 = at high)
    price_range = roll_high - roll_low
    df["range_position"] = (df["Close"] - roll_low) / price_range.replace(0, np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optimal Trade Entry (OTE — Fibonacci 0.618 / 0.705)
# ─────────────────────────────────────────────────────────────────────────────

def add_ote(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    OTE zone: between 61.8% and 70.5% retracement of the last significant swing.
    Bullish OTE: price retraces to 0.618-0.705 of a bullish swing → buy zone.
    Bearish OTE: price retraces to 0.618-0.705 of a bearish swing → sell zone.
    """
    df = df.copy()
    swing_high = df["High"].rolling(lookback).max()
    swing_low  = df["Low"].rolling(lookback).min()

    bull_ote_high = swing_high - 0.618 * (swing_high - swing_low)
    bull_ote_low  = swing_high - 0.705 * (swing_high - swing_low)

    bear_ote_low  = swing_low  + 0.618 * (swing_high - swing_low)
    bear_ote_high = swing_low  + 0.705 * (swing_high - swing_low)

    df["in_bull_ote"] = (
        (df["Close"] >= bull_ote_low) & (df["Close"] <= bull_ote_high)
    ).astype(int)

    df["in_bear_ote"] = (
        (df["Close"] >= bear_ote_low) & (df["Close"] <= bear_ote_high)
    ).astype(int)

    # Fibonacci retracement level of current price relative to swing
    df["fib_level"] = (swing_high - df["Close"]) / (swing_high - swing_low + 1e-8)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kill Zone Flags
# ─────────────────────────────────────────────────────────────────────────────

def add_kill_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for each ICT Kill Zone (based on UTC hour)."""
    df = df.copy()
    # Ensure we have UTC hour
    if df.index.tz is not None:
        hour = df.index.tz_convert("UTC").hour
    else:
        hour = df.index.hour

    hour_series = pd.Series(hour, index=df.index)

    for zone_name, (start_h, end_h) in KILL_ZONES.items():
        col = "kz_" + zone_name.lower().replace(" ", "_")
        df[col] = ((hour_series >= start_h) & (hour_series < end_h)).astype(int)

    df["in_kill_zone"] = df[[c for c in df.columns if c.startswith("kz_")]].max(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Premium / Discount & Power of 3
# ─────────────────────────────────────────────────────────────────────────────

def add_premium_discount(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Premium  = price above 50% of daily/weekly range → selling zone.
    Discount = price below 50%                       → buying zone.
    Equilibrium = near 50%
    """
    df = df.copy()
    high = df["High"].rolling(lookback).max()
    low  = df["Low"].rolling(lookback).min()
    mid  = (high + low) / 2

    df["premium"]    = (df["Close"] > mid).astype(int)
    df["discount"]   = (df["Close"] < mid).astype(int)
    df["equilibrium"] = ((df["Close"] - mid).abs() / (high - low + 1e-8) < 0.05).astype(int)
    return df


def add_power_of_three(df: pd.DataFrame) -> pd.DataFrame:
    """
    Power of 3 (AMD — Accumulation, Manipulation, Distribution):
    • Accumulation: low volume, tight range (early session)
    • Manipulation: a sweep (false breakout) — captured by ssl_sweep / bsl_sweep
    • Distribution: strong directional move after manipulation
    We encode these as session-phase features.
    """
    df = df.copy()
    if df.index.tz is not None:
        hour = df.index.tz_convert("UTC").hour
    else:
        hour = df.index.hour

    hour_s = pd.Series(hour, index=df.index)
    # Accumulation: Asian session (0-5 UTC)
    df["phase_accumulation"]  = ((hour_s >= 0)  & (hour_s < 5)).astype(int)
    # Manipulation: London open spike (7-9 UTC)
    df["phase_manipulation"]  = ((hour_s >= 7)  & (hour_s < 9)).astype(int)
    # Distribution: NY session (13-17 UTC)
    df["phase_distribution"]  = ((hour_s >= 13) & (hour_s < 17)).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Technical Indicators (momentum / volatility context)
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Standard indicators used as ML context features."""
    df = df.copy()

    # EMA stack
    for period in [8, 21, 50, 200]:
        df[f"ema{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

    # EMA alignment (bull if 8>21>50>200)
    df["ema_bull_stack"] = (
        (df["ema8"] > df["ema21"]) &
        (df["ema21"] > df["ema50"]) &
        (df["ema50"] > df["ema200"])
    ).astype(int)

    df["ema_bear_stack"] = (
        (df["ema8"] < df["ema21"]) &
        (df["ema21"] < df["ema50"]) &
        (df["ema50"] < df["ema200"])
    ).astype(int)

    # RSI (14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)

    # ATR (14) — normalised as % of price
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["Close"] * 100

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_bullish"] = (df["macd_hist"] > 0).astype(int)

    # Candle body characteristics
    df["body_size"]    = (df["Close"] - df["Open"]).abs()
    df["upper_wick"]   = df["High"] - df[["Open","Close"]].max(axis=1)
    df["lower_wick"]   = df[["Open","Close"]].min(axis=1) - df["Low"]
    df["is_bullish_candle"] = (df["Close"] > df["Open"]).astype(int)
    df["is_bearish_candle"] = (df["Close"] < df["Open"]).astype(int)

    # Day of week (0=Mon … 4=Fri) — ICT favors Tue/Wed/Thu for entries
    df["day_of_week"] = df.index.dayofweek

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. Label Generation (Bias Target)
# ─────────────────────────────────────────────────────────────────────────────

def generate_bias_label(df: pd.DataFrame, forward_bars: int = 5,
                         threshold_pct: float = 0.3) -> pd.DataFrame:
    """
    Creates the training target:
      1 = BULLISH  (future close goes up > threshold_pct within forward_bars)
      0 = BEARISH  (future close goes down > threshold_pct)
      2 = CONSOLIDATION (neither)
    """
    df = df.copy()
    future_close = df["Close"].shift(-forward_bars)
    pct_change   = (future_close - df["Close"]) / df["Close"] * 100

    df["bias_label"] = 2  # default: consolidation
    df.loc[pct_change >  threshold_pct, "bias_label"] = 1   # BULLISH
    df.loc[pct_change < -threshold_pct, "bias_label"] = 0   # BEARISH

    # Drop the last forward_bars rows (no future to label)
    df = df.iloc[:-forward_bars]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Order Blocks
    "ob_bullish", "ob_bearish", "ob_strength", "ob_bull_nearby", "ob_bear_nearby",
    # FVG
    "fvg_bullish", "fvg_bearish", "fvg_size_pct", "fvg_bull_nearby", "fvg_bear_nearby",
    # Market Structure
    "hh", "hl", "lh", "ll", "bos_bull", "bos_bear", "mss_bull", "mss_bear",
    "structure_score",
    # Liquidity
    "bsl_sweep", "ssl_sweep", "range_position",
    # OTE
    "in_bull_ote", "in_bear_ote", "fib_level",
    # Kill Zones
    "kz_asia", "kz_london", "kz_new_york_am", "kz_new_york_pm", "in_kill_zone",
    # Premium / Discount
    "premium", "discount", "equilibrium",
    # Power of 3
    "phase_accumulation", "phase_manipulation", "phase_distribution",
    # Indicators
    "ema_bull_stack", "ema_bear_stack",
    "rsi", "rsi_overbought", "rsi_oversold",
    "atr_pct", "macd_hist", "macd_bullish",
    "body_size", "upper_wick", "lower_wick",
    "is_bullish_candle", "is_bearish_candle",
    "day_of_week",
]


def build_features(df: pd.DataFrame, label: bool = True) -> pd.DataFrame:
    """
    Run the full ICT feature-engineering pipeline on a raw OHLCV DataFrame.

    Args:
        df:    Raw OHLCV DataFrame from data_fetcher
        label: If True, append bias_label column (for training)

    Returns:
        DataFrame with all ICT features (and label if requested), NaN rows dropped.
    """
    df = add_order_blocks(df)
    df = add_fvg(df)
    df = add_market_structure(df)
    df = add_liquidity(df)
    df = add_ote(df)
    df = add_kill_zones(df)
    df = add_premium_discount(df)
    df = add_power_of_three(df)
    df = add_indicators(df)

    if label:
        df = generate_bias_label(df)

    # Keep only feature columns (+ label)
    cols = FEATURE_COLUMNS.copy()
    if label and "bias_label" in df.columns:
        cols.append("bias_label")

    existing = [c for c in cols if c in df.columns]
    df = df[existing].dropna()
    return df
