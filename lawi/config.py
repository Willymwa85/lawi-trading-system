"""
LAWI Trading System - Configuration
Company: LAWI
Author: Willy Mwangi Kiarie
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Telegram ───────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── Yahoo Finance Tickers ───────────────────────────────────────────────────
SYMBOLS = {
    "XAUUSD": "GC=F",      # Gold Futures
    "GBPUSD": "GBPUSD=X",  # GBP/USD Forex
    "SP500":  "^GSPC",     # S&P 500 Index
    "US30":   "^DJI",      # Dow Jones / US30
}

# Human-friendly names for Telegram messages
SYMBOL_NAMES = {
    "XAUUSD": "🥇 Gold (XAUUSD)",
    "GBPUSD": "💷 GBP/USD",
    "SP500":  "📈 S&P 500",
    "US30":   "🏛️ Dow Jones (US30)",
}

# ─── Data Settings ───────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 60    # days of history to fetch for feature engineering
TRAIN_DAYS      = 365   # days of history to train the model on

# Timeframes to fetch (pandas/yfinance interval strings)
INTERVAL_1H  = "1h"
INTERVAL_4H  = "60m"   # yfinance uses '60m' for 4h we resample from 1h
INTERVAL_D1  = "1d"

# ─── ICT Kill Zones (UTC hours) ──────────────────────────────────────────────
KILL_ZONES = {
    "Asia":        (0,  5),    # 00:00 – 05:00 UTC
    "London":      (7,  10),   # 07:00 – 10:00 UTC
    "New York AM": (13, 16),   # 13:00 – 16:00 UTC
    "New York PM": (18, 20),   # 18:00 – 20:00 UTC
}

# ─── Model Settings ──────────────────────────────────────────────────────────
MODEL_DIR          = os.path.join(os.path.dirname(__file__), "..", "models")
BIAS_LABELS        = {0: "BEARISH", 1: "BULLISH", 2: "CONSOLIDATION"}
CONFIDENCE_MIN     = 0.55   # minimum confidence to send signal

# ─── Scheduler ───────────────────────────────────────────────────────────────
# Morning brief: 07:00 Toronto (Eastern) = 12:00 UTC (EST) / 11:00 UTC (EDT)
SCHEDULER_HOUR_UTC   = 12
SCHEDULER_MINUTE_UTC = 0

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "lawi_trading.db")
