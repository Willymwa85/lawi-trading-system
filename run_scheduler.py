"""
LAWI Trading System - Scheduler Entry Point
Runs daily predictions at 07:00 Toronto time and sends Telegram signals.

Usage:
    python run_scheduler.py

For cloud deployment (Railway / Render): this is your start command.
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("lawi_scheduler.log"),
    ]
)

from lawi.db import init_db
from lawi.alerts import test_connection
from lawi.scheduler import start_scheduler

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LAWI Trading System — Scheduler")
    print("  Daily predictions + Telegram signals")
    print("="*60)

    init_db()

    print("\n  Testing Telegram connection …")
    ok = test_connection()
    if ok:
        print("  ✅ Telegram connected!")
    else:
        print("  ⚠️  Telegram not connected — check your .env file")
        print("     Scheduler will start anyway, but signals won't send.\n")

    print("\n  Starting LAWI Scheduler …")
    print("  Press Ctrl+C to stop.\n")
    start_scheduler()
