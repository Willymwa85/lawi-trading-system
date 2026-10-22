"""
LAWI Trading System - Run Predictions Right Now
Use this to get an immediate signal without waiting for the scheduler.

Usage:
    python predict_now.py
    python predict_now.py --symbol XAUUSD
"""

import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from lawi.config import SYMBOLS, SYMBOL_NAMES, BIAS_LABELS
from lawi.model import predict_all, predict_bias
from lawi.alerts import send_daily_brief, send_signal
from lawi.db import init_db, save_prediction, mark_sent

BIAS_EMOJI = {"BULLISH":"🟢","BEARISH":"🔴","CONSOLIDATION":"🟡"}

def print_prediction(symbol: str, pred: dict):
    name  = SYMBOL_NAMES.get(symbol, symbol)
    if "error" in pred:
        print(f"\n  ❌ {name}: {pred['error']}")
        return

    label = pred.get("label","?")
    conf  = pred.get("confidence_pct","?")
    confl = pred.get("confluence",0)
    emoji = BIAS_EMOJI.get(label,"❓")

    print(f"\n  {emoji} {name}")
    print(f"     Bias:       {label}")
    print(f"     Confidence: {conf}")
    print(f"     Confluence: {confl}/8 ICT signals")

    ict = pred.get("ict_signals", {})
    print("     ─── ICT Breakdown ───")
    for comp, val in ict.items():
        print(f"       • {comp}: {val}")


def main():
    parser = argparse.ArgumentParser(description="LAWI — Run predictions now")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol (XAUUSD, GBPUSD, SP500, US30)")
    parser.add_argument("--telegram", action="store_true",
                        help="Send signals to Telegram after predicting")
    args = parser.parse_args()

    init_db()

    print("\n" + "="*60)
    print("  LAWI Trading System — Live Predictions")
    print("  ICT 2022 Mentorship Model | XGBoost")
    print("="*60)

    if args.symbol:
        symbol = args.symbol.upper()
        if symbol not in SYMBOLS:
            print(f"  ❌ Unknown symbol '{symbol}'. Choose from: {', '.join(SYMBOLS.keys())}")
            return
        ticker = SYMBOLS[symbol]
        pred = predict_bias(symbol, ticker)
        print_prediction(symbol, pred)
        if args.telegram:
            ok = send_signal(symbol, pred)
            if ok:
                print(f"\n  ✅ Signal sent to Telegram!")
    else:
        preds = predict_all(SYMBOLS)
        for symbol, pred in preds.items():
            print_prediction(symbol, pred)
            row_id = save_prediction(symbol, pred)
            if args.telegram and "error" not in pred and pred.get("above_threshold"):
                ok = send_signal(symbol, pred)
                if ok:
                    mark_sent(row_id)

        if args.telegram:
            print("\n  Sending daily brief to Telegram …")
            send_daily_brief(preds)
            print("  ✅ Daily brief sent!")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
