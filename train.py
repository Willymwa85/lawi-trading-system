"""
LAWI Trading System - Train All Models
Run this ONCE before starting the system.

Usage:
    python train.py
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from lawi.config import SYMBOLS
from lawi.model import train_all
from lawi.db import init_db, save_training_log

def main():
    print("\n" + "="*60)
    print("  LAWI Trading System — Model Training")
    print("  ICT 2022 Mentorship Model | XGBoost")
    print("="*60 + "\n")

    init_db()
    results = train_all(SYMBOLS)

    print("\n" + "="*60)
    print("  TRAINING RESULTS SUMMARY")
    print("="*60)

    for symbol, res in results.items():
        save_training_log(symbol, res)
        if "error" in res:
            print(f"  ❌ {symbol:8s} — FAILED: {res['error']}")
        else:
            acc    = res["accuracy"] * 100
            cv_acc = res["cv_accuracy"] * 100
            n      = res["n_samples"]
            flag   = "✅" if acc >= 60 else "⚠️"
            print(f"  {flag} {symbol:8s} — Accuracy: {acc:.1f}%  CV: {cv_acc:.1f}%  Samples: {n}")

    print("\n  ✅ Training complete!")
    print("  Next steps:")
    print("    1. Start dashboard:  streamlit run lawi/dashboard.py")
    print("    2. Start scheduler:  python run_scheduler.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
