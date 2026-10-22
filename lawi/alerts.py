"""
LAWI Trading System - Telegram Alerts
Sends formatted ICT bias signals to Kiarie's Telegram channel.
"""

import logging
import requests
from datetime import datetime
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOL_NAMES, BIAS_LABELS

logger = logging.getLogger(__name__)

BIAS_EMOJI = {
    "BULLISH":       "🟢",
    "BEARISH":       "🔴",
    "CONSOLIDATION": "🟡",
}

BIAS_ARROW = {
    "BULLISH":       "⬆️  BUY",
    "BEARISH":       "⬇️  SELL",
    "CONSOLIDATION": "↔️  WAIT",
}


def _send_message(text: str) -> bool:
    """Send a raw Telegram message. Returns True if OK."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured. Check .env file.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram message sent ✓")
            return True
        else:
            logger.error(f"Telegram error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def format_signal(symbol: str, prediction: dict) -> str:
    """
    Format a prediction dict into a nicely styled Telegram message.
    """
    label       = prediction.get("label", "UNKNOWN")
    confidence  = prediction.get("confidence_pct", "?%")
    confluence  = prediction.get("confluence", 0)
    ict_signals = prediction.get("ict_signals", {})
    proba       = prediction.get("probabilities", {})
    ts          = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    name  = SYMBOL_NAMES.get(symbol, symbol)
    emoji = BIAS_EMOJI.get(label, "❓")
    arrow = BIAS_ARROW.get(label, label)

    # Build ICT confluence table
    ict_lines = ""
    for component, value in ict_signals.items():
        ict_lines += f"  • <b>{component}:</b> {value}\n"

    # Probability bar
    bull_pct = round(proba.get("BULLISH", 0) * 100)
    bear_pct = round(proba.get("BEARISH", 0) * 100)
    cons_pct = round(proba.get("CONSOLIDATION", 0) * 100)

    msg = f"""
╔══════════════════════════╗
  🤖 <b>LAWI Trading Signal</b>
╚══════════════════════════╝

📊 <b>{name}</b>
{emoji} <b>Bias: {arrow}</b>
🎯 Confidence: <b>{confidence}</b>
⚡ Confluence: <b>{confluence}/8 signals</b>

📐 <b>ICT 2022 Analysis</b>
{ict_lines}
📈 <b>Probabilities</b>
  🟢 Bullish:       {bull_pct}%
  🔴 Bearish:       {bear_pct}%
  🟡 Consolidation: {cons_pct}%

⏰ {ts}
🏢 <i>LAWI Trading System</i>
""".strip()
    return msg


def send_signal(symbol: str, prediction: dict) -> bool:
    """Format and send a single symbol signal to Telegram."""
    if "error" in prediction:
        logger.warning(f"[{symbol}] Skipping alert — prediction error: {prediction['error']}")
        return False

    if not prediction.get("above_threshold", False):
        logger.info(f"[{symbol}] Confidence below threshold — skipping Telegram alert")
        return False

    text = format_signal(symbol, prediction)
    return _send_message(text)


def send_daily_brief(predictions: dict):
    """
    Send a consolidated daily morning brief for all 4 assets.
    predictions = {symbol: prediction_dict}
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    for symbol, pred in predictions.items():
        if "error" in pred:
            lines.append(f"  {SYMBOL_NAMES.get(symbol, symbol)}: ❌ Error")
            continue
        label  = pred.get("label", "UNKNOWN")
        emoji  = BIAS_EMOJI.get(label, "❓")
        conf   = pred.get("confidence_pct", "?")
        confl  = pred.get("confluence", 0)
        lines.append(
            f"  {emoji} <b>{SYMBOL_NAMES.get(symbol, symbol)}</b>\n"
            f"     Bias: <b>{label}</b> | {conf} | {confl}/8 signals"
        )

    summary = "\n\n".join(lines)

    header = f"""
🌅 <b>LAWI Morning Brief — {ts}</b>
━━━━━━━━━━━━━━━━━━━━━

{summary}

━━━━━━━━━━━━━━━━━━━━━
📌 <i>Trade in the direction of the bias.
Wait for your ICT model entry during Kill Zones.</i>
🏢 <i>LAWI Trading System — ICT 2022 Model</i>
""".strip()

    return _send_message(header)


def send_error_alert(message: str) -> bool:
    """Send a simple error/system alert to Telegram."""
    text = f"⚠️ <b>LAWI System Alert</b>\n\n{message}\n\n🏢 <i>LAWI Trading System</i>"
    return _send_message(text)


def test_connection() -> bool:
    """Send a test message to verify Telegram is working."""
    msg = "✅ <b>LAWI System</b> — Telegram connection verified!\n🏢 <i>LAWI Trading System</i>"
    ok = _send_message(msg)
    if ok:
        logger.info("Telegram test message sent successfully.")
    else:
        logger.error("Telegram test FAILED. Check BOT_TOKEN and CHAT_ID in .env")
    return ok
