"""
LAWI Trading System - Forex Factory News Calendar
Fetches high-impact news events for today from Forex Factory RSS feed.
Used at 4AM to warn about red-flag news days (NFP, CPI, FOMC, etc.)
"""

import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date
import pytz

logger = logging.getLogger(__name__)

FOREX_FACTORY_RSS = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
TORONTO_TZ = pytz.timezone("America/Toronto")
TARGET_CURRENCIES = {"USD", "GBP"}


def fetch_news_today() -> list[dict]:
    """
    Fetch today's high-impact news events from Forex Factory.
    Returns a list of event dicts with: title, time, currency, impact.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (LAWI Trading System)"}
        resp = requests.get(FOREX_FACTORY_RSS, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Could not fetch Forex Factory calendar: {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        logger.warning(f"Could not parse Forex Factory XML: {e}")
        return []

    today = date.today()
    events = []

    for item in root.iter("event"):
        try:
            title    = item.findtext("title", "").strip()
            currency = item.findtext("country", "").strip().upper()
            impact   = item.findtext("impact", "").strip().lower()
            date_str = item.findtext("date", "").strip()
            time_str = item.findtext("time", "").strip()

            if impact not in ("high",):
                continue
            if currency not in TARGET_CURRENCIES:
                continue

            try:
                event_date = datetime.strptime(date_str, "%m-%d-%Y").date()
            except ValueError:
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

            if event_date != today:
                continue

            display_time = time_str if time_str else "All Day"
            events.append({
                "title":    title,
                "currency": currency,
                "time":     display_time,
                "impact":   "HIGH",
            })

        except Exception:
            continue

    events.sort(key=lambda x: x["time"])
    return events


def format_news_warning(events: list[dict]) -> str:
    """Format news events into a Telegram warning block."""
    if not events:
        return ""

    lines = ["🔴 <b>HIGH IMPACT NEWS TODAY</b>"]
    lines.append("━━━━━━━━━━━━━━━━━━━━━")

    for ev in events:
        flag = "🇺🇸" if ev["currency"] == "USD" else "🇬🇧"
        lines.append(f"{flag} <b>{ev['time']}</b> — {ev['title']} ({ev['currency']})")

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⏸️ <i>Consider waiting for the news spike to settle before entering.</i>")
    lines.append("✅ <i>Best entries: 15–30 min after release.</i>")

    return "\n".join(lines)


def is_high_impact_day() -> bool:
    """Returns True if today has any high-impact USD or GBP news."""
    return len(fetch_news_today()) > 0