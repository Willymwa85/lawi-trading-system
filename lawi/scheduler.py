"""
LAWI Trading System - Scheduler
Runs daily predictions at 07:00 Toronto (Eastern) time = 12:00 UTC.
Uses APScheduler so it keeps running when deployed to the cloud.
"""

import logging
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import SYMBOLS, SCHEDULER_HOUR_UTC, SCHEDULER_MINUTE_UTC
from .model import predict_all
from .alerts import send_daily_brief, send_signal, send_error_alert
from .db import init_db, save_prediction, mark_sent, save_training_log

logger = logging.getLogger(__name__)


def run_daily_predictions():
    """
    Core daily job:
    1. Fetch latest data for all 4 symbols via Yahoo Finance
    2. Run XGBoost bias prediction
    3. Save predictions to SQLite
    4. Send consolidated morning brief to Telegram
    5. Send individual signals for high-confidence calls
    """
    logger.info("=" * 60)
    logger.info("LAWI Daily Prediction Job STARTED")
    logger.info("=" * 60)

    try:
        predictions = predict_all(SYMBOLS)

        for symbol, pred in predictions.items():
            if "error" not in pred:
                row_id = save_prediction(symbol, pred)
                logger.info(
                    f"[{symbol}] {pred['label']} — "
                    f"{pred['confidence_pct']} confidence | "
                    f"{pred['confluence']}/8 confluence"
                )

                # Send individual signal if confidence is high
                if pred.get("above_threshold"):
                    ok = send_signal(symbol, pred)
                    if ok:
                        mark_sent(row_id)
            else:
                logger.error(f"[{symbol}] Prediction failed: {pred['error']}")

        # Send daily summary brief
        send_daily_brief(predictions)
        logger.info("LAWI Daily Brief sent to Telegram ✓")

    except Exception as e:
        logger.exception(f"LAWI daily job FAILED: {e}")
        send_error_alert(f"Daily prediction job failed:\n{e}")


def retrain_weekly():
    """
    Weekly retraining job (runs every Sunday 00:00 UTC).
    Re-trains XGBoost models with the latest data.
    """
    from .model import train_all
    logger.info("LAWI Weekly Retraining STARTED")
    try:
        results = train_all(SYMBOLS)
        for symbol, res in results.items():
            save_training_log(symbol, res)
            if "error" not in res:
                logger.info(f"[{symbol}] Retrained — accuracy: {res['accuracy']}")
            else:
                logger.error(f"[{symbol}] Retraining failed: {res['error']}")
        logger.info("LAWI Weekly Retraining COMPLETE")
    except Exception as e:
        logger.exception(f"LAWI retraining FAILED: {e}")
        send_error_alert(f"Weekly retraining failed:\n{e}")


def start_scheduler():
    """Start the APScheduler blocking scheduler."""
    init_db()
    scheduler = BlockingScheduler(timezone=pytz.utc)

    # Daily prediction at 07:00 Toronto (12:00 UTC winter / 11:00 UTC summer)
    scheduler.add_job(
        run_daily_predictions,
        CronTrigger(hour=SCHEDULER_HOUR_UTC, minute=SCHEDULER_MINUTE_UTC, timezone=pytz.utc),
        id="daily_predictions",
        name="LAWI Daily Bias Prediction",
        replace_existing=True,
    )

    # Weekly retraining: Sunday at 02:00 UTC
    scheduler.add_job(
        retrain_weekly,
        CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=pytz.utc),
        id="weekly_retrain",
        name="LAWI Weekly Model Retrain",
        replace_existing=True,
    )

    logger.info("LAWI Scheduler started:")
    logger.info(f"  • Daily predictions: {SCHEDULER_HOUR_UTC:02d}:{SCHEDULER_MINUTE_UTC:02d} UTC")
    logger.info("  • Weekly retraining: Sunday 02:00 UTC")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("LAWI Scheduler stopped.")
