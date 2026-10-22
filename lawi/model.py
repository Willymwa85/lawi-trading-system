"""
LAWI Trading System - ML Model
XGBoost classifier trained on ICT features to predict Daily Bias:
  0 = BEARISH | 1 = BULLISH | 2 = CONSOLIDATION
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from .config import MODEL_DIR, BIAS_LABELS, CONFIDENCE_MIN
from .features import FEATURE_COLUMNS, build_features
from .data_fetcher import fetch_daily

logger = logging.getLogger(__name__)

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────

def _model_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f"lawi_bias_{symbol}.pkl")


def save_model(model, symbol: str):
    path = _model_path(symbol)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"[{symbol}] Model saved → {path}")


def load_model(symbol: str):
    path = _model_path(symbol)
    if not os.path.exists(path):
        logger.warning(f"[{symbol}] No saved model found at {path}")
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"[{symbol}] Model loaded from {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(symbol: str, ticker: str, days: int = 365) -> dict:
    """
    Train an XGBoost bias model for the given symbol.

    Returns:
        dict with keys: accuracy, report, model_path, n_samples
    """
    logger.info(f"[{symbol}] Starting training — fetching {days} days of daily data …")

    raw = fetch_daily(ticker, days=days)
    if raw.empty or len(raw) < 60:
        return {"error": f"Not enough data for {symbol} (got {len(raw)} rows)"}

    df = build_features(raw, label=True)
    if len(df) < 30:
        return {"error": f"Not enough feature rows after engineering for {symbol}"}

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values
    y = df["bias_label"].values.astype(int)

    # Time-series cross-validation (do NOT shuffle — preserve time order)
    tscv = TimeSeriesSplit(n_splits=5)
    acc_list = []

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        acc   = accuracy_score(y_val, preds)
        acc_list.append(acc)
        logger.info(f"[{symbol}] Fold {fold+1}/5 accuracy: {acc:.3f}")

    # Final fit on all data
    model.fit(X, y, verbose=False)
    final_preds = model.predict(X)
    final_acc   = accuracy_score(y, final_preds)
    report      = classification_report(y, final_preds,
                                        target_names=["BEARISH","BULLISH","CONSOLIDATION"],
                                        output_dict=True)

    save_model(model, symbol)

    result = {
        "symbol":       symbol,
        "accuracy":     round(final_acc, 4),
        "cv_accuracy":  round(float(np.mean(acc_list)), 4),
        "n_samples":    len(df),
        "report":       report,
        "model_path":   _model_path(symbol),
        "feature_cols": feature_cols,
    }
    logger.info(f"[{symbol}] Training complete — accuracy: {final_acc:.3f} | CV: {np.mean(acc_list):.3f}")
    return result


def train_all(symbols: dict) -> dict:
    """Train models for all LAWI symbols. symbols = {symbol: ticker}"""
    results = {}
    for symbol, ticker in symbols.items():
        logger.info(f"\n{'='*50}\nTraining {symbol}\n{'='*50}")
        results[symbol] = train(symbol, ticker)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_bias(symbol: str, ticker: str, days: int = 60) -> dict:
    """
    Predict today's bias for a symbol.

    Returns:
        dict with bias, label, confidence, probabilities, confluence_score,
              ict_signals (individual component signals), timestamp
    """
    model = load_model(symbol)
    if model is None:
        return {"error": f"No model for {symbol}. Run training first."}

    raw = fetch_daily(ticker, days=days)
    if raw.empty or len(raw) < 30:
        return {"error": f"Insufficient recent data for {symbol}"}

    df = build_features(raw, label=False)
    if df.empty:
        return {"error": f"Feature engineering returned empty DataFrame for {symbol}"}

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_latest = df[feature_cols].values[-1:]   # most recent bar

    proba   = model.predict_proba(X_latest)[0]
    classes = model.classes_
    proba_dict = {BIAS_LABELS[int(c)]: round(float(p), 4) for c, p in zip(classes, proba)}

    pred_class   = int(classes[np.argmax(proba)])
    confidence   = float(np.max(proba))
    bias_label   = BIAS_LABELS[pred_class]

    # ── ICT component summary (last bar values) ──────────────────────────────
    last = df.iloc[-1]
    ict_signals = {
        "Order Block":         "🟢 Bullish" if last.get("ob_bull_nearby",0) else ("🔴 Bearish" if last.get("ob_bear_nearby",0) else "➖ None"),
        "Fair Value Gap":      "🟢 Bullish" if last.get("fvg_bull_nearby",0) else ("🔴 Bearish" if last.get("fvg_bear_nearby",0) else "➖ None"),
        "Market Structure":    "🟢 Bullish" if last.get("structure_score",0) > 1 else ("🔴 Bearish" if last.get("structure_score",0) < -1 else "➖ Neutral"),
        "Liquidity":           "🟢 SSL Sweep" if last.get("ssl_sweep",0) else ("🔴 BSL Sweep" if last.get("bsl_sweep",0) else "➖ None"),
        "OTE Zone":            "🟢 Bull OTE" if last.get("in_bull_ote",0) else ("🔴 Bear OTE" if last.get("in_bear_ote",0) else "➖ None"),
        "Premium/Discount":    "🔴 Premium" if last.get("premium",0) else ("🟢 Discount" if last.get("discount",0) else "➖ Equil."),
        "Kill Zone":           "✅ Active" if last.get("in_kill_zone",0) else "❌ Inactive",
        "EMA Stack":           "🟢 Bullish" if last.get("ema_bull_stack",0) else ("🔴 Bearish" if last.get("ema_bear_stack",0) else "➖ Mixed"),
        "RSI":                 f"{last.get('rsi', 50):.1f} {'(OB)' if last.get('rsi_overbought',0) else '(OS)' if last.get('rsi_oversold',0) else ''}",
        "MACD":                "🟢 Bullish" if last.get("macd_bullish",0) else "🔴 Bearish",
    }

    # Confluence score: how many ICT signals agree with the predicted direction
    if bias_label == "BULLISH":
        bull_signals = sum([
            int(last.get("ob_bull_nearby", 0)),
            int(last.get("fvg_bull_nearby", 0)),
            int(last.get("structure_score", 0) > 1),
            int(last.get("ssl_sweep", 0)),
            int(last.get("in_bull_ote", 0)),
            int(last.get("discount", 0)),
            int(last.get("ema_bull_stack", 0)),
            int(last.get("macd_bullish", 0)),
        ])
        confluence_score = bull_signals
    elif bias_label == "BEARISH":
        bear_signals = sum([
            int(last.get("ob_bear_nearby", 0)),
            int(last.get("fvg_bear_nearby", 0)),
            int(last.get("structure_score", 0) < -1),
            int(last.get("bsl_sweep", 0)),
            int(last.get("in_bear_ote", 0)),
            int(last.get("premium", 0)),
            int(last.get("ema_bear_stack", 0)),
            int(1 - last.get("macd_bullish", 1)),
        ])
        confluence_score = bear_signals
    else:
        confluence_score = 0

    return {
        "symbol":           symbol,
        "bias":             pred_class,
        "label":            bias_label,
        "confidence":       round(confidence, 4),
        "confidence_pct":   f"{confidence*100:.1f}%",
        "above_threshold":  confidence >= CONFIDENCE_MIN,
        "probabilities":    proba_dict,
        "confluence":       confluence_score,
        "ict_signals":      ict_signals,
        "timestamp":        pd.Timestamp.utcnow().isoformat(),
    }


def predict_all(symbols: dict) -> dict:
    """Run predict_bias for all symbols. Returns {symbol: prediction_dict}"""
    results = {}
    for symbol, ticker in symbols.items():
        logger.info(f"[{symbol}] Predicting bias …")
        results[symbol] = predict_bias(symbol, ticker)
    return results
