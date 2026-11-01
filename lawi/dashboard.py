"""
LAWI Trading System - Streamlit Dashboard
Full ICT 2022 bias dashboard for all 4 assets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os

# Allow running as: streamlit run lawi/dashboard.py  OR  python -m streamlit run lawi/dashboard.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lawi.config import SYMBOLS, SYMBOL_NAMES, BIAS_LABELS, CONFIDENCE_MIN
from lawi.data_fetcher import fetch_daily, fetch_hourly, get_current_price
from lawi.model import predict_bias, load_model
from lawi.features import build_features
from lawi.db import init_db, get_recent_predictions, get_accuracy_stats
from lawi.alerts import test_connection

# ─── Admin Auth ───────────────────────────────────────────────────────────────
def is_admin() -> bool:
    return st.session_state.get("is_admin", False)

def show_admin_login():
    """Show a compact admin login in the sidebar."""
    if is_admin():
        st.sidebar.success("🔓 Admin mode")
        if st.sidebar.button("🔒 Lock", key="lock_btn", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()
    else:
        with st.sidebar.expander("🔐 Admin Login", expanded=False):
            pwd = st.sidebar.text_input("Password", type="password", key="admin_pwd")
            if st.sidebar.button("Unlock", key="unlock_btn", use_container_width=True):
                correct = st.secrets.get("DASHBOARD_PASSWORD", "lawi2024")
                if pwd == correct:
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.sidebar.error("Wrong password")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LAWI Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px; border-radius: 12px; margin-bottom: 20px;
        text-align: center; color: white;
    }
    .signal-card {
        background: #1e1e2e; border-radius: 12px; padding: 16px;
        border: 1px solid #333; margin-bottom: 12px;
    }
    .bullish  { border-left: 4px solid #00ff88; }
    .bearish  { border-left: 4px solid #ff4757; }
    .consol   { border-left: 4px solid #ffa502; }
    .metric-card {
        background: #16213e; border-radius: 8px; padding: 12px;
        text-align: center; border: 1px solid #0f3460;
    }
    .stMetric > div { background: #1e1e2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Initialise DB ────────────────────────────────────────────────────────────
init_db()

# ─── Admin login (always shown in sidebar) ────────────────────────────────────
show_admin_login()

# ─── Header ──────────────────────────────────────────────────────────────────
import pytz
_toronto = pytz.timezone("America/Toronto")
_now_tor  = datetime.now(_toronto)
_day_name = _now_tor.strftime("%A")
_date_str = _now_tor.strftime("%B %d, %Y")
_is_weekend = _now_tor.weekday() >= 5

st.markdown(f"""
<div class="main-header">
    <h1>🏦 LAWI Trading System</h1>
    <p>ICT 2022 Mentorship Model · Smart Money Concepts · AI-Powered Bias Prediction</p>
    <p style="font-size:12px; opacity:0.7;">Gold · GBP/USD · S&P 500 · US30/Dow Jones</p>
    <p style="font-size:13px; opacity:0.9;">📅 {_day_name}, {_date_str}</p>
</div>
""", unsafe_allow_html=True)

if _is_weekend:
    st.warning("⚠️ **Markets are closed today (Weekend).** Signals shown reflect Friday's closing data — use this as your **Monday outlook**. No trades on weekends!")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0f3460/ffffff?text=LAWI+Trading", width=200)
    st.markdown("---")
    st.markdown("### ⚙️ Controls")

    selected_symbol = st.selectbox(
        "Select Asset",
        list(SYMBOLS.keys()),
        format_func=lambda s: SYMBOL_NAMES.get(s, s)
    )

    if st.button("🔄 Refresh Predictions", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if is_admin():
        if st.button("📨 Test Telegram", use_container_width=True):
            with st.spinner("Sending test message …"):
                ok = test_connection()
            if ok:
                st.success("✅ Telegram connected!")
            else:
                st.error("❌ Telegram failed. Check .env")

    st.markdown("---")
    st.markdown("### 📐 ICT Kill Zones (EST)")
    st.markdown("""
    - 🌏 **Asia:**    7:00 PM – 12:00 AM
    - 🇬🇧 **London:** 2:00 AM – 5:00 AM
    - 🇺🇸 **NY AM:**  7:00 AM – 11:00 AM
    - 🇺🇸 **NY PM:**  2:00 PM – 4:00 PM
    """)

    st.markdown("---")
    utc_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"Last refresh: {utc_now}")


# ─── Load Predictions ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_all_predictions():
    results = {}
    for symbol, ticker in SYMBOLS.items():
        model = load_model(symbol)
        if model is None:
            results[symbol] = {"error": f"Model not trained yet for {symbol}"}
            continue
        results[symbol] = predict_bias(symbol, ticker)
    return results


@st.cache_data(ttl=300)
def load_price_chart(symbol: str):
    ticker = SYMBOLS.get(symbol)
    return fetch_daily(ticker, days=60)


# ─── TABS ────────────────────────────────────────────────────────────────────
if is_admin():
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Signals", "📈 Charts", "📋 History", "🎯 Accuracy"])
else:
    tab1, tab2, tab3 = st.tabs(["📊 Live Signals", "📈 Charts", "📋 History"])
    tab4 = None

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Signals
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🤖 Today's ICT Bias — All Assets")

    with st.spinner("Loading predictions …"):
        preds = load_all_predictions()

    cols = st.columns(4)
    for i, (symbol, pred) in enumerate(preds.items()):
        with cols[i]:
            name  = SYMBOL_NAMES.get(symbol, symbol)
            if "error" in pred:
                st.error(f"**{name}**\n\n❌ {pred['error']}")
            else:
                label = pred.get("label", "?")
                conf  = pred.get("confidence_pct", "?")
                emoji = {"BULLISH":"🟢","BEARISH":"🔴","CONSOLIDATION":"🟡"}.get(label,"❓")
                color = {"BULLISH":"#00ff88","BEARISH":"#ff4757","CONSOLIDATION":"#ffa502"}.get(label,"#888")
                price = get_current_price(symbol)
                price_str = f"${price:,.2f}" if price else "N/A"

                st.markdown(f"""
                <div class="signal-card {'bullish' if label=='BULLISH' else 'bearish' if label=='BEARISH' else 'consol'}">
                    <h4 style="margin:0; color:{color};">{emoji} {name}</h4>
                    <h2 style="margin:4px 0; color:{color};">{label}</h2>
                    <p style="margin:2px 0;">🎯 {conf} confidence</p>
                    <p style="margin:2px 0;">⚡ {pred.get('confluence',0)}/8 confluence</p>
                    <p style="margin:2px 0; color:#888;">💰 {price_str}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader(f"🔍 Detail — {SYMBOL_NAMES.get(selected_symbol, selected_symbol)}")
    pred = preds.get(selected_symbol, {})

    if "error" in pred:
        st.warning(f"⚠️ {pred['error']}")

    # Admin: always show Train button
    if is_admin():
        from lawi.config import TRAIN_DAYS
        if st.button(f"🚀 Train Model Now ({TRAIN_DAYS} days)", key="train_btn"):
            with st.spinner(f"Training {selected_symbol} — fetching {TRAIN_DAYS} days of data … (2–5 min)"):
                from lawi.model import train
                result = train(selected_symbol, SYMBOLS[selected_symbol], days=TRAIN_DAYS)
                if "error" not in result:
                    st.success(f"✅ Training complete! Accuracy: {result['accuracy']*100:.1f}%  |  CV: {result['cv_accuracy']*100:.1f}%  |  Samples: {result['n_samples']}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(result["error"])
        st.markdown("---")

    if "error" not in pred:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bias",        pred.get("label","?"))
        c2.metric("Confidence",  pred.get("confidence_pct","?"))
        c3.metric("Confluence",  f"{pred.get('confluence',0)}/8")
        c4.metric("Signal",      "✅ SEND" if pred.get("above_threshold") else "⏸️ WAIT")

        st.markdown("#### 📐 ICT 2022 Component Analysis")
        ict = pred.get("ict_signals", {})
        ict_cols = st.columns(2)
        items = list(ict.items())
        for j, (component, value) in enumerate(items):
            ict_cols[j % 2].markdown(f"**{component}:** {value}")

        st.markdown("#### 📊 Bias Probabilities")
        proba = pred.get("probabilities", {})
        prob_df = pd.DataFrame(list(proba.items()), columns=["Bias", "Probability"])
        prob_df["Probability_pct"] = prob_df["Probability"] * 100
        fig_bar = px.bar(
            prob_df, x="Bias", y="Probability_pct",
            color="Bias",
            color_discrete_map={"BULLISH":"#00ff88","BEARISH":"#ff4757","CONSOLIDATION":"#ffa502"},
            labels={"Probability_pct": "Probability (%)"},
            title="Prediction Probability Distribution",
        )
        fig_bar.update_layout(
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="white", showlegend=False, height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Charts
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"📈 Price Chart — {SYMBOL_NAMES.get(selected_symbol, selected_symbol)}")

    with st.spinner("Loading chart data …"):
        df_chart = load_price_chart(selected_symbol)

    if df_chart.empty:
        st.warning("No chart data available.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"],  close=df_chart["Close"],
            name="Price",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4757",
        ))
        for period, color in [(21,"#3498db"),(50,"#e67e22"),(200,"#9b59b6")]:
            ema = df_chart["Close"].ewm(span=period).mean()
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=ema, name=f"EMA{period}",
                line=dict(color=color, width=1), opacity=0.8
            ))
        fig.update_layout(
            title=f"{SYMBOL_NAMES.get(selected_symbol,'?')} — Daily Chart (60 days)",
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="white", height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig_vol = go.Figure(go.Bar(
            x=df_chart.index, y=df_chart["Volume"],
            marker_color="#3498db", name="Volume", opacity=0.7
        ))
        fig_vol.update_layout(
            plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
            font_color="white", height=200, title="Volume",
        )
        st.plotly_chart(fig_vol, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — History
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📋 Prediction History")
    history = get_recent_predictions(limit=50)
    if not history:
        st.info("No predictions saved yet. Run your first prediction!")
    else:
        df_hist = pd.DataFrame(history)
        df_hist["created_at"] = pd.to_datetime(df_hist["created_at"])
        df_hist["confidence"] = (df_hist["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            df_hist[["created_at","symbol","label","confidence","confluence","sent_telegram"]]
            .rename(columns={
                "created_at":    "Time (UTC)",
                "symbol":        "Asset",
                "label":         "Bias",
                "confidence":    "Confidence",
                "confluence":    "Confluence",
                "sent_telegram": "Sent to Telegram",
            }),
            use_container_width=True,
        )
        label_counts = pd.DataFrame(history)["label"].value_counts()
        fig_pie = px.pie(
            values=label_counts.values, names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={"BULLISH":"#00ff88","BEARISH":"#ff4757","CONSOLIDATION":"#ffa502"},
            title="Historical Bias Distribution",
        )
        fig_pie.update_layout(paper_bgcolor="#1e1e2e", font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Accuracy (Admin only)
# ──────────────────────────────────────────────────────────────────────────────
if tab4 is not None:
    with tab4:
        st.subheader("🎯 Model Accuracy Tracker")
        st.info("📌 Accuracy is calculated when you log the actual market outcome after each trading session.")

        acc_cols = st.columns(4)
        for i, symbol in enumerate(SYMBOLS.keys()):
            stats = get_accuracy_stats(symbol)
            with acc_cols[i]:
                name  = SYMBOL_NAMES.get(symbol, symbol)
                total = stats["total"]
                acc   = stats["accuracy_pct"]
                color = "#00ff88" if acc >= 60 else "#ffa502" if acc >= 50 else "#ff4757"
                st.markdown(f"""
                <div class="metric-card">
                    <h5>{name}</h5>
                    <h2 style="color:{color};">{acc:.1f}%</h2>
                    <p style="color:#888;">{total} trades logged</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Confusion Matrix ─────────────────────────────────────────────────
        st.markdown("#### 🔢 Confusion Matrix — Model Self-Evaluation")
        st.caption("Shows how often the model predicts each class correctly on training data.")

        cm_sym = st.selectbox("Select asset for analysis", list(SYMBOLS.keys()),
                               key="cm_sym", format_func=lambda s: SYMBOL_NAMES.get(s, s))
        model_obj = load_model(cm_sym)
        if model_obj is None:
            st.warning(f"No trained model found for {cm_sym}. Train it first.")
        else:
            from lawi.data_fetcher import fetch_daily
            from lawi.features import build_features, FEATURE_COLUMNS
            from lawi.config import TRAIN_DAYS
            from sklearn.metrics import confusion_matrix as sk_cm
            with st.spinner("Loading data for analysis…"):
                raw_cm = fetch_daily(SYMBOLS[cm_sym], days=TRAIN_DAYS)
                df_cm  = build_features(raw_cm, label=True)
                feat_cols = [c for c in FEATURE_COLUMNS if c in df_cm.columns]
                X_cm = df_cm[feat_cols].values
                y_cm = df_cm["bias_label"].values.astype(int)
                y_pred_cm = model_obj.predict(X_cm)
            labels = ["BEARISH", "BULLISH", "CONSOLIDATION"]
            cm = sk_cm(y_cm, y_pred_cm)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            fig_cm = go.Figure(go.Heatmap(
                z=cm_pct,
                x=[f"Pred {l}" for l in labels],
                y=[f"True {l}" for l in labels],
                colorscale="RdYlGn", zmin=0, zmax=100,
                text=[[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(3)] for i in range(3)],
                texttemplate="%{text}", textfont={"size": 13}, showscale=True,
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — {SYMBOL_NAMES.get(cm_sym, cm_sym)}",
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="white", height=380,
                xaxis_title="Predicted", yaxis_title="Actual",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            k1, k2, k3 = st.columns(3)
            for col, i, label in zip([k1, k2, k3], [0, 1, 2], labels):
                correct = cm[i][i]
                total   = cm[i].sum()
                pct     = (correct / total * 100) if total > 0 else 0
                emoji   = "🟢" if pct >= 60 else "🟡" if pct >= 45 else "🔴"
                col.metric(f"{emoji} {label}", f"{pct:.1f}%", f"{correct}/{total} correct")

        st.markdown("---")

        # ── Feature Importance ────────────────────────────────────────────────
        st.markdown("#### 🏆 Feature Importance — Which ICT Signals Matter Most?")
        st.caption("Higher score = more influence on the model's prediction.")

        fi_sym = st.selectbox("Select asset for feature importance", list(SYMBOLS.keys()),
                               key="fi_sym", format_func=lambda s: SYMBOL_NAMES.get(s, s))
        model_fi = load_model(fi_sym)
        if model_fi is None:
            st.warning(f"No trained model found for {fi_sym}. Train it first.")
        else:
            try:
                importances = model_fi.feature_importances_
                from lawi.data_fetcher import fetch_daily
                from lawi.features import build_features, FEATURE_COLUMNS
                raw_fi = fetch_daily(SYMBOLS[fi_sym], days=60)
                df_fi  = build_features(raw_fi, label=False)
                fi_cols = [c for c in FEATURE_COLUMNS if c in df_fi.columns]
                if len(importances) == len(fi_cols):
                    fi_df = pd.DataFrame({"Feature": fi_cols, "Importance": importances})
                    fi_df = fi_df.sort_values("Importance", ascending=True).tail(20)
                    fig_fi = go.Figure(go.Bar(
                        x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                        marker_color="#00ff88",
                        text=[f"{v:.4f}" for v in fi_df["Importance"]],
                        textposition="outside",
                    ))
                    fig_fi.update_layout(
                        title=f"Top 20 Features — {SYMBOL_NAMES.get(fi_sym, fi_sym)}",
                        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                        font_color="white", height=550,
                        xaxis_title="Importance Score", yaxis_title="",
                        margin=dict(l=160),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.warning("Feature count mismatch — retrain the model to refresh.")
            except Exception as e:
                st.error(f"Could not load feature importance: {e}")

        st.markdown("---")
        st.markdown("#### 📝 Log Trade Outcome")
        log_sym  = st.selectbox("Asset", list(SYMBOLS.keys()), key="log_sym",
                                 format_func=lambda s: SYMBOL_NAMES.get(s, s))
        log_pred = st.selectbox("Your prediction was", ["BULLISH","BEARISH","CONSOLIDATION"])
        log_act  = st.selectbox("Actual market moved", ["BULLISH","BEARISH","CONSOLIDATION"])
        log_note = st.text_input("Notes (optional)")

        if st.button("✅ Save Outcome"):
            from lawi.db import get_conn
            from datetime import datetime as dt
            correct = 1 if log_pred == log_act else 0
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO accuracy_log (logged_at, symbol, prediction, actual, correct, notes)
                       VALUES (?,?,?,?,?,?)""",
                    (dt.utcnow().isoformat(), log_sym, log_pred, log_act, correct, log_note)
                )
            st.success("✅ Outcome saved!" if correct else "📊 Outcome saved (incorrect prediction logged).")
            st.rerun()