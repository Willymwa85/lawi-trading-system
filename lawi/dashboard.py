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

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 LAWI Trading System</h1>
    <p>ICT 2022 Mentorship Model · Smart Money Concepts · AI-Powered Bias Prediction</p>
    <p style="font-size:12px; opacity:0.7;">Gold · GBP/USD · S&P 500 · US30/Dow Jones</p>
</div>
""", unsafe_allow_html=True)

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
@st.cache_data(ttl=600)   # Cache for 10 minutes
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Signals", "📈 Charts", "📋 History", "🎯 Accuracy"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Signals
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🤖 Today's ICT Bias — All Assets")

    with st.spinner("Loading predictions …"):
        preds = load_all_predictions()

    # Summary row
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

    # Detailed view for selected symbol
    st.subheader(f"🔍 Detail — {SYMBOL_NAMES.get(selected_symbol, selected_symbol)}")
    pred = preds.get(selected_symbol, {})

    if "error" in pred:
        st.warning(f"⚠️ {pred['error']}")
        if st.button("🚀 Train Model Now"):
            with st.spinner(f"Training {selected_symbol} model (this takes a minute) …"):
                from lawi.model import train
                result = train(selected_symbol, SYMBOLS[selected_symbol])
                if "error" not in result:
                    st.success(f"✅ Trained! Accuracy: {result['accuracy']*100:.1f}%")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(result["error"])
    else:
        # Metric row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bias",        pred.get("label","?"))
        c2.metric("Confidence",  pred.get("confidence_pct","?"))
        c3.metric("Confluence",  f"{pred.get('confluence',0)}/8")
        c4.metric("Signal",      "✅ SEND" if pred.get("above_threshold") else "⏸️ WAIT")

        # ICT Signal breakdown
        st.markdown("#### 📐 ICT 2022 Component Analysis")
        ict = pred.get("ict_signals", {})
        ict_cols = st.columns(2)
        items = list(ict.items())
        for j, (component, value) in enumerate(items):
            ict_cols[j % 2].markdown(f"**{component}:** {value}")

        # Probability gauge
        st.markdown("#### 📊 Bias Probabilities")
        proba = pred.get("probabilities", {})
        prob_df = pd.DataFrame(
            list(proba.items()), columns=["Bias", "Probability"]
        )
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
            font_color="white", showlegend=False,
            height=300,
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

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="Price",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4757",
        ))

        # EMA overlays
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

        # Volume bar
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
                "created_at":     "Time (UTC)",
                "symbol":         "Asset",
                "label":          "Bias",
                "confidence":     "Confidence",
                "confluence":     "Confluence",
                "sent_telegram":  "Sent to Telegram",
            }),
            use_container_width=True,
        )

        # Bias distribution pie
        label_counts = pd.DataFrame(history)["label"].value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={"BULLISH":"#00ff88","BEARISH":"#ff4757","CONSOLIDATION":"#ffa502"},
            title="Historical Bias Distribution",
        )
        fig_pie.update_layout(paper_bgcolor="#1e1e2e", font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Accuracy
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🎯 Model Accuracy Tracker")
    st.info("📌 Accuracy is calculated when you log the actual market outcome after each trading session.")

    acc_cols = st.columns(4)
    for i, symbol in enumerate(SYMBOLS.keys()):
        stats = get_accuracy_stats(symbol)
        with acc_cols[i]:
            name = SYMBOL_NAMES.get(symbol, symbol)
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
