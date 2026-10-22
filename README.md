# 🏦 LAWI Trading System

**ICT 2022 Mentorship Model · Smart Money Concepts · AI-Powered Bias Prediction**

Built for: **Willy Mwangi Kiarie** | Company: **LAWI**

---

## What is LAWI?

LAWI is a fully automated trading signal system that:
- 📡 **Fetches data from Yahoo Finance** — no MT5, no laptop needed 24/7
- 🧠 **Applies ICT 2022 Mentorship concepts** (Order Blocks, FVG, OTE, Liquidity, Kill Zones, Market Structure)
- 🤖 **Predicts daily bias** (BULLISH / BEARISH / CONSOLIDATION) using XGBoost ML
- 📨 **Sends Telegram signals** every morning at 07:00 Toronto time
- 📊 **Shows a live Streamlit dashboard** for all 4 assets

### 4 Assets Covered
| Asset | Yahoo Finance Ticker | Description |
|-------|---------------------|-------------|
| Gold  | `GC=F` | XAUUSD Gold Futures |
| GBP/USD | `GBPUSD=X` | British Pound vs US Dollar |
| S&P 500 | `^GSPC` | S&P 500 Index |
| US30 | `^DJI` | Dow Jones Industrial Average |

---

## 📋 ICT Concepts Used

| Concept | What It Does |
|---------|-------------|
| **Order Blocks (OB)** | Last opposing candle before a displacement — institutional supply/demand zones |
| **Fair Value Gaps (FVG)** | Price gaps (imbalance) that market tends to fill — entry zones |
| **Optimal Trade Entry (OTE)** | 61.8%–70.5% Fibonacci retracement — best entry after a swing |
| **Market Structure** | Higher Highs/Lows (bullish) vs Lower Highs/Lows (bearish) + BOS/MSS |
| **Liquidity Sweeps** | BSL (buy-side) and SSL (sell-side) sweep detection |
| **Kill Zones** | London (07–10 UTC), NY AM (13–16 UTC), Asia (00–05 UTC) — best trade times |
| **Premium/Discount** | Above 50% of range = Premium (sell); Below 50% = Discount (buy) |
| **Power of 3** | Accumulation (Asia) → Manipulation (London) → Distribution (NY) |

---

## ⚡ Quick Start (Windows — VS Code)

### Step 1: Install Python
Download Python 3.11 from https://python.org and install.
Make sure to tick **"Add Python to PATH"**.

### Step 2: Open project in VS Code
Extract the `lawi-system` folder.
In VS Code: **File → Open Folder → select lawi-system**

### Step 3: Open Terminal in VS Code
Press **Ctrl + `** (backtick)

### Step 4: Install dependencies
```
pip install -r requirements.txt
```

### Step 5: Set up your .env file
```
copy .env.example .env
```
Open `.env` in VS Code and fill in:
```
TELEGRAM_BOT_TOKEN=your_actual_token
TELEGRAM_CHAT_ID=your_actual_chat_id
```

> **How to get these?**
> - Open Telegram, search for **@BotFather**
> - Type `/newbot`, follow steps → copy your **Bot Token**
> - For Chat ID: message your bot once, then visit:
>   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
>   and copy the `"id"` number

### Step 6: Train the AI models (one time only)
```
python train.py
```
This downloads 1 year of data and trains XGBoost for all 4 assets.
Takes about 2-5 minutes.

### Step 7: Test your Telegram connection
```
python predict_now.py --telegram
```
You should receive a message on Telegram!

### Step 8: Start the dashboard
```
streamlit run lawi/dashboard.py
```
Opens at: http://localhost:8501

### Step 9: Start the automatic scheduler
Open a SECOND terminal in VS Code (click **+** in terminal panel):
```
python run_scheduler.py
```
This sends signals every morning at 07:00 Toronto time.

---

## 📱 Telegram Commands

| Command | What it does |
|---------|-------------|
| `python predict_now.py` | Run predictions right now (no Telegram) |
| `python predict_now.py --telegram` | Run now AND send to Telegram |
| `python predict_now.py --symbol XAUUSD --telegram` | Single asset signal |

---

## ☁️ Cloud Deployment (Signals 24/7 — laptop OFF)

### Option 1: Railway (Recommended — Free tier available)
1. Create account at https://railway.app
2. Click **New Project → Deploy from GitHub repo**
3. Push the `lawi-system` folder to your GitHub
4. In Railway: **Variables → Add variables**:
   - `TELEGRAM_BOT_TOKEN` = your token
   - `TELEGRAM_CHAT_ID` = your chat ID
5. Railway will automatically deploy and run 24/7!

### Option 2: Render
1. Create account at https://render.com
2. New → **Web Service** → connect GitHub
3. Build command: `pip install -r requirements.txt && python train.py`
4. Start command: `streamlit run lawi/dashboard.py --server.port=$PORT --server.address=0.0.0.0`
5. Add environment variables in Render dashboard

### Option 3: GitHub Actions (Scheduler only — free)
Create `.github/workflows/daily_signal.yml`:
```yaml
name: LAWI Daily Signal
on:
  schedule:
    - cron: '0 12 * * 1-5'   # 12:00 UTC Mon-Fri = 07:00 Toronto
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python train.py
      - run: python predict_now.py --telegram
    env:
      TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
      TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
```
Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in GitHub → Settings → Secrets.

---

## 📂 Project Structure

```
lawi-system/
│
├── lawi/
│   ├── __init__.py        # Package init
│   ├── config.py          # All settings (symbols, kill zones, thresholds)
│   ├── data_fetcher.py    # Yahoo Finance data download
│   ├── features.py        # ICT 2022 feature engineering (OB, FVG, OTE, etc.)
│   ├── model.py           # XGBoost ML training and prediction
│   ├── alerts.py          # Telegram signal formatting and sending
│   ├── scheduler.py       # APScheduler daily/weekly jobs
│   ├── dashboard.py       # Streamlit dashboard
│   └── db.py              # SQLite database (predictions, accuracy)
│
├── models/                # Trained .pkl model files (created after training)
│
├── train.py               # One-time training script
├── run_scheduler.py       # Start the daily scheduler
├── predict_now.py         # Manual prediction trigger
├── requirements.txt       # Python dependencies
├── .env.example           # Template for credentials
├── Procfile               # For Railway/Heroku deployment
└── railway.json           # Railway deployment config
```

---

## 🔧 Troubleshooting

**Q: `pip` not found**
→ Use `pip3` instead of `pip`, or reinstall Python and tick "Add to PATH"

**Q: Models not found when running dashboard**
→ Run `python train.py` first

**Q: Telegram not sending**
→ Check `.env` file — make sure no spaces around the `=` sign
→ Run `python predict_now.py --telegram` to see the error message

**Q: No data for a symbol**
→ Yahoo Finance may be temporarily down. Wait 10 minutes and retry.

**Q: Training takes very long**
→ Normal — it downloads 365 days × 4 symbols and trains 5-fold CV. Takes 3-10 min.

---

## 📊 Understanding the Signal

Each morning you receive:
```
🤖 LAWI Morning Brief
📊 Gold (XAUUSD)
🟢 Bias: ⬆️ BUY
🎯 Confidence: 72.3%
⚡ Confluence: 6/8 signals
```

**How to use it:**
1. **BUY signal** → Look for long entries during London or NY AM Kill Zone
2. **SELL signal** → Look for short entries during Kill Zones
3. **WAIT signal** → Stay out — market is consolidating
4. **Confluence ≥ 5** = strong signal | **< 3** = weak, be cautious
5. **Confidence ≥ 65%** = reliable signal

Always combine with your own ICT analysis — the model gives you the BIAS, you find the precise entry using OTE / Order Blocks on the lower timeframe.

---

*LAWI Trading System · ICT 2022 Mentorship Model · Built with ❤️ for Willy Kiarie*
