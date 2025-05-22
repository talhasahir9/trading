# ------------------ PART 1: Imports, UI, Data ------------------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import feedparser
from textblob import TextBlob
import asyncio
import websockets
import threading
import json
from datetime import datetime

st.set_page_config(page_title="Smart Crypto Bot", layout="wide")

# --- App Title
st.title("🧠 Smart Crypto Bot v3 – Modular Edition")

# ------------------ Sidebar Config ------------------
with st.sidebar:
    st.header("⚙️ Strategy Settings")

    token_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "BNBUSDT": "BNB-USD"}
    interval_map = {"15m": "15m", "1h": "60m", "4h": "4h", "1d": "1d"}

    token = st.selectbox("Token", list(token_map.keys()), index=0)
    interval = st.selectbox("Interval", list(interval_map.keys()), index=1)
    strategy = st.selectbox("Strategy", ["MA Crossover", "RSI", "Bollinger Bands"])
    tp_sl_method = st.selectbox("TP/SL Method", ["ATR", "Fixed R:R", "Price Structure", "Volatility Range", "Trailing"])
    use_sentiment = st.checkbox("📰 Use News Sentiment?", value=True)
    filter_by_trend = st.checkbox("📉 Filter by HTF Trend?", value=True)

    st.markdown("---")

    capital = st.number_input("💰 Starting Capital ($)", value=10000)
    risk_percent = st.slider("Risk per Trade (%)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    atr_mult = st.slider("ATR/Volatility Multiplier", 1.0, 5.0, 2.0)

    if tp_sl_method == "Fixed R:R":
        rr_ratio = st.slider("Risk:Reward Ratio", 1.0, 5.0, 2.0)

    # Strategy parameters
    st.markdown("---")
    st.subheader("Indicators")
    short_ma = st.slider("MA Short", 5, 20, 10) if strategy == "MA Crossover" else None
    long_ma = st.slider("MA Long", 20, 50, 20) if strategy == "MA Crossover" else None
    rsi_window = st.slider("RSI Window", 5, 30, 14) if strategy == "RSI" else None
    bb_window = st.slider("BB Window", 10, 30, 20) if strategy == "Bollinger Bands" else None

# ------------------ LIVE BINANCE PRICE ------------------
st.subheader("📡 Live Binance Price")
live_price_box = st.empty()
binance_symbol = token.lower()

def run_socket():
    async def listen():
        uri = f"wss://stream.binance.com:9443/ws/{binance_symbol}@trade"
        async with websockets.connect(uri) as ws:
            while True:
                data = json.loads(await ws.recv())
                price = float(data["p"])
                live_price_box.markdown(f"### 💵 Live Price: ${price:,.2f}")
    try:
        asyncio.run(listen())
    except:
        pass

threading.Thread(target=run_socket, daemon=True).start()

# ------------------ Function: Load Price Data ------------------
@st.cache_data
def load_data(symbol, interval):
    return yf.download(symbol, period="10d", interval=interval).dropna()

# ------------------ Function: Load MTF Trend ------------------
def get_htf_trend(symbol, timeframe="1h"):
    df = yf.download(symbol, period="30d", interval=timeframe).dropna()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    return "Bullish" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "Bearish"

# ------------------ Function: Load Sentiment ------------------
def fetch_sentiment():
    url = "https://cryptopanic.com/feed/rss/"
    feed = feedparser.parse(url)
    headlines = feed.entries[:5]
    scores = [TextBlob(article.title).sentiment.polarity for article in headlines]
    return sum(scores) / len(scores) if scores else 0, headlines
    # ------------------ Function: Generate Strategy Signals ------------------
def generate_signals(df, strategy, short_ma=None, long_ma=None, rsi_window=None, bb_window=None):
    df["Signal"] = 0

    if strategy == "MA Crossover":
        df["MA1"] = df["Close"].rolling(short_ma).mean()
        df["MA2"] = df["Close"].rolling(long_ma).mean()
        df["Signal"] = np.where((df["MA1"] > df["MA2"]) & (df["MA1"].shift(1) <= df["MA2"].shift(1)), 1,
                        np.where((df["MA1"] < df["MA2"]) & (df["MA1"].shift(1) >= df["MA2"].shift(1)), -1, 0))

    elif strategy == "RSI":
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], rsi_window).rsi()
        df["Signal"] = np.where((df["RSI"].shift(1) < 30) & (df["RSI"] > 30), 1,
                        np.where((df["RSI"].shift(1) > 70) & (df["RSI"] < 70), -1, 0))

    elif strategy == "Bollinger Bands":
        bb = ta.volatility.BollingerBands(df["Close"], bb_window, 2)
        df["Upper"] = bb.bollinger_hband()
        df["Lower"] = bb.bollinger_lband()
        df["Signal"] = np.where((df["Close"].shift(1) < df["Lower"].shift(1)) & (df["Close"] > df["Lower"]), 1,
                        np.where((df["Close"].shift(1) > df["Upper"].shift(1)) & (df["Close"] < df["Upper"]), -1, 0))

    return df[df["Signal"] != 0]


# ------------------ Function: Calculate TP/SL ------------------
def calculate_tp_sl(row, df, method, atr_mult, rr_ratio):
    price = row["Price"]
    atr = row["ATR"]
    i = df.index.get_loc(row.name)

    if method == "ATR":
        tp = price + atr_mult * atr if row["Type"] == "Buy" else price - atr_mult * atr
        sl = price - atr_mult * atr if row["Type"] == "Buy" else price + atr_mult * atr

    elif method == "Fixed R:R":
        risk = 0.01
        tp = price * (1 + risk * rr_ratio) if row["Type"] == "Buy" else price * (1 - risk * rr_ratio)
        sl = price * (1 - risk) if row["Type"] == "Buy" else price * (1 + risk)

    elif method == "Price Structure":
        lookback = df.iloc[max(0, i - 10):i]
        low, high = lookback["Low"].min(), lookback["High"].max()
        tp = price + atr_mult * atr if row["Type"] == "Buy" else price - atr_mult * atr
        sl = low if row["Type"] == "Buy" else high

    elif method == "Volatility Range":
        range_val = (df["High"].rolling(10).mean() - df["Low"].rolling(10).mean()).iloc[i]
        tp = price + 2 * range_val if row["Type"] == "Buy" else price - 2 * range_val
        sl = price - range_val if row["Type"] == "Buy" else price + range_val

    elif method == "Trailing":
        tp = None
        sl = price - atr if row["Type"] == "Buy" else price + atr

    return pd.Series([tp, sl])


# ------------------ Function: Run Backtest ------------------
def run_backtest(signals, df, capital, risk_percent, sentiment_score, use_sentiment,
                 trend_1h, filter_by_trend):
    trades = []
    cap = capital

    for i in range(len(signals) - 1):
        s = signals.iloc[i]
        next_time = signals.index[i + 1]

        if filter_by_trend:
            if trend_1h == "Bullish" and s["Type"] == "Sell":
                continue
            if trend_1h == "Bearish" and s["Type"] == "Buy":
                continue

        if use_sentiment:
            if s["Type"] == "Buy" and sentiment_score < -0.1:
                continue
            if s["Type"] == "Sell" and sentiment_score > 0.1:
                continue

        entry_price = s["Price"]
        risk = cap * (risk_percent / 100)
        size = risk / abs(entry_price - s["SL"])
        df_slice = df[(df.index > s.name) & (df.index <= next_time)]

        result, exit_price = "Loss", s["SL"]
        for ts, row in df_slice.iterrows():
            if s["Type"] == "Buy":
                if row["Low"] <= s["SL"]:
                    exit_time, pnl = ts, -risk
                    break
                if s["TP"] and row["High"] >= s["TP"]:
                    exit_price, exit_time, result = s["TP"], ts, "Win"
                    break
            else:
                if row["High"] >= s["SL"]:
                    exit_time, pnl = ts, -risk
                    break
                if s["TP"] and row["Low"] <= s["TP"]:
                    exit_price, exit_time, result = s["TP"], ts, "Win"
                    break
        else:
            exit_price = df.loc[next_time]["Close"]
            exit_time = next_time
            result = "Win" if ((s["Type"] == "Buy" and exit_price > entry_price) or
                               (s["Type"] == "Sell" and exit_price < entry_price)) else "Loss"

        pnl = (exit_price - entry_price) * size if s["Type"] == "Buy" else (entry_price - exit_price) * size
        cap += pnl

        trades.append({
            "Entry": s.name, "Exit": exit_time,
            "Type": s["Type"], "Entry Price": entry_price,
            "Exit Price": exit_price, "PnL": pnl,
            "Result": result, "Capital": cap
        })

    df_trades = pd.DataFrame(trades)
    df_trades["Cumulative PnL"] = df_trades["PnL"].cumsum()
    return df_trades
    # ------------------ RUN THE BOT ------------------
st.subheader("📊 Strategy Results")

df = load_data(yf_symbol, yf_interval)
df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()

sentiment_score, news_headlines = fetch_sentiment() if use_sentiment else (0, [])
trend_1h = get_htf_trend(yf_symbol, "60m")
trend_1d = get_htf_trend(yf_symbol, "1d")

st.markdown(f"**📈 Sentiment:** `{sentiment_score:.2f}` — **{('😃 Bullish' if sentiment_score > 0.1 else '😡 Bearish' if sentiment_score < -0.1 else '😐 Neutral')}`")
st.markdown(f"**⏱ 1H Trend:** `{trend_1h}` | **📅 1D Trend:** `{trend_1d}`")

for news in news_headlines:
    st.markdown(f"- [{news.title}]({news.link})")

signals = generate_signals(df, strategy, short_ma, long_ma, rsi_window, bb_window)
signals["Type"] = signals["Signal"].apply(lambda x: "Buy" if x == 1 else "Sell")
signals["Price"] = signals["Close"]
signals[["TP", "SL"]] = signals.apply(calculate_tp_sl, axis=1,
    args=(df, tp_sl_method, atr_mult, rr_ratio if tp_sl_method == "Fixed R:R" else 2), result_type='expand'
)

results_df = run_backtest(signals, df, capital, risk_percent, sentiment_score, use_sentiment, trend_1h, filter_by_trend)

# ------------------ VISUAL OUTPUT ------------------
st.markdown("### 📈 Equity Curve")
fig, ax = plt.subplots()
results_df["Cumulative PnL"].plot(ax=ax)
ax.set_ylabel("Profit ($)")
ax.grid()
st.pyplot(fig)

# ------------------ STATS ------------------
st.write(f"**📊 Total Trades:** {len(results_df)}")
st.write(f"**🏆 Win Rate:** {(results_df['Result'] == 'Win').mean():.2%}")
st.write(f"**💰 Final Capital:** ${results_df['Capital'].iloc[-1]:,.2f}")

# ------------------ DOWNLOADS ------------------
col1, col2 = st.columns(2)
col1.download_button("📥 Download Trades CSV", results_df.to_csv(index=False), "trades.csv")
col2.download_button("📥 Download Signals CSV", signals.to_csv(index=False), "signals.csv")