import streamlit as st
import pandas as pd
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import numpy as np

# ========== CONFIG ==========
SYMBOLS = ['DOGEUSDT', '1000SHIBUSDT', '1000PEPEUSDT', '1000FLOKIUSDT']
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '1d']
BASE_URL = "https://fapi.binance.com"
LIMIT = 200

st.set_page_config(page_title="Meme Coin Analyzer", layout="wide")
st.title("🚀 Meme Coin Trading Dashboard")

# ========== DATA FETCHING ==========
def get_klines(symbol, interval, limit=LIMIT):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    return df

def apply_indicators(df):
    df['EMA_9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['ATR'] = atr.average_true_range()
    df['min'] = df['low'].rolling(window=20).min()
    df['max'] = df['high'].rolling(window=20).max()
    return df

def make_decision(df, strategy="Combined"):
    last = df.iloc[-1]
    entry = last['close']
    signal = "Neutral"
    tp = sl = None

    if last['EMA_9'] > last['EMA_21'] and last['MACD'] > last['MACD_Signal'] and last['RSI'] < 70:
        signal = "Bullish"
    elif last['EMA_9'] < last['EMA_21'] and last['MACD'] < last['MACD_Signal'] and last['RSI'] > 30:
        signal = "Bearish"

    if strategy == "BB":
        tp = last['BB_High'] if signal == "Bullish" else last['BB_Low']
        sl = last['BB_Low'] if signal == "Bullish" else last['BB_High']
    elif strategy == "ATR":
        tp = entry + 2 * last['ATR'] if signal == "Bullish" else entry - 2 * last['ATR']
        sl = entry - 1.5 * last['ATR'] if signal == "Bullish" else entry + 1.5 * last['ATR']
    elif strategy == "SupportResistance":
        tp = last['max'] if signal == "Bullish" else last['min']
        sl = last['min'] if signal == "Bullish" else last['max']
    elif strategy == "Combined":
        bb_tp = last['BB_High'] if signal == "Bullish" else last['BB_Low']
        atr_tp = entry + 2 * last['ATR'] if signal == "Bullish" else entry - 2 * last['ATR']
        sr_tp = last['max'] if signal == "Bullish" else last['min']
        tp = min(bb_tp, atr_tp, sr_tp) if signal == "Bullish" else max(bb_tp, atr_tp, sr_tp)

        bb_sl = last['BB_Low'] if signal == "Bullish" else last['BB_High']
        atr_sl = entry - 1.5 * last['ATR'] if signal == "Bullish" else entry + 1.5 * last['ATR']
        sr_sl = last['min'] if signal == "Bullish" else last['max']
        sl = max(bb_sl, atr_sl, sr_sl) if signal == "Bullish" else min(bb_sl, atr_sl, sr_sl)
    elif strategy == "AI":
        # Basic linear regression AI-driven TP/SL
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values
        model = LinearRegression().fit(X, y)
        trend = model.predict([[len(df) + 5]])[0]
        tp = trend
        sl = entry - (tp - entry) * 0.5
        signal = "Bullish" if tp > entry else "Bearish"

    return signal, entry, round(tp, 8) if tp else None, round(sl, 8) if sl else None

def get_news_sentiment():
    url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"
    try:
        headlines = requests.get(url).json()['results']
        headlines = [n['title'] for n in headlines][:5]
    except:
        headlines = ["News unavailable"]
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    avg_sentiment = round(sum(scores) / len(scores), 2) if scores else 0
    return headlines, avg_sentiment

def backtest_signals(df, strategy, lookahead=10, capital=1000):
    balance = capital
    for i in range(30, len(df) - lookahead):
        sub_df = df.iloc[:i + 1]
        signal, entry, tp, sl = make_decision(sub_df, strategy)
        if signal not in ["Bullish", "Bearish"] or not tp or not sl:
            continue
        for j in range(1, lookahead):
            price = df.iloc[i + j]['close']
            if (signal == "Bullish" and price >= tp) or (signal == "Bearish" and price <= tp):
                balance *= (1 + abs(tp - entry) / entry)
                break
            elif (signal == "Bullish" and price <= sl) or (signal == "Bearish" and price >= sl):
                balance *= (1 - abs(entry - sl) / entry)
                break
    profit_pct = round((balance - capital) / capital * 100, 2)
    return round(balance, 2), profit_pct

# ========== STREAMLIT UI ==========
selected_symbol = st.selectbox("Select Coin", SYMBOLS)
strategy = st.selectbox("TP/SL Strategy", ["BB", "ATR", "SupportResistance", "Combined", "AI"])

news, sentiment_score = get_news_sentiment()
st.subheader("📰 Sentiment Analysis")
st.write("**Sentiment Score:**", sentiment_score)
for headline in news:
    st.write("•", headline)

st.divider()
cols = st.columns(len(TIMEFRAMES))

for idx, tf in enumerate(TIMEFRAMES):
    with cols[idx]:
        st.markdown(f"### ⏱️ {tf}")
        df = get_klines(selected_symbol, tf)
        df = apply_indicators(df)
        signal, entry, tp, sl = make_decision(df, strategy)
        rr = round(abs(tp - entry) / abs(entry - sl), 2) if tp and sl and sl != entry else 0
        st.metric("Signal", signal)
        st.write(f"Entry: {entry:.8f}")
        st.write(f"TP: {tp}")
        st.write(f"SL: {sl}")
        st.write(f"R:R: {rr}")
        final_balance, profit_pct = backtest_signals(df, strategy)
        st.write(f"Backtest Final Capital: ${final_balance} (+{profit_pct}%)")
