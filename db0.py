import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# CONFIG
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
TIMEFRAMES = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '1d': '1d'}
BASE_URL = "https://fapi.binance.com"
LIMIT = 200

st.set_page_config(layout="wide")
st.title("📊 Multi-Timeframe Crypto Dashboard")

# FUNCTIONS
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
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
    df['ATR'] = atr.average_true_range()
    df['min'] = df['low'].rolling(window=30).min()
    df['max'] = df['high'].rolling(window=30).max()
    return df

def make_decision(df, strategy="Combined"):
    last = df.iloc[-1]
    signal = "Neutral"
    entry = last['close']
    tp = sl = None

    if last['EMA_9'] > last['EMA_21'] and last['RSI'] < 70 and last['MACD'] > last['MACD_Signal']:
        signal = "Bullish"
    elif last['EMA_9'] < last['EMA_21'] and last['RSI'] > 30 and last['MACD'] < last['MACD_Signal']:
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
        tp = np.mean([bb_tp, atr_tp, sr_tp])
        bb_sl = last['BB_Low'] if signal == "Bullish" else last['BB_High']
        atr_sl = entry - 1.5 * last['ATR'] if signal == "Bullish" else entry + 1.5 * last['ATR']
        sr_sl = last['min'] if signal == "Bullish" else last['max']
        sl = np.mean([bb_sl, atr_sl, sr_sl])
    elif strategy == "AI":
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values
        model = LinearRegression().fit(X, y)
        pred = model.predict(np.array([[len(df) + 5]]))[0]
        signal = "Bullish" if pred > y[-1] else "Bearish"
        tp = pred
        sl = entry - (tp - entry) if signal == "Bullish" else entry + (entry - tp)

    return signal, entry, round(tp, 8) if tp else None, round(sl, 8) if sl else None

def backtest_signals(df, strategy="Combined", lookahead=10, initial_balance=1000):
    balance = initial_balance
    trades = 0
    wins = 0
    rr_list = []

    for i in range(30, len(df) - lookahead):
        sliced = df.iloc[:i + 1]
        signal, entry, tp, sl = make_decision(sliced, strategy)
        if signal not in ["Bullish", "Bearish"] or not tp or not sl:
            continue

        success = fail = False
        for j in range(1, lookahead + 1):
            price = df.iloc[i + j]['close']
            if signal == "Bullish":
                if price >= tp:
                    success = True
                    break
                elif price <= sl:
                    fail = True
                    break
            elif signal == "Bearish":
                if price <= tp:
                    success = True
                    break
                elif price >= sl:
                    fail = True
                    break

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk else 0
        rr_list.append(rr)

        if success:
            balance *= (1 + (reward / entry))
            wins += 1
        elif fail:
            balance *= (1 - (risk / entry))
        trades += 1

    win_rate = round(100 * wins / trades, 2) if trades else 0
    rr_avg = round(sum(rr_list) / len(rr_list), 2) if rr_list else 0
    return trades, win_rate, round(balance, 2), rr_avg

def get_news():
    try:
        # Replace 'your_api_key' with your actual CryptoPanic API key
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token=be6cad5fb0879ce0cc363e5025fbec980082d6f4&public=true"
        news_data = requests.get(url).json()['results'][:5]  # Fetch top 5 news articles
        headlines = [n['title'] for n in news_data]  # Extract the title of each news article
    except:
        headlines = ["News unavailable"]
    return headlines

def analyze_sentiment(news):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in news]
    return sum(scores) / len(scores) if scores else 0

# UI
selected_symbol = st.selectbox("Select Coin", SYMBOLS)
strategy = st.selectbox("Select TP/SL Method", ["BB", "ATR", "SupportResistance", "Combined", "AI"])

for tf in TIMEFRAMES:
    st.subheader(f"{selected_symbol} - {tf.upper()}")
    df = get_klines(selected_symbol, TIMEFRAMES[tf])
    df = apply_indicators(df)
    signal, entry, tp, sl = make_decision(df, strategy)
    st.metric(label="Signal", value=signal)
    st.write(f"**Entry:** {entry:.8f} | **TP:** {tp:.8f} | **SL:** {sl:.8f}")
    rr = round(abs(tp - entry) / abs(entry - sl), 2) if tp and sl and entry else "N/A"
    st.write(f"**Reward:Risk:** {rr}")

    t, w, final, rra = backtest_signals(df, strategy)
    st.write(f"**Backtest - Trades:** {t} | **Win Rate:** {w}% | **Final Capital:** ${final} | **Avg R:R:** {rra}")

# Sentiment Analysis
st.subheader("🗞️ News Sentiment")
news = get_news()
sentiment = analyze_sentiment(news)
st.write(f"**Overall Sentiment Score:** {round(sentiment, 2)}")
st.write("Recent Headlines:")
for n in news:
    st.markdown(f"- {n}")
