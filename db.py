import streamlit as st
import pandas as pd
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ========== CONFIG ==========
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
TIMEFRAMES = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d'}
LIMIT = 100
BASE_URL = "https://fapi.binance.com"

# ========== FUNCTIONS ==========
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
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    return df

def find_support_resistance(df, lookback=20):
    highs = df['high'].rolling(window=lookback).max()
    lows = df['low'].rolling(window=lookback).min()
    return highs.iloc[-1], lows.iloc[-1]

def make_decision(df, strategy="Bollinger Bands"):
    last = df.iloc[-1]
    signal = "Neutral"
    entry = last['close']

    if last['EMA_9'] > last['EMA_21'] and last['RSI'] < 70 and last['MACD'] > last['MACD_Signal']:
        signal = "Bullish"
    elif last['EMA_9'] < last['EMA_21'] and last['RSI'] > 30 and last['MACD'] < last['MACD_Signal']:
        signal = "Bearish"

    if strategy == "Bollinger Bands":
        bb = BollingerBands(df['close'])
        tp = bb.bollinger_hband().iloc[-1] if signal == "Bullish" else bb.bollinger_lband().iloc[-1]
        sl = entry - 1.5 * (entry * 0.01) if signal == "Bullish" else entry + 1.5 * (entry * 0.01)

    elif strategy == "ATR":
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
        tp = entry + 2 * atr if signal == "Bullish" else entry - 2 * atr
        sl = entry - 1.5 * atr if signal == "Bullish" else entry + 1.5 * atr

    elif strategy == "Support/Resistance":
        resistance, support = find_support_resistance(df)
        tp = resistance if signal == "Bullish" else support
        sl = entry - (entry * 0.015) if signal == "Bullish" else entry + (entry * 0.015)

    elif strategy == "AI (Coming Soon)":
        tp = entry * 1.02
        sl = entry * 0.99

    else:
        tp = sl = None

    return signal, entry, round(tp, 2), round(sl, 2)

def get_news():
    url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"
    try:
        news_data = requests.get(url).json()['results'][:5]
        headlines = [n['title'] for n in news_data]
    except:
        headlines = ["News unavailable"]
    return headlines

def analyze_sentiment(news):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in news]
    return sum(scores) / len(scores) if scores else 0

# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Crypto Trading Dashboard", layout="wide")
st.title("Crypto Trading Signal Dashboard")

strategy_option = st.selectbox("Select TP/SL Strategy", ["Bollinger Bands", "ATR", "Support/Resistance", "AI (Coming Soon)"])

selected_symbol = st.selectbox("Select Coin", SYMBOLS)
st.subheader(f"{selected_symbol}")
cols = st.columns(len(TIMEFRAMES))

for i, (label, tf) in enumerate(TIMEFRAMES.items()):
    df = get_klines(selected_symbol, tf)
    df = apply_indicators(df)
    signal, entry, tp, sl = make_decision(df, strategy=strategy_option)
    with cols[i]:
        st.metric(label=f"{label} Signal", value=signal)
        st.text(f"Entry: ${entry:.2f}\nTP: ${tp} | SL: ${sl}")

# ========== SENTIMENT SECTION ==========
st.markdown("---")
st.header("Market News Sentiment")
news = get_news()
sentiment = analyze_sentiment(news)
st.write("**Headlines:**")
for h in news:
    st.write(f"- {h}")

sentiment_status = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
st.metric(label="Sentiment Score", value=f"{sentiment:.2f}", delta=sentiment_status)

st.caption("Powered by Binance Futures + TA + Sentiment + Smart TP/SL")
