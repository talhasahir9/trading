import streamlit as st
import pandas as pd
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ========== CONFIG ==========
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
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
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

def make_decision(df):
    last = df.iloc[-1]
    signal = "Neutral"
    if last['EMA_9'] > last['EMA_21'] and last['RSI'] < 70 and last['MACD'] > last['MACD_Signal']:
        signal = "Bullish"
    elif last['EMA_9'] < last['EMA_21'] and last['RSI'] > 30 and last['MACD'] < last['MACD_Signal']:
        signal = "Bearish"
    return signal, last['close'], round(last['close'] * 1.02, 2), round(last['close'] * 0.99, 2)

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

for symbol in SYMBOLS:
    st.subheader(symbol)
    cols = st.columns(len(TIMEFRAMES))
    for i, (label, tf) in enumerate(TIMEFRAMES.items()):
        df = get_klines(symbol, tf)
        df = apply_indicators(df)
        signal, entry, tp, sl = make_decision(df)
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

st.caption("Powered by Binance Futures + Technical Analysis + News Sentiment")
