# app.py - Complete Trading App with FRAMA + DeMark + Quad
# Save this file as app.py

"""
Risk Range Trading System - Professional Grade
Features: FRAMA, DeMark TD Sequential, Quad Framework
Version: 2.0
"""

from __future__ import annotations
import os, math, requests
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote_plus
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Risk Range Pro - FRAMA + DeMark + Quad",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Risk Range Pro - Fractal Adaptive + DeMark + Market Regime")
st.caption(f"Advanced Trading System | Build: {datetime.now().strftime('%Y.%m.%d')} | Stocks Only")

# API Key
API_KEY = os.getenv("TWELVE_DATA_API_KEY") or st.secrets.get("TWELVE_DATA_API_KEY", None)
if not API_KEY:
    st.error("⚠️ No API key detected! Add TWELVE_DATA_API_KEY to Streamlit secrets or environment.")
    st.info("Get free API key at: https://twelvedata.com (800 calls/day free)")
    API_KEY = st.text_input("Or enter your API key here:", type="password")
    if not API_KEY:
        st.stop()

# Constants
DEFAULT_OUTPUTSIZE = 800
DEFAULT_FRAMA_LEN = 20

# Sector mapping
SYMBOL_SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'JPM': 'Financials', 'BAC': 'Financials', 'XLF': 'Financials',
    'XOM': 'Energy', 'CVX': 'Energy', 'XLE': 'Energy',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'WMT': 'Consumer Staples',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
    'SPY': 'Market', 'QQQ': 'Technology', 'TLT': 'Bonds', 'GLD': 'Gold'
}

@st.cache_data(show_spinner=False, ttl=300)
def fetch_ohlcv(symbol: str, interval: str = "1day", outputsize: int = 800) -> pd.DataFrame:
    """Fetch OHLCV data from Twelve Data API"""
    url = f"https://api.twelvedata.com/time_series?symbol={quote_plus(symbol)}&interval={interval}&outputsize={outputsize}&order=ASC&timezone=America/New_York&apikey={API_KEY}"
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            raise RuntimeError("Rate limit hit. Wait 1 minute or reduce symbols.")
        if r.status_code != 200:
            raise RuntimeError(f"API Error {r.status_code}")
        
        j = r.json()
        if j.get("status") == "error":
            raise RuntimeError(j.get("message", "Unknown error"))
        
        vals = j.get("values", [])
        if not vals:
            raise RuntimeError("No data returned")
        
        df = pd.DataFrame(vals)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {symbol}: {str(e)}")

def calculate_frama(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20):
    """Calculate Fractal Adaptive Moving Average"""
    n = len(close)
    frama = np.full(n, np.nan)
    D = np.full(n, np.nan)
    
    if n == 0:
        return pd.Series(frama), pd.Series(D)
    
    v = float(close.iloc[0])
    
    for i in range(n):
        if i >= length - 1:
            half = length // 2
            s = i - length + 1
            
            h1 = np.max(high[s:s+half])
            l1 = np.min(low[s:s+half])
            h2 = np.max(high[s+half:i+1])
            l2 = np.min(low[s+half:i+1])
            h3 = np.max(high[s:i+1])
            l3 = np.min(low[s:i+1])
            
            n1 = (h1 - l1) / max(half, 1)
            n2 = (h2 - l2) / max(half, 1)
            n3 = (h3 - l3) / max(length, 1)
            
            if n1 > 0 and n2 > 0 and n3 > 0:
                d = (math.log(n1 + n2) - math.log(n3)) / math.log(2.0)
                d = np.clip(d, 1.0, 2.0)
                D[i] = d
                
                alpha = math.exp(-4.6 * (d - 1.0))
                alpha = np.clip(alpha, 0.01, 1.0)
                
                v = alpha * float(close.iloc[i]) + (1 - alpha) * v
        
        frama[i] = v
    
    return pd.Series(frama, index=close.index), pd.Series(D, index=close.index)

def calculate_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TD Sequential Setup and Countdown"""
    df = df.copy()
    n = len(df)
    
    df['td_setup'] = 0
    df['td_countdown'] = 0
    df['td_nine'] = False
    df['td_thirteen'] = False
    
    for i in range(4, n):
        if df['close'].iloc[i] < df['close'].iloc[i-4]:
            if df['td_setup'].iloc[i-1] > 0:
                df['td_setup'].iloc[i] = 1
            else:
                df['td_setup'].iloc[i] = df['td_setup'].iloc[i-1] - 1
        elif df['close'].iloc[i] > df['close'].iloc[i-4]:
            if df['td_setup'].iloc[i-1] < 0:
                df['td_setup'].iloc[i] = 1
            else:
                df['td_setup'].iloc[i] = df['td_setup'].iloc[i-1] + 1
        else:
            df['td_setup'].iloc[i] = 0
    
    df['td_nine'] = (abs(df['td_setup']) == 9)
    
    countdown_active = False
    countdown_count = 0
    countdown_direction = 0
    
    for i in range(2, n):
        if df['td_nine'].iloc[i]:
            countdown_active = True
            countdown_count = 0
            countdown_direction = 1 if df['td_setup'].iloc[i] < 0 else -1
        
        if countdown_active and i >= 2:
            if countdown_direction == 1 and df['close'].iloc[i] <= df['low'].iloc[i-2]:
                countdown_count += 1
            elif countdown_direction == -1 and df['close'].iloc[i] >= df['high'].iloc[i-2]:
                countdown_count += 1
            
            df['td_countdown'].iloc[i] = countdown_count
            
            if countdown_count >= 13:
                df['td_thirteen'].iloc[i] = True
                countdown_active = False
    
    return df

def detect_market_quad(spy_df: pd.DataFrame, tlt_df: pd.DataFrame) -> Tuple[int, Dict]:
    """Detect current market Quad"""
    if len(spy_df) < 60 or len(tlt_df) < 60:
        return 1, {'description': 'Insufficient data'}
    
    spy_roc_20 = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-20] - 1) * 100
    spy_roc_60 = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-60] - 1) * 100
    tlt_roc_20 = (tlt_df['close'].iloc[-1] / tlt_df['close'].iloc[-20] - 1) * 100
    
    growth_up = spy_roc_20 > 0 and spy_roc_20 > spy_roc_60
    inflation_up = tlt_roc_20 < -1
    
    if growth_up and not inflation_up:
        quad = 1
    elif growth_up and inflation_up:
        quad = 2
    elif not growth_up and inflation_up:
        quad = 3
    else:
        quad = 4
    
    descriptions = {
        1: "Goldilocks (Growth↑ Inflation↓) - Best for tech/growth",
        2: "Overheating (Growth↑ Inflation↑) - Energy/materials win",
        3: "Stagflation (Growth↓ Inflation↑) - Get defensive",
        4: "Deflation (Growth↓ Inflation↓) - Bonds and cash"
    }
    
    return quad, {
        'growth_score': spy_roc_20,
        'inflation_score': -tlt_roc_20,
        'description': descriptions[quad]
    }

def generate_signals(df: pd.DataFrame, quad: int = 1) -> pd.DataFrame:
    """Generate combined trading signals"""
    df = df.copy()
    
    df['frama'], df['D'] = calculate_frama(df['high'], df['low'], df['close'])
    df['frama_fast'], _ = calculate_frama(df['high'], df['low'], df['close'], 10)
    df['frama_slow'], _ = calculate_frama(df['high'], df['low'], df['close'], 30)
    
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    df['rr_upper'] = df['frama'] + df['atr']
    df['rr_lower'] = df['frama'] - df['atr']
    
    df = calculate_td_sequential(df)
    
    df['signal_score'] = 0
    df['signal_type'] = 'NEUTRAL'
    
    for i in range(len(df)):
        score = 0
        
        if df['frama_fast'].iloc[i] > df['frama_slow'].iloc[i]:
            score += 25
        else:
            score -= 25
        
        if df['td_nine'].iloc[i]:
            if df['td_setup'].iloc[i] == -9:
                score += 40
            elif df['td_setup'].iloc[i] == 9:
                score -= 40
        
        if df['td_thirteen'].iloc[i]:
            score += 60 if df['td_countdown'].iloc[i] > 0 else -60
        
        price_pos = (df['close'].iloc[i] - df['rr_lower'].iloc[i]) / (df['rr_upper'].iloc[i] - df['rr_lower'].iloc[i] + 0.0001)
        if price_pos < 0.2:
            score += 20
        elif price_pos > 0.8:
            score -= 20
        
        quad_mult = {1: 1.3, 2: 1.0, 3: 0.7, 4: 0.5}
        score *= quad_mult.get(quad, 1.0)
        
        if score >= 50:
            df.loc[i, 'signal_type'] = 'STRONG BUY'
        elif score >= 25:
            df.loc[i, 'signal_type'] = 'BUY'
        elif score <= -50:
            df.loc[i, 'signal_type'] = 'STRONG SELL'
        elif score <= -25:
            df.loc[i, 'signal_type'] = 'SELL'
        
        df.loc[i, 'signal_score'] = score
    
    return df

# Main UI
col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
symbols_text = col1.text_input("📊 Symbols", value="AAPL, MSFT, SPY")
interval = col2.selectbox("⏱️ Interval", ["1day", "4h", "1h"], index=0)
quad_mode = col3.selectbox("🎯 Quad", ["Auto", "1", "2", "3", "4"], index=0)
run_btn = col4.button("▶️ Analyze", type="primary", use_container_width=True)

with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        use_demark = st.checkbox("Enable DeMark", value=True)
        show_signals = st.checkbox("Show signals on chart", value=True)
    with col2:
        signal_threshold = st.slider("Signal threshold", 0, 100, 25)
        risk_percent = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)

if run_btn:
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    
    if not symbols:
        st.error("Enter at least one symbol")
        st.stop()
    
    current_quad = 1
    if quad_mode == "Auto":
        with st.spinner("Detecting market regime..."):
            try:
                spy_df = fetch_ohlcv("SPY", interval)
                tlt_df = fetch_ohlcv("TLT", interval)
                current_quad, quad_info = detect_market_quad(spy_df, tlt_df)
                st.success(f"**Quad {current_quad}** - {quad_info['description']}")
            except:
                st.warning("Could not detect Quad, using default")
    else:
        current_quad = int(quad_mode)
    
    tabs = st.tabs(symbols)
    
    for sym, tab in zip(symbols, tabs):
        with tab:
            try:
                with st.spinner(f"Analyzing {sym}..."):
                    df = fetch_ohlcv(sym, interval)
                    df = generate_signals(df, current_quad)
                    
                    latest = df.iloc[-1]
                    
                    st.markdown(f"### {sym} Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${latest['close']:.2f}")
                    
                    trend = "📈 Up" if latest['frama_fast'] > latest['frama_slow'] else "📉 Down"
                    col2.metric("Trend", trend)
                    
                    signal_emoji = "🟢" if "BUY" in latest['signal_type'] else "🔴" if "SELL" in latest['signal_type'] else "⚪"
                    col3.metric("Signal", f"{latest['signal_type']} {signal_emoji}")
                    col4.metric("Score", f"{latest['signal_score']:.0f}")
                    
                    if use_demark:
                        st.markdown("#### TD Sequential Status")
                        col1, col2, col3 = st.columns(3)
                        
                        td_setup = int(latest['td_setup'])
                        td_text = f"{abs(td_setup)}/9"
                        col1.metric("TD Setup", td_text)
                        col2.metric("TD Countdown", f"{int(latest['td_countdown'])}/13")
                        
                        if latest['td_nine']:
                            col3.success("✅ TD9 Signal!")
                        elif latest['td_thirteen']:
                            col3.success("⚡ TD13 Signal!")
                        else:
                            col3.info("Waiting...")
                    
                    st.markdown("#### 🎯 Trading Levels")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.info(f"**Entry:** ${latest['rr_lower']:.2f}")
                    col2.success(f"**Target 1:** ${latest['frama']:.2f}")
                    col3.success(f"**Target 2:** ${latest['rr_upper']:.2f}")
                    col4.warning(f"**Stop:** ${(latest['rr_lower'] - latest['atr']*0.5):.2f}")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=df['datetime'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=sym
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['datetime'], y=df['rr_upper'],
                        mode='lines', name='Upper',
                        line=dict(color='rgba(255,0,0,0.3)', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['datetime'], y=df['frama'],
                        mode='lines', name='FRAMA',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['datetime'], y=df['rr_lower'],
                        mode='lines', name='Lower',
                        line=dict(color='rgba(0,255,0,0.3)', width=1)
                    ))
                    
                    if use_demark and show_signals:
                        td9_buys = df[df['td_nine'] & (df['td_setup'] == -9)]
                        td9_sells = df[df['td_nine'] & (df['td_setup'] == 9)]
                        
                        if not td9_buys.empty:
                            fig.add_trace(go.Scatter(
                                x=td9_buys['datetime'],
                                y=td9_buys['low'] * 0.99,
                                mode='markers+text',
                                name='TD9 Buy',
                                marker=dict(color='green', size=10, symbol='triangle-up'),
                                text=['9'] * len(td9_buys),
                                textposition='bottom center'
                            ))
                        
                        if not td9_sells.empty:
                            fig.add_trace(go.Scatter(
                                x=td9_sells['datetime'],
                                y=td9_sells['high'] * 1.01,
                                mode='markers+text',
                                name='TD9 Sell',
                                marker=dict(color='red', size=10, symbol='triangle-down'),
                                text=['9'] * len(td9_sells),
                                textposition='top center'
                            ))
                    
                    fig.update_layout(
                        title=f"{sym} - FRAMA + DeMark (Quad {current_quad})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=600,
                        xaxis_rangeslider_visible=False,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    sector = SYMBOL_SECTORS.get(sym, 'Unknown')
                    quad_scores = {
                        1: {'Technology': 100, 'Consumer Discretionary': 90, 'Financials': 70, 'Energy': 30},
                        2: {'Energy': 100, 'Materials': 95, 'Financials': 85, 'Technology': 50},
                        3: {'Consumer Staples': 100, 'Utilities': 95, 'Healthcare': 90, 'Technology': 20},
                        4: {'Utilities': 100, 'Consumer Staples': 95, 'Healthcare': 85, 'Technology': 30}
                    }
                    
                    score = quad_scores.get(current_quad, {}).get(sector, 50)
                    
                    if score >= 80:
                        st.success(f"✅ {sym} ({sector}) - EXCELLENT for Quad {current_quad}")
                    elif score >= 60:
                        st.info(f"👍 {sym} ({sector}) - Good for Quad {current_quad}")
                    else:
                        st.warning(f"⚠️ {sym} ({sector}) - Not ideal for Quad {current_quad}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("💡 Best signals: FRAMA + TD9/13 + Right Quad | ⚠️ Risk: Past ≠ Future | 🔑 API: twelvedata.com")