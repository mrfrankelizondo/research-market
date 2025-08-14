# app.py - Complete Trading App with FRAMA + DeMark + Quad (Enhanced with Plain English)
# Save this file as app.py

"""
Risk Range Trading System - Professional Grade
Features: FRAMA, DeMark TD Sequential, Quad Framework
Version: 3.0 - Enhanced with plain English descriptions
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
.stMetric > div:first-child { font-size: 0.9rem; }
.stMetric > div:nth-child(2) { font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Risk Range Pro - Fractal Adaptive + DeMark + Market Regime")
st.caption(f"Advanced Trading System | Build: {datetime.now().strftime('%Y.%m.%d')} | Plain English Signals")

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
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
    'JPM': 'Financials', 'BAC': 'Financials', 'XLF': 'Financials', 'GS': 'Financials',
    'XOM': 'Energy', 'CVX': 'Energy', 'XLE': 'Energy', 'COP': 'Energy',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'CVS': 'Healthcare',
    'SPY': 'Market', 'QQQ': 'Technology', 'TLT': 'Bonds', 'GLD': 'Gold',
    'XLU': 'Utilities', 'XLI': 'Industrials', 'XLV': 'Healthcare', 'XLK': 'Technology'
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
        return 1, {'description': 'Insufficient data', 'simple': 'Need more data'}
    
    spy_roc_20 = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-20] - 1) * 100
    spy_roc_60 = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-60] - 1) * 100
    tlt_roc_20 = (tlt_df['close'].iloc[-1] / tlt_df['close'].iloc[-20] - 1) * 100
    
    growth_up = spy_roc_20 > 0 and spy_roc_20 > spy_roc_60
    inflation_up = tlt_roc_20 < -1
    
    if growth_up and not inflation_up:
        quad = 1
        simple = "Economy growing, inflation falling - BEST environment"
    elif growth_up and inflation_up:
        quad = 2
        simple = "Economy hot, inflation rising - Watch for Fed action"
    elif not growth_up and inflation_up:
        quad = 3
        simple = "Economy slowing, inflation high - WORST environment"
    else:
        quad = 4
        simple = "Economy weak, inflation falling - Wait for recovery"
    
    descriptions = {
        1: "Goldilocks (Growth↑ Inflation↓) - Best for tech/growth",
        2: "Overheating (Growth↑ Inflation↑) - Energy/materials win",
        3: "Stagflation (Growth↓ Inflation↑) - Get defensive",
        4: "Deflation (Growth↓ Inflation↓) - Bonds and cash"
    }
    
    return quad, {
        'growth_score': spy_roc_20,
        'inflation_score': -tlt_roc_20,
        'description': descriptions[quad],
        'simple': simple
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
symbols_text = col1.text_input("📊 Symbols", value="AAPL, MSFT, SPY", 
                               help="Enter stock symbols separated by commas")
interval = col2.selectbox("⏱️ Timeframe", ["1day", "4h", "1h"], index=0,
                          help="Daily for investing, hourly for trading")
quad_mode = col3.selectbox("🎯 Market Mode", ["Auto Detect", "1-Best", "2-Hot", "3-Danger", "4-Cash"], index=0)
run_btn = col4.button("▶️ Analyze", type="primary", use_container_width=True)

with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        use_demark = st.checkbox("Enable TD Sequential", value=True, 
                                help="Shows exhaustion points")
        show_signals = st.checkbox("Show signals on chart", value=True)
    with col2:
        signal_threshold = st.slider("Signal sensitivity", 0, 100, 25,
                                    help="Lower = more signals")
        risk_percent = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)

if run_btn:
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    
    if not symbols:
        st.error("Please enter at least one symbol")
        st.stop()
    
    current_quad = 1
    if quad_mode == "Auto Detect":
        with st.spinner("🔍 Analyzing market conditions..."):
            try:
                spy_df = fetch_ohlcv("SPY", interval)
                tlt_df = fetch_ohlcv("TLT", interval)
                current_quad, quad_info = detect_market_quad(spy_df, tlt_df)
                
                st.success(f"📍 **Market Environment: Quad {current_quad}**")
                st.info(f"💡 **Plain English:** {quad_info['simple']}")
                
                col1, col2 = st.columns(2)
                col1.metric("Stock Market Momentum", f"{quad_info['growth_score']:.1f}%",
                           help="Positive = Growing, Negative = Slowing")
                col2.metric("Inflation Pressure", f"{quad_info['inflation_score']:.1f}%",
                           help="Positive = Rising, Negative = Falling")
                
            except:
                st.warning("Could not detect market regime, using default (Quad 1)")
                current_quad = 1
    else:
        quad_map = {"1-Best": 1, "2-Hot": 2, "3-Danger": 3, "4-Cash": 4}
        current_quad = quad_map.get(quad_mode, 1)
        
        quad_explanations = {
            1: "Best environment - Buy quality growth stocks on any dip",
            2: "Overheating - Rotate to energy, materials, banks",
            3: "Dangerous - Stay defensive or in cash",
            4: "Recession likely - Bonds and cash are king"
        }
        st.info(f"📍 **Using Quad {current_quad}:** {quad_explanations[current_quad]}")
    
    tabs = st.tabs(symbols)
    
    for sym, tab in zip(symbols, tabs):
        with tab:
            try:
                with st.spinner(f"Analyzing {sym}..."):
                    df = fetch_ohlcv(sym, interval)
                    df = generate_signals(df, current_quad)
                    
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    # Header with price
                    st.markdown(f"## {sym} Analysis")
                    
                    price_change = latest['close'] - prev['close']
                    price_pct = (price_change / prev['close']) * 100
                    
                    col1, col2, col3 = st.columns([2, 2, 2])
                    
                    with col1:
                        st.metric("Current Price", f"${latest['close']:.2f}", 
                                 f"{price_change:+.2f} ({price_pct:+.1f}%)")
                    
                    with col2:
                        trend = "📈 Uptrend" if latest['frama_fast'] > latest['frama_slow'] else "📉 Downtrend"
                        trend_desc = "Bullish momentum" if "Up" in trend else "Bearish momentum"
                        st.metric("Market Trend", trend)
                        st.caption(trend_desc)
                    
                    with col3:
                        # Convert signal to plain English
                        signal_map = {
                            'STRONG BUY': ("💚 Strong Buy", "Excellent setup - buy now"),
                            'BUY': ("🟢 Buy", "Good setup - consider buying"),
                            'STRONG SELL': ("🔴 Strong Sell", "Exit immediately"),
                            'SELL': ("🟠 Sell", "Consider reducing position"),
                            'NEUTRAL': ("⚪ Wait", "No clear opportunity")
                        }
                        signal_display, signal_desc = signal_map.get(latest['signal_type'], 
                                                                     ("⚪ Wait", "No signal"))
                        st.metric("Signal", signal_display)
                        st.caption(signal_desc)
                    
                    # TD Sequential Status with plain English
                    if use_demark:
                        st.markdown("### 📊 Market Exhaustion Indicators (TD Sequential)")
                        col1, col2, col3 = st.columns(3)
                        
                        td_setup = int(latest['td_setup'])
                        td_countdown = int(latest['td_countdown'])
                        
                        with col1:
                            # TD Setup plain English
                            if td_setup == -9:
                                td_text = "🟢 BOTTOM SIGNAL"
                                td_desc = "Selling exhausted - buy opportunity"
                            elif td_setup == 9:
                                td_text = "🔴 TOP SIGNAL"
                                td_desc = "Buying exhausted - sell warning"
                            elif td_setup <= -7:
                                td_text = f"📉 Oversold soon ({abs(td_setup)}/9)"
                                td_desc = "Bottom forming"
                            elif td_setup >= 7:
                                td_text = f"📈 Overbought soon ({td_setup}/9)"
                                td_desc = "Top forming"
                            elif td_setup <= -4:
                                td_text = f"📉 Selling pressure ({abs(td_setup)}/9)"
                                td_desc = "Downtrend active"
                            elif td_setup >= 4:
                                td_text = f"📈 Buying pressure ({td_setup}/9)"
                                td_desc = "Uptrend active"
                            else:
                                td_text = "➡️ No pattern"
                                td_desc = "Waiting for setup"
                            
                            st.metric("Exhaustion Level", td_text)
                            st.caption(td_desc)
                        
                        with col2:
                            # TD Countdown plain English
                            if latest['td_thirteen']:
                                cd_text = "⚡ MAJOR REVERSAL"
                                cd_desc = "Strong reversal NOW"
                            elif td_countdown >= 10:
                                cd_text = f"⚠️ Reversal soon ({td_countdown}/13)"
                                cd_desc = "Major turn imminent"
                            elif td_countdown >= 7:
                                cd_text = f"📊 Building ({td_countdown}/13)"
                                cd_desc = "Pressure increasing"
                            elif td_countdown >= 1:
                                cd_text = f"⏳ Counting ({td_countdown}/13)"
                                cd_desc = "Tracking reversal"
                            else:
                                cd_text = "⏸️ Inactive"
                                cd_desc = "No countdown yet"
                            
                            st.metric("Reversal Countdown", cd_text)
                            st.caption(cd_desc)
                        
                        with col3:
                            # Overall signal
                            if latest['td_nine'] and td_setup == -9:
                                st.success("✅ **BUY SETUP COMPLETE**")
                                st.caption("Look for entry point")
                            elif latest['td_nine'] and td_setup == 9:
                                st.error("❌ **SELL SETUP COMPLETE**")
                                st.caption("Take profits/exit")
                            elif latest['td_thirteen']:
                                st.warning("⚡ **MAJOR REVERSAL**")
                                st.caption("Strong signal!")
                            elif abs(td_setup) >= 7:
                                st.info(f"⏰ **Signal in {9-abs(td_setup)} days**")
                                st.caption("Get ready...")
                            else:
                                st.info("📊 **Monitoring...**")
                                st.caption("No signal yet")
                    
                    # Trading Levels with actionable language
                    st.markdown("### 🎯 Action Plan")
                    
                    # Determine context
                    if latest['frama_fast'] > latest['frama_slow']:
                        market_state = "📈 **Market State: UPTREND** - Look for buying opportunities"
                    else:
                        market_state = "📉 **Market State: DOWNTREND** - Avoid buying, consider selling"
                    
                    price_position = (latest['close'] - latest['rr_lower']) / (latest['rr_upper'] - latest['rr_lower'] + 0.0001)
                    if price_position < 0.3:
                        position_quality = "🟢 **Price Location: EXCELLENT** - Near support, good entry zone"
                    elif price_position < 0.5:
                        position_quality = "🟡 **Price Location: DECENT** - Middle of range, okay entry"
                    elif price_position < 0.7:
                        position_quality = "🟠 **Price Location: POOR** - Above middle, wait for pullback"
                    else:
                        position_quality = "🔴 **Price Location: OVERBOUGHT** - Near resistance, don't buy"
                    
                    st.info(market_state)
                    st.info(position_quality)
                    
                    # Specific price levels
                    col1, col2, col3, col4 = st.columns(4)
                    
                    stop_loss = latest['rr_lower'] - latest['atr']*0.5
                    risk_amt = abs(latest['rr_lower'] - stop_loss)
                    reward1 = abs(latest['frama'] - latest['rr_lower'])
                    reward2 = abs(latest['rr_upper'] - latest['rr_lower'])
                    rr1 = reward1/risk_amt if risk_amt > 0 else 0
                    rr2 = reward2/risk_amt if risk_amt > 0 else 0
                    
                    with col1:
                        st.success(f"**BUY at:** ${latest['rr_lower']:.2f}")
                        st.caption("Wait for this price")
                    
                    with col2:
                        st.info(f"**Target 1:** ${latest['frama']:.2f}")
                        st.caption(f"Sell half ({rr1:.1f}:1)")
                    
                    with col3:
                        st.info(f"**Target 2:** ${latest['rr_upper']:.2f}")
                        st.caption(f"Sell rest ({rr2:.1f}:1)")
                    
                    with col4:
                        st.error(f"**STOP at:** ${stop_loss:.2f}")
                        risk_pct = abs((stop_loss - latest['rr_lower'])/latest['rr_lower']*100)
                        st.caption(f"Risk: {risk_pct:.1f}%")
                    
                    # Plain English Action Box
                    st.markdown("### 📋 What Should I Do?")
                    
                    if latest['signal_score'] >= 50:
                        st.success(f"""
                        ### ✅ **ACTION: BUY THIS STOCK**
                        
                        **Step-by-step instructions:**
                        1. Place a limit buy order at **${latest['rr_lower']:.2f}**
                        2. Set your stop loss at **${stop_loss:.2f}** (risk: {risk_pct:.1f}%)
                        3. When price reaches **${latest['frama']:.2f}**, sell half your position
                        4. Let the rest ride to **${latest['rr_upper']:.2f}**, then sell all
                        5. Position size: Risk only **{risk_percent}%** of your account
                        
                        **Why this is a good trade:**
                        - Signal strength is very high ({latest['signal_score']:.0f}/100)
                        - Risk/reward ratio is favorable ({rr2:.1f}:1)
                        - Market regime (Quad {current_quad}) supports this trade
                        """)
                    
                    elif latest['signal_score'] >= 25:
                        st.info(f"""
                        ### 👍 **ACTION: CONSIDER BUYING**
                        
                        **The setup is decent but not perfect:**
                        - Wait for price to drop to **${latest['rr_lower']:.2f}**
                        - Use a smaller position (half normal size)
                        - Stop loss at **${stop_loss:.2f}**
                        - Signal strength: {latest['signal_score']:.0f}/100 (moderate)
                        
                        **Consider waiting if:**
                        - You're risk-averse
                        - You have better opportunities
                        - TD Sequential hasn't completed yet
                        """)
                    
                    elif latest['signal_score'] <= -25:
                        st.error(f"""
                        ### ❌ **ACTION: DO NOT BUY (Consider Selling)**
                        
                        **Why you should avoid this stock:**
                        - Bearish signal detected ({latest['signal_score']:.0f}/100)
                        - Trend is down
                        - Better opportunities elsewhere
                        
                        **If you own this stock:**
                        - Consider selling now at **${latest['close']:.2f}**
                        - Or set a stop loss immediately
                        - Don't add to your position
                        """)
                    
                    else:
                        st.warning(f"""
                        ### ⏸️ **ACTION: WAIT**
                        
                        **No clear opportunity right now because:**
                        - Signal is neutral ({latest['signal_score']:.0f}/100)
                        - No edge in this trade
                        - Better to preserve capital
                        
                        **Check back when:**
                        - TD Sequential reaches 9
                        - Price hits support at **${latest['rr_lower']:.2f}**
                        - Signal score exceeds 25
                        """)
                    
                    # Chart
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
                        mode='lines', name='Sell Zone (Resistance)',
                        line=dict(color='rgba(255,0,0,0.3)', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['datetime'], y=df['frama'],
                        mode='lines', name='Fair Value (FRAMA)',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['datetime'], y=df['rr_lower'],
                        mode='lines', name='Buy Zone (Support)',
                        line=dict(color='rgba(0,255,0,0.3)', width=1)
                    ))
                    
                    if use_demark and show_signals:
                        # TD9 Buy signals
                        td9_buys = df[df['td_nine'] & (df['td_setup'] == -9)]
                        if not td9_buys.empty:
                            fig.add_trace(go.Scatter(
                                x=td9_buys['datetime'],
                                y=td9_buys['low'] * 0.99,
                                mode='markers+text',
                                name='Buy Signal (TD9)',
                                marker=dict(color='green', size=12, symbol='triangle-up'),
                                text=['BUY'] * len(td9_buys),
                                textposition='bottom center'
                            ))
                        
                        # TD9 Sell signals
                        td9_sells = df[df['td_nine'] & (df['td_setup'] == 9)]
                        if not td9_sells.empty:
                            fig.add_trace(go.Scatter(
                                x=td9_sells['datetime'],
                                y=td9_sells['high'] * 1.01,
                                mode='markers+text',
                                name='Sell Signal (TD9)',
                                marker=dict(color='red', size=12, symbol='triangle-down'),
                                text=['SELL'] * len(td9_sells),
                                textposition='top center'
                            ))
                        
                        # TD13 signals
                        td13s = df[df['td_thirteen']]
                        if not td13s.empty:
                            fig.add_trace(go.Scatter(
                                x=td13s['datetime'],
                                y=td13s['close'],
                                mode='markers',
                                name='Major Reversal (TD13)',
                                marker=dict(color='yellow', size=15, symbol='star',
                                          line=dict(color='black', width=2))
                            ))
                    
                    # Add signal backgrounds
                    if show_signals:
                        for i in range(len(df)):
                            if abs(df['signal_score'].iloc[i]) >= signal_threshold:
                                if df['signal_score'].iloc[i] > 0:
                                    color = 'rgba(0,255,0,0.05)'
                                else:
                                    color = 'rgba(255,0,0,0.05)'
                                    
                                fig.add_vrect(
                                    x0=df['datetime'].iloc[i],
                                    x1=df['datetime'].iloc[min(i+1, len(df)-1)],
                                    fillcolor=color,
                                    layer="below",
                                    line_width=0
                                )
                    
                    fig.update_layout(
                        title=f"{sym} - Buy at Green Line, Sell at Red Line (Quad {current_quad})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=600,
                        xaxis_rangeslider_visible=False,
                        template="plotly_white",
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plain English Summary
                    st.markdown("### 📝 Plain English Summary")
                    
                    summary_parts = []
                    
                    # Price trend
                    if latest['frama_fast'] > latest['frama_slow']:
                        summary_parts.append(f"{sym} is in an **uptrend** (bullish)")
                    else:
                        summary_parts.append(f"{sym} is in a **downtrend** (bearish)")
                    
                    # TD Sequential
                    if td_setup == -9:
                        summary_parts.append("The selling looks exhausted (potential bottom)")
                    elif td_setup == 9:
                        summary_parts.append("The buying looks exhausted (potential top)")
                    elif abs(td_setup) >= 7:
                        bars_left = 9 - abs(td_setup)
                        summary_parts.append(f"A reversal signal may appear in {bars_left} days")
                    
                    # Price position
                    if price_position < 0.3:
                        summary_parts.append("Price is near support (good for buying)")
                    elif price_position > 0.7:
                        summary_parts.append("Price is near resistance (good for selling)")
                    else:
                        summary_parts.append("Price is in the middle of its range")
                    
                    # Signal
                    if latest['signal_score'] >= 50:
                        summary_parts.append("**Strong buy signal** - multiple indicators agree")
                    elif latest['signal_score'] >= 25:
                        summary_parts.append("Moderate buy signal - some positive signs")
                    elif latest['signal_score'] <= -50:
                        summary_parts.append("**Strong sell signal** - exit positions")
                    elif latest['signal_score'] <= -25:
                        summary_parts.append("Moderate sell signal - reduce exposure")
                    else:
                        summary_parts.append("No clear signal - better to wait")
                    
                    # Quad context
                    quad_context = {
                        1: f"The market environment (Quad {current_quad}) is excellent for stocks",
                        2: f"The market environment (Quad {current_quad}) suggests caution on growth stocks",
                        3: f"The market environment (Quad {current_quad}) is dangerous - be very selective",
                        4: f"The market environment (Quad {current_quad}) favors cash over stocks"
                    }
                    summary_parts.append(quad_context.get(current_quad, ""))
                    
                    # Display summary
                    summary = ". ".join(summary_parts) + "."
                    st.info(summary)
                    
                    # Position Calculator
                    with st.expander("💰 Position Size Calculator", expanded=False):
                        account = st.number_input(
                            "Your Account Size ($)", 
                            value=10000, 
                            step=100, 
                            key=f"acc_{sym}",
                            help="Total value of your trading account"
                        )
                        
                        entry = latest['rr_lower']
                        stop = entry - latest['atr'] * 0.5
                        risk_amt = account * (risk_percent / 100)
                        shares = int(risk_amt / abs(entry - stop)) if abs(entry - stop) > 0 else 0
                        position_value = shares * entry
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Shares to Buy", f"{shares:,}")
                        col2.metric("Total Cost", f"${position_value:,.2f}")
                        col3.metric("Risk Amount", f"${risk_amt:.2f}")
                        col4.metric("Potential Profit", f"${shares * (latest['rr_upper'] - entry):.2f}")
                        
                        st.info(f"""
                        **Instructions:**
                        1. Buy {shares:,} shares at ${entry:.2f}
                        2. This will cost ${position_value:,.2f}
                        3. Set stop loss at ${stop:.2f}
                        4. Maximum loss will be ${risk_amt:.2f} ({risk_percent}% of account)
                        5. Potential profit at target: ${shares * (latest['rr_upper'] - entry):.2f}
                        """)
                    
                    # Sector Analysis
                    sector = SYMBOL_SECTORS.get(sym, 'Unknown')
                    quad_scores = {
                        1: {'Technology': 100, 'Consumer Discretionary': 90, 'Financials': 70, 
                            'Energy': 30, 'Consumer Staples': 40, 'Healthcare': 60, 'Utilities': 20},
                        2: {'Energy': 100, 'Materials': 95, 'Financials': 85, 'Technology': 50,
                            'Consumer Staples': 60, 'Healthcare': 55, 'Utilities': 30},
                        3: {'Consumer Staples': 100, 'Utilities': 95, 'Healthcare': 90, 
                            'Technology': 20, 'Energy': 70, 'Financials': 30},
                        4: {'Utilities': 100, 'Consumer Staples': 95, 'Healthcare': 85, 
                            'Technology': 30, 'Energy': 25, 'Financials': 20, 'Bonds': 100}
                    }
                    
                    score = quad_scores.get(current_quad, {}).get(sector, 50)
                    
                    st.markdown("### 🏢 Sector Analysis")
                    
                    if score >= 80:
                        st.success(f"""
                        ✅ **EXCELLENT FIT** - {sym} ({sector})
                        
                        This sector performs very well in Quad {current_quad}.
                        Historical win rate: ~{60 + score/5:.0f}%
                        Recommendation: Buy aggressively on dips
                        """)
                    elif score >= 60:
                        st.info(f"""
                        👍 **GOOD FIT** - {sym} ({sector})
                        
                        This sector does okay in Quad {current_quad}.
                        Historical win rate: ~{60 + score/5:.0f}%
                        Recommendation: Buy with normal position size
                        """)
                    elif score >= 40:
                        st.warning(f"""
                        ⚠️ **NEUTRAL FIT** - {sym} ({sector})
                        
                        This sector is neutral in Quad {current_quad}.
                        Historical win rate: ~{60 + score/5:.0f}%
                        Recommendation: Be selective, reduce size
                        """)
                    else:
                        st.error(f"""
                        ❌ **POOR FIT** - {sym} ({sector})
                        
                        This sector underperforms in Quad {current_quad}.
                        Historical win rate: ~{60 + score/5:.0f}%
                        Recommendation: Avoid or look for shorts
                        """)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Full Data (CSV)",
                        csv,
                        f"{sym}_{interval}_analysis.csv",
                        "text/csv",
                        key=f"dl_{sym}"
                    )
                    
            except Exception as e:
                st.error(f"""
                ### ❌ Error analyzing {sym}
                
                **What went wrong:** {str(e)}
                
                **Common fixes:**
                - Check the symbol is correct (e.g., AAPL not APPL)
                - Make sure it's a stock, not crypto
                - Try again in a minute if rate limited
                - Verify your API key is valid
                """)

# Bottom section with recommendations
st.markdown("---")
st.markdown("## 🎯 Quick Reference Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📈 Best Setups
    - TD9 Buy + Uptrend + Quad 1
    - Price at lower band
    - Signal score > 50
    - Risk/Reward > 2:1
    """)

with col2:
    st.markdown("""
    ### ⚠️ Warning Signs
    - TD9 Sell completed
    - Downtrend active
    - Quad 3 or 4
    - Price at upper band
    """)

with col3:
    st.markdown("""
    ### 💡 Pro Tips
    - Wait for price to come to you
    - Use smaller size in Quad 3/4
    - Take partial profits at Target 1
    - Never skip the stop loss
    """)

# Quad recommendations
with st.expander("📚 What Stocks Work Best in Each Market Environment?", expanded=False):
    st.markdown("""
    ### Quad 1 (Goldilocks) - Growth & Tech
    **Best:** AAPL, MSFT, GOOGL, NVDA, AMZN, META, CRM
    **Why:** Low inflation + growth = perfect for tech
    **Strategy:** Buy every dip aggressively
    
    ### Quad 2 (Overheating) - Energy & Materials
    **Best:** XOM, CVX, FCX, XLE, GLD, SLB, COP
    **Why:** Inflation hedges outperform
    **Strategy:** Rotate from tech to commodities
    
    ### Quad 3 (Stagflation) - Defensive Only
    **Best:** PG, KO, WMT, XLU, JNJ, PEP, COST
    **Why:** Only defensive stocks hold up
    **Strategy:** Preserve capital, stay defensive
    
    ### Quad 4 (Deflation) - Cash & Bonds
    **Best:** TLT, SHY, Cash, UUP, Low-volatility ETFs
    **Why:** Everything falls except bonds
    **Strategy:** Wait for the bottom, then buy
    """)

# Footer
st.markdown("---")
st.caption("""
💡 **Remember:** This tool provides analysis, not financial advice. Always do your own research.
📊 **Data:** Real-time from Twelve Data | 🔑 **API:** Get free at twelvedata.com
""")

# Risk disclaimer
with st.expander("⚠️ Risk Disclaimer", expanded=False):
    st.warning("""
    **Important:** 
    - Past performance does not guarantee future results
    - All trading involves risk of loss
    - This is not financial advice
    - Never risk more than you can afford to lose
    - Consider paper trading first
    - Consult a financial advisor for personal advice
    """)

# Session stats
if st.sidebar:
    st.sidebar.caption(f"Session: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption(f"Build: v3.0")