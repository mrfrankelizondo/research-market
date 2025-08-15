# app.py — The Research Report (FRAMA + DeMark, Enhanced)
"""
The Research Report — Macro + Technical Trading Dashboard
Features: FRAMA, DeMark TD Sequential, Risk Ranges, Volume Analysis
Version: 4.0 — Enhanced with all fixes and improvements
"""

from __future__ import annotations

import os, math, requests
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote_plus
from datetime import datetime

# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="The Research Report",
    page_icon="📊",
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

st.title("📊 The Research Report")
st.caption(f"Technical Trading Dashboard | Build: {datetime.now().strftime('%Y.%m.%d')} | v4.0 Enhanced")

# ------------------------- API Key with Validation -------------------------
try:
    _secret = st.secrets.get("TWELVE_DATA_API_KEY", None)
except Exception:
    _secret = None
API_KEY = os.getenv("TWELVE_DATA_API_KEY") or _secret
if not API_KEY:
    st.error("⚠️ No API key detected! Add TWELVE_DATA_API_KEY to Streamlit secrets or environment.")
    st.info("Get a free API key at: https://twelvedata.com (800 calls/day free)")
    API_KEY = st.text_input("Or enter your API key here:", type="password")
    if not API_KEY:
        st.stop()
    # Validate API key format
    elif len(API_KEY) < 10:
        st.error("❌ Invalid API key format - key seems too short")
        st.stop()

# ------------------------- Constants & Sector Map -------------------------
DEFAULT_OUTPUTSIZE = 800
DEFAULT_FRAMA_LEN = 20
MIN_DATA_POINTS = 30  # Minimum bars needed for reliable signals

SYMBOL_SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
    'META': 'Technology', 'CRM': 'Technology', 'QQQ': 'Technology', 'XLK': 'Technology',
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
    'JPM': 'Financials', 'BAC': 'Financials', 'XLF': 'Financials', 'GS': 'Financials',
    'XOM': 'Energy', 'CVX': 'Energy', 'XLE': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'CVS': 'Healthcare', 'XLV': 'Healthcare',
    'XLI': 'Industrials', 'XLU': 'Utilities',
    'FCX': 'Materials',
    'GLD': 'Gold',
    'TLT': 'Bonds', 'SHY': 'Bonds',
    'UUP': 'Cash/Dollar',
    'SPY': 'Market'
}

# ------------------------- Data Fetch -------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_ohlcv(symbol: str, interval: str = "1day", outputsize: int = DEFAULT_OUTPUTSIZE, apikey: str = "") -> "pd.DataFrame":
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={quote_plus(symbol)}&interval={interval}&outputsize={outputsize}"
        f"&order=ASC&timezone=America/New_York&apikey={apikey}"
    )
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

# ------------------------- Indicators -------------------------
def calculate_frama(high: pd.Series, low: pd.Series, close: pd.Series, length: int = DEFAULT_FRAMA_LEN) -> "Tuple[pd.Series, pd.Series]":
    n = len(close)
    frama = np.full(n, np.nan)
    D = np.full(n, np.nan)
    if n == 0:
        return pd.Series(frama, index=close.index), pd.Series(D, index=close.index)
    v = float(close.iloc[0])
    for i in range(n):
        if i >= length - 1:
            half = length // 2
            s = i - length + 1
            h1 = np.max(high[s:s+half]); l1 = np.min(low[s:s+half])
            h2 = np.max(high[s+half:i+1]); l2 = np.min(low[s+half:i+1])
            h3 = np.max(high[s:i+1]);     l3 = np.min(low[s:i+1])
            n1 = (h1 - l1) / max(half, 1)
            n2 = (h2 - l2) / max(half, 1)
            n3 = (h3 - l3) / max(length, 1)
            if n1 > 0 and n2 > 0 and n3 > 0:
                d = (math.log(n1 + n2) - math.log(n3)) / math.log(2.0)
                d = np.clip(d, 1.0, 2.0); D[i] = d
                alpha = np.clip(math.exp(-4.6 * (d - 1.0)), 0.01, 1.0)
                v = alpha * float(close.iloc[i]) + (1 - alpha) * v
        frama[i] = v
    return pd.Series(frama, index=close.index), pd.Series(D, index=close.index)

def calculate_td_sequential(df: "pd.DataFrame") -> "pd.DataFrame":
    """Fixed TD Sequential with correct countdown logic"""
    df = df.copy()
    n = len(df)
    df['td_setup'] = 0
    df['td_countdown'] = 0
    df['td_nine'] = False
    df['td_thirteen'] = False
    
    # Setup phase
    for i in range(4, n):
        if df['close'].iloc[i] < df['close'].iloc[i-4]:
            df.loc[i, 'td_setup'] = 1 if df['td_setup'].iloc[i-1] > 0 else df['td_setup'].iloc[i-1] - 1
        elif df['close'].iloc[i] > df['close'].iloc[i-4]:
            df.loc[i, 'td_setup'] = 1 if df['td_setup'].iloc[i-1] < 0 else df['td_setup'].iloc[i-1] + 1
        else:
            df.loc[i, 'td_setup'] = 0
    
    df['td_nine'] = (df['td_setup'].abs() == 9)
    
    # Countdown phase (FIXED LOGIC)
    countdown_active = False
    countdown_count = 0
    countdown_direction = 0
    
    for i in range(2, n):
        if df['td_nine'].iloc[i]:
            countdown_active = True
            countdown_count = 0
            # Fixed: After buy setup (-9), look for bullish confirmation
            # After sell setup (+9), look for bearish confirmation
            if df['td_setup'].iloc[i] == -9:
                countdown_direction = 1  # Buy setup, look for bullish bars
            else:  # td_setup == 9
                countdown_direction = -1  # Sell setup, look for bearish bars
        
        if countdown_active and i >= 2:
            # Fixed countdown logic
            if countdown_direction == 1 and df['close'].iloc[i] >= df['high'].iloc[i-2]:
                # Bullish confirmation after buy setup
                countdown_count += 1
            elif countdown_direction == -1 and df['close'].iloc[i] <= df['low'].iloc[i-2]:
                # Bearish confirmation after sell setup
                countdown_count += 1
            
            df.loc[i, 'td_countdown'] = countdown_count
            
            if countdown_count >= 13:
                df.loc[i, 'td_thirteen'] = True
                countdown_active = False
    
    return df

# ------------------------- Enhanced Signal Engine with Volume -------------------------
@st.cache_data(ttl=60)  # Cache for 1 minute for performance
def calculate_all_signals(df: "pd.DataFrame", symbol: str) -> "pd.DataFrame":
    """Cached signal calculation for performance"""
    return generate_signals(df)

def generate_signals(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.copy()
    
    # Check minimum data requirement
    if len(df) < MIN_DATA_POINTS:
        df['signal_score'] = 0.0
        df['signal_type'] = 'INSUFFICIENT_DATA'
        df['confidence'] = 'Low'
        return df
    
    # FRAMA variants
    df['frama'], df['D'] = calculate_frama(df['high'], df['low'], df['close'])
    df['frama_fast'], _ = calculate_frama(df['high'], df['low'], df['close'], 10)
    df['frama_slow'], _ = calculate_frama(df['high'], df['low'], df['close'], 30)

    # ATR (EWMA of True Range)
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Risk ranges
    df['rr_upper'] = df['frama'] + df['atr']
    df['rr_lower'] = df['frama'] - df['atr']

    # Volume Analysis (NEW)
    df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_spike'] = df['volume'] > (df['volume_ma'] * 1.5)
    
    # TD Sequential
    df = calculate_td_sequential(df)

    # Safe price position in range [0,1]
    denom = (df['rr_upper'] - df['rr_lower']).replace(0, np.nan)
    price_pos_series = ((df['close'] - df['rr_lower']) / denom).clip(0, 1).fillna(0.5)

    # Enhanced Scoring with Volume and Confidence
    df['signal_score'] = 0.0
    df['signal_type'] = 'NEUTRAL'
    df['confidence'] = 'Low'

    for i in range(len(df)):
        score = 0.0
        confidence = 0

        # Trend via FRAMA cross
        ff = df['frama_fast'].iloc[i]
        fs = df['frama_slow'].iloc[i]
        trend_up = pd.notna(ff) and pd.notna(fs) and (ff > fs)
        
        score += 25 if trend_up else -25
        if abs(ff - fs) / fs > 0.02 if pd.notna(fs) and fs != 0 else False:  # Strong trend
            confidence += 1

        # TD signals
        if bool(df['td_nine'].iloc[i]):
            if df['td_setup'].iloc[i] == -9:
                score += 40
                confidence += 1
            elif df['td_setup'].iloc[i] == 9:
                score -= 40
                confidence += 1

        if bool(df['td_thirteen'].iloc[i]):
            score += 60 if df['td_countdown'].iloc[i] > 0 else -60
            confidence += 2  # TD13 is high confidence

        # Price location in range
        ppos = float(price_pos_series.iloc[i])
        if ppos < 0.2:
            score += 20
            if trend_up: confidence += 1
        elif ppos > 0.8:
            score -= 20
            if not trend_up: confidence += 1

        # Volume confirmation (NEW)
        if df['volume_spike'].iloc[i]:
            if score > 0:
                score += 10  # Volume confirms bullish signal
                confidence += 1
            elif score < 0:
                score -= 10  # Volume confirms bearish signal
                confidence += 1

        # Signal classification
        if score >= 50:
            sig = 'STRONG BUY'
        elif score >= 25:
            sig = 'BUY'
        elif score <= -50:
            sig = 'STRONG SELL'
        elif score <= -25:
            sig = 'SELL'
        else:
            sig = 'NEUTRAL'

        # Confidence level
        conf_level = ['Low', 'Medium', 'High', 'Very High'][min(confidence, 3)]

        df.loc[i, 'signal_type'] = sig
        df.loc[i, 'signal_score'] = score
        df.loc[i, 'confidence'] = conf_level

    return df

# ------------------------- Main UI -------------------------
col1, col2, col3 = st.columns([3, 1.5, 1.5])
symbols_text = col1.text_input("📊 Symbols", value="AAPL, MSFT, SPY", help="Enter stock symbols separated by commas")
interval = col2.selectbox("⏱️ Timeframe", ["1day", "4h", "1h"], index=0, help="Daily for investing, hourly for trading")
run_btn = col3.button("▶️ Analyze", type="primary", use_container_width=True)

with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        use_demark = st.checkbox("Enable TD Sequential", value=True, help="Shows exhaustion points")
        show_signals = st.checkbox("Show signals on chart", value=True)
        show_volume = st.checkbox("Show volume analysis", value=True, help="Volume confirmation")
    with c2:
        signal_threshold = st.slider("Signal sensitivity", 0, 100, 25, help="Lower = more signals")
        risk_percent = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)
        max_position_pct = st.slider("Max position size (% of account)", 10, 50, 25, 5)

if run_btn:
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    if not symbols:
        st.error("Please enter at least one symbol")
        st.stop()

    tabs = st.tabs(symbols)
    for sym, tab in zip(symbols, tabs):
        with tab:
            try:
                with st.spinner(f"Analyzing {sym}..."):
                    df = fetch_ohlcv(sym, interval, DEFAULT_OUTPUTSIZE, API_KEY)
                    
                    # Data validation
                    if len(df) < MIN_DATA_POINTS:
                        st.warning(f"⚠️ Insufficient data: {len(df)} bars (need {MIN_DATA_POINTS}+). Some features may be limited.")
                    
                    df = calculate_all_signals(df, sym)

                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest

                    # Header with price
                    st.markdown(f"## {sym} Analysis")

                    price_change = latest['close'] - prev['close']
                    price_pct = (price_change / prev['close']) * 100 if prev['close'] else 0.0

                    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.5])
                    with c1:
                        st.metric("Current Price", f"${latest['close']:.2f}",
                                  f"{price_change:+.2f} ({price_pct:+.1f}%)")
                    with c2:
                        trend_up = (pd.notna(latest['frama_fast']) and pd.notna(latest['frama_slow']) 
                                   and latest['frama_fast'] > latest['frama_slow'])
                        st.metric("Market Trend", "📈 Uptrend" if trend_up else "📉 Downtrend")
                        st.caption("Bullish momentum" if trend_up else "Bearish momentum")
                    with c3:
                        signal_map = {
                            'STRONG BUY': ("💚 Strong Buy", "Excellent setup - buy now"),
                            'BUY': ("🟢 Buy", "Good setup - consider buying"),
                            'STRONG SELL': ("🔴 Strong Sell", "Exit immediately"),
                            'SELL': ("🟠 Sell", "Consider reducing position"),
                            'NEUTRAL': ("⚪ Wait", "No clear opportunity"),
                            'INSUFFICIENT_DATA': ("⚠️ Limited", "Need more data")
                        }
                        signal_display, signal_desc = signal_map.get(latest['signal_type'], ("⚪ Wait", "No signal"))
                        st.metric("Signal", signal_display)
                        st.caption(signal_desc)
                    with c4:
                        # Confidence indicator (NEW)
                        conf_color = {'Low': '🔴', 'Medium': '🟡', 'High': '🟢', 'Very High': '💚'}
                        st.metric("Confidence", conf_color.get(latest['confidence'], '🔴') + " " + latest['confidence'])

                    # Volume Analysis Section (NEW)
                    if show_volume and 'volume_spike' in df.columns:
                        if latest['volume_spike']:
                            st.info(f"📊 **Volume Alert:** Current volume is {(latest['volume']/latest['volume_ma']):.1f}x average - Strong confirmation!")

                    # Short-history guard for trading levels
                    critical_cols = ['rr_lower', 'rr_upper', 'frama', 'atr']
                    has_levels = not pd.isna(latest[critical_cols]).any()

                    # TD Sequential Status with plain English
                    if use_demark:
                        st.markdown("### 📊 Market Exhaustion Indicators (TD Sequential)")
                        c1, c2, c3 = st.columns(3)

                        td_setup = int(latest['td_setup'])
                        td_countdown = int(latest['td_countdown'])

                        with c1:
                            if td_setup == -9:
                                st.metric("Exhaustion Level", "🟢 BOTTOM SIGNAL")
                                st.caption("Selling exhausted - buy opportunity")
                            elif td_setup == 9:
                                st.metric("Exhaustion Level", "🔴 TOP SIGNAL")
                                st.caption("Buying exhausted - sell warning")
                            elif td_setup <= -7:
                                st.metric("Exhaustion Level", f"📉 Oversold soon ({abs(td_setup)}/9)")
                                st.caption("Bottom forming")
                            elif td_setup >= 7:
                                st.metric("Exhaustion Level", f"📈 Overbought soon ({td_setup}/9)")
                                st.caption("Top forming")
                            elif td_setup <= -4:
                                st.metric("Exhaustion Level", f"📉 Selling pressure ({abs(td_setup)}/9)")
                                st.caption("Downtrend active")
                            elif td_setup >= 4:
                                st.metric("Exhaustion Level", f"📈 Buying pressure ({td_setup}/9)")
                                st.caption("Uptrend active")
                            else:
                                st.metric("Exhaustion Level", "➡️ No pattern")
                                st.caption("Waiting for setup")

                        with c2:
                            if bool(latest['td_thirteen']):
                                st.metric("Reversal Countdown", "⚡ MAJOR REVERSAL")
                                st.caption("Strong reversal NOW")
                            elif td_countdown >= 10:
                                st.metric("Reversal Countdown", f"⚠️ Reversal soon ({td_countdown}/13)")
                                st.caption("Major turn imminent")
                            elif td_countdown >= 7:
                                st.metric("Reversal Countdown", f"📊 Building ({td_countdown}/13)")
                                st.caption("Pressure increasing")
                            elif td_countdown >= 1:
                                st.metric("Reversal Countdown", f"⏳ Counting ({td_countdown}/13)")
                                st.caption("Tracking reversal")
                            else:
                                st.metric("Reversal Countdown", "⏸️ Inactive")
                                st.caption("No countdown yet")

                        with c3:
                            if bool(latest['td_nine']) and td_setup == -9:
                                st.success("✅ **BUY SETUP COMPLETE**")
                                st.caption("Look for entry point")
                            elif bool(latest['td_nine']) and td_setup == 9:
                                st.error("❌ **SELL SETUP COMPLETE**")
                                st.caption("Take profits/exit")
                            elif bool(latest['td_thirteen']):
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

                    price_position_value = 0.5
                    if has_levels:
                        denom = (latest['rr_upper'] - latest['rr_lower'])
                        price_position_value = ((latest['close'] - latest['rr_lower']) / denom) if denom else 0.5
                        if not (0 <= price_position_value <= 1) or pd.isna(price_position_value):
                            price_position_value = 0.5

                    st.info("📈 **Market State: UPTREND** - Look for buying opportunities" if trend_up
                            else "📉 **Market State: DOWNTREND** - Avoid buying, consider selling")

                    st.info("🟢 **Price Location: EXCELLENT** - Near support, good entry zone" if price_position_value < 0.3 else
                            "🟡 **Price Location: DECENT** - Middle of range, okay entry" if price_position_value < 0.5 else
                            "🟠 **Price Location: POOR** - Above middle, wait for pullback" if price_position_value < 0.7 else
                            "🔴 **Price Location: OVERBOUGHT** - Near resistance, don't buy")

                    if not has_levels:
                        st.warning(f"Not enough data to compute targets/stop reliably yet (need {MIN_DATA_POINTS}+ bars).")
                    else:
                        c1, c2, c3, c4 = st.columns(4)
                        stop_loss = latest['rr_lower'] - latest['atr'] * 0.5
                        risk_amt = abs(latest['rr_lower'] - stop_loss)
                        reward1 = abs(latest['frama'] - latest['rr_lower'])
                        reward2 = abs(latest['rr_upper'] - latest['rr_lower'])
                        rr1 = (reward1 / risk_amt) if risk_amt > 0 else 0.0
                        rr2 = (reward2 / risk_amt) if risk_amt > 0 else 0.0

                        with c1:
                            st.success(f"**BUY at:** ${latest['rr_lower']:.2f}")
                            st.caption("Wait for this price")
                        with c2:
                            st.info(f"**Target 1:** ${latest['frama']:.2f}")
                            st.caption(f"Sell half ({rr1:.1f}:1)")
                        with c3:
                            st.info(f"**Target 2:** ${latest['rr_upper']:.2f}")
                            st.caption(f"Sell rest ({rr2:.1f}:1)")
                        with c4:
                            st.error(f"**STOP at:** ${stop_loss:.2f}")
                            risk_pct = abs((stop_loss - latest['rr_lower']) / latest['rr_lower'] * 100) if latest['rr_lower'] else 0.0
                            st.caption(f"Risk: {risk_pct:.1f}%")

                    # Plain English Action Box
                    st.markdown("### 📋 What Should I Do?")
                    if has_levels:
                        risk_pct = abs((stop_loss - latest['rr_lower']) / latest['rr_lower'] * 100) if latest['rr_lower'] else 0.0
                        if latest['signal_score'] >= 50:
                            conf_text = f" (Confidence: {latest['confidence']})"
                            st.success(f"""
                            ### ✅ **ACTION: BUY THIS STOCK** {conf_text}
                            1. Place a limit buy at **${latest['rr_lower']:.2f}**
                            2. Stop loss **${stop_loss:.2f}** (risk: {risk_pct:.1f}%)
                            3. Take half at **${latest['frama']:.2f}**
                            4. Exit remainder at **${latest['rr_upper']:.2f}**
                            5. Risk only **{risk_percent}%** of your account
                            """)
                        elif latest['signal_score'] >= 25:
                            st.info(f"""
                            ### 👍 **ACTION: CONSIDER BUYING** (Confidence: {latest['confidence']})
                            - Prefer an entry near **${latest['rr_lower']:.2f}**
                            - Use a smaller position (half size)
                            - Stop **${stop_loss:.2f}**
                            - Signal strength: {latest['signal_score']:.0f}/100
                            """)
                        elif latest['signal_score'] <= -25:
                            st.error(f"""
                            ### ❌ **ACTION: DO NOT BUY (Consider Selling)**
                            - Bearish signal ({latest['signal_score']:.0f}/100)
                            - If long, consider trimming or placing a stop
                            - Confidence: {latest['confidence']}
                            """)
                        else:
                            st.warning("### ⏸️ **ACTION: WAIT**\nNo clear edge right now. Low confidence setup.")
                    else:
                        st.info("⚙️ Collecting more data. Signals and levels will populate once enough history is available.")

                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=sym
                    ))
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['rr_upper'], mode='lines',
                                             name='Sell Zone (Resistance)', line=dict(color='rgba(255,0,0,0.3)', width=1)))
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['frama'], mode='lines',
                                             name='Fair Value (FRAMA)', line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['rr_lower'], mode='lines',
                                             name='Buy Zone (Support)', line=dict(color='rgba(0,255,0,0.3)', width=1)))

                    if use_demark and show_signals:
                        td9_buys = df[df['td_nine'] & (df['td_setup'] == -9)]
                        if not td9_buys.empty:
                            fig.add_trace(go.Scatter(
                                x=td9_buys['datetime'], y=td9_buys['low'] * 0.99,
                                mode='markers+text', name='Buy Signal (TD9)',
                                marker=dict(color='green', size=12, symbol='triangle-up'),
                                text=['BUY'] * len(td9_buys), textposition='bottom center'
                            ))
                        td9_sells = df[df['td_nine'] & (df['td_setup'] == 9)]
                        if not td9_sells.empty:
                            fig.add_trace(go.Scatter(
                                x=td9_sells['datetime'], y=td9_sells['high'] * 1.01,
                                mode='markers+text', name='Sell Signal (TD9)',
                                marker=dict(color='red', size=12, symbol='triangle-down'),
                                text=['SELL'] * len(td9_sells), textposition='top center'
                            ))
                        td13s = df[df['td_thirteen']]
                        if not td13s.empty:
                            fig.add_trace(go.Scatter(
                                x=td13s['datetime'], y=td13s['close'],
                                mode='markers', name='Major Reversal (TD13)',
                                marker=dict(color='yellow', size=15, symbol='star',
                                            line=dict(color='black', width=2))
                            ))

                    # Volume spike markers (NEW)
                    if show_volume and 'volume_spike' in df.columns:
                        vol_spikes = df[df['volume_spike']]
                        if not vol_spikes.empty:
                            fig.add_trace(go.Scatter(
                                x=vol_spikes['datetime'], y=vol_spikes['low'] * 0.98,
                                mode='markers', name='Volume Spike',
                                marker=dict(color='purple', size=8, symbol='diamond'),
                                hovertext=['Vol Spike'] * len(vol_spikes)
                            ))

                    if show_signals:
                        for i in range(len(df)):
                            sc = df['signal_score'].iloc[i]
                            if pd.notna(sc) and (abs(sc) >= signal_threshold):
                                color = 'rgba(0,255,0,0.05)' if sc > 0 else 'rgba(255,0,0,0.05)'
                                fig.add_vrect(
                                    x0=df['datetime'].iloc[i],
                                    x1=df['datetime'].iloc[min(i+1, len(df)-1)],
                                    fillcolor=color, layer="below", line_width=0
                                )

                    fig.update_layout(
                        title=f"The Research Report — {sym}",
                        xaxis_title="Date", yaxis_title="Price ($)",
                        height=600, xaxis_rangeslider_visible=False,
                        template="plotly_white", hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Plain English Summary with Volume info
                    st.markdown("### 📝 Plain English Summary")
                    parts = []
                    parts.append(f"{sym} is in an **uptrend** (bullish)" if trend_up else f"{sym} is in a **downtrend** (bearish)")
                    
                    if use_demark:
                        if td_setup == -9: parts.append("Selling looks exhausted (potential bottom)")
                        elif td_setup == 9: parts.append("Buying looks exhausted (potential top)")
                        elif abs(td_setup) >= 7: parts.append(f"A reversal setup may complete in {9-abs(td_setup)} bars")
                    
                    if price_position_value < 0.3: parts.append("Price is near support (good for buying)")
                    elif price_position_value > 0.7: parts.append("Price is near resistance (good for selling)")
                    else: parts.append("Price is in the middle of its range")
                    
                    # Add volume info
                    if latest['volume_spike']:
                        parts.append("**Volume is confirming the move** (high activity)")
                    
                    sc = latest['signal_score']
                    parts.append("**Strong buy signal**" if sc >= 50 else
                                 "Moderate buy signal" if sc >= 25 else
                                 "**Strong sell signal**" if sc <= -50 else
                                 "Moderate sell signal" if sc <= -25 else
                                 "No clear signal - better to wait")
                    
                    parts.append(f"Signal confidence: **{latest['confidence']}**")
                    
                    sector = SYMBOL_SECTORS.get(sym, 'Unknown')
                    parts.append(f"Sector: {sector}")
                    st.info(". ".join(parts) + ".")

                    # Enhanced Position Calculator with Max Position Check
                    with st.expander("💰 Position Size Calculator", expanded=False):
                        account = st.number_input("Your Account Size ($)", value=10000, step=100, key=f"acc_{sym}",
                                                  help="Total value of your trading account")
                        if has_levels:
                            entry = latest['rr_lower']
                            stop = entry - latest['atr'] * 0.5
                            risk_amt = account * (risk_percent / 100)
                            per_share_risk = abs(entry - stop)
                            shares = int(risk_amt / per_share_risk) if per_share_risk > 0 else 0
                            position_value = shares * entry
                            
                            # Max position size check (NEW)
                            max_position_value = account * (max_position_pct / 100)
                            if position_value > max_position_value:
                                shares = int(max_position_value / entry)
                                position_value = shares * entry
                                st.warning(f"⚠️ Position capped at {max_position_pct}% of account for safety")
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Shares to Buy", f"{shares:,}")
                            c2.metric("Total Cost", f"${position_value:,.2f}")
                            c3.metric("Risk Amount", f"${risk_amt:.2f}")
                            c4.metric("Potential Profit", f"${shares * (latest['rr_upper'] - entry):.2f}")
                            
                            # Risk/Reward display
                            rr_ratio = (latest['rr_upper'] - entry) / per_share_risk if per_share_risk > 0 else 0
                            if rr_ratio >= 2:
                                st.success(f"✅ Risk/Reward: {rr_ratio:.1f}:1 - Great trade!")
                            elif rr_ratio >= 1.5:
                                st.info(f"👍 Risk/Reward: {rr_ratio:.1f}:1 - Acceptable trade")
                            else:
                                st.warning(f"⚠️ Risk/Reward: {rr_ratio:.1f}:1 - Poor trade setup")
                            
                            st.info(f"""
                            **Trade Summary:**
                            - Buy {shares:,} shares @ ${entry:.2f}
                            - Stop loss @ ${stop:.2f}
                            - Maximum loss: ${risk_amt:.2f} ({risk_percent}% of account)
                            - Position size: {(position_value/account*100):.1f}% of account
                            """)
                        else:
                            st.caption("Need more history to compute entry/stop/targets.")

                    # Data Quality Indicator (NEW)
                    with st.expander("📊 Data Quality & Reliability", expanded=False):
                        data_points = len(df)
                        quality_score = min(100, (data_points / 200) * 100)
                        
                        st.progress(quality_score / 100)
                        st.caption(f"Data Quality: {quality_score:.0f}% ({data_points} bars)")
                        
                        if data_points < 30:
                            st.error("❌ Insufficient data - signals unreliable")
                        elif data_points < 100:
                            st.warning("⚠️ Limited data - use caution")
                        elif data_points < 200:
                            st.info("✅ Good data - signals reliable")
                        else:
                            st.success("💚 Excellent data - high confidence")
                        
                        # Show confidence factors
                        st.markdown("**Confidence Factors:**")
                        checks = {
                            "Enough historical data": data_points >= MIN_DATA_POINTS,
                            "FRAMA calculated": pd.notna(latest['frama']),
                            "TD Sequential active": abs(latest['td_setup']) > 0,
                            "Volume data available": pd.notna(latest['volume_ma']),
                            "Risk levels computed": has_levels
                        }
                        for check, passed in checks.items():
                            st.write(f"{'✅' if passed else '❌'} {check}")

                    # Download with enhanced data
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Full Data (CSV)",
                        csv,
                        f"{sym}_{interval}_enhanced_analysis.csv",
                        "text/csv",
                        key=f"dl_{sym}"
                    )

            except Exception as e:
                st.error(f"""
                ### ❌ Error analyzing {sym}

                **What went wrong:** {str(e)}

                **Common fixes:**
                - Check the symbol (e.g., AAPL not APPL)
                - Ensure it's a supported stock/ETF
                - Try again if rate limited (wait 60 seconds)
                - Verify your API key is valid
                - Check if you have enough API calls remaining
                """)
                
                # Debug info
                with st.expander("🔧 Debug Information"):
                    st.code(f"""
                    Symbol: {sym}
                    Interval: {interval}
                    API Key Length: {len(API_KEY) if API_KEY else 0}
                    Error Type: {type(e).__name__}
                    Error Details: {str(e)}
                    """)

# ------------------------- Reference & Footer -------------------------
st.markdown("---")
st.markdown("## 🎯 Quick Reference Guide")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""### 📈 Best Setups
- TD9 Buy + Uptrend
- Price at lower band
- Signal score > 50
- High confidence level
- Volume confirmation
- Risk/Reward > 2:1""")
with c2:
    st.markdown("""### ⚠️ Warning Signs
- TD9 Sell completed
- Strong downtrend
- Price at upper band
- Low confidence signals
- No volume support
- Poor risk/reward (<1.5:1)""")
with c3:
    st.markdown("""### 💡 Pro Tips
- Wait for high confidence
- Check volume spikes
- Use position limits
- Scale in/out of trades
- Always use stops
- Review data quality""")

st.markdown("---")

# Enhanced footer with version info
col1, col2 = st.columns(2)
with col1:
    st.caption(f"""
    📊 **The Research Report v4.0 Enhanced**
    Session: {datetime.now().strftime('%H:%M:%S')} • Build: {datetime.now().strftime('%Y.%m.%d')}
    Data: Twelve Data • API: twelvedata.com
    """)
with col2:
    st.caption("""
    **What's New in v4.0:**
    ✅ Fixed TD Sequential countdown logic
    ✅ Added volume confirmation signals
    ✅ Signal confidence indicators
    ✅ Position size safety limits
    ✅ Data quality metrics
    """)

with st.expander("⚠️ Risk Disclaimer", expanded=False):
    st.warning("""
    **IMPORTANT RISK DISCLOSURE:**
    - Past performance does not guarantee future results
    - All trading involves substantial risk of loss
    - This tool provides analysis, NOT financial advice
    - Never risk more than you can afford to lose
    - Consider paper trading before using real money
    - Consult a financial advisor for personalized advice
    - The developers assume no liability for trading losses
    
    **By using this tool, you acknowledge:**
    - You understand the risks involved in trading
    - You are solely responsible for your trading decisions
    - Technical indicators can and do fail
    - No trading system is perfect or guaranteed
    """)

# Performance tips
with st.expander("🚀 Performance Tips", expanded=False):
    st.info("""
    **To improve app performance:**
    - Analyze fewer symbols at once (3-5 max)
    - Use daily timeframe for slower updates
    - Clear cache if data seems stale (refresh page)
    - Reduce outputsize if hitting rate limits
    - Consider upgrading API plan for more calls
    
    **Optimal Usage:**
    - Run analysis pre-market or after hours
    - Save CSV exports for offline analysis
    - Focus on high-confidence signals only
    - Combine with fundamental analysis
    """)

