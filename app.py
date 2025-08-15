# app.py — The Research Report (Performance Optimized)
"""
The Research Report — Macro + Technical Trading Dashboard
Features: FRAMA, DeMark TD Sequential, Risk Ranges, Volume Analysis
Version: 4.1 — Performance Optimized for Speed
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
import time

# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="The Research Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar and optimize CSS
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none !important; }
.stMetric > div:first-child { font-size: 0.9rem; }
.stMetric > div:nth-child(2) { font-size: 1.5rem; font-weight: bold; }
/* Optimize rendering */
.element-container { transition: none !important; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 The Research Report")
st.caption(f"Technical Trading Dashboard | Build: {datetime.now().strftime('%Y.%m.%d')} | v4.1 Optimized")

# ------------------------- Performance Settings -------------------------
# Reduce default data points for faster loading
DEFAULT_OUTPUTSIZE = 500  # Reduced from 800
DEFAULT_FRAMA_LEN = 20
MIN_DATA_POINTS = 30
CACHE_TTL = 300  # 5 minutes cache

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
    elif len(API_KEY) < 10:
        st.error("❌ Invalid API key format - key seems too short")
        st.stop()

# ------------------------- Sector Map (Reduced) -------------------------
SYMBOL_SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
    'META': 'Technology', 'QQQ': 'Technology', 'AMZN': 'Consumer', 'TSLA': 'Consumer',
    'JPM': 'Financials', 'XLF': 'Financials', 'XOM': 'Energy', 'XLE': 'Energy',
    'SPY': 'Market', 'IWM': 'Market', 'DIA': 'Market'
}

# ------------------------- Optimized Data Fetch -------------------------
@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def fetch_ohlcv(symbol: str, interval: str = "1day", outputsize: int = DEFAULT_OUTPUTSIZE, apikey: str = "") -> pd.DataFrame:
    """Optimized data fetching with better error handling"""
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={quote_plus(symbol)}&interval={interval}&outputsize={outputsize}"
        f"&order=ASC&timezone=America/New_York&apikey={apikey}"
    )
    try:
        r = requests.get(url, timeout=15)  # Reduced timeout
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

        # Vectorized operations for speed
        df = pd.DataFrame(vals)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])

        return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    except requests.Timeout:
        raise RuntimeError(f"Request timeout for {symbol} - try again")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {symbol}: {str(e)}")

# ------------------------- Optimized Indicators -------------------------
@st.cache_data(ttl=CACHE_TTL)
def calculate_frama_vectorized(high: pd.Series, low: pd.Series, close: pd.Series, length: int = DEFAULT_FRAMA_LEN) -> Tuple[pd.Series, pd.Series]:
    """Vectorized FRAMA calculation for better performance"""
    n = len(close)
    if n < length:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    
    # Pre-allocate arrays
    frama = np.full(n, np.nan)
    D = np.full(n, np.nan)
    
    # Vectorized operations where possible
    half = length // 2
    
    # Initialize with first close
    v = close.iloc[0]
    
    for i in range(length - 1, n):
        s = i - length + 1
        
        # Use numpy for faster min/max
        h1 = high[s:s+half].max()
        l1 = low[s:s+half].min()
        h2 = high[s+half:i+1].max()
        l2 = low[s+half:i+1].min()
        h3 = high[s:i+1].max()
        l3 = low[s:i+1].min()
        
        n1 = (h1 - l1) / half
        n2 = (h2 - l2) / half
        n3 = (h3 - l3) / length
        
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            d = np.clip(d, 1.0, 2.0)
            D[i] = d
            alpha = np.exp(-4.6 * (d - 1.0))
            alpha = np.clip(alpha, 0.01, 1.0)
            v = alpha * close.iloc[i] + (1 - alpha) * v
        
        frama[i] = v
    
    return pd.Series(frama, index=close.index), pd.Series(D, index=close.index)

@st.cache_data(ttl=CACHE_TTL)
def calculate_td_sequential_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized TD Sequential calculation"""
    df = df.copy()
    n = len(df)
    
    # Pre-allocate columns
    df['td_setup'] = 0
    df['td_countdown'] = 0
    df['td_nine'] = False
    df['td_thirteen'] = False
    
    # Vectorized close comparison
    close_arr = df['close'].values
    
    # Setup phase - partially vectorized
    for i in range(4, n):
        if close_arr[i] < close_arr[i-4]:
            df.iloc[i, df.columns.get_loc('td_setup')] = 1 if df.iloc[i-1]['td_setup'] > 0 else df.iloc[i-1]['td_setup'] - 1
        elif close_arr[i] > close_arr[i-4]:
            df.iloc[i, df.columns.get_loc('td_setup')] = 1 if df.iloc[i-1]['td_setup'] < 0 else df.iloc[i-1]['td_setup'] + 1
    
    # Vectorized TD9 detection
    df['td_nine'] = df['td_setup'].abs() == 9
    
    # Countdown phase (fixed logic)
    countdown_active = False
    countdown_count = 0
    countdown_direction = 0
    
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    for i in range(2, n):
        if df.iloc[i]['td_nine']:
            countdown_active = True
            countdown_count = 0
            countdown_direction = 1 if df.iloc[i]['td_setup'] == -9 else -1
        
        if countdown_active and i >= 2:
            if countdown_direction == 1 and close_arr[i] >= high_arr[i-2]:
                countdown_count += 1
            elif countdown_direction == -1 and close_arr[i] <= low_arr[i-2]:
                countdown_count += 1
            
            df.iloc[i, df.columns.get_loc('td_countdown')] = countdown_count
            
            if countdown_count >= 13:
                df.iloc[i, df.columns.get_loc('td_thirteen')] = True
                countdown_active = False
    
    return df

# ------------------------- Optimized Signal Engine -------------------------
@st.cache_data(ttl=CACHE_TTL)
def generate_signals_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized signal generation with vectorized operations"""
    df = df.copy()
    
    if len(df) < MIN_DATA_POINTS:
        df['signal_score'] = 0.0
        df['signal_type'] = 'INSUFFICIENT_DATA'
        df['confidence'] = 'Low'
        return df
    
    # Calculate indicators
    df['frama'], df['D'] = calculate_frama_vectorized(df['high'], df['low'], df['close'])
    df['frama_fast'], _ = calculate_frama_vectorized(df['high'], df['low'], df['close'], 10)
    df['frama_slow'], _ = calculate_frama_vectorized(df['high'], df['low'], df['close'], 30)
    
    # Vectorized ATR calculation
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
    
    # Risk ranges
    df['rr_upper'] = df['frama'] + df['atr']
    df['rr_lower'] = df['frama'] - df['atr']
    
    # Volume analysis - vectorized
    df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_spike'] = df['volume'] > (df['volume_ma'] * 1.5)
    
    # TD Sequential
    df = calculate_td_sequential_optimized(df)
    
    # Vectorized price position
    price_range = df['rr_upper'] - df['rr_lower']
    df['price_position'] = ((df['close'] - df['rr_lower']) / price_range.where(price_range != 0, np.nan)).clip(0, 1).fillna(0.5)
    
    # Vectorized scoring where possible
    df['signal_score'] = 0.0
    
    # Trend scoring - vectorized
    trend_score = np.where(
        (df['frama_fast'].notna()) & (df['frama_slow'].notna()) & (df['frama_fast'] > df['frama_slow']),
        25, -25
    )
    df['signal_score'] += trend_score
    
    # TD9 scoring
    td9_buy_score = np.where((df['td_nine']) & (df['td_setup'] == -9), 40, 0)
    td9_sell_score = np.where((df['td_nine']) & (df['td_setup'] == 9), -40, 0)
    df['signal_score'] += td9_buy_score + td9_sell_score
    
    # TD13 scoring
    td13_score = np.where(df['td_thirteen'], 
                          np.where(df['td_countdown'] > 0, 60, -60), 0)
    df['signal_score'] += td13_score
    
    # Price position scoring
    price_score = np.where(df['price_position'] < 0.2, 20,
                           np.where(df['price_position'] > 0.8, -20, 0))
    df['signal_score'] += price_score
    
    # Volume confirmation
    volume_boost = np.where(df['volume_spike'], 
                            np.where(df['signal_score'] > 0, 10, -10), 0)
    df['signal_score'] += volume_boost
    
    # Signal classification - vectorized
    df['signal_type'] = np.where(df['signal_score'] >= 50, 'STRONG BUY',
                        np.where(df['signal_score'] >= 25, 'BUY',
                        np.where(df['signal_score'] <= -50, 'STRONG SELL',
                        np.where(df['signal_score'] <= -25, 'SELL', 'NEUTRAL'))))
    
    # Confidence calculation - simplified for speed
    df['confidence'] = np.where(df['signal_score'].abs() >= 50, 'High',
                       np.where(df['signal_score'].abs() >= 25, 'Medium', 'Low'))
    
    return df

# ------------------------- Optimized Chart Creation -------------------------
def create_chart_optimized(df: pd.DataFrame, sym: str, show_signals: bool, show_volume: bool, signal_threshold: float) -> go.Figure:
    """Create chart with optimized rendering"""
    
    # Downsample data if too many points
    max_points = 200
    if len(df) > max_points:
        # Keep recent data at full resolution, downsample older data
        recent = df.iloc[-100:]
        older = df.iloc[:-100].iloc[::2]  # Take every other point
        df_plot = pd.concat([older, recent])
    else:
        df_plot = df
    
    fig = go.Figure()
    
    # Main candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot['datetime'], 
        open=df_plot['open'], 
        high=df_plot['high'], 
        low=df_plot['low'], 
        close=df_plot['close'],
        name=sym,
        increasing_line_width=1,
        decreasing_line_width=1
    ))
    
    # Lines with reduced points
    fig.add_trace(go.Scatter(
        x=df_plot['datetime'], 
        y=df_plot['rr_upper'],
        mode='lines',
        name='Resistance',
        line=dict(color='rgba(255,0,0,0.3)', width=1),
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['datetime'], 
        y=df_plot['frama'],
        mode='lines',
        name='FRAMA',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['datetime'], 
        y=df_plot['rr_lower'],
        mode='lines',
        name='Support',
        line=dict(color='rgba(0,255,0,0.3)', width=1),
        hoverinfo='skip'
    ))
    
    # Only add signal markers if enabled
    if show_signals:
        # TD9 signals - only recent ones for performance
        recent_df = df.iloc[-100:]  # Last 100 bars only
        
        td9_buys = recent_df[recent_df['td_nine'] & (recent_df['td_setup'] == -9)]
        if not td9_buys.empty:
            fig.add_trace(go.Scatter(
                x=td9_buys['datetime'],
                y=td9_buys['low'] * 0.99,
                mode='markers',
                name='TD9 Buy',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                hovertext=['Buy'] * len(td9_buys)
            ))
        
        td9_sells = recent_df[recent_df['td_nine'] & (recent_df['td_setup'] == 9)]
        if not td9_sells.empty:
            fig.add_trace(go.Scatter(
                x=td9_sells['datetime'],
                y=td9_sells['high'] * 1.01,
                mode='markers',
                name='TD9 Sell',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                hovertext=['Sell'] * len(td9_sells)
            ))
    
    # Optimize layout
    fig.update_layout(
        title=f"{sym} - Technical Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,  # Reduced height for faster rendering
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode='x unified',
        showlegend=False,  # Hide legend for cleaner look
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            fixedrange=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"])  # Hide weekends
            ]
        ),
        yaxis=dict(fixedrange=False)
    )
    
    # Disable some interactions for performance
    fig.update_traces(xaxis="x", yaxis="y")
    
    return fig

# ------------------------- Main UI -------------------------
col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
symbols_text = col1.text_input("📊 Symbols (max 5)", value="AAPL, MSFT, SPY", help="Enter up to 5 symbols for best performance")
interval = col2.selectbox("⏱️ Timeframe", ["1day", "4h", "1h"], index=0)
data_points = col3.selectbox("📈 Data Points", [200, 500, 800], index=0, help="Fewer points = faster loading")
run_btn = col4.button("▶️ Analyze", type="primary", use_container_width=True)

# Quick settings (simplified)
use_advanced = st.checkbox("Show advanced features", value=False)
if use_advanced:
    c1, c2, c3 = st.columns(3)
    with c1:
        show_signals = st.checkbox("Show TD signals", value=True)
    with c2:
        show_volume = st.checkbox("Volume analysis", value=False)
    with c3:
        signal_threshold = st.slider("Signal threshold", 0, 100, 25)
else:
    show_signals = True
    show_volume = False
    signal_threshold = 25

if run_btn:
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()][:5]  # Limit to 5
    
    if not symbols:
        st.error("Please enter at least one symbol")
        st.stop()
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create tabs
    tabs = st.tabs(symbols)
    
    for idx, (sym, tab) in enumerate(zip(symbols, tabs)):
        with tab:
            try:
                # Update progress
                progress = (idx + 0.5) / len(symbols)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {sym}...")
                
                # Fetch and process data
                df = fetch_ohlcv(sym, interval, data_points, API_KEY)
                
                if len(df) < MIN_DATA_POINTS:
                    st.warning(f"⚠️ Limited data: {len(df)} bars")
                    continue
                
                # Generate signals
                df = generate_signals_optimized(df)
                
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                # Simplified metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price_change = latest['close'] - prev['close']
                    price_pct = (price_change / prev['close'] * 100) if prev['close'] else 0
                    st.metric("Price", f"${latest['close']:.2f}", f"{price_pct:+.1f}%")
                
                with col2:
                    trend_up = (latest.get('frama_fast', 0) > latest.get('frama_slow', 0))
                    st.metric("Trend", "📈 UP" if trend_up else "📉 DOWN")
                
                with col3:
                    signal_emoji = {
                        'STRONG BUY': '💚', 'BUY': '🟢', 
                        'STRONG SELL': '🔴', 'SELL': '🟠',
                        'NEUTRAL': '⚪', 'INSUFFICIENT_DATA': '⚠️'
                    }
                    st.metric("Signal", signal_emoji.get(latest['signal_type'], '⚪') + " " + latest['signal_type'].replace('_', ' '))
                
                with col4:
                    st.metric("Score", f"{latest['signal_score']:.0f}")
                
                # Quick action box
                if latest['signal_score'] >= 50:
                    st.success(f"✅ **BUY** - Strong signal ({latest['signal_score']:.0f}/100)")
                elif latest['signal_score'] >= 25:
                    st.info(f"👍 **Consider buying** - Moderate signal")
                elif latest['signal_score'] <= -25:
                    st.error(f"❌ **SELL/AVOID** - Negative signal")
                else:
                    st.warning("⏸️ **WAIT** - No clear signal")
                
                # Trading levels (simplified)
                if pd.notna(latest.get('rr_lower')) and pd.notna(latest.get('rr_upper')):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Entry:** ${latest['rr_lower']:.2f}")
                    with col2:
                        st.info(f"**Target:** ${latest['rr_upper']:.2f}")
                    with col3:
                        stop = latest['rr_lower'] - latest.get('atr', 0) * 0.5
                        st.error(f"**Stop:** ${stop:.2f}")
                
                # Chart
                with st.container():
                    chart = create_chart_optimized(df, sym, show_signals, show_volume, signal_threshold)
                    st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                
                # TD Sequential status (if enabled)
                if use_advanced and 'td_setup' in latest:
                    td_setup = int(latest.get('td_setup', 0))
                    if abs(td_setup) >= 7:
                        st.info(f"📊 TD Sequential: {abs(td_setup)}/9 - Signal soon!")
                    if latest.get('td_nine', False):
                        st.warning("⚡ TD9 Signal Complete!")
                
                # Update progress
                progress = (idx + 1) / len(symbols)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Error with {sym}: {str(e)}")
                continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

# ------------------------- Simplified Footer -------------------------
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🟢 Buy Signals:** Score > 25")
with c2:
    st.markdown("**⚪ Neutral:** Score -25 to 25")
with c3:
    st.markdown("**🔴 Sell Signals:** Score < -25")

st.caption(f"v4.1 Performance Optimized • Data: Twelve Data • {datetime.now().strftime('%H:%M:%S')}")

# Cache management
if st.button("🔄 Clear Cache", help="Click if data seems stale"):
    st.cache_data.clear()
    st.success("Cache cleared! Refresh to reload.")

