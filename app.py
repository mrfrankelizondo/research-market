# app.py — The Research Report Pro (Advanced Features)
"""
The Research Report Pro — Advanced Trading Dashboard
Features: Market Context, Backtesting, Smart Alerts, Relative Strength, 
         Sector Rotation, Multi-Timeframe Analysis, Correlation Matrix
Version: 5.0 Pro — Full-Featured Trading System
"""

from __future__ import annotations

import os, math, requests
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import time

# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="The Research Report Pro",
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
.element-container { transition: none !important; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Title with Analyze Button -------------------------
col_title, col_button = st.columns([10, 2])
with col_title:
    st.title("📊 The Research Report Pro")
    st.caption(f"Advanced Trading System | v5.0 | {datetime.now().strftime('%Y.%m.%d %H:%M')}")
with col_button:
    st.write("")  # Spacer
    run_btn = st.button("▶️ ANALYZE", type="primary", use_container_width=True, key="main_analyze")

# ------------------------- Performance Settings -------------------------
DEFAULT_OUTPUTSIZE = 500
DEFAULT_FRAMA_LEN = 20
MIN_DATA_POINTS = 30
CACHE_TTL = 300

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
        st.error("❌ Invalid API key format")
        st.stop()

# ------------------------- Sector ETF Mapping -------------------------
SECTOR_ETFS = {
    'XLK': ('Technology', '💻'),
    'XLF': ('Financials', '🏦'),
    'XLE': ('Energy', '⚡'),
    'XLV': ('Healthcare', '🏥'),
    'XLI': ('Industrials', '🏭'),
    'XLP': ('Staples', '🛒'),
    'XLY': ('Discretionary', '🛍️'),
    'XLRE': ('Real Estate', '🏢'),
    'XLB': ('Materials', '⚒️'),
    'XLU': ('Utilities', '💡')
}

# ------------------------- Data Fetch -------------------------
@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def fetch_ohlcv(symbol: str, interval: str = "1day", outputsize: int = DEFAULT_OUTPUTSIZE, apikey: str = "") -> pd.DataFrame:
    """Optimized data fetching with better error handling"""
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={quote_plus(symbol)}&interval={interval}&outputsize={outputsize}"
        f"&order=ASC&timezone=America/New_York&apikey={apikey}"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 429:
            raise RuntimeError("Rate limit hit. Wait 1 minute.")
        if r.status_code != 200:
            raise RuntimeError(f"API Error {r.status_code}")

        j = r.json()
        if j.get("status") == "error":
            raise RuntimeError(j.get("message", "Unknown error"))

        vals = j.get("values", [])
        if not vals:
            raise RuntimeError("No data returned")

        df = pd.DataFrame(vals)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])

        return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    except requests.Timeout:
        raise RuntimeError(f"Request timeout for {symbol}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {symbol}: {str(e)}")

# ------------------------- Indicators -------------------------
@st.cache_data(ttl=CACHE_TTL)
def calculate_frama_vectorized(high: pd.Series, low: pd.Series, close: pd.Series, length: int = DEFAULT_FRAMA_LEN) -> Tuple[pd.Series, pd.Series]:
    """Vectorized FRAMA calculation"""
    n = len(close)
    if n < length:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    
    frama = np.full(n, np.nan)
    D = np.full(n, np.nan)
    
    half = length // 2
    v = close.iloc[0]
    
    for i in range(length - 1, n):
        s = i - length + 1
        
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
    """Fixed TD Sequential calculation"""
    df = df.copy()
    n = len(df)
    
    df['td_setup'] = 0
    df['td_countdown'] = 0
    df['td_nine'] = False
    df['td_thirteen'] = False
    
    close_arr = df['close'].values
    
    for i in range(4, n):
        if close_arr[i] < close_arr[i-4]:
            df.iloc[i, df.columns.get_loc('td_setup')] = 1 if df.iloc[i-1]['td_setup'] > 0 else df.iloc[i-1]['td_setup'] - 1
        elif close_arr[i] > close_arr[i-4]:
            df.iloc[i, df.columns.get_loc('td_setup')] = 1 if df.iloc[i-1]['td_setup'] < 0 else df.iloc[i-1]['td_setup'] + 1
    
    df['td_nine'] = df['td_setup'].abs() == 9
    
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

# ------------------------- Signal Generation -------------------------
@st.cache_data(ttl=CACHE_TTL)
def generate_signals_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized signal generation"""
    df = df.copy()
    
    if len(df) < MIN_DATA_POINTS:
        df['signal_score'] = 0.0
        df['signal_type'] = 'INSUFFICIENT_DATA'
        df['confidence'] = 'Low'
        return df
    
    df['frama'], df['D'] = calculate_frama_vectorized(df['high'], df['low'], df['close'])
    df['frama_fast'], _ = calculate_frama_vectorized(df['high'], df['low'], df['close'], 10)
    df['frama_slow'], _ = calculate_frama_vectorized(df['high'], df['low'], df['close'], 30)
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
    
    df['rr_upper'] = df['frama'] + df['atr']
    df['rr_lower'] = df['frama'] - df['atr']
    
    df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_spike'] = df['volume'] > (df['volume_ma'] * 1.5)
    
    df = calculate_td_sequential_optimized(df)
    
    price_range = df['rr_upper'] - df['rr_lower']
    df['price_position'] = ((df['close'] - df['rr_lower']) / price_range.where(price_range != 0, np.nan)).clip(0, 1).fillna(0.5)
    
    df['signal_score'] = 0.0
    
    trend_score = np.where(
        (df['frama_fast'].notna()) & (df['frama_slow'].notna()) & (df['frama_fast'] > df['frama_slow']),
        25, -25
    )
    df['signal_score'] += trend_score
    
    td9_buy_score = np.where((df['td_nine']) & (df['td_setup'] == -9), 40, 0)
    td9_sell_score = np.where((df['td_nine']) & (df['td_setup'] == 9), -40, 0)
    df['signal_score'] += td9_buy_score + td9_sell_score
    
    td13_score = np.where(df['td_thirteen'], 
                          np.where(df['td_countdown'] > 0, 60, -60), 0)
    df['signal_score'] += td13_score
    
    price_score = np.where(df['price_position'] < 0.2, 20,
                           np.where(df['price_position'] > 0.8, -20, 0))
    df['signal_score'] += price_score
    
    volume_boost = np.where(df['volume_spike'], 
                            np.where(df['signal_score'] > 0, 10, -10), 0)
    df['signal_score'] += volume_boost
    
    df['signal_type'] = np.where(df['signal_score'] >= 50, 'STRONG BUY',
                        np.where(df['signal_score'] >= 25, 'BUY',
                        np.where(df['signal_score'] <= -50, 'STRONG SELL',
                        np.where(df['signal_score'] <= -25, 'SELL', 'NEUTRAL'))))
    
    df['confidence'] = np.where(df['signal_score'].abs() >= 50, 'High',
                       np.where(df['signal_score'].abs() >= 25, 'Medium', 'Low'))
    
    return df

# ========================= NEW FEATURES =========================

# Feature 1: Market Context (Risk-On/Risk-Off)
@st.cache_data(ttl=CACHE_TTL)
def get_market_regime(spy_data: pd.DataFrame) -> Dict:
    """Identify market regime using SPY data"""
    if len(spy_data) < 20:
        return {"regime": "UNKNOWN", "description": "Insufficient data", "color": "gray"}
    
    latest = spy_data.iloc[-1]
    
    # Calculate volatility (ATR as % of price)
    volatility = (latest['atr'] / latest['close']) * 100 if 'atr' in spy_data else 2.0
    
    # Trend strength
    trend_up = latest.get('frama_fast', 0) > latest.get('frama_slow', 0)
    above_frama = latest['close'] > latest.get('frama', latest['close'])
    
    # 20-day performance
    perf_20d = (latest['close'] / spy_data['close'].iloc[-20] - 1) * 100 if len(spy_data) >= 20 else 0
    
    # Determine regime
    if volatility > 3:
        return {
            "regime": "⚠️ HIGH VOLATILITY",
            "description": "Reduce position sizes, widen stops",
            "volatility": volatility,
            "color": "orange",
            "action": "CAUTION"
        }
    elif trend_up and above_frama and perf_20d > 0:
        return {
            "regime": "✅ RISK-ON",
            "description": "Favorable for longs, add to winners",
            "volatility": volatility,
            "color": "green",
            "action": "BULLISH"
        }
    elif not trend_up and not above_frama and perf_20d < 0:
        return {
            "regime": "🔴 RISK-OFF",
            "description": "Defensive mode, consider cash/shorts",
            "volatility": volatility,
            "color": "red",
            "action": "BEARISH"
        }
    else:
        return {
            "regime": "➡️ NEUTRAL",
            "description": "Mixed signals, wait for clarity",
            "volatility": volatility,
            "color": "gray",
            "action": "WAIT"
        }

# Feature 2: Relative Strength vs SPY
@st.cache_data(ttl=CACHE_TTL)
def calculate_relative_strength(symbol_df: pd.DataFrame, spy_df: pd.DataFrame, symbol: str) -> Dict:
    """Compare symbol performance vs SPY"""
    if len(symbol_df) < 20 or len(spy_df) < 20:
        return {"rs": 0, "message": "Insufficient data", "strength": "UNKNOWN"}
    
    # Multiple timeframe comparison
    timeframes = {
        '5D': 5,
        '20D': 20,
        '60D': 60 if len(symbol_df) >= 60 else len(symbol_df)-1
    }
    
    rs_data = {}
    for label, days in timeframes.items():
        if days > 0 and len(symbol_df) >= days and len(spy_df) >= days:
            symbol_return = (symbol_df['close'].iloc[-1] / symbol_df['close'].iloc[-days] - 1) * 100
            spy_return = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-days] - 1) * 100
            rs_data[label] = symbol_return - spy_return
    
    # Average RS
    avg_rs = np.mean(list(rs_data.values())) if rs_data else 0
    
    # Classification
    if avg_rs > 5:
        strength = "💪 OUTPERFORMING"
        message = f"Beating SPY by {avg_rs:.1f}% avg"
        color = "green"
    elif avg_rs > 0:
        strength = "👍 PERFORMING"
        message = f"Slightly ahead of SPY (+{avg_rs:.1f}%)"
        color = "lightgreen"
    elif avg_rs > -5:
        strength = "➡️ LAGGING"
        message = f"Behind SPY by {abs(avg_rs):.1f}%"
        color = "orange"
    else:
        strength = "📉 UNDERPERFORMING"
        message = f"Weak vs SPY ({avg_rs:.1f}%)"
        color = "red"
    
    return {
        "rs": avg_rs,
        "message": message,
        "strength": strength,
        "color": color,
        "details": rs_data
    }

# Feature 3: Backtest Signals
@st.cache_data(ttl=3600)
def backtest_signals(df: pd.DataFrame, symbol: str) -> Dict:
    """Quick backtest to show signal reliability"""
    if len(df) < 50:
        return None
    
    trades = []
    
    for i in range(30, len(df)-10):  # Leave room for exit
        if df['signal_type'].iloc[i] in ['BUY', 'STRONG BUY']:
            entry = df['close'].iloc[i]
            
            # Check multiple exit points
            exits = {
                '3D': df['close'].iloc[min(i+3, len(df)-1)],
                '5D': df['close'].iloc[min(i+5, len(df)-1)],
                '10D': df['close'].iloc[min(i+10, len(df)-1)]
            }
            
            for period, exit_price in exits.items():
                profit = (exit_price - entry) / entry * 100
                trades.append({
                    'period': period,
                    'profit': profit,
                    'signal_score': df['signal_score'].iloc[i]
                })
    
    if not trades:
        return None
    
    # Calculate statistics
    results = {}
    for period in ['3D', '5D', '10D']:
        period_trades = [t for t in trades if t['period'] == period]
        if period_trades:
            profits = [t['profit'] for t in period_trades]
            results[period] = {
                'win_rate': sum(1 for p in profits if p > 0) / len(profits) * 100,
                'avg_profit': np.mean(profits),
                'best': max(profits),
                'worst': min(profits),
                'total_trades': len(profits)
            }
    
    # Overall stats
    all_profits = [t['profit'] for t in trades if t['period'] == '5D']  # Use 5D as default
    
    return {
        'overall_win_rate': sum(1 for p in all_profits if p > 0) / len(all_profits) * 100 if all_profits else 0,
        'overall_avg_profit': np.mean(all_profits) if all_profits else 0,
        'total_signals': len(all_profits),
        'by_period': results,
        'symbol': symbol
    }

# Feature 4: Smart Alerts
def check_alerts(df: pd.DataFrame, symbol: str, market_regime: Dict) -> List[str]:
    """Generate actionable alerts"""
    if len(df) < 2:
        return []
    
    alerts = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Critical alerts
    
    # Price near support
    if 'rr_lower' in latest and latest['close'] and latest['rr_lower']:
        distance_to_support = abs(latest['close'] - latest['rr_lower']) / latest['close']
        if distance_to_support < 0.02:  # Within 2% of support
            alerts.append(f"🎯 **{symbol}** at SUPPORT (${latest['rr_lower']:.2f}) - Watch for bounce!")
    
    # TD9 completion
    if latest.get('td_nine', False):
        if latest.get('td_setup') == -9:
            alerts.append(f"⚡ **{symbol}** TD9 BUY signal complete - Reversal imminent!")
        elif latest.get('td_setup') == 9:
            alerts.append(f"⚠️ **{symbol}** TD9 SELL signal - Consider taking profits!")
    
    # TD13 major reversal
    if latest.get('td_thirteen', False):
        alerts.append(f"🚨 **{symbol}** TD13 MAJOR REVERSAL - Strong signal!")
    
    # Volume spike with signal
    if latest.get('volume_spike', False) and latest.get('signal_score', 0) > 25:
        alerts.append(f"📊 **{symbol}** VOLUME SURGE confirms buy signal!")
    
    # Trend change
    if 'frama_fast' in df.columns and 'frama_slow' in df.columns:
        if (prev['frama_fast'] <= prev['frama_slow'] and 
            latest['frama_fast'] > latest['frama_slow']):
            alerts.append(f"🔄 **{symbol}** TREND CHANGE - Turned bullish!")
        elif (prev['frama_fast'] >= prev['frama_slow'] and 
              latest['frama_fast'] < latest['frama_slow']):
            alerts.append(f"📉 **{symbol}** TREND CHANGE - Turned bearish!")
    
    # Market regime alerts
    if market_regime.get('action') == 'CAUTION':
        if latest.get('signal_score', 0) > 50:
            alerts.append(f"⚠️ High volatility - Reduce {symbol} position size!")
    
    return alerts

# Feature 5: Correlation Analysis
@st.cache_data(ttl=CACHE_TTL)
def find_correlations(symbols_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Find which symbols move together"""
    if len(symbols_data) < 2:
        return None
    
    # Build price matrix
    prices = pd.DataFrame()
    for sym, df in symbols_data.items():
        if len(df) >= 20:
            prices[sym] = df['close'].pct_change().iloc[-20:]  # Last 20 days
    
    if len(prices.columns) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = prices.corr()
    
    # Find significant correlations
    correlations = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Significant correlation
                correlations.append({
                    'Symbol 1': corr_matrix.index[i],
                    'Symbol 2': corr_matrix.index[j],
                    'Correlation': corr_val,
                    'Relationship': 'Strong Positive' if corr_val > 0.7 else 
                                  'Positive' if corr_val > 0 else 
                                  'Negative' if corr_val > -0.7 else 
                                  'Strong Negative'
                })
    
    return corr_matrix, correlations

# Feature 6: Sector Rotation Monitor
@st.cache_data(ttl=CACHE_TTL)
def sector_rotation_analysis(api_key: str) -> Dict:
    """Show which sectors are hot"""
    performances = {}
    
    with st.spinner("Analyzing sector rotation..."):
        for etf, (sector, emoji) in SECTOR_ETFS.items():
            try:
                df = fetch_ohlcv(etf, '1day', 100, api_key)
                if len(df) >= 20:
                    perf_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                    perf_20d = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                    
                    # Generate signals for sector
                    df = generate_signals_optimized(df)
                    
                    performances[sector] = {
                        'emoji': emoji,
                        'etf': etf,
                        '5D': perf_5d,
                        '20D': perf_20d,
                        'signal': df['signal_type'].iloc[-1] if 'signal_type' in df else 'NEUTRAL',
                        'momentum': perf_5d > 0 and perf_20d > 0
                    }
            except:
                continue
    
    if not performances:
        return None
    
    # Sort by 20D performance
    sorted_sectors = sorted(performances.items(), key=lambda x: x[1]['20D'], reverse=True)
    
    return {
        'performances': performances,
        'sorted': sorted_sectors,
        'top': sorted_sectors[:3],
        'bottom': sorted_sectors[-3:],
        'timestamp': datetime.now()
    }

# Feature 8: Multi-Timeframe Analysis
@st.cache_data(ttl=CACHE_TTL)
def multi_timeframe_alignment(symbol: str, api_key: str) -> Dict:
    """Check if multiple timeframes agree"""
    signals = {}
    scores = {}
    
    timeframes = ['1h', '4h', '1day']
    
    for tf in timeframes:
        try:
            df = fetch_ohlcv(symbol, tf, 100, api_key)
            if len(df) >= MIN_DATA_POINTS:
                df = generate_signals_optimized(df)
                signals[tf] = df['signal_type'].iloc[-1]
                scores[tf] = df['signal_score'].iloc[-1]
        except:
            signals[tf] = 'ERROR'
            scores[tf] = 0
    
    # Calculate alignment
    buy_signals = sum(1 for s in signals.values() if 'BUY' in str(s))
    sell_signals = sum(1 for s in signals.values() if 'SELL' in str(s))
    
    avg_score = np.mean([s for s in scores.values() if s != 0])
    
    if buy_signals == 3:
        alignment = "💎 PERFECT ALIGNMENT"
        message = "All timeframes bullish - Strong buy!"
        color = "green"
        strength = 100
    elif buy_signals == 2:
        alignment = "✅ GOOD ALIGNMENT"
        message = "2/3 timeframes bullish"
        color = "lightgreen"
        strength = 66
    elif sell_signals >= 2:
        alignment = "🔴 BEARISH ALIGNMENT"
        message = "Multiple timeframes bearish"
        color = "red"
        strength = -66
    else:
        alignment = "⚠️ MIXED SIGNALS"
        message = "Timeframes disagree - wait"
        color = "orange"
        strength = 0
    
    return {
        'alignment': alignment,
        'message': message,
        'color': color,
        'strength': strength,
        'signals': signals,
        'scores': scores,
        'avg_score': avg_score
    }

# ------------------------- Main UI -------------------------
# Input section
col1, col2, col3 = st.columns([4, 2, 2])
with col1:
    symbols_text = st.text_input("📊 Symbols (max 5)", value="AAPL, MSFT, SPY", 
                                 help="Enter up to 5 symbols separated by commas")
with col2:
    interval = st.selectbox("⏱️ Timeframe", ["1day", "4h", "1h"], index=0)
with col3:
    data_points = st.selectbox("📈 Data Points", [200, 500, 800], index=0)

# Main analysis
if run_btn:
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()][:5]
    
    if not symbols:
        st.error("Please enter at least one symbol")
        st.stop()
    
    # Always fetch SPY for market context
    if 'SPY' not in symbols:
        symbols.append('SPY')
    
    # Store all data for correlation analysis
    all_data = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch all data first
    status_text.text("Fetching market data...")
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_ohlcv(sym, interval, data_points, API_KEY)
            df = generate_signals_optimized(df)
            all_data[sym] = df
            progress_bar.progress((idx + 1) / len(symbols) * 0.3)
        except Exception as e:
            st.error(f"Failed to fetch {sym}: {str(e)}")
            continue
    
    # Get SPY data for market context
    spy_data = all_data.get('SPY')
    if not spy_data:
        st.warning("Could not fetch SPY data for market context")
        market_regime = {"regime": "UNKNOWN", "description": "SPY data unavailable", "color": "gray"}
    else:
        market_regime = get_market_regime(spy_data)
    
    # Sector Rotation Analysis
    status_text.text("Analyzing sector rotation...")
    progress_bar.progress(0.4)
    sector_data = sector_rotation_analysis(API_KEY)
    
    # =================== MARKET OVERVIEW SECTION ===================
    st.markdown("---")
    st.header("🌍 Market Overview")
    
    # Market Context Display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Market Regime", market_regime['regime'])
        st.caption(market_regime['description'])
    
    with col2:
        if spy_data is not None and len(spy_data) > 0:
            spy_latest = spy_data.iloc[-1]
            spy_prev = spy_data.iloc[-2] if len(spy_data) > 1 else spy_latest
            spy_change = (spy_latest['close'] / spy_prev['close'] - 1) * 100
            st.metric("S&P 500", f"${spy_latest['close']:.2f}", f"{spy_change:+.1f}%")
        else:
            st.metric("S&P 500", "N/A")
    
    with col3:
        st.metric("Volatility", f"{market_regime.get('volatility', 0):.1f}%")
        st.caption("Market volatility (ATR%)")
    
    with col4:
        # Count bullish vs bearish signals
        bull_count = sum(1 for sym, df in all_data.items() 
                        if len(df) > 0 and 'BUY' in df.iloc[-1].get('signal_type', ''))
        bear_count = sum(1 for sym, df in all_data.items() 
                        if len(df) > 0 and 'SELL' in df.iloc[-1].get('signal_type', ''))
        st.metric("Signal Balance", f"{bull_count}🟢 / {bear_count}🔴")
        st.caption("Bullish vs Bearish")
    
    # Sector Rotation Display
    if sector_data:
        st.subheader("🔄 Sector Rotation")
        
        # Top and bottom sectors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔥 Hot Sectors (Leaders)**")
            for sector, data in sector_data['top']:
                emoji = data['emoji']
                perf = data['20D']
                signal = data['signal']
                color = "🟢" if 'BUY' in signal else "🔴" if 'SELL' in signal else "⚪"
                st.success(f"{emoji} **{sector}**: +{perf:.1f}% {color}")
        
        with col2:
            st.markdown("**❄️ Cold Sectors (Laggards)**")
            for sector, data in sector_data['bottom']:
                emoji = data['emoji']
                perf = data['20D']
                signal = data['signal']
                color = "🟢" if 'BUY' in signal else "🔴" if 'SELL' in signal else "⚪"
                st.error(f"{emoji} **{sector}**: {perf:.1f}% {color}")
    
    # Correlation Matrix
    if len(all_data) > 1:
        corr_matrix, correlations = find_correlations(all_data)
        
        if correlations:
            st.subheader("🔗 Symbol Correlations")
            
            # Show significant correlations
            for corr in correlations[:3]:  # Show top 3
                if corr['Correlation'] > 0.7:
                    st.info(f"**{corr['Symbol 1']}** ↔️ **{corr['Symbol 2']}**: {corr['Correlation']:.2f} (Move together)")
                elif corr['Correlation'] < -0.5:
                    st.warning(f"**{corr['Symbol 1']}** ↔️ **{corr['Symbol 2']}**: {corr['Correlation']:.2f} (Inverse)")
    
    # Smart Alerts Section
    st.markdown("---")
    all_alerts = []
    for sym, df in all_data.items():
        if sym != 'SPY' or 'SPY' in symbols[:5]:  # Only show SPY alerts if explicitly requested
            alerts = check_alerts(df, sym, market_regime)
            all_alerts.extend(alerts)
    
    if all_alerts:
        st.header("🚨 Smart Alerts")
        alert_container = st.container()
        with alert_container:
            for alert in all_alerts[:5]:  # Show max 5 alerts
                if "TD13" in alert or "MAJOR" in alert:
                    st.error(alert)
                elif "TD9" in alert or "SUPPORT" in alert:
                    st.warning(alert)
                else:
                    st.info(alert)
    
    # =================== INDIVIDUAL SYMBOL ANALYSIS ===================
    st.markdown("---")
    st.header("📊 Symbol Analysis")
    
    # Remove SPY from tabs if it wasn't explicitly requested
    display_symbols = [s for s in symbols if s != 'SPY' or 'SPY' in symbols_text.upper()]
    
    tabs = st.tabs(display_symbols)
    
    for idx, (sym, tab) in enumerate(zip(display_symbols, tabs)):
        with tab:
            status_text.text(f"Analyzing {sym}...")
            progress_bar.progress(0.5 + (idx + 1) / len(display_symbols) * 0.5)
            
            if sym not in all_data:
                st.error(f"No data available for {sym}")
                continue
            
            df = all_data[sym]
            
            if len(df) < MIN_DATA_POINTS:
                st.warning(f"⚠️ Limited data: {len(df)} bars")
                continue
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Header metrics with enhanced features
            col1, col2, col3, col4, col5 = st.columns(5)
            
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
            
            with col5:
                # Relative Strength vs SPY
                if spy_data is not None and sym != 'SPY':
                    rs_data = calculate_relative_strength(df, spy_data, sym)
                    st.metric("vs SPY", rs_data['strength'].split()[0])
                    st.caption(f"{rs_data['rs']:.1f}%")
            
            # Multi-timeframe alignment
            mtf_col1, mtf_col2 = st.columns([1, 3])
            with mtf_col1:
                st.markdown("**📈 Multi-Timeframe**")
            with mtf_col2:
                with st.spinner("Checking timeframes..."):
                    mtf = multi_timeframe_alignment(sym, API_KEY)
                    if mtf['strength'] == 100:
                        st.success(f"{mtf['alignment']}: {mtf['message']}")
                    elif mtf['strength'] > 0:
                        st.info(f"{mtf['alignment']}: {mtf['message']}")
                    elif mtf['strength'] < 0:
                        st.error(f"{mtf['alignment']}: {mtf['message']}")
                    else:
                        st.warning(f"{mtf['alignment']}: {mtf['message']}")
            
            # Backtest Results
            backtest = backtest_signals(df, sym)
            if backtest:
                st.subheader("📊 Historical Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Win Rate", f"{backtest['overall_win_rate']:.0f}%")
                    st.caption(f"Last {backtest['total_signals']} signals")
                
                with col2:
                    st.metric("Avg Return", f"{backtest['overall_avg_profit']:.1f}%")
                    st.caption("Per trade (5 days)")
                
                with col3:
                    if '5D' in backtest['by_period']:
                        best = backtest['by_period']['5D']['best']
                        st.metric("Best Trade", f"+{best:.1f}%")
                        st.caption("5-day period")
                
                with col4:
                    if '5D' in backtest['by_period']:
                        worst = backtest['by_period']['5D']['worst']
                        st.metric("Worst Trade", f"{worst:.1f}%")
                        st.caption("5-day period")
                
                # Performance by holding period
                if backtest['by_period']:
                    st.markdown("**Performance by Holding Period:**")
                    perf_cols = st.columns(3)
                    for idx, (period, data) in enumerate(backtest['by_period'].items()):
                        with perf_cols[idx % 3]:
                            color = "🟢" if data['win_rate'] > 60 else "🟡" if data['win_rate'] > 50 else "🔴"
                            st.info(f"**{period}**: {data['win_rate']:.0f}% wins, {data['avg_profit']:+.1f}% avg {color}")
            
            # Trading Levels
            if pd.notna(latest.get('rr_lower')) and pd.notna(latest.get('rr_upper')):
                st.subheader("🎯 Trading Levels")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.success(f"**Entry:** ${latest['rr_lower']:.2f}")
                    distance = ((latest['rr_lower'] - latest['close']) / latest['close'] * 100)
                    st.caption(f"{distance:+.1f}% away")
                
                with col2:
                    st.info(f"**Target 1:** ${latest['frama']:.2f}")
                    upside = ((latest['frama'] - latest['close']) / latest['close'] * 100)
                    st.caption(f"{upside:+.1f}% upside")
                
                with col3:
                    st.info(f"**Target 2:** ${latest['rr_upper']:.2f}")
                    upside2 = ((latest['rr_upper'] - latest['close']) / latest['close'] * 100)
                    st.caption(f"{upside2:+.1f}% upside")
                
                with col4:
                    stop = latest['rr_lower'] - latest.get('atr', 0) * 0.5
                    st.error(f"**Stop:** ${stop:.2f}")
                    risk = ((stop - latest['close']) / latest['close'] * 100)
                    st.caption(f"{risk:.1f}% risk")
            
            # Action Box with context
            st.subheader("📋 Action Plan")
            
            # Consider market regime in recommendations
            if market_regime.get('action') == 'CAUTION':
                st.warning("⚠️ Market is volatile - use smaller position sizes!")
            
            if latest['signal_score'] >= 50:
                if backtest and backtest['overall_win_rate'] > 60:
                    st.success(f"""
                    ### ✅ **HIGH CONFIDENCE BUY**
                    - Signal Score: {latest['signal_score']:.0f}/100
                    - Historical Win Rate: {backtest['overall_win_rate']:.0f}%
                    - Multi-timeframe: {mtf['alignment']}
                    - Entry: ${latest.get('rr_lower', 0):.2f}
                    - Stop: ${latest.get('rr_lower', 0) - latest.get('atr', 0) * 0.5:.2f}
                    """)
                else:
                    st.info(f"""
                    ### 👍 **BUY SIGNAL**
                    - Signal Score: {latest['signal_score']:.0f}/100
                    - Use smaller position (moderate confidence)
                    """)
            elif latest['signal_score'] >= 25:
                st.info(f"""
                ### 🤔 **CONSIDER BUYING**
                - Signal Score: {latest['signal_score']:.0f}/100
                - Wait for better entry or confirmation
                """)
            elif latest['signal_score'] <= -25:
                st.error(f"""
                ### ❌ **AVOID / SELL**
                - Bearish Signal: {latest['signal_score']:.0f}/100
                - Consider exiting longs
                """)
            else:
                st.warning("### ⏸️ **WAIT** - No clear signal")
            
            # Chart
            st.subheader("📈 Price Chart")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['datetime'], 
                open=df['open'], 
                high=df['high'], 
                low=df['low'], 
                close=df['close'],
                name=sym
            ))
            
            # FRAMA and bands
            fig.add_trace(go.Scatter(
                x=df['datetime'], y=df['rr_upper'],
                mode='lines', name='Resistance',
                line=dict(color='rgba(255,0,0,0.3)', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['datetime'], y=df['frama'],
                mode='lines', name='FRAMA',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['datetime'], y=df['rr_lower'],
                mode='lines', name='Support',
                line=dict(color='rgba(0,255,0,0.3)', width=1)
            ))
            
            # TD Sequential markers
            td9_buys = df[df['td_nine'] & (df['td_setup'] == -9)]
            if not td9_buys.empty:
                fig.add_trace(go.Scatter(
                    x=td9_buys['datetime'],
                    y=td9_buys['low'] * 0.99,
                    mode='markers',
                    name='TD9 Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            td9_sells = df[df['td_nine'] & (df['td_setup'] == 9)]
            if not td9_sells.empty:
                fig.add_trace(go.Scatter(
                    x=td9_sells['datetime'],
                    y=td9_sells['high'] * 1.01,
                    mode='markers',
                    name='TD9 Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            fig.update_layout(
                title=f"{sym} - Technical Analysis",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500,
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                hovermode='x unified',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()

# =================== FOOTER ===================
st.markdown("---")
st.markdown("### 📚 Quick Reference")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **🟢 Best Setups**
    - Perfect MTF alignment
    - Win rate > 60%
    - Near support level
    - Volume confirmation
    - Risk-on market
    """)

with col2:
    st.markdown("""
    **🔴 Avoid When**
    - Mixed timeframes
    - Win rate < 50%
    - High volatility
    - Bearish market
    - No clear levels
    """)

with col3:
    st.markdown("""
    **📊 Indicators**
    - TD9: Exhaustion
    - TD13: Major reversal
    - FRAMA: Dynamic trend
    - RS: vs Market
    """)

with col4:
    st.markdown("""
    **💡 Pro Tips**
    - Check sector rotation
    - Watch correlations
    - Use smaller size in volatile markets
    - Wait for alerts
    """)

st.caption(f"v5.0 Pro | Data: Twelve Data | {datetime.now().strftime('%H:%M:%S')}")

# Cache management
if st.button("🔄 Clear Cache", help="Click if data seems stale"):
    st.cache_data.clear()
    st.success("Cache cleared! Refresh to reload.")

