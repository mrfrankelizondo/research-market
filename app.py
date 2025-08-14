# app.py — The Research Report (FRAMA + DeMark, no Quads)
"""
The Research Report — Macro + Technical Trading Dashboard
Features: FRAMA, DeMark TD Sequential, Risk Ranges (no Quad framework)
Version: 3.2 — Quads removed; reliability & math safety retained
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
st.caption(f"Technical Trading Dashboard | Build: {datetime.now().strftime('%Y.%m.%d')} | Plain-English Signals")

# ------------------------- API Key -------------------------
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

# ------------------------- Constants & Sector Map -------------------------
DEFAULT_OUTPUTSIZE = 800
DEFAULT_FRAMA_LEN = 20

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
    df = df.copy(); n = len(df)
    df['td_setup'] = 0; df['td_countdown'] = 0; df['td_nine'] = False; df['td_thirteen'] = False
    for i in range(4, n):
        if df['close'].iloc[i] < df['close'].iloc[i-4]:
            df.loc[i, 'td_setup'] = 1 if df['td_setup'].iloc[i-1] > 0 else df['td_setup'].iloc[i-1] - 1
        elif df['close'].iloc[i] > df['close'].iloc[i-4]:
            df.loc[i, 'td_setup'] = 1 if df['td_setup'].iloc[i-1] < 0 else df['td_setup'].iloc[i-1] + 1
        else:
            df.loc[i, 'td_setup'] = 0
    df['td_nine'] = (df['td_setup'].abs() == 9)
    countdown_active = False; countdown_count = 0; countdown_direction = 0
    for i in range(2, n):
        if df['td_nine'].iloc[i]:
            countdown_active = True; countdown_count = 0
            countdown_direction = 1 if df['td_setup'].iloc[i] < 0 else -1
        if countdown_active and i >= 2:
            if countdown_direction == 1 and df['close'].iloc[i] <= df['low'].iloc[i-2]: countdown_count += 1
            elif countdown_direction == -1 and df['close'].iloc[i] >= df['high'].iloc[i-2]: countdown_count += 1
            df.loc[i, 'td_countdown'] = countdown_count
            if countdown_count >= 13:
                df.loc[i, 'td_thirteen'] = True; countdown_active = False
    return df

# ------------------------- Signal Engine (no Quads) -------------------------
def generate_signals(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.copy()
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

    # TD Sequential
    df = calculate_td_sequential(df)

    # Safe price position in range [0,1], neutral to 0.5 if unknown
    denom = (df['rr_upper'] - df['rr_lower']).replace(0, np.nan)
    price_pos_series = ((df['close'] - df['rr_lower']) / denom).clip(0, 1).fillna(0.5)

    # Scoring (no quad multiplier)
    df['signal_score'] = 0.0
    df['signal_type'] = 'NEUTRAL'

    for i in range(len(df)):
        score = 0.0

        # Trend via FRAMA cross
        ff = df['frama_fast'].iloc[i]; fs = df['frama_slow'].iloc[i]
        score += 25 if (pd.notna(ff) and pd.notna(fs) and (ff > fs)) else -25

        # TD signals
        if bool(df['td_nine'].iloc[i]):
            if df['td_setup'].iloc[i] == -9: score += 40
            elif df['td_setup'].iloc[i] == 9: score -= 40

        if bool(df['td_thirteen'].iloc[i]):
            score += 60 if df['td_countdown'].iloc[i] > 0 else -60

        # Price location in range
        ppos = float(price_pos_series.iloc[i])
        if ppos < 0.2: score += 20
        elif ppos > 0.8: score -= 20

        # Label
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

        df.loc[i, 'signal_type'] = sig
        df.loc[i, 'signal_score'] = score

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
    with c2:
        signal_threshold = st.slider("Signal sensitivity", 0, 100, 25, help="Lower = more signals")
        risk_percent = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)

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
                    df = generate_signals(df)

                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest

                    # Header with price
                    st.markdown(f"## {sym} Analysis")

                    price_change = latest['close'] - prev['close']
                    price_pct = (price_change / prev['close']) * 100 if prev['close'] else 0.0

                    c1, c2, c3 = st.columns([2, 2, 2])
                    with c1:
                        st.metric("Current Price", f"${latest['close']:.2f}",
                                  f"{price_change:+.2f} ({price_pct:+.1f}%)")
                    with c2:
                        trend_up = (pd.notna(latest['frama_fast']) and pd.notna(latest['frama_slow']) and latest['frama_fast'] > latest['frama_slow'])
                        st.metric("Market Trend", "📈 Uptrend" if trend_up else "📉 Downtrend")
                        st.caption("Bullish momentum" if trend_up else "Bearish momentum")
                    with c3:
                        signal_map = {
                            'STRONG BUY': ("💚 Strong Buy", "Excellent setup - buy now"),
                            'BUY': ("🟢 Buy", "Good setup - consider buying"),
                            'STRONG SELL': ("🔴 Strong Sell", "Exit immediately"),
                            'SELL': ("🟠 Sell", "Consider reducing position"),
                            'NEUTRAL': ("⚪ Wait", "No clear opportunity")
                        }
                        signal_display, signal_desc = signal_map.get(latest['signal_type'], ("⚪ Wait", "No signal"))
                        st.metric("Signal", signal_display)
                        st.caption(signal_desc)

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
                                st.metric("Exhaustion Level", "🟢 BOTTOM SIGNAL"); st.caption("Selling exhausted - buy opportunity")
                            elif td_setup == 9:
                                st.metric("Exhaustion Level", "🔴 TOP SIGNAL"); st.caption("Buying exhausted - sell warning")
                            elif td_setup <= -7:
                                st.metric("Exhaustion Level", f"📉 Oversold soon ({abs(td_setup)}/9)"); st.caption("Bottom forming")
                            elif td_setup >= 7:
                                st.metric("Exhaustion Level", f"📈 Overbought soon ({td_setup}/9)"); st.caption("Top forming")
                            elif td_setup <= -4:
                                st.metric("Exhaustion Level", f"📉 Selling pressure ({abs(td_setup)}/9)"); st.caption("Downtrend active")
                            elif td_setup >= 4:
                                st.metric("Exhaustion Level", f"📈 Buying pressure ({td_setup}/9)"); st.caption("Uptrend active")
                            else:
                                st.metric("Exhaustion Level", "➡️ No pattern"); st.caption("Waiting for setup")

                        with c2:
                            if bool(latest['td_thirteen']):
                                st.metric("Reversal Countdown", "⚡ MAJOR REVERSAL"); st.caption("Strong reversal NOW")
                            elif td_countdown >= 10:
                                st.metric("Reversal Countdown", f"⚠️ Reversal soon ({td_countdown}/13)"); st.caption("Major turn imminent")
                            elif td_countdown >= 7:
                                st.metric("Reversal Countdown", f"📊 Building ({td_countdown}/13)"); st.caption("Pressure increasing")
                            elif td_countdown >= 1:
                                st.metric("Reversal Countdown", f"⏳ Counting ({td_countdown}/13)"); st.caption("Tracking reversal")
                            else:
                                st.metric("Reversal Countdown", "⏸️ Inactive"); st.caption("No countdown yet")

                        with c3:
                            if bool(latest['td_nine']) and td_setup == -9:
                                st.success("✅ **BUY SETUP COMPLETE**"); st.caption("Look for entry point")
                            elif bool(latest['td_nine']) and td_setup == 9:
                                st.error("❌ **SELL SETUP COMPLETE**"); st.caption("Take profits/exit")
                            elif bool(latest['td_thirteen']):
                                st.warning("⚡ **MAJOR REVERSAL**"); st.caption("Strong signal!")
                            elif abs(td_setup) >= 7:
                                st.info(f"⏰ **Signal in {9-abs(td_setup)} days**"); st.caption("Get ready...")
                            else:
                                st.info("📊 **Monitoring...**"); st.caption("No signal yet")

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
                        st.warning("Not enough data to compute targets/stop reliably yet (need ~30+ bars).")
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
                            st.success(f"""
                            ### ✅ **ACTION: BUY THIS STOCK**
                            1. Place a limit buy at **${latest['rr_lower']:.2f}**
                            2. Stop loss **${stop_loss:.2f}** (risk: {risk_pct:.1f}%)
                            3. Take half at **${latest['frama']:.2f}**
                            4. Exit remainder at **${latest['rr_upper']:.2f}**
                            5. Risk only **{risk_percent}%** of your account
                            """)
                        elif latest['signal_score'] >= 25:
                            st.info(f"""
                            ### 👍 **ACTION: CONSIDER BUYING**
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
                            """)
                        else:
                            st.warning("### ⏸️ **ACTION: WAIT**\nNo clear edge right now.")
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

                    # Plain English Summary
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
                    sc = latest['signal_score']
                    parts.append("**Strong buy signal**" if sc >= 50 else
                                 "Moderate buy signal" if sc >= 25 else
                                 "**Strong sell signal**" if sc <= -50 else
                                 "Moderate sell signal" if sc <= -25 else
                                 "No clear signal - better to wait")
                    sector = SYMBOL_SECTORS.get(sym, 'Unknown')
                    parts.append(f"Sector: {sector}")
                    st.info(". ".join(parts) + ".")

                    # Position Calculator
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
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Shares to Buy", f"{shares:,}")
                            c2.metric("Total Cost", f"${position_value:,.2f}")
                            c3.metric("Risk Amount", f"${risk_amt:.2f}")
                            c4.metric("Potential Profit", f"${shares * (latest['rr_upper'] - entry):.2f}")
                            st.info(f"Buy {shares:,} @ ${entry:.2f} • Stop ${stop:.2f} • Max loss ${risk_amt:.2f} ({risk_percent}%)")
                        else:
                            st.caption("Need more history to compute entry/stop/targets.")

                    # Download
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
                - Check the symbol (e.g., AAPL not APPL)
                - Ensure it's a supported stock/ETF
                - Try again if rate limited
                - Verify your API key is valid
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
- Risk/Reward > 2:1""")
with c2:
    st.markdown("""### ⚠️ Warning Signs
- TD9 Sell completed
- Strong downtrend
- Price at/near upper band
- Weak risk/reward""")
with c3:
    st.markdown("""### 💡 Pro Tips
- Let price come to you
- Use fixed % risk per trade
- Take partial profits at Target 1
- Always place the stop""")

st.markdown("---")
st.caption(f"Session: {datetime.now().strftime('%H:%M:%S')} • Build: v3.2 — The Research Report")
st.caption("""
💡 This tool provides analysis, not financial advice. Always do your own research.
📊 Data: Twelve Data • 🔑 API: Free at twelvedata.com
""")

with st.expander("⚠️ Risk Disclaimer", expanded=False):
    st.warning("""
    - Past performance does not guarantee future results
    - All trading involves risk of loss
    - This is not financial advice
    - Never risk more than you can afford to lose
    - Consider paper trading first
    """)

