#!/usr/bin/env python3
"""NIFTY 50 Web Dashboard — Streamlit + Plotly"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY 50 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* Light background */
  .stApp { background-color: #ffffff; }
  section[data-testid="stSidebar"] { background-color: #f5f5f5; }

  /* All text black */
  .stApp, .stApp p, .stApp span, .stApp div,
  .stApp label, .stApp li, .stMarkdown { color: #111111 !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] * { color: #111111 !important; }

  /* Header */
  h1, h2, h3 { color: #111111 !important; letter-spacing: -0.5px; }

  /* Caption */
  .stCaptionContainer, .stCaptionContainer p { color: #555555 !important; }

  /* Dataframe tweaks */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Info box */
  div[data-testid="stInfo"] { background: #e8f4fd; color: #111111 !important; }
  div[data-testid="stInfo"] * { color: #111111 !important; }

  /* Expander */
  details { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; }
  summary { color: #111111 !important; }

  /* Hide streamlit branding */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Timeframe configs ─────────────────────────────────────────────────────────
TIMEFRAMES = {
    "📈 Swing Trade": {
        "period": "1y",  "interval": "1d",  "sma_fast": 50,  "sma_slow": 200,
        "min_bars": 200, "desc": "Daily candles · holds days to weeks",
    },
    "🏦 Long Term": {
        "period": "5y",  "interval": "1wk", "sma_fast": 50,  "sma_slow": 200,
        "min_bars": 200, "desc": "Weekly candles · holds months to years",
    },
    "⚡ Day Trade": {
        "period": "5d",  "interval": "15m", "sma_fast": 20,  "sma_slow": 50,
        "min_bars": 50,  "desc": "15-min candles · intraday positions",
    },
}

# ── Backtest configs (extended history) ──────────────────────────────────────
BT_CONFIGS = {
    "📈 Swing Trade": {"period": "3y",  "interval": "1d",  "sma_fast": 50, "sma_slow": 200, "min_bars": 210, "max_hold": 60},
    "🏦 Long Term":   {"period": "15y", "interval": "1wk", "sma_fast": 50, "sma_slow": 200, "min_bars": 210, "max_hold": 26},
    "⚡ Day Trade":   {"period": "60d", "interval": "15m", "sma_fast": 20, "sma_slow": 50,  "min_bars": 60,  "max_hold": 25},
}

# ── NIFTY 50 universe ─────────────────────────────────────────────────────────
NIFTY50 = [
    "ADANIENT.NS",  "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS",  "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
    "BEL.NS",       "BHARTIARTL.NS", "BPCL.NS",       "BRITANNIA.NS",
    "CIPLA.NS",     "COALINDIA.NS",  "DIVISLAB.NS",   "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS",     "HCLTECH.NS",    "HDFCBANK.NS",
    "HDFCLIFE.NS",  "HEROMOTOCO.NS", "HINDALCO.NS",   "HINDUNILVR.NS",
    "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",       "ITC.NS",
    "JSWSTEEL.NS",  "KOTAKBANK.NS",  "LT.NS",         "M&M.NS",
    "MARUTI.NS",    "NESTLEIND.NS",  "NTPC.NS",       "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS",   "SBILIFE.NS",    "SBIN.NS",
    "SHRIRAMFIN.NS","SUNPHARMA.NS",  "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TCS.NS",        "TECHM.NS",      "TITAN.NS",
    "ULTRACEMCO.NS","WIPRO.NS",
]

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd(close):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist   = line - signal
    return line, signal, hist


def get_signal(price, rsi, sma_fast, sma_slow, macd_hist, prev_rsi=None, prev_macd_hist=None):
    above_slow  = price > sma_slow
    rsi_rising  = prev_rsi       is None or rsi       > prev_rsi
    macd_rising = prev_macd_hist is None or macd_hist > prev_macd_hist
    rsi_falling = prev_rsi is not None and rsi < prev_rsi
    # BUY: oversold + both momentum recovery signals (no SMA200 required —
    # corrections happen in all trends; RSI+MACD confirmation filters quality)
    if rsi < 40 and rsi_rising and macd_rising:
        return "BUY"
    # SELL: overbought OR in a downtrend AND actively still falling (not recovering)
    elif rsi > 75 or (not above_slow and rsi_falling):
        return "SELL"
    return "HOLD"


def calc_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_stop_target(df, price, atr, signal, interval):
    """
    Stop  = recent swing low (last N bars), capped at 2×ATR below price.
    Target = recent swing high (last N bars), minimum 1.5×R above price.
    For SELL signals, flip: stop is recent swing HIGH, target is recent swing LOW.
    """
    n = {"15m": 20, "1d": 14, "1wk": 20}.get(interval, 14)

    recent_low  = round(df["Low"].iloc[-n:].min(),  2)
    recent_high = round(df["High"].iloc[-n:].max(), 2)

    if signal == "SELL":
        # Short / exit view: danger level above, downside below
        atr_stop   = round(price + 2 * atr, 2)
        stop       = round(min(recent_high, atr_stop), 2)   # nearest resistance above
        risk       = stop - price
        min_target = round(price - 1.5 * risk, 2)
        target     = round(min(recent_low, min_target), 2)  # nearest support below
        stop_pct   = round((stop - price) / price * 100, 1)
        target_pct = round((price - target) / price * 100, 1)
    else:
        # Long / buy view: support below, resistance above
        atr_stop   = round(price - 2 * atr, 2)
        stop       = round(max(recent_low, atr_stop), 2)    # nearest support below
        risk       = price - stop
        min_target = round(price + 1.5 * risk, 2)
        target     = round(max(recent_high, min_target), 2) # nearest resistance above
        stop_pct   = round(-risk / price * 100, 1)
        target_pct = round((target - price) / price * 100, 1)

    return stop, target, stop_pct, target_pct


# ── Backtesting engine ────────────────────────────────────────────────────────
def _compute_quality_score(win_rate, avg_win, avg_loss, n_trades):
    if n_trades < 3:
        return None
    exp = win_rate * avg_win + (1 - win_rate) * avg_loss
    exp_score  = max(0.0, min(50.0, (exp + 3.0) / 6.0 * 50.0))
    wr_score   = max(0.0, min(30.0, (win_rate - 0.40) / 0.20 * 30.0))
    size_score = min(20.0, n_trades / 10.0 * 20.0)
    return round(exp_score + wr_score + size_score)


def backtest_single(ticker, tf_key):
    try:
        cfg = BT_CONFIGS[tf_key]
        df  = yf.Ticker(ticker).history(
            period=cfg["period"], interval=cfg["interval"], auto_adjust=True
        )
        if df.empty or len(df) < cfg["min_bars"]:
            return {"ticker": ticker, "n_trades": 0, "win_rate": 0, "avg_win": 0,
                    "avg_loss": 0, "expectancy": 0, "quality_score": None, "error": "insufficient_data"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        close      = df["Close"]
        fast, slow = cfg["sma_fast"], cfg["sma_slow"]
        interval   = cfg["interval"]
        n_lb       = {"15m": 20, "1d": 14, "1wk": 20}.get(interval, 14)

        sma_fast_s = close.rolling(fast).mean()
        sma_slow_s = close.rolling(slow).mean()
        rsi_s      = calc_rsi(close)
        atr_s      = calc_atr(df)
        _, _, hist_s = calc_macd(close)

        valid_mask = sma_slow_s.notna() & rsi_s.notna() & atr_s.notna()
        start_i    = int(valid_mask.values.argmax())

        trades     = []
        in_trade   = False
        prev_sig   = "HOLD"
        entry_price = stop = target = 0.0
        entry_idx   = -1

        for i in range(start_i, len(df)):
            sf  = sma_fast_s.iloc[i]
            sw  = sma_slow_s.iloc[i]
            rsi = rsi_s.iloc[i]
            atr = atr_s.iloc[i]
            mh  = hist_s.iloc[i]
            if pd.isna(sf) or pd.isna(sw) or pd.isna(rsi) or pd.isna(atr) or pd.isna(mh):
                continue

            price_i  = float(close.iloc[i])
            high_i   = float(df["High"].iloc[i])
            low_i    = float(df["Low"].iloc[i])
            prev_rsi = float(rsi_s.iloc[i - 1]) if i > start_i and not pd.isna(rsi_s.iloc[i - 1]) else None
            prev_mh  = float(hist_s.iloc[i - 1]) if i > start_i and not pd.isna(hist_s.iloc[i - 1]) else None
            cur_sig  = get_signal(price_i, float(rsi), float(sf), float(sw), float(mh), prev_rsi, prev_mh)

            # Exit check before entry
            if in_trade and i > entry_idx:
                bars_held = i - entry_idx
                if high_i >= target:
                    ret = (target - entry_price) / entry_price * 100
                    trades.append({"result": "WIN",  "ret": ret, "bars": bars_held})
                    in_trade = False
                elif low_i <= stop:
                    ret = (stop - entry_price) / entry_price * 100
                    trades.append({"result": "LOSS", "ret": ret, "bars": bars_held})
                    in_trade = False
                elif cur_sig == "SELL":
                    ret = (price_i - entry_price) / entry_price * 100
                    trades.append({"result": "EXIT", "ret": ret, "bars": bars_held})
                    in_trade = False
                elif bars_held >= cfg["max_hold"]:
                    ret = (price_i - entry_price) / entry_price * 100
                    trades.append({"result": "TIMEOUT", "ret": ret, "bars": bars_held})
                    in_trade = False

            # Entry check
            if not in_trade and cur_sig == "BUY" and prev_sig != "BUY":
                df_slice    = df.iloc[:i + 1]
                recent_low  = float(df_slice["Low"].iloc[-n_lb:].min())
                recent_high = float(df_slice["High"].iloc[-n_lb:].max())
                atr_stop    = price_i - 2 * float(atr)
                stop_val    = max(recent_low, atr_stop)
                risk        = price_i - stop_val
                if risk > 0 and stop_val < price_i:
                    target_val = max(recent_high, price_i + 1.5 * risk)
                    if target_val > price_i:
                        entry_price = price_i
                        stop        = stop_val
                        target      = target_val
                        entry_idx   = i
                        in_trade    = True

            prev_sig = cur_sig

        if not trades:
            return {"ticker": ticker, "n_trades": 0, "win_rate": 0, "avg_win": 0,
                    "avg_loss": 0, "expectancy": 0, "quality_score": None, "error": "no_trades"}

        wins   = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] != "WIN"]
        n      = len(trades)
        wr     = len(wins) / n
        aw     = sum(t["ret"] for t in wins)   / len(wins)   if wins   else 0.0
        al     = sum(t["ret"] for t in losses) / len(losses) if losses else 0.0
        exp    = wr * aw + (1 - wr) * al
        return {
            "ticker":        ticker,
            "n_trades":      n,
            "win_rate":      wr,
            "avg_win":       round(aw, 2),
            "avg_loss":      round(al, 2),
            "expectancy":    round(exp, 2),
            "quality_score": _compute_quality_score(wr, aw, al, n),
            "error":         None,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(tf_key: str):
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(backtest_single, t, tf_key): t for t in NIFTY50}
        for fut in as_completed(futures):
            data = fut.result()
            if data:
                results.append(data)
    results.sort(key=lambda d: (-(d["quality_score"] or 0), -d["win_rate"]))
    return results


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_single(ticker, cfg):
    try:
        df = yf.Ticker(ticker).history(
            period=cfg["period"], interval=cfg["interval"], auto_adjust=True
        )
        if df.empty or len(df) < cfg["min_bars"]:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close    = df["Close"].dropna()
        fast, slow = cfg["sma_fast"], cfg["sma_slow"]

        price    = round(float(close.iloc[-1]), 2)
        sma_fast = round(close.rolling(fast).mean().iloc[-1], 2)
        sma_slow = round(close.rolling(slow).mean().iloc[-1], 2)
        rsi_s    = calc_rsi(close)
        rsi      = round(float(rsi_s.iloc[-1]), 1)
        prev_rsi = round(float(rsi_s.iloc[-2]), 1) if rsi_s.dropna().shape[0] >= 2 else None
        ml, sl, mh = calc_macd(close)
        macd_hist  = round(float(mh.iloc[-1]), 2)
        prev_mh    = round(float(mh.iloc[-2]), 2) if mh.dropna().shape[0] >= 2 else None
        signal     = get_signal(price, rsi, sma_fast, sma_slow, macd_hist, prev_rsi, prev_mh)

        atr_s  = calc_atr(df)
        atr    = round(float(atr_s.iloc[-1]), 2)
        stop, target, stop_pct, target_pct = calc_stop_target(df, price, atr, signal, cfg["interval"])

        df = df.copy()
        df["SMAfast"] = close.rolling(fast).mean()
        df["SMAslow"] = close.rolling(slow).mean()
        df["RSI"]     = rsi_s
        df["MACD"], df["SignalLine"], df["Hist"] = calc_macd(close)

        return {
            "ticker":     ticker,
            "name":       ticker.replace(".NS", "").replace(".BO", ""),
            "df":         df,
            "price":      price,
            "sma_fast":   sma_fast,
            "sma_slow":   sma_slow,
            "sma_fast_label": f"SMA {fast}",
            "sma_slow_label": f"SMA {slow}",
            "rsi":        rsi,
            "macd_hist":  macd_hist,
            "vs_slow":    round((price - sma_slow) / sma_slow * 100, 1),
            "signal":     signal,
            "atr":        atr,
            "stop":       stop,
            "target":     target,
            "stop_pct":   stop_pct,
            "target_pct": target_pct,
        }
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_all(timeframe_key: str):
    cfg     = TIMEFRAMES[timeframe_key]
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_single, t, cfg): t for t in NIFTY50}
        for fut in as_completed(futures):
            data = fut.result()
            if data:
                results.append(data)
    order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    results.sort(key=lambda d: (order[d["signal"]], d["rsi"]))
    return results


# ── Plotly chart ──────────────────────────────────────────────────────────────
SIG_COLOR = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#f39c12"}

def build_chart(data):
    df     = data["df"].dropna(subset=["SMAslow"])
    ticker = data["name"]
    signal = data["signal"]
    price  = data["price"]
    fl     = data["sma_fast_label"]
    sl     = data["sma_slow_label"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(f"Price  •  {fl}  •  {sl}", "MACD", "RSI (14)"),
    )

    # ── Price panel ───────────────────────────────────────────────────────────
    # Green fill: price above slow SMA
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMAslow"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", fill="tonexty",
        fillcolor="rgba(46,204,113,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Red fill: slow SMA above price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMAslow"],
        mode="lines", fill="tonexty",
        fillcolor="rgba(231,76,60,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # SMA lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMAfast"],
        name=fl, line=dict(color="#e67e22", width=1.5, dash="dot"),
        hovertemplate=f"{fl}: ₹%{{y:,.2f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMAslow"],
        name=sl, line=dict(color="#8e44ad", width=1.5, dash="dot"),
        hovertemplate=f"{sl}: ₹%{{y:,.2f}}<extra></extra>",
    ), row=1, col=1)

    # Price line (on top)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Price", line=dict(color="#1a6faf", width=2),
        hovertemplate="₹%{y:,.2f}<extra>Price</extra>",
    ), row=1, col=1)

    # Stop loss and target lines
    stop   = data["stop"]
    target = data["target"]
    fig.add_hline(y=stop,   line_dash="dash", line_color="#e74c3c", line_width=1.5,
                  annotation_text=f"Stop ₹{stop:,.2f}", annotation_position="right",
                  annotation_font_color="#e74c3c", row=1, col=1)
    fig.add_hline(y=target, line_dash="dash", line_color="#27ae60", line_width=1.5,
                  annotation_text=f"Target ₹{target:,.2f}", annotation_position="right",
                  annotation_font_color="#27ae60", row=1, col=1)

    # ── MACD panel ────────────────────────────────────────────────────────────
    hist_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Hist"],
        name="Histogram", marker_color=hist_colors, opacity=0.7,
        hovertemplate="%{y:.3f}<extra>Hist</extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#1a6faf", width=1.2),
        hovertemplate="%{y:.3f}<extra>MACD</extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SignalLine"],
        name="Signal", line=dict(color="#e67e22", width=1.2),
        hovertemplate="%{y:.3f}<extra>Signal</extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="#30363d", line_width=1, row=2, col=1)

    # ── RSI panel ─────────────────────────────────────────────────────────────
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(231,76,60,0.08)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(46,204,113,0.08)", line_width=0, row=3, col=1)
    fig.add_hline(y=75, line_dash="dash", line_color="#e74c3c", line_width=1, row=3, col=1)
    fig.add_hline(y=40, line_dash="dash", line_color="#2ecc71", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI", line=dict(color="#1a6faf", width=1.5),
        hovertemplate="%{y:.1f}<extra>RSI</extra>",
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        height=680,
        margin=dict(l=10, r=10, t=60, b=10),
        title=dict(
            text=f"<b>{ticker}</b>   ₹{price:,.2f}   "
                 f"<span style='color:{SIG_COLOR[signal]}'>{signal}</span>",
            font=dict(size=18, color="#111111"),
            x=0.01,
        ),
        legend=dict(
            orientation="h", x=0, y=1.02,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#111111"),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        font=dict(color="#111111"),
    )
    # Consistent grid style across all subplots
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="#dee2e6", gridwidth=0.5,
                         zeroline=False, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#dee2e6", gridwidth=0.5,
                         zeroline=False, row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


# ── Build display table ───────────────────────────────────────────────────────
SIGNAL_LABEL = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}

def build_table(results):
    rows = []
    for d in results:
        rows.append({
            "Ticker":    d["name"],
            "Price (₹)": d["price"],
            "RSI":       d["rsi"],
            "vs Slow SMA": d["vs_slow"],
            "MACD Hist": d["macd_hist"],
            "Stop Loss": d["stop"],
            "SL %":      d["stop_pct"],
            "Target":    d["target"],
            "Tgt %":     d["target_pct"],
            "Signal":    SIGNAL_LABEL[d["signal"]],
        })
    return pd.DataFrame(rows)


# ── Portfolio helpers ─────────────────────────────────────────────────────────

NIFTY50_NAMES = sorted([t.replace(".NS", "").replace(".BO", "") for t in NIFTY50])

def portfolio_recommendation(signal, pnl_pct):
    if signal == "BUY":
        return "🟢 Add more — oversold dip in an uptrend"
    if signal == "SELL":
        if pnl_pct is None:
            return "🔴 Consider exiting — bearish signal"
        if pnl_pct > 5:
            return f"🔴 Lock in profits (+{pnl_pct:.1f}%) — trend turning bearish"
        if pnl_pct < -5:
            return f"🔴 Cut losses ({pnl_pct:.1f}%) — trend broken"
        return "🔴 Exit near breakeven — bearish signal"
    # HOLD
    if pnl_pct is not None and pnl_pct > 15:
        return "🟡 Hold — strong gains, keep trailing stop-loss"
    return "🟡 Hold — no clear direction yet"


# ── News fetch ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker: str):
    try:
        items = yf.Ticker(ticker).news or []
        out   = []
        for n in items[:5]:
            # yfinance v0.2.50+ wraps everything under n["content"]
            c = n.get("content", n)
            title = c.get("title", "")
            link  = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url", "#")
            src   = (c.get("provider") or {}).get("displayName", "")
            pub   = c.get("pubDate", "")
            try:
                date_str = datetime.strptime(pub[:10], "%Y-%m-%d").strftime("%d %b") if pub else ""
            except Exception:
                date_str = ""
            if title:
                out.append({"title": title, "link": link, "source": src, "date": date_str})
        return out
    except Exception:
        return []


# ── Plain-English indicator breakdown ─────────────────────────────────────────

def render_indicator_breakdown(data):
    sig        = data["signal"]
    price      = data["price"]
    rsi        = data["rsi"]
    sma_fast   = data["sma_fast"]
    sma_slow   = data["sma_slow"]
    fl         = data["sma_fast_label"]
    sl         = data["sma_slow_label"]
    stop       = data["stop"]
    target     = data["target"]
    stop_pct   = abs(data["stop_pct"])
    target_pct = data["target_pct"]
    macd_hist  = data["macd_hist"]
    above_slow = price > sma_slow
    above_fast = price > sma_fast
    color      = SIG_COLOR[sig]

    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price",  f"₹{price:,.2f}")
    c2.metric("RSI (14)",       f"{rsi}")
    c3.metric(fl,               f"₹{sma_fast:,.0f}")
    c4.metric(sl,               f"₹{sma_slow:,.0f}")
    if sig == "SELL":
        c5.metric("Danger Zone",    f"₹{stop:,.2f}",   delta=f"+{stop_pct:.1f}%")
        c6.metric("Downside Target",f"₹{target:,.2f}", delta=f"-{target_pct:.1f}%", delta_color="inverse")
    else:
        c5.metric("Stop Loss",      f"₹{stop:,.2f}",   delta=f"{stop_pct:.1f}%",   delta_color="inverse")
        c6.metric("Target",         f"₹{target:,.2f}", delta=f"+{target_pct:.1f}%")

    st.write("")

    # ── Overall verdict ───────────────────────────────────────────────────────
    if sig == "BUY":
        verdict = (
            f"**This stock looks like a buying opportunity right now.** "
            f"It has been heavily sold off and is now showing early recovery signs — "
            f"RSI is turning up from oversold levels and momentum is shifting from bearish to bullish. "
            f"This combination often marks a good entry point for a bounce."
        )
    elif sig == "SELL":
        if rsi > 75 and not above_slow:
            verdict = (
                f"**This stock is flashing two warning signals at once.** "
                f"It has risen too fast too soon (overbought) *and* has slipped below its long-term average. "
                f"Consider exiting or reducing your position."
            )
        elif rsi > 75:
            verdict = (
                f"**This stock has rallied very hard, very quickly.** "
                f"At an RSI of {rsi}, it's in 'overbought' territory — like a rubber band stretched too far. "
                f"It may be due for a pullback. Avoid chasing it higher."
            )
        else:
            verdict = (
                f"**The long-term trend has turned negative.** "
                f"The stock has dropped below its {sl} of ₹{sma_slow:,.0f}, which acts like the floor of the market. "
                f"When a stock breaks this level, it often continues lower. Consider exiting."
            )
    else:
        verdict = (
            f"**No strong signal right now — best to hold and wait.** "
            f"The stock is moving normally without any extreme buying or selling pressure. "
            f"It's like waiting at a traffic signal — not the right moment to accelerate or brake hard."
        )

    st.markdown(f"### {SIGNAL_LABEL[sig]}  —  What this means")
    st.markdown(f"<p style='font-size:16px'>{verdict}</p>", unsafe_allow_html=True)
    st.divider()

    # ── Detailed plain-English explanations ───────────────────────────────────
    st.markdown("#### 📊 Breaking it down")

    # RSI
    if rsi < 40:
        rsi_text = (
            f"**Momentum (RSI {rsi}) — Oversold 🟢**  \n"
            f"The RSI measures how fast a stock is moving. At {rsi}, it's below 40, which means "
            f"the stock has been sold off heavily and is now in 'oversold' territory. "
            f"Think of it like a spring compressed too far down — it often bounces back. "
            f"A BUY signal also requires the RSI to be rising (recovery confirmed) and MACD momentum turning up."
        )
    elif rsi > 75:
        rsi_text = (
            f"**Momentum (RSI {rsi}) — Overbought 🔴**  \n"
            f"At {rsi}, the RSI is above 75, meaning the stock has been bought aggressively in a short time. "
            f"Think of it like a rubber band stretched too far up — it tends to snap back. "
            f"Buyers may start taking profits, pushing the price down."
        )
    else:
        rsi_text = (
            f"**Momentum (RSI {rsi}) — Neutral 🟡**  \n"
            f"At {rsi}, the RSI sits comfortably between 40 and 75 — the 'neutral zone'. "
            f"There's no extreme buying or selling pressure. "
            f"The stock is simply moving normally, with neither bulls nor bears in full control."
        )
    st.markdown(rsi_text)

    # SMA trend
    if above_slow and above_fast:
        sma_text = (
            f"**Trend ({fl} & {sl}) — Uptrend 🟢**  \n"
            f"At ₹{price:,.0f}, the stock is above both its short-term average (₹{sma_fast:,.0f}) "
            f"and long-term average (₹{sma_slow:,.0f}). "
            f"Think of these averages as the 'normal' price level. Being above both means "
            f"the stock is trending healthily upward on both short and long timescales."
        )
    elif above_slow and not above_fast:
        sma_text = (
            f"**Trend ({fl} & {sl}) — Mixed ⚠️**  \n"
            f"The stock has dipped below its short-term average (₹{sma_fast:,.0f}) but is still "
            f"above its long-term average (₹{sma_slow:,.0f}). "
            f"The big picture trend is intact, but there's short-term weakness — a yellow flag worth watching."
        )
    else:
        sma_text = (
            f"**Trend ({fl} & {sl}) — Downtrend 🔴**  \n"
            f"At ₹{price:,.0f}, the stock has fallen below its long-term average of ₹{sma_slow:,.0f}. "
            f"This long-term average acts like a floor — once a stock falls through it, "
            f"it often continues lower. This is one of the strongest sell signals in technical analysis."
        )
    st.markdown(sma_text)

    # MACD
    if macd_hist > 0:
        macd_text = (
            f"**Momentum trend (MACD +{macd_hist}) — Building 🟢**  \n"
            f"The MACD histogram is positive, meaning short-term buying pressure is picking up. "
            f"Think of it as the wind moving in the right direction — it supports a move upward."
        )
    else:
        macd_text = (
            f"**Momentum trend (MACD {macd_hist}) — Fading 🔴**  \n"
            f"The MACD histogram is negative, meaning short-term momentum is losing steam. "
            f"Think of it as a car slowing down even before you see the brake lights — "
            f"sellers are quietly gaining the upper hand."
        )
    st.markdown(macd_text)

    st.divider()

    # ── Risk management ───────────────────────────────────────────────────────
    st.markdown("#### 🛡️ Risk management")
    if sig == "SELL":
        st.markdown(
            f"- **Danger zone above ₹{stop:,.2f}** (+{stop_pct:.1f}% from current price)  \n"
            f"  This is the nearest resistance level. If the stock rallies back above this, the bearish case weakens — re-evaluate before shorting or holding a short position."
        )
        st.markdown(
            f"- **Downside target ₹{target:,.2f}** (-{target_pct:.1f}% from current price)  \n"
            f"  This is the nearest support level the stock could fall to. If you're holding this stock, this is a realistic exit point to watch."
        )
    else:
        st.markdown(
            f"- **Stop Loss at ₹{stop:,.2f}** ({stop_pct:.1f}% below current price)  \n"
            f"  This is the most recent support level — the lowest price the stock has traded at recently. "
            f"If it breaks below here, the buyers are no longer holding. Exit to protect your capital."
        )
        st.markdown(
            f"- **Target at ₹{target:,.2f}** (+{target_pct:.1f}% from current price)  \n"
            f"  This is the most recent resistance level — the highest price the stock has reached recently. "
            f"Breaking above it cleanly would be a positive signal. This gives a **reward-to-risk ratio of at least 1.5:1**."
        )

    st.divider()

    # ── Latest news ───────────────────────────────────────────────────────────
    st.markdown("#### 📰 Latest news")
    ticker_full = data["ticker"]
    with st.spinner("Fetching latest news …"):
        news = fetch_news(ticker_full)

    if news:
        for item in news:
            date_tag = f" · {item['date']}" if item["date"] else ""
            src_tag  = f" · *{item['source']}*" if item["source"] else ""
            st.markdown(f"- [{item['title']}]({item['link']}){date_tag}{src_tag}")
    else:
        st.caption("No recent news found for this stock.")


def render_dashboard(results):
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for d in results:
        counts[d["signal"]] += 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks Loaded", len(results), f"of {len(NIFTY50)}")
    m2.metric("🟢 BUY",  counts["BUY"])
    m3.metric("🟡 HOLD", counts["HOLD"])
    m4.metric("🔴 SELL", counts["SELL"])

    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}  •  Data cached for 15 min  •  Click any row to view chart")
    st.divider()

    # Search bar
    search = st.text_input("🔍 Search stock", placeholder="e.g. RELIANCE, INFY, TCS …", label_visibility="collapsed")
    filtered = [r for r in results if search.upper() in r["name"]] if search else results
    if search and not filtered:
        st.warning(f"No stocks matched '{search}'.")
        return

    df_table = build_table(filtered)
    event = st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Ticker":      st.column_config.TextColumn("Ticker", width="small"),
            "Price (₹)":   st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
            "RSI":         st.column_config.NumberColumn("RSI", format="%.1f"),
            "vs Slow SMA": st.column_config.NumberColumn("vs Slow SMA", format="%.1f%%"),
            "MACD Hist":   st.column_config.NumberColumn("MACD Hist", format="%.2f"),
            "Stop Loss":   st.column_config.NumberColumn("Stop Loss ₹", format="₹%.2f"),
            "SL %":        st.column_config.NumberColumn("SL %", format="%.1f%%"),
            "Target":      st.column_config.NumberColumn("Target ₹", format="₹%.2f"),
            "Tgt %":       st.column_config.NumberColumn("Tgt %", format="+%.1f%%"),
            "Signal":      st.column_config.TextColumn("Signal", width="small"),
        },
    )

    selected_rows = event.selection.rows
    if selected_rows:
        idx  = selected_rows[0]
        data = filtered[idx]
        st.divider()
        st.plotly_chart(build_chart(data), use_container_width=True)

        with st.expander("📊 Indicator breakdown", expanded=True):
            render_indicator_breakdown(data)
    else:
        st.info("👆 Click any row in the table above to view its price chart.", icon="📊")


def render_portfolio(results):
    lookup = {d["name"]: d for d in results}

    st.subheader("Enter your holdings")
    st.caption("Add the stocks you own, how many shares, and your average buy price. Buy price is optional but enables P&L tracking.")

    # Init session state
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(
            [{"Ticker": "RELIANCE", "Shares": 10.0, "Avg Buy Price (₹)": 0.0}]
        )

    edited = st.data_editor(
        st.session_state.portfolio,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.SelectboxColumn(
                "Ticker", options=NIFTY50_NAMES, required=True
            ),
            "Shares": st.column_config.NumberColumn(
                "Shares", min_value=0.0, step=1.0, required=True
            ),
            "Avg Buy Price (₹)": st.column_config.NumberColumn(
                "Avg Buy Price (₹)", min_value=0.0, step=0.5,
                help="Leave 0 if unknown — P&L columns will be hidden"
            ),
        },
        hide_index=True,
    )

    if st.button("📊 Analyse Portfolio", type="primary", use_container_width=False):
        st.session_state.portfolio = edited
        st.session_state.show_analysis = True

    if not st.session_state.get("show_analysis"):
        return

    portfolio = st.session_state.portfolio.dropna(subset=["Ticker"])
    portfolio = portfolio[portfolio["Shares"] > 0]
    if portfolio.empty:
        st.warning("Add at least one holding with shares > 0.")
        return

    has_buy_price = portfolio["Avg Buy Price (₹)"].gt(0).any()

    # Build analysis rows
    rows = []
    for _, row in portfolio.iterrows():
        name = row["Ticker"]
        if name not in lookup:
            continue
        d          = lookup[name]
        shares     = row["Shares"]
        buy_price  = row["Avg Buy Price (₹)"]
        cur_price  = d["price"]
        cur_value  = round(cur_price * shares, 2)

        pnl_amt  = round((cur_price - buy_price) * shares, 2) if buy_price > 0 else None
        pnl_pct  = round((cur_price - buy_price) / buy_price * 100, 2) if buy_price > 0 else None
        rec      = portfolio_recommendation(d["signal"], pnl_pct)

        stop       = d["stop"]
        target     = d["target"]
        risk_ps    = round(cur_price - stop, 2)
        reward_ps  = round(target - cur_price, 2)
        total_risk = round(risk_ps * shares, 2)
        total_rwd  = round(reward_ps * shares, 2)

        r = {
            "Ticker":        name,
            "Shares":        shares,
            "Current Price": cur_price,
            "Current Value": cur_value,
            "Stop Loss":     stop,
            "SL %":          d["stop_pct"],
            "Target":        target,
            "Tgt %":         d["target_pct"],
            "Risk (₹)":      total_risk,
            "Reward (₹)":    total_rwd,
            "Signal":        SIGNAL_LABEL[d["signal"]],
            "Action":        rec,
        }
        if has_buy_price:
            r["Buy Price"] = buy_price if buy_price > 0 else None
            r["P&L (₹)"]  = pnl_amt
            r["P&L (%)"]  = pnl_pct
        rows.append(r)

    if not rows:
        st.error("None of your tickers matched NIFTY 50 data. Check spelling.")
        return

    df_port = pd.DataFrame(rows)

    st.divider()
    st.subheader("Portfolio Analysis")

    # Summary metrics
    total_value = df_port["Current Value"].sum()
    sig_counts  = df_port["Signal"].value_counts()

    col_config = {
        "Ticker":        st.column_config.TextColumn("Ticker"),
        "Shares":        st.column_config.NumberColumn("Shares", format="%.0f"),
        "Current Price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
        "Current Value": st.column_config.NumberColumn("Current Value", format="₹%.2f"),
        "Stop Loss":     st.column_config.NumberColumn("Stop Loss ₹", format="₹%.2f"),
        "SL %":          st.column_config.NumberColumn("SL %", format="%.1f%%"),
        "Target":        st.column_config.NumberColumn("Target ₹", format="₹%.2f"),
        "Tgt %":         st.column_config.NumberColumn("Tgt %", format="+%.1f%%"),
        "Risk (₹)":      st.column_config.NumberColumn("Total Risk ₹", format="₹%.0f"),
        "Reward (₹)":    st.column_config.NumberColumn("Total Reward ₹", format="₹%.0f"),
        "Signal":        st.column_config.TextColumn("Signal"),
        "Action":        st.column_config.TextColumn("Action", width="large"),
    }

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Portfolio Value", f"₹{total_value:,.0f}")

    if has_buy_price:
        valid_pnl = df_port["P&L (₹)"].dropna()
        total_pnl = valid_pnl.sum()
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        s2.metric("Total P&L", f"₹{total_pnl:,.0f}", delta_color=pnl_color)
        col_config["Buy Price"] = st.column_config.NumberColumn("Buy Price", format="₹%.2f")
        col_config["P&L (₹)"]  = st.column_config.NumberColumn("P&L (₹)", format="₹%.2f")
        col_config["P&L (%)"]  = st.column_config.NumberColumn("P&L (%)", format="%.2f%%")
    else:
        s2.metric("Holdings", len(rows))

    s3.metric("🔴 Sell signals", sig_counts.get("🔴 SELL", 0))
    s4.metric("🟢 Buy signals",  sig_counts.get("🟢 BUY",  0))

    st.dataframe(df_port, use_container_width=True, hide_index=True, column_config=col_config)

    # Urgent alerts
    sell_holds = df_port[df_port["Signal"] == "🔴 SELL"]
    buy_holds  = df_port[df_port["Signal"] == "🟢 BUY"]

    if not sell_holds.empty:
        st.divider()
        st.markdown("### ⚠️ Stocks requiring attention")
        for _, r in sell_holds.iterrows():
            st.error(f"**{r['Ticker']}** — {r['Action']}", icon="🔴")

    if not buy_holds.empty:
        st.divider()
        st.markdown("### 💡 Opportunities in your portfolio")
        for _, r in buy_holds.iterrows():
            st.success(f"**{r['Ticker']}** — {r['Action']}", icon="🟢")

    # Chart for selected holding
    st.divider()
    st.subheader("View chart for a holding")
    tickers_in_portfolio = df_port["Ticker"].tolist()
    chosen = st.selectbox("Select stock", tickers_in_portfolio)
    if chosen and chosen in lookup:
        st.plotly_chart(build_chart(lookup[chosen]), use_container_width=True)


# ── Backtest renderer ─────────────────────────────────────────────────────────
def render_backtest(results, tf_key):
    cfg = BT_CONFIGS[tf_key]

    st.info(
        "**What is this?**  \n"
        "This tab replays every BUY signal generated by the app over the past "
        f"{'3 years' if cfg['interval']=='1d' else '15 years' if cfg['interval']=='1wk' else '60 days'} "
        f"of {tf_key} data and checks: did the stock reach its target price before hitting the stop loss?  \n"
        "A higher quality score means the signals have been more reliable historically.  \n"
        "⚠️ Past performance does not guarantee future results."
    )

    valid = [r for r in results if r["n_trades"] >= 3 and r["quality_score"] is not None]
    if not valid:
        st.warning("Not enough trades found to score signals. Try a different timeframe.")
        return

    avg_q      = sum(r["quality_score"] for r in valid) / len(valid)
    reliable   = sum(1 for r in valid if r["win_rate"] > 0.55)
    marginal   = sum(1 for r in valid if 0.45 <= r["win_rate"] <= 0.55)
    unreliable = sum(1 for r in valid if r["win_rate"] < 0.45)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Stocks Scored", len(valid))
    m2.metric("Avg Signal Quality", f"{avg_q:.0f} / 100")
    m3.metric("🟢 Reliable (>55%)", reliable)
    m4.metric("🟡 Marginal (45–55%)", marginal)
    m5.metric("🔴 Unreliable (<45%)", unreliable)

    st.divider()

    def _wr_label(wr, n):
        if n < 3:
            return "—"
        pct = wr * 100
        if wr > 0.55:    return f"🟢 {pct:.1f}%"
        elif wr >= 0.45: return f"🟡 {pct:.1f}%"
        else:            return f"🔴 {pct:.1f}%"

    rows = []
    for r in results:
        n = r["n_trades"]
        rows.append({
            "Ticker":     r["ticker"].replace(".NS", "").replace(".BO", ""),
            "Trades":     n,
            "Win Rate":   _wr_label(r["win_rate"], n),
            "Avg Win":    f"+{r['avg_win']:.2f}%" if n >= 3 else "—",
            "Avg Loss":   f"{r['avg_loss']:.2f}%"  if n >= 3 else "—",
            "Expectancy": f"{r['expectancy']:+.2f}%" if n >= 3 else "—",
            "Quality":    r["quality_score"] if n >= 3 else None,
        })

    df_bt = pd.DataFrame(rows)
    st.dataframe(
        df_bt,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker":     st.column_config.TextColumn("Ticker",    width="small"),
            "Trades":     st.column_config.NumberColumn("Trades",  format="%d", width="small"),
            "Win Rate":   st.column_config.TextColumn("Win Rate",  width="small"),
            "Avg Win":    st.column_config.TextColumn("Avg Win",   width="small"),
            "Avg Loss":   st.column_config.TextColumn("Avg Loss",  width="small"),
            "Expectancy": st.column_config.TextColumn("Expectancy"),
            "Quality":    st.column_config.ProgressColumn(
                "Signal Quality (0–100)", min_value=0, max_value=100, width="medium"
            ),
        },
    )

    skipped = [r for r in results if r["n_trades"] < 3]
    if skipped:
        names = ", ".join(r["ticker"].replace(".NS","") for r in skipped)
        st.caption(
            f"⚠️ {len(skipped)} stocks had fewer than 3 historical BUY signals and are unscored "
            f"({names}). This means RSI < 40 rising + MACD rising + price above SMA {cfg['sma_slow']} was rarely triggered for them."
        )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.title("📈 NIFTY 50 Dashboard")
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Timeframe selector
    tf_key = st.radio(
        "Timeframe",
        options=list(TIMEFRAMES.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="tf_selector",
    )
    st.caption(f"**{tf_key}** — {TIMEFRAMES[tf_key]['desc']}")

    with st.spinner("Fetching live data for all 50 stocks …"):
        results = fetch_all(tf_key)

    # Invalidate cache if data schema is outdated (missing new keys)
    if results and "stop" not in results[0]:
        st.cache_data.clear()
        st.rerun()

    if not results:
        st.error("Failed to fetch data. Please check your internet connection and try again.")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Market Dashboard", "💼 My Portfolio", "🔬 Backtest"])

    with tab1:
        render_dashboard(results)

    with tab2:
        render_portfolio(results)

    with tab3:
        if "bt_results" not in st.session_state or st.session_state.get("bt_tf") != tf_key:
            st.markdown("### 🔬 Signal Reliability Backtest")
            st.write(
                "Run a historical simulation to see how reliable the BUY signals have been. "
                "For each signal in the past, it checks whether the stock hit its target before its stop loss."
            )
            if st.button("▶️ Run Backtest", type="primary"):
                with st.spinner(f"Backtesting all 50 stocks on {tf_key} data … this takes ~15–30 seconds"):
                    st.session_state.bt_results = run_backtest(tf_key)
                    st.session_state.bt_tf      = tf_key
                st.rerun()
        else:
            render_backtest(st.session_state.bt_results, tf_key)
            if st.button("🔄 Re-run Backtest"):
                del st.session_state["bt_results"]
                st.rerun()


if __name__ == "__main__":
    main()
