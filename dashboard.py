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


def get_signal(price, rsi, sma50, sma200, macd_hist):
    above_200 = price > sma200
    if rsi < 35 and above_200:
        return "BUY"
    elif rsi > 75 or not above_200:
        return "SELL"
    return "HOLD"


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_single(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if df.empty or len(df) < 200:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"].dropna()
        price  = round(float(close.iloc[-1]), 2)
        sma50  = round(close.rolling(50).mean().iloc[-1], 2)
        sma200 = round(close.rolling(200).mean().iloc[-1], 2)
        rsi_s  = calc_rsi(close)
        rsi    = round(float(rsi_s.iloc[-1]), 1)
        ml, sl, mh = calc_macd(close)
        macd_hist  = round(float(mh.iloc[-1]), 2)
        signal     = get_signal(price, rsi, sma50, sma200, macd_hist)

        # Attach full series for charting
        df = df.copy()
        df["SMA50"]  = close.rolling(50).mean()
        df["SMA200"] = close.rolling(200).mean()
        df["RSI"]    = rsi_s
        df["MACD"], df["SignalLine"], df["Hist"] = calc_macd(close)

        return {
            "ticker":    ticker,
            "name":      ticker.replace(".NS", "").replace(".BO", ""),
            "df":        df,
            "price":     price,
            "rsi":       rsi,
            "sma50":     sma50,
            "sma200":    sma200,
            "macd_hist": macd_hist,
            "vs200":     round((price - sma200) / sma200 * 100, 1),
            "signal":    signal,
        }
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_all():
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_single, t): t for t in NIFTY50}
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
    df     = data["df"].dropna(subset=["SMA200"])
    ticker = data["name"]
    signal = data["signal"]
    price  = data["price"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Price  •  SMA 50  •  SMA 200", "MACD", "RSI (14)"),
    )

    # ── Price panel ───────────────────────────────────────────────────────────
    # Green fill: price above SMA200
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA200"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", fill="tonexty",
        fillcolor="rgba(46,204,113,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Red fill: SMA200 above price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA200"],
        mode="lines", fill="tonexty",
        fillcolor="rgba(231,76,60,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # SMA lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA50"],
        name="SMA 50", line=dict(color="#e67e22", width=1.5, dash="dot"),
        hovertemplate="SMA50: ₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA200"],
        name="SMA 200", line=dict(color="#8e44ad", width=1.5, dash="dot"),
        hovertemplate="SMA200: ₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Price line (on top)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Price", line=dict(color="#1a6faf", width=2),
        hovertemplate="₹%{y:,.2f}<extra>Price</extra>",
    ), row=1, col=1)

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
    fig.add_hrect(y0=0,  y1=35,  fillcolor="rgba(46,204,113,0.08)", line_width=0, row=3, col=1)
    fig.add_hline(y=75, line_dash="dash", line_color="#e74c3c", line_width=1, row=3, col=1)
    fig.add_hline(y=35, line_dash="dash", line_color="#2ecc71", line_width=1, row=3, col=1)
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
            "SMA 50":    d["sma50"],
            "SMA 200":   d["sma200"],
            "vs SMA200": d["vs200"],
            "MACD Hist": d["macd_hist"],
            "Signal":    SIGNAL_LABEL[d["signal"]],
        })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Header
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.title("📈 NIFTY 50 Dashboard")
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Fetch data
    with st.spinner("Fetching live data for all 50 stocks …"):
        results = fetch_all()

    if not results:
        st.error("Failed to fetch data. Please check your internet connection and try again.")
        return

    # Summary metrics
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

    # Table
    df_table = build_table(results)

    event = st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Ticker":    st.column_config.TextColumn("Ticker", width="small"),
            "Price (₹)": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
            "RSI":       st.column_config.NumberColumn("RSI", format="%.1f"),
            "SMA 50":    st.column_config.NumberColumn("SMA 50",  format="₹%.0f"),
            "SMA 200":   st.column_config.NumberColumn("SMA 200", format="₹%.0f"),
            "vs SMA200": st.column_config.NumberColumn("vs SMA200", format="%.1f%%"),
            "MACD Hist": st.column_config.NumberColumn("MACD Hist", format="%.2f"),
            "Signal":    st.column_config.TextColumn("Signal", width="small"),
        },
    )

    # Chart on row click
    selected_rows = event.selection.rows
    if selected_rows:
        idx  = selected_rows[0]
        data = results[idx]
        st.divider()
        fig = build_chart(data)
        st.plotly_chart(fig, use_container_width=True)

        # Indicator breakdown below chart
        with st.expander("📊 Indicator breakdown", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Current Price", f"₹{data['price']:,.2f}")
            c2.metric("RSI (14)",      f"{data['rsi']}")
            c3.metric("SMA 50",        f"₹{data['sma50']:,.0f}")
            c4.metric("SMA 200",       f"₹{data['sma200']:,.0f}")
            c5.metric("vs SMA 200",    f"{data['vs200']:+.1f}%",
                      delta_color="normal" if data["vs200"] >= 0 else "inverse")

            st.write("")
            sig   = data["signal"]
            color = SIG_COLOR[sig]
            above = data["price"] > data["sma200"]

            st.markdown(f"**Decision: <span style='color:{color}'>{SIGNAL_LABEL[sig]}</span>**", unsafe_allow_html=True)
            if sig == "BUY":
                st.markdown(f"- RSI **{data['rsi']}** < 35 → oversold territory")
                st.markdown(f"- Price **₹{data['price']:,}** > SMA200 **₹{data['sma200']:,}** → long-term uptrend intact")
            elif sig == "SELL":
                if data["rsi"] > 75:
                    st.markdown(f"- RSI **{data['rsi']}** > 75 → overbought territory")
                if not above:
                    st.markdown(f"- Price **₹{data['price']:,}** < SMA200 **₹{data['sma200']:,}** → long-term downtrend")
            else:
                st.markdown(f"- RSI **{data['rsi']}** in neutral zone (35–75)")
                st.markdown(f"- Price is **{'above' if above else 'below'}** SMA200 → awaiting clearer signal")
    else:
        st.info("👆 Click any row in the table above to view its price chart.", icon="📊")


if __name__ == "__main__":
    main()
