#!/usr/bin/env python3
"""
NIFTY 50 Stock Signal Analyzer
  python3 nifty_analyzer.py              → interactive mode
  python3 nifty_analyzer.py RELIANCE     → single stock report + chart
  python3 nifty_analyzer.py --all        → full NIFTY 50 table
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("MacOSX")          # native macOS window, no Tk needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
from datetime import datetime

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

# ── Indicator calculations ────────────────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return round((100 - 100 / (1 + rs)).iloc[-1], 2)


def calc_sma(close: pd.Series, window: int) -> float:
    return round(close.rolling(window).mean().iloc[-1], 2)


def calc_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line  = ema12 - ema26
    sig   = line.ewm(span=9, adjust=False).mean()
    hist  = line - sig
    return round(line.iloc[-1], 4), round(sig.iloc[-1], 4), round(hist.iloc[-1], 4)


# ── Decision logic ────────────────────────────────────────────────────────────

def get_signal(price, rsi, sma50, sma200, macd_hist):
    """
    BUY  — RSI < 35 AND price > SMA200  (oversold dip in uptrend)
    SELL — RSI > 75 OR  price < SMA200  (overbought or trend break)
    HOLD — neutral / conflicting
    """
    above_200 = price > sma200
    reasons   = []

    if rsi < 35 and above_200:
        signal = "BUY"
        reasons.append(f"RSI {rsi} < 35 → oversold")
        reasons.append(f"Price > SMA200 ₹{sma200} → uptrend intact")
        if macd_hist > 0:
            reasons.append("MACD histogram positive → momentum confirming")
    elif rsi > 75 or not above_200:
        signal = "SELL"
        if rsi > 75:
            reasons.append(f"RSI {rsi} > 75 → overbought")
        if not above_200:
            reasons.append(f"Price < SMA200 ₹{sma200} → downtrend")
        if macd_hist < 0:
            reasons.append("MACD histogram negative → bearish momentum")
    else:
        signal = "HOLD"
        reasons.append("RSI in neutral zone (35–75)")
        reasons.append(f"Price {'above' if above_200 else 'below'} SMA200 → trend {'positive' if above_200 else 'negative'}")
        reasons.append("Awaiting clearer entry/exit signal")

    return signal, reasons


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch(ticker: str):
    """Download 1-year OHLCV and return computed indicators, or None on error."""
    ticker = ticker.upper().strip()
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker += ".NS"

    # Use Ticker.history() — thread-safe unlike yf.download() in parallel contexts
    df = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    if len(close) < 200:
        return None

    price      = round(float(close.iloc[-1]), 2)
    rsi        = calc_rsi(close)
    sma50      = calc_sma(close, 50)
    sma200     = calc_sma(close, 200)
    ml, sl, mh = calc_macd(close)
    signal, reasons = get_signal(price, rsi, sma50, sma200, mh)

    # also return full df for charting
    df["SMA50"]  = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    return {
        "ticker": ticker,
        "df":     df,
        "price":  price,
        "rsi":    rsi,
        "sma50":  sma50,
        "sma200": sma200,
        "macd":   ml,
        "signal_line": sl,
        "macd_hist": mh,
        "signal": signal,
        "reasons": reasons,
    }


# ── Price chart ───────────────────────────────────────────────────────────────

SIG_COLOR = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#f39c12"}

def show_chart(data: dict) -> None:
    df     = data["df"]
    ticker = data["ticker"]
    signal = data["signal"]

    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    fig.suptitle(
        f"{ticker}  —  ₹{data['price']}   [{signal}]",
        color=SIG_COLOR[signal], fontsize=16, fontweight="bold", y=0.97,
    )

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[5, 2, 2], hspace=0.06)
    ax_price = fig.add_subplot(gs[0])
    ax_macd  = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi   = fig.add_subplot(gs[2], sharex=ax_price)

    for ax in (ax_price, ax_macd, ax_rsi):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.yaxis.label.set_color("#8b949e")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(color="#21262d", linewidth=0.6, linestyle="--")

    close = df["Close"]

    # ── Price panel ──────────────────────────────────────────────────────────
    ax_price.plot(df.index, close,        color="#58a6ff", linewidth=1.2, label="Price")
    ax_price.plot(df.index, df["SMA50"],  color="#f0883e", linewidth=1,   label="SMA 50",  linestyle="--")
    ax_price.plot(df.index, df["SMA200"], color="#bc8cff", linewidth=1,   label="SMA 200", linestyle="--")
    ax_price.fill_between(df.index, close, df["SMA200"],
                          where=(close >= df["SMA200"]),
                          alpha=0.08, color="#2ecc71", interpolate=True)
    ax_price.fill_between(df.index, close, df["SMA200"],
                          where=(close < df["SMA200"]),
                          alpha=0.08, color="#e74c3c", interpolate=True)
    ax_price.set_ylabel("Price (₹)", color="#8b949e")
    ax_price.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#c9d1d9", fontsize=8)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── MACD panel ───────────────────────────────────────────────────────────
    ax_macd.plot(df.index, df["MACD"],   color="#58a6ff", linewidth=1,   label="MACD")
    ax_macd.plot(df.index, df["Signal"], color="#f0883e", linewidth=1,   label="Signal")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["Hist"]]
    ax_macd.bar(df.index, df["Hist"], color=colors, alpha=0.6, width=1)
    ax_macd.axhline(0, color="#30363d", linewidth=0.8)
    ax_macd.set_ylabel("MACD", color="#8b949e")
    ax_macd.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=7)
    plt.setp(ax_macd.get_xticklabels(), visible=False)

    # ── RSI panel ────────────────────────────────────────────────────────────
    ax_rsi.plot(df.index, df["RSI"], color="#58a6ff", linewidth=1, label="RSI (14)")
    ax_rsi.axhline(75, color="#e74c3c", linewidth=0.8, linestyle="--", label="Overbought 75")
    ax_rsi.axhline(35, color="#2ecc71", linewidth=0.8, linestyle="--", label="Oversold 35")
    ax_rsi.fill_between(df.index, df["RSI"], 75,
                        where=(df["RSI"] >= 75), alpha=0.15, color="#e74c3c", interpolate=True)
    ax_rsi.fill_between(df.index, df["RSI"], 35,
                        where=(df["RSI"] <= 35), alpha=0.15, color="#2ecc71", interpolate=True)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", color="#8b949e")
    ax_rsi.set_xlabel("Date", color="#8b949e")
    ax_rsi.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=7)
    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_rsi.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_rsi.get_xticklabels(), rotation=30, ha="right", color="#8b949e")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ── Single-stock report ───────────────────────────────────────────────────────

ANSI = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m", "RST": "\033[0m"}

def print_report(data: dict) -> None:
    signal = data["signal"]
    c, r   = ANSI[signal], ANSI["RST"]
    sep    = "─" * 56

    print(f"\n{sep}")
    print(f"  SIGNAL REPORT  »  {data['ticker']}")
    print(sep)
    print(f"  Current Price    : ₹{data['price']}")
    print(f"  RSI (14)         : {data['rsi']}")
    print(f"  SMA 50           : ₹{data['sma50']}")
    print(f"  SMA 200          : ₹{data['sma200']}")
    print(f"  MACD Line        : {data['macd']}")
    print(f"  Signal Line      : {data['signal_line']}")
    print(f"  MACD Histogram   : {data['macd_hist']}")
    print(sep)
    print(f"  DECISION         : {c}{signal}{r}")
    print(sep)
    print("  Reasoning:")
    for reason in data["reasons"]:
        print(f"    • {reason}")
    print(sep)


# ── Full NIFTY 50 table ───────────────────────────────────────────────────────

def _sig_label(s):
    return {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}[s]

def show_all_table() -> None:
    print(f"\nFetching data for all {len(NIFTY50)} NIFTY 50 stocks (parallel) …\n")

    results = []
    errors  = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch, t): t for t in NIFTY50}
        done = 0
        for fut in as_completed(futures):
            done += 1
            ticker = futures[fut]
            try:
                data = fut.result()
                if data:
                    results.append(data)
                else:
                    errors.append(ticker)
            except Exception:
                errors.append(ticker)
            print(f"\r  Progress: {done}/{len(NIFTY50)}  ", end="", flush=True)

    print()

    # Sort: BUY first, then HOLD, then SELL; within group by RSI asc
    order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    results.sort(key=lambda d: (order[d["signal"]], d["rsi"]))

    rows = []
    for d in results:
        name   = d["ticker"].replace(".NS", "")
        vs200  = f"{((d['price'] - d['sma200']) / d['sma200'] * 100):+.1f}%"
        rows.append([
            name,
            f"₹{d['price']:,.2f}",
            f"{d['rsi']:.1f}",
            f"₹{d['sma50']:,.0f}",
            f"₹{d['sma200']:,.0f}",
            vs200,
            f"{d['macd_hist']:+.2f}",
            _sig_label(d["signal"]),
        ])

    headers = ["Ticker", "Price", "RSI", "SMA50", "SMA200", "vs SMA200", "MACD Hist", "Signal"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline", colalign=(
        "left", "right", "right", "right", "right", "right", "right", "center"
    )))

    # Summary counts
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for d in results:
        counts[d["signal"]] += 1

    print(f"\n  Summary  →  "
          f"{ANSI['BUY']}BUY {counts['BUY']}{ANSI['RST']}  |  "
          f"{ANSI['HOLD']}HOLD {counts['HOLD']}{ANSI['RST']}  |  "
          f"{ANSI['SELL']}SELL {counts['SELL']}{ANSI['RST']}"
          f"   (as of {datetime.now().strftime('%d %b %Y %H:%M')})")

    if errors:
        print(f"\n  Failed to fetch: {', '.join(errors)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if "--all" in args:
        show_all_table()
        return

    if args:
        for t in args:
            print(f"\nFetching data for {t.upper()} …")
            data = fetch(t)
            if data:
                print_report(data)
                show_chart(data)
            else:
                print(f"  ERROR: No data for '{t}'. Check the ticker symbol.")
        return

    # Interactive
    print("=" * 56)
    print("   NIFTY 50 Stock Analyzer")
    print("   Commands: <ticker>  |  all  |  quit")
    print("   Example : RELIANCE  or  INFY.NS")
    print("=" * 56)

    while True:
        try:
            cmd = input("\nEnter ticker / 'all' / 'quit': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if cmd.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        elif cmd.lower() == "all":
            show_all_table()
        elif cmd:
            print(f"\nFetching data for {cmd.upper()} …")
            data = fetch(cmd)
            if data:
                print_report(data)
                show_chart(data)
            else:
                print(f"  ERROR: No data for '{cmd}'. Check the ticker symbol.")


if __name__ == "__main__":
    main()
