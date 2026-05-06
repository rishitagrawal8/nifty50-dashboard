# NIFTY 50 Dashboard

A real-time technical analysis dashboard for all 50 NIFTY stocks — live data, buy/sell/hold signals, and interactive charts.

## Features

- **Live data** via `yfinance` — 1 year of daily OHLCV for all 50 NIFTY stocks
- **Technical indicators** — RSI (14), SMA 50, SMA 200, MACD
- **Buy / Sell / Hold signals** based on indicator logic
- **Web dashboard** — sortable table, click any stock to view its interactive chart
- **CLI tool** — run analysis and view charts from the terminal

## Signal Logic

| Signal | Condition |
|--------|-----------|
| 🟢 BUY  | RSI < 35 AND Price > SMA 200 (oversold dip in an uptrend) |
| 🔴 SELL | RSI > 75 OR Price < SMA 200 (overbought or trend breakdown) |
| 🟡 HOLD | All other cases — neutral or conflicting signals |

## Getting Started

```bash
pip install -r requirements.txt
```

### Web Dashboard
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`. Click any row to see a 3-panel chart (Price + SMAs, MACD, RSI).

### CLI
```bash
# Single stock — report + matplotlib chart
python3 nifty_analyzer.py RELIANCE

# All 50 stocks as a table
python3 nifty_analyzer.py --all

# Interactive mode
python3 nifty_analyzer.py
```

## Chart Panels

| Panel | Contents |
|-------|----------|
| Price | Close price + SMA 50 + SMA 200, shaded green/red vs SMA 200 |
| MACD  | MACD line, signal line, histogram (green = bullish, red = bearish) |
| RSI   | RSI (14) with overbought (>75) and oversold (<35) zones shaded |
