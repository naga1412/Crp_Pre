# SMC Pro — Crypto & Stock Analyzer

A professional-grade technical analysis platform with Smart Money Concepts (SMC), real-time price streaming, AI-powered candle prediction, and live market news.

---

## Features

- **Live Charts** — Real-time candlestick charts via Binance WebSocket (crypto) and yFinance polling (stocks)
- **Smart Money Concepts** — Order Blocks, Fair Value Gaps (FVG), Break of Structure (BOS/CHoCH), Buy/Sell-Side Liquidity levels
- **Wyckoff Analysis** — Accumulation, Markup, Distribution, Markdown phase detection
- **Multi-Timeframe (MTF) Analysis** — HTF-weighted bias scoring (1w/1d dominate over 1m)
- **Advanced Pattern Detection** — Triple Top/Bottom, Head & Shoulders, Inverse H&S, Rising/Falling Wedge, Ascending/Descending/Symmetrical Triangle, Rounding Bottom/Top
- **AI Candle Prediction** — Future candle projections (neon yellow) anchored to last real candle, count configurable per timeframe
- **Live Market News** — 11 RSS sources (CNBC, Yahoo Finance, MarketWatch, CoinDesk, Google News) with 60s refresh, category tabs (FED/RATES, WAR/GEO, CRYPTO REG, EARNINGS, MACRO/ECONOMY, CRYPTO MARKET, STOCK MARKET), HIGH/MEDIUM/LOW impact scoring
- **Symbol Search** — 50+ crypto pairs, 100+ stocks/ETFs, commodities (Gold, Silver, Oil), indices (S&P 500, NASDAQ, VIX)
- **Liquidation Heatmap** — Estimated liquidation zones for leveraged positions

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Python |
| Data (Crypto) | Binance REST API + WebSocket |
| Data (Stocks) | yFinance |
| TA Engine | pandas, numpy, custom SMC logic |
| Frontend | React 18 (CDN), LightweightCharts 4.1.1, Tailwind CSS |
| News | RSS (CNBC, Yahoo, MarketWatch, CoinDesk, Google News) |

---

## Project Structure

```
claude/
├── backend/
│   ├── main.py          # FastAPI app — REST endpoints, WebSocket proxy, news fetcher
│   └── ta_logic.py      # TA engine — SMC, Wyckoff, MTF, pattern detection, prediction
└── frontend/
    └── index.html       # Single-file React app with LightweightCharts
```

---

## Setup & Run

### Requirements
```
pip install fastapi uvicorn websockets requests yfinance pandas numpy ta
```

### Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Start Frontend
```bash
# Serve index.html on port 3000 (any static server)
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

---

## Usage

- **Symbol Search** — Click the symbol button (top-left) to search crypto, stocks, ETFs, commodities, indices
- **Timeframes** — Switch between 1m / 5m / 15m / 1h / 4h / 1d / 1w
- **Future Candles** — Enter a number in the `🕯 Candles:` input to control how many predicted candles appear (leave blank for auto)
- **News** — Right panel shows live news filtered by category; HIGH IMPACT stories highlighted in red

---

## Disclaimer

This tool is for educational and informational purposes only. Not financial advice. Always do your own research.
