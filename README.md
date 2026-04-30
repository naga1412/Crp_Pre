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
.
├── backend/
│   ├── main.py            # FastAPI app — REST endpoints, WebSocket proxy, news fetcher
│   ├── ta_logic.py        # TA engine — SMC, Wyckoff, MTF, pattern detection, projection
│   └── requirements.txt   # Pinned Python dependencies
├── frontend/
│   └── index.html         # Single-file React app served against the backend
└── docs/                  # Standalone GitHub Pages build (separate, no backend)
    └── README.md
```

The `docs/` directory contains a different, self-contained build of the app
that talks directly to Binance/yFinance from the browser. It is not used by
the backend-driven flow described below — see `docs/README.md`.

---

## Setup & Run

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated list of allowed origins |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `STOCK_POLL_INTERVAL` | `15` | Seconds between yFinance polls per stock WS client |

### 2. Frontend

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

To point the frontend at a different backend, set `window.__SMC_CONFIG__`
**before** the bundled scripts run, e.g. by editing `index.html`:

```html
<script>
  window.__SMC_CONFIG__ = {
    api: 'https://api.example.com',
    ws:  'wss://api.example.com',
  };
</script>
```

---

## Usage

- **Symbol Search** — click the symbol button (top-left) to search crypto, stocks, ETFs, commodities, indices.
- **Timeframes** — switch between 1m / 5m / 15m / 1h / 4h / 1d / 1w.
- **Future Candles** — enter a number in the `🕯 Candles:` input to control how many projected candles appear (leave blank for auto).
- **News** — right panel shows aggregated headlines filtered by category; HIGH IMPACT items highlighted in red.

> The "future candle" projection is a deterministic walk toward TP1 with
> small noise — useful as a visual guide, not an AI forecast.

---

## Disclaimer

This tool is for educational and informational purposes only. Not financial
advice. Always do your own research. The displayed liquidation levels are
approximations and do not reflect any specific exchange's margin tiers.
