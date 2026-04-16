# SMC Pro — Crypto & Stock Analyzer
## Complete Project Documentation & Recreation Guide

> **Single-file React 18 in-browser app** — no build step, no server, no npm.  
> Open `docs/index.html` in Chrome/Edge and it runs fully in the browser.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technical Analysis (TA) Engine](#2-technical-analysis-ta-engine)
3. [Market Structure Detectors](#3-market-structure-detectors)
4. [The 8 Scoring Engines](#4-the-8-scoring-engines)
5. [Master Bias Score System](#5-master-bias-score-system)
6. [Deep Learning Supervisor (Neural Network)](#6-deep-learning-supervisor-neural-network)
7. [Ghost Candle Prediction System](#7-ghost-candle-prediction-system)
8. [Volume Profile (VP) Engine](#8-volume-profile-vp-engine)
9. [News Intelligence System](#9-news-intelligence-system)
10. [Signal Stability System](#10-signal-stability-system)
11. [AI Supervisor Override](#11-ai-supervisor-override)
12. [Data Sources & WebSocket Feed](#12-data-sources--websocket-feed)
13. [File System Persistence](#13-file-system-persistence)
14. [Scanner System](#14-scanner-system)
15. [Background Validation](#15-background-validation)
16. [UI Layout & Tabs](#16-ui-layout--tabs)
17. [How to Use — Trading Guidelines](#17-how-to-use--trading-guidelines)
18. [Full Recreation Prompt](#18-full-recreation-prompt)

---

## 1. Architecture Overview

```
index.html (single file, ~3500 lines)
├── HTML/CSS — TailwindCSS + custom dark-theme styles (TradingView color palette)
├── JavaScript (Babel JSX — transpiled in-browser at runtime)
│   ├── TA Engine          — pure-JS indicators (EMA, RSI, ATR, MACD, etc.)
│   ├── Market Structure   — Swing detection, FVG, BOS, Liquidity Map
│   ├── Scoring Engines    — 8 engines → weighted Master Bias Score
│   ├── Neural Network     — 96→64→32→3 feedforward NN (pure JS)
│   ├── Ghost Candle Sim   — NN-powered structural next-candle prediction
│   ├── Volume Profile     — 100-bucket histogram, POC/VAH/VAL/HVN/LVN
│   ├── News System        — 11 RSS feeds, keyword scoring, NN memory
│   ├── FS Storage         — File System Access API → A:\crypto_prediaction_model
│   ├── Scanner            — Multi-symbol parallel analysis
│   └── React Components   — App, Chart, Sidebar, Scanner, News tabs
└── CDN dependencies
    ├── React 18 (development UMD)
    ├── Babel Standalone (in-browser JSX transpile)
    ├── Tailwind CSS (CDN)
    └── LightweightCharts 4.1.1 (TradingView chart library)
```

**Key design principle:** Everything runs client-side. No backend. The NN trains, validates, and self-corrects entirely in the browser using localStorage + File System Access API for persistence.

---

## 2. Technical Analysis (TA) Engine

All indicator functions are pure JavaScript, operating on arrays of OHLCV data.

### 2.1 EMA (Exponential Moving Average)
```
calcEMA(arr, period)
```
- Standard EMA with smoothing factor `k = 2/(period+1)`
- Seed: first `period` values averaged as SMA
- Returns same-length array (null-padded at start)
- Used for: EMA9, EMA21, EMA50 (trend bias), MACD, Keltner Channel

### 2.2 RSI (Relative Strength Index, period=14)
```
calcRSI(closes, period=14)
```
- Wilder's smoothed RSI
- Signals:
  - RSI > 70 → overbought (bearish score -15)
  - RSI < 30 → oversold (bullish score +15)
  - RSI > 55 → mild bullish (+8), RSI < 45 → mild bearish (-8)
- Also used in: StochRSI, RSI Divergence detection

### 2.3 ATR (Average True Range, period=14)
```
calcATR(highs, lows, closes, period=14)
```
- True Range = max(H-L, |H-prevC|, |L-prevC|)
- ATR = Wilder's smoothed TR
- Used as: volatility unit for SL/TP levels, ghost candle sizing, signal thresholds

### 2.4 MACD (12/26/9)
```
calcMACD(closes, fast=12, slow=26, signal=9)
```
- Returns: `{macd, signal, hist, prevHist, cross, expansion}`
- `cross`: 'bull' (MACD crossed above signal), 'bear' (crossed below)
- `expansion`: |hist| > |prevHist| (momentum growing)
- Scores: Bull cross +15, Bear cross -15, hist>0 + expanding +8, etc.

### 2.5 StochRSI (14/3/3)
```
calcStochRSI(closes, per=14, smoothK=3, smoothD=3)
```
- RSI → Stochastic of RSI → double-smoothed K and D lines
- Returns: `{k, d}` (0-100)
- Scores: k<20 → bullish +10, k>80 → bearish -10, k>d → +4

### 2.6 CMF (Chaikin Money Flow, period=20)
```
calcCMF(candles, period=20)
```
- Money Flow Multiplier = `(close-low - (high-close)) / (high-low)`
- CMF = sum(MFM × volume) / sum(volume) over period
- Range: -1 to +1
- Scores: >0.05 → bullish +10, <-0.05 → bearish -10

### 2.7 Bollinger Bands (20/2)
```
calcBB(closes, period=20, mult=2)
```
- Returns: `{upper, middle, lower, bandwidth}`
- Bandwidth = (upper-lower)/middle
- Used for: BB squeeze detection, ghost candle resistance dampening, RSI divergence

### 2.8 Keltner Channel (20/1.5 ATR)
```
calcKeltner(candles, period=20, mult=1.5)
```
- Middle = EMA(20), Bands = EMA ± 1.5×ATR
- Used for: BB Squeeze (when BB is inside Keltner = low volatility → big move coming)

### 2.9 RSI Divergence
```
detectRSIDivergence(closes, rsi)
```
- Compares last 12 bars split into two 6-bar windows
- Detects:
  - **Bullish Regular**: price makes lower low, RSI makes higher low (+15)
  - **Bearish Regular**: price makes higher high, RSI makes lower high (-15)
  - **Bullish Hidden**: price makes higher low, RSI makes lower low (+15)
  - **Bearish Hidden**: price makes lower high, RSI makes higher high (-15)

### 2.10 Candle Type Classifier
```
candleType(candle)
```
- Body/range > 75% → Marubozu (strong momentum)
- Body/range < 10% → Doji (indecision)
- Lower wick > 2×body, tiny upper wick → Hammer (bullish reversal)
- Upper wick > 2×body, tiny lower wick → Shooting Star (bearish)

### 2.11 Delta (Volume Imbalance)
```
calcDelta(candles)
```
- Estimates buying vs selling pressure per candle
- `buyFraction = (close-low)/(high-low)` — proxy for buy pressure
- Buy vol = volume × buyFraction, Sell vol = volume × (1-buyFraction)
- Delta = buy - sell (positive = buyers dominated)
- Cumulative delta used to detect divergence with price

### 2.12 Volume-Weighted BVIX Proxy
```
calcBVIXProxy(candles, period=14)
```
- `BVIX = ATR / price × 100`
- Measures volatility as a % of price
- Shown in microstructure panel as a volatility warning gauge

---

## 3. Market Structure Detectors

### 3.1 Swing Highs/Lows
```
detectSwings(highs, lows, lookback=5)
```
- A bar is a swing high if it's higher than 5 bars on each side
- A bar is a swing low if it's lower than 5 bars on each side
- Keeps last 5 swings in each direction
- Used for: BOS detection, liquidity level mapping, BSL/SSL

### 3.2 Fair Value Gaps (FVG)
```
detectFVG(candles)
```
- Bullish FVG: `candle[i].low > candle[i-2].high` (gap up, unfilled)
- Bearish FVG: `candle[i].high < candle[i-2].low` (gap down, unfilled)
- Marks imbalances where price typically returns to fill
- Shown on chart as horizontal bands

### 3.3 Break of Structure (BOS)
```
detectBOS(candles, swings)
```
- Bullish BOS: close crosses above a recent swing high
- Bearish BOS: close crosses below a recent swing low
- Confirms trend direction change — high-weight signal

### 3.4 Liquidity Map
```
calcLiquidityMap(candles)
```
- BSL (Buy-Side Liquidity): above swing highs — where stop-losses of shorts rest
- SSL (Sell-Side Liquidity): below swing lows — where stop-losses of longs rest
- Liquidity grabs: spike beyond a level that immediately reverses (stop hunt)
- Used for entry/exit zone refinement

### 3.5 Absorption Zones
```
detectAbsorption(candles, atr)
```
- High volume + small range candle (>2× avg vol, range < 0.5×ATR)
- Indicates smart money absorbing supply/demand
- Adds +5 to microstructure score when detected

---

## 4. The 8 Scoring Engines

Each engine outputs a score (-100 to +100) and detailed signals. All feed the Master Bias Score.

### Engine 1 — SMC/EMA Structure (weight 20%)
- Based on EMA alignment: price vs EMA9/21/50
- Bull%: price>EMA9 (+25), price>EMA21 (+20), price>EMA50 (+15), RSI>50 (+20), BOS bull (+20), VP context (+5)
- Raw: `(bull_pct - 50) × 2` → range -100 to +100

### Engine 2 — Wyckoff Phase (weight 10%)
- Phases: **Accumulation** (+15), **Markup** (+30), **Distribution** (-15), **Markdown** (-30)
- Phase detection:
  - Markup: bullBias AND RSI>55
  - Distribution: price>EMA50 AND RSI>60
  - Markdown: bearBias AND RSI<45
  - Default: Accumulation

### Engine 3 — Volume Profile (weight 10%)
- Multi-timeframe VP: 1w(40%), 1d(30%), 4h(20%), 1h(10%)
- Price above POC → bullish contribution weighted by timeframe
- Price below POC → half-weight bearish penalty

### Engine 4 — Momentum Score (weight 10%)
- Combines: RSI14, MACD(12/26/9), StochRSI(14/3/3), CMF(20), BB Squeeze
- Each sub-indicator scores individually and sums
- BB Squeeze state reported separately (expansion imminent)

### Engine 5 — Microstructure Score (weight 10%)
- Volume delta dominance: buy-dominant +20, sell-dominant -20
- Cumulative delta divergence: ±15
- Absorption zones: +5
- BVIX (volatility index): displayed, not scored

### Engine 6 — OI/Funding Score (weight 10%)
- Only for crypto perpetual futures (Binance FAPI)
- Funding rate signals:
  - >0.001 (1/10 of 1%) → overleveraged longs → bearish -20
  - <-0.001 → overleveraged shorts → bullish +20
  - 0.0003 to 0.001 → slight long bias → mild bearish -10
  - -0.001 to -0.0003 → slight short bias → mild bullish +10

### Engine 7 — Sentiment Score (weight 5%)
- Fear & Greed Index (alternative.me API):
  - ≤25 (Extreme Fear) → bullish +20
  - ≥75 (Extreme Greed) → bearish -20
- News sentiment (bullish count vs bearish count): ±10

### Engine 8 — Intermarket Score (weight 5%)
- DXY (US Dollar Index): DXY up → crypto/risk assets down
- SPX (S&P 500): SPX up → risk-on bullish signal
- Gold: tracked but not scored directly
- BTC dominance: tracked (from CoinGecko)
- Weights differ for crypto vs stocks

---

## 5. Master Bias Score System

```
calcMasterBias(setup, momSig, microSig, oiSig, imSig, sentSig, mtfVP, price)
```

**Weighted blend:**
| Engine           | Weight |
|------------------|--------|
| SMC/Structure    | 20%    |
| Wyckoff Phase    | 10%    |
| Volume Profile   | 10%    |
| OI/Funding       | 10%    |
| Momentum         | 10%    |
| Microstructure   | 10%    |
| Sentiment        | 5%     |
| Intermarket      | 5%     |
| ML Boost         | +10% of weighted sum |

**Output:** Score -100 to +100  
- > +60 → **Strong Bullish** (teal)
- > +30 → **Bullish**
- -30 to +30 → **Neutral** (gray)
- < -30 → **Bearish**
- < -60 → **Strong Bearish** (red)

**Signal thresholds:** Score ≥ +8 → LONG, Score ≤ -8 → SHORT, else NEUTRAL

### Signal Stability (EMA + Hold Lock)
1. **EMA Smoothing**: `biasEMA = biasEMA×0.65 + newScore×0.35`  
   Prevents single-candle ticks from flipping the signal
2. **Hold Lock**: Signal only flips after **2 consecutive readings** in the new direction  
   Prevents false flips from temporary 30-second poll noise

---

## 6. Deep Learning Supervisor (Neural Network)

### Architecture
```
Input Layer:  96 neurons  (12 features × 8 candles)
Hidden Layer: 64 neurons  (sigmoid activation)
Hidden Layer: 32 neurons  (sigmoid activation)
Output Layer:  3 neurons  (sigmoid activation)
  → Output[0]: direction probability (0=bearish, 1=bullish)
  → Output[1]: magnitude (0-1 scaled move size relative to 2×ATR)
  → Output[2]: volatility (0-1 scaled range relative to 3×ATR)
```

### 12 Features Per Candle
| # | Feature | Range | Source |
|---|---------|-------|--------|
| 1 | RSI(14) | 0-1 | TA Engine |
| 2 | MACD histogram / ATR (normalized) | 0-1 | TA Engine |
| 3 | Bollinger Band position (where in band) | 0-1 | TA Engine |
| 4 | StochRSI K | 0-1 | TA Engine |
| 5 | Volume ratio (vs 20-bar average, capped 3×) | 0-1 | TA Engine |
| 6 | Body direction (0=bear, 1=bull) | 0 or 1 | Candle |
| 7 | Body fraction (body/range) | 0-1 | Candle |
| 8 | Price vs EMA9 | 0 or 1 | TA Engine |
| 9 | Price vs EMA21 | 0 or 1 | TA Engine |
| 10 | Price vs EMA50 | 0 or 1 | TA Engine |
| 11 | 5-bar slope (normalized ÷ ATR) | 0-1 | TA Engine |
| 12 | CMF (money flow, normalized) | 0-1 | TA Engine |

**Sequence window:** Last 8 candles → flattened 96-element input vector

### Training Process
- **Full training:** 20 epochs, LR=0.02, on entire candle history
- **Guard:** Skip if candle count hasn't grown by ≥2 since last training
- **Error-correction samples:** Weighted 5× (wrong predictions retrained harder)
- **Immediate self-correction:** 8 epochs, LR=0.03, triggered when validation finds wrong prediction

### Self-Correction Flow
```
Predict → log {input, prediction, anchorPrice, anchorTime} → wait for next candle
→ Validate: compare predicted direction vs actual candle direction
→ If WRONG: weight=5 retrain (8 epochs, LR=0.03) immediately
→ _nnStats.corrections++ for tracking
→ Wrong samples also included in next full training run
```

### Prediction Output
```javascript
{
  prob: 0.72,          // direction probability (>0.58 → LONG, <0.42 → SHORT)
  magnitude: 0.45,     // expected move size (0-1)
  volatility: 0.38,    // expected range (0-1)
  signal: 'LONG',
  confidence: 44,      // |prob - 0.5| × 200
  accuracy: 67,        // training accuracy %
  recentAcc: 71,       // rolling-50 live validation accuracy
  corrections: 23,     // total self-corrections applied
  immediateRetrains: 8 // times immediate retraining triggered
}
```

### Accuracy Tracking
- **Training accuracy:** batch accuracy at training time (smoothed: 60% old, 40% new)
- **Live accuracy (recentAcc):** rolling window of last 50 validated predictions vs actual outcome (ground truth)
- **Session count:** how many chart loads triggered a training run

### Storage Keys (localStorage)
| Key | Contents |
|-----|----------|
| `smc_deep_nn_v7` | Neural network weights & biases (JSON) |
| `smc_deep_stats_v7` | Training stats, accuracy, session count |
| `smc_deep_outcomes_v7` | Pending predictions awaiting validation |
| `smc_deep_news_v7` | News→price reaction memory |

---

## 7. Ghost Candle Prediction System

Ghost candles are semi-transparent future candle predictions displayed on the chart after the live candle.

### How It Works

**Step 1 — NN Inference:**
- Runs NN on current candle sequence
- Gets `nnProb` (direction), `nnMag` (move size), `nnVol` (volatility)
- Falls back to technical signal direction if NN not trained enough (<150 candles)

**Step 2 — Candle Sizing (realistic floor):**
```javascript
const avgBody = last20 candles avg |close-open|
const avgRange = last20 candles avg (high-low)
const baseBody = Math.max(avgBody × 0.70, nnMag × 2 × ATR)
const baseRange = Math.max(avgRange × 0.60, nnVol × 3 × ATR)
```
> Ghost candles are always at least 70% of recent actual candle body size — prevents tiny unreadable ghosts when NN is early in training.

**Step 3 — Structure Sequence Selection:**
| NN Confidence | Pattern Sequence |
|---------------|-----------------|
| >35% | Impulse → Continuation → Continuation → Pullback (repeat) |
| 15-35% | Impulse → Inside → Inside → Breakout (repeat) |
| <15% | Range → Inside → Range → Inside (market indecisive) |

**Step 4 — Candle Type OHLC Construction:**
| Pattern | Body Size | Wick Behavior |
|---------|-----------|---------------|
| Impulse | 55-100% of base (NN confidence scales) | Small counter-wicks |
| Continuation | 55% of base | Small counter-wicks |
| Pullback | 38% of base (Fibonacci retracement) | Equal wicks |
| Inside | 18% of base | Large equal wicks |
| Breakout | 88% of base | Minimal wicks |
| Pinbar | 7% of base | 72% wick in rejection direction |
| Range | 22% of base | 22% equal wicks |

**Step 5 — Special Rules:**
- **RSI Dampening:** RSI>78 or <22 → multiply body sizes by 0.5 (exhaustion → smaller candles)
- **BB Dampening:** Price at/beyond BB band → multiply by 0.35 (mean-reversion expected)
- **Pinbar Injection:** Near POC/VAH/VAL/BB bands → replace impulse/continuation with pinbar
- **Decay:** Each successive ghost candle shrinks: `decay = max(0.22, 1 - i×0.045)`

**Color Coding:**
| Pattern | Color | Meaning |
|---------|-------|---------|
| Impulse | Yellow (#FFD700) | Strong directional move |
| Continuation | Yellow | Same direction as impulse |
| Pullback | Orange (#FF9800) | Counter-trend retracement |
| Inside | Gray (#787b86) | Consolidation/indecision |
| Breakout | Blue (#2962FF) | Breakout after consolidation |
| Pinbar | Purple (#AB47BC) | Rejection at key level |
| Range | Gray | Sideways movement |

**Markers:** Small shape-only dots (size=0.5) below/above ghost candles — no text labels to avoid covering candle bodies.

---

## 8. Volume Profile (VP) Engine

```
calcVolumeProfile(candles, numBuckets=100)
```

### How It's Built
1. Find price range (min-max) across all candles
2. Divide into 100 equal price buckets
3. For each candle, distribute volume proportionally across overlapping buckets
4. Find the highest-volume bucket → **POC** (Point of Control)
5. Expand outward from POC until 70% of total volume is covered → **Value Area** (VAH/VAL)
6. Mark buckets >2× average volume → **HVN** (High Volume Nodes, strong support/resistance)
7. Mark buckets <0.3× average volume → **LVN** (Low Volume Nodes, price moves fast through these)

### Key Levels
| Level | Description | Trading Use |
|-------|-------------|-------------|
| **POC** | Price of Control — highest traded volume | Strong magnet, expect price to test |
| **VAH** | Value Area High — top of 70% volume zone | Resistance above value area |
| **VAL** | Value Area Low — bottom of 70% volume zone | Support below value area |
| **HVN** | High Volume Node | Price stalls here (congestion) |
| **LVN** | Low Volume Node | Price moves quickly (gap) |

### Canvas Rendering
- Drawn on an HTML5 Canvas overlay positioned exactly on top of the chart
- **Colors:** POC=red line, VAH/VAL=teal lines, In-VA bars=teal fill, Out-VA bars=gray fill
- **VP drift fix:** Canvas redrawn on every chart viewport change:
  - `subscribeVisibleLogicalRangeChange` (pan/zoom)
  - `wheel` event (mouse scroll)
  - `mouseup` / `touchend` events (drag release)
- `priceToCoordinate()` is called fresh on each redraw to get correct pixel position

### Multi-Timeframe VP
Computed independently for 1w, 1d, 4h, 1h timeframes. Each TF's POC feeds the Master Bias VP score at weighted contribution (1w=40%, 1d=30%, 4h=20%, 1h=10%).

---

## 9. News Intelligence System

### RSS Feed Sources (11 feeds)
- Yahoo Finance (markets + crypto)
- CNBC (markets, world economy, crypto)
- CoinDesk
- Google News (Fed/FOMC, geopolitical, crypto, markets, macro/CPI)

### Processing Pipeline
1. **Fetch:** All feeds in parallel via 3 CORS proxies (fallback chain)
2. **Parse:** DOMParser extracts `<item>` elements (title, link, pubDate)
3. **Score:** Keyword matching for bullish/bearish sentiment and impact level
4. **Categorize:** Into 8 categories (FED/RATES, WAR/GEO, CRYPTO REG, etc.)
5. **Deduplicate:** By first 65 chars of headline
6. **Sort:** By timestamp (newest first)
7. **Display:** In News tab with color-coded categories and sentiment badges

### Keyword Sets
**Bullish keywords:** rate cut, fed cut, pivot, dovish, stimulus, rally, surge, upgrade, ceasefire, recovery, beats, approves bitcoin, trade deal, soft landing

**Bearish keywords:** rate hike, hawkish, recession, war, attack, invasion, airstrike, crash, plunge, ban, default, stagflation, crackdown, layoffs, sanctions, nuclear, collapse, tariff

**High-impact multiplier (2×):** war, invasion, attack, rate cut, rate hike, crash, recession, ceasefire, default, nuclear

### Impact Levels
| Impact Score | Label | NN Training? |
|-------------|-------|-------------|
| ≥3 | HIGH | ✅ Logged to NN memory |
| 1.5-3 | MEDIUM | ❌ Skipped |
| <1.5 | LOW | ❌ Skipped |

> Only HIGH-impact + directional (bullish/bearish, not neutral) news gets logged to the NN's news memory. This prevents noise from polluting the model.

### News Reaction Memory
- On HIGH-impact news arrival: log feature snapshot (`_buildSequenceInput`) + news metadata
- After 5 candles: validate — compare price at news vs price 5 candles later
- Record actual price reaction % for each high-impact news event
- Stored in `smc_deep_news_v7` (localStorage) + `news_reactions.json` (disk)

---

## 10. Signal Stability System

**Problem solved:** On a 30-second poll, a small price tick could shift RSI by 0.1 and tip the master bias from LONG to SHORT, then back on the next poll. This made the signal unusable for real trading.

### Layer 1 — EMA Smoothing on Master Bias
```javascript
biasEMA = Math.round(biasEMA × 0.65 + newScore × 0.35)
```
- New readings only have 35% influence
- History (previous EMA) has 65% influence
- Smooths micro-fluctuations from 30s polls

### Layer 2 — Signal Hold Lock
```javascript
if (newSignal !== currentSignal) {
  holdCount++
  if (holdCount < 2 && currentSignal !== null) {
    return currentSignal  // hold — don't flip yet
  }
}
// Only flip after 2 consecutive readings in new direction
```
- Signal only changes if 2 polls in a row agree on the new direction
- Prevents one-off spikes from changing your trade direction

### Result
- Signal is stable for 15+ minutes under normal volatility
- Only flips when market has genuinely changed direction (sustained)

---

## 11. AI Supervisor Override

### How It Works
The Neural Network's prediction **blends into** the Master Bias Score. The higher the NN's live accuracy, the more weight it gets.

```javascript
const maxWeight = liveAcc >= 65 ? 0.55 : liveAcc >= 60 ? 0.48 : 0.40
const nnWeight = Math.min(maxWeight,
  Math.max(0.05, nnConf × 0.6 + Math.max(0, (liveAcc-50)/80))
)
// aiAdjustedBias = masterBias × (1-nnWeight) + nnDirectional × nnWeight
```

| Live Accuracy | Max NN Weight | Meaning |
|---------------|--------------|---------|
| ≥ 65% | 55% | NN dominates the signal |
| ≥ 60% | 48% | NN has strong influence |
| < 60% | 40% | NN has moderate influence |
| < 50 candles | 5% | NN barely matters (not trained) |

### Status Badges on Signal Button
- **"⚠️ NN flipped signal"** — NN disagreed with raw Master Bias and changed the final signal
- **"✓ NN confirms"** — NN agrees with the direction from the 8 engines
- **"🧠 X% NN weight"** — shows current blend percentage

### Confidence-Weighted Direction
```
nnDirectional = (nnProb - 0.5) × 200  (→ -100 to +100 like master bias)
```

---

## 12. Data Sources & WebSocket Feed

### Crypto (Binance)
- **REST Spot:** `https://api.binance.com/api/v3/klines` (up to 1000 candles per request)
- **REST Futures:** `https://fapi.binance.com/fapi/v1/klines` (fallback for futures-only pairs)
- **OI + Funding:** `https://fapi.binance.com/fapi/v1/openInterest` + `/fundingRate`
- **WebSocket live:** `wss://stream.binance.com:9443/ws/{symbol}@kline_{tf}`
- Spot tried first; if 404, futures used and cached in `FUTURES_ONLY` Set

### Stocks/ETFs/Commodities/Indices (Yahoo Finance)
- Via CORS proxy: `https://api.allorigins.win/raw?url=...` or `https://corsproxy.io/`
- Yahoo Finance chart API: `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`
- 4h candles are aggregated from 1h data (group 4 consecutive 1h candles)

### Fear & Greed Index
- `https://api.alternative.me/fng/?limit=2` (current + yesterday)

### Intermarket Data
- DXY, SPX, Gold via Yahoo Finance
- BTC dominance from CoinGecko global API

### Poll Frequency
- **Main poll:** Every 30 seconds (`quiet=true`)
- **Quiet poll optimization:** Fetches only 100 candles; if last candle timestamp unchanged, skips all TA computation (returns early, no recalculation)
- **Full load:** On symbol/TF switch, fetches 600 candles + full TA

---

## 13. File System Persistence

### Setup
Uses the **File System Access API** (Chrome/Edge 86+).
1. User clicks "Connect Folder" → browser shows folder picker
2. User selects `A:\crypto_prediaction_model` (or any folder)
3. Directory handle stored in IndexedDB for persistence across browser sessions
4. On next app start: auto-reconnect if permission still granted

### Files Written
| File | Contents | When Written |
|------|----------|-------------|
| `model_weights.json` | NN layers, weights, biases | On session count or correction count change |
| `training_stats.json` | Accuracy, sessions, corrections, news logged | Same condition |
| `prediction_outcomes.json` | All pending + validated predictions | Same condition |
| `news_reactions.json` | News → price reaction log | Same condition |
| `predictions_log.json` | Every prediction made (up to 2000) | Each prediction |
| `training_history.json` | Training batch history (up to 5000) | Each training run |

### Smart Write Guard
```javascript
async function fsSaveModel(force=false) {
  const sessChanged = _nnStats.sessions !== _fsDirtySessionStamp
  const corChanged = _nnStats.corrections !== _fsDirtyCorStamp
  if (!force && !sessChanged && !corChanged) return  // skip — nothing changed
  // ... write files ...
  _fsDirtySessionStamp = _nnStats.sessions
  _fsDirtyCorStamp = _nnStats.corrections
}
```
Only writes to disk when the model has actually learned something new. Prevents constant disk writes on every 30-second poll.

### Immediate Write on Connect
When folder is connected (either button click or auto-reconnect), `fsSaveModel(true)` is called immediately — force-writing the current model state to disk.

---

## 14. Scanner System

Scans multiple symbols in parallel and displays ranked LONG/SHORT signals.

### Configuration (persisted to localStorage)
- **Asset count:** 1-500 symbols to scan (default: 50)
- **Auto-rescan interval:** 0.5-60 minutes (default: 5 min)
- **Timeframe filter:** Which TF to score on
- **Minimum strength:** Weak/Medium/Strong threshold

### How It Scans
1. Takes first N symbols from `ALL_SYMBOLS` list (or fetches live Binance list)
2. For each symbol: fetches candles → runs `generateSetup` → computes all 8 engine scores → calculates Master Bias
3. Results sorted by |score| descending (strongest signals first)
4. Displayed in 2-column grid (LONG left, SHORT right) with signal cards

### Signal Cards Show
- Symbol name + exchange badge
- Master Bias score + label + colored bar
- Bull% indicator
- Sparkline mini-chart (10 bars)
- Dominant scoring engine
- Click → navigates main chart to that symbol

### Performance
- Scans run with 300ms stagger between symbols to avoid API rate limiting
- Status indicator shows scan progress and last scan time

---

## 15. Background Validation

### Purpose
When you switch from BTC/5m to SOL/5m, the app continues to track what happened to BTC predictions — it follows up whether the predicted candle formed correctly.

### Implementation
```javascript
backgroundValidateAll(currentSym, currentTf)
// Triggered on every chart symbol/TF switch
```

1. Reads `smc_deep_outcomes_v7` — all pending unvalidated predictions
2. Finds unique symbol+TF combinations (excluding current chart, max 5)
3. For each: fetches 80 candles from Binance/YF
4. Runs `validatePendingPredictions` → marks correct/wrong
5. If wrong → immediate 8-epoch correction retrain
6. Saves model to FS if anything changed

**Throttle:** 5-minute cooldown between runs (`_bgValidateLastRun` ref) — prevents API flood when rapidly switching charts.

---

## 16. UI Layout & Tabs

### Left Sidebar (~320px)
**Scrollable panel with accordions:**
- Asset info (price, change, leverage suggestion)
- Signal button (LONG/SHORT/NEUTRAL with AI supervisor badge)
- Trade setup (entry zone, SL, TP1, TP2, RR ratios)
- Next candle prediction (direction, type, size, confidence)
- AI Supervisor card (NN stats: accuracy, live accuracy, corrections, sessions)
- SMC Analysis (BOS, FVG, liquidity)
- Wyckoff + Volume Profile
- Multi-timeframe bias grid
- Momentum indicators panel (RSI, MACD, StochRSI, CMF)
- Microstructure panel (delta dominance, absorption, BVIX)
- OI/Funding panel (crypto only)
- Intermarket panel (DXY, SPX, Gold, BTC dominance)
- Sentiment panel (Fear & Greed, news sentiment)

### Main Chart Area
- LightweightCharts 4.1.1 candlestick chart (dark TV theme)
- EMA lines: EMA9 (blue), EMA21 (orange), EMA50 (purple)
- FVG zones: horizontal bands (teal/red)
- Liquidity levels: BSL/SSL horizontal lines
- Volume Profile: Canvas overlay (right side, 15% chart width)
- Ghost candles: semi-transparent predicted candles after live price
- Ghost markers: small shape-only dots (no text)

### Top Tab Bar
| Tab | Content |
|-----|---------|
| **Chart** | Main candlestick chart + analysis |
| **Scanner** | Multi-symbol signal scanner |
| **News** | RSS news feed with sentiment analysis |

### Scanner Tab Controls
- Symbol count input (1-500)
- Auto-rescan interval (0.5-60 min)
- Timeframe selector
- Strength filter
- Start/Stop scan button
- FS folder status + connect button

---

## 17. How to Use — Trading Guidelines

### Initial Setup
1. Open `docs/index.html` in Chrome or Edge (NOT Firefox — File System API needed)
2. Click **"📁 Connect Folder"** → select or create `A:\crypto_prediaction_model`
3. The app auto-loads any existing model from disk
4. Select your asset from the search bar (Ctrl+K or click search)
5. Select your timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)

### Reading the Signal
1. **Signal Button Color:**
   - Green LONG = bullish bias (≥+8 score)
   - Red SHORT = bearish bias (≤-8 score)
   - Gray NEUTRAL = no clear direction

2. **AI Supervisor Badge:**
   - "⚠️ NN flipped signal" = NN disagrees with engines — trust the NN if accuracy > 60%
   - "✓ NN confirms" = all systems agree — higher confidence trade

3. **Master Bias Score Breakdown:**
   - Check which engines are contributing most
   - All engines pointing same direction = high conviction

### Ghost Candle Reading
- **Yellow impulse candles** = strong move expected
- **Orange pullback** = brief counter-move before continuation
- **Gray inside bars** = consolidation, wait for breakout
- **Blue breakout candle** = breakout from consolidation incoming
- **Purple pinbar** = rejection at key level (POC/BB band), possible reversal

### Volume Profile Usage
- **Price at POC** = expect slow/choppy movement (high interest zone)
- **Price above VAH** = strong breakout territory — momentum play
- **Price below VAL** = breakdown territory — short bias
- **LVN gaps** = price will move quickly through these zones (no resistance)
- **HVN** = expect price to stall here

### Trade Setup (SL/TP)
- **Stop Loss:** 1.5×ATR from entry (on the wrong side)
- **TP1:** 2×ATR (1:1.33 minimum RR)
- **TP2:** 4×ATR (1:2.67 aggressive target)
- **Suggested leverage:** 3× (RSI extreme), 5× (normal), 10× (low volatility ATR)

### When NOT to Trade
- Master Bias score between -8 and +8 (NEUTRAL) — no edge
- Ghost candles showing all "inside/range" patterns — market consolidating
- RSI > 78 or < 22 — exhaustion, avoid momentum trades
- OI Funding rate extreme (>0.001 or <-0.001) — counter-trend risk
- News tab showing HIGH-impact bearish news AND your signal is LONG (fade the signal)

### NN Learning Period
- First 100 candles: NN not active (too little data)
- 100-500 candles: NN starting to learn, weight it less
- 500+ candles + multiple sessions: NN starts being reliable
- Watch `recentAcc` (rolling-50): if >60% trust NN override; if <50% ignore it

---

## 18. Full Recreation Prompt

Copy this prompt into any AI assistant to recreate this exact project from scratch:

---

### PROMPT TO RECREATE SMC PRO ANALYZER

```
Build a single-file React 18 trading analyzer called "SMC Pro — Crypto & Stock Analyzer" 
as docs/index.html with NO build step. Use CDN scripts only:
- React 18 (UMD development)
- Babel Standalone (in-browser JSX)
- Tailwind CSS (CDN with custom config)
- LightweightCharts 4.1.1 (TradingView)

TECH STACK:
- All JavaScript in a single <script type="text/babel"> tag
- Dark TradingView theme: background #131722, panel #1e222d, border #2a2e39
- Bull color: #26a69a, Bear: #ef5350, Primary: #2962ff

══════════════════════════════════════════
SECTION 1: TECHNICAL ANALYSIS ENGINE
══════════════════════════════════════════

Implement these pure-JS functions operating on OHLCV candle arrays:

calcEMA(arr, period): Standard EMA with k=2/(p+1), SMA seed
calcRSI(closes, period=14): Wilder's smoothed RSI
calcATR(highs, lows, closes, period=14): True Range with Wilder smoothing
calcMACD(closes, fast=12, slow=26, signal=9): Returns {macd, signal, hist, prevHist, cross, expansion}
calcStochRSI(closes, per=14, smoothK=3, smoothD=3): Double-smoothed StochRSI {k, d}
calcCMF(candles, period=20): Chaikin Money Flow using (close-low-(high-close))/(high-low) × volume
calcBB(closes, period=20, mult=2): Bollinger Bands {upper, middle, lower, bandwidth}
calcKeltner(candles, period=20, mult=1.5): EMA ± 1.5×ATR bands
detectRSIDivergence(closes, rsi): Compare last 12 bars (2×6 windows) for bullish/bearish regular/hidden divergence
candleType(candle): Classify as Marubozu, Doji, Hammer, Shooting Star, Bullish/Bearish Candle
calcDelta(candles): Estimate buy/sell volume per candle using (close-low)/(high-low) proxy
detectAbsorption(candles, atr): High volume (>2× avg) + small range (<0.5×ATR) zones
calcBVIXProxy(candles, period=14): ATR/price × 100 (volatility % of price)

══════════════════════════════════════════
SECTION 2: MARKET STRUCTURE DETECTION
══════════════════════════════════════════

detectSwings(highs, lows, lookback=5): Find pivot highs/lows (higher than 5 bars each side), keep last 5
detectFVG(candles): Fair Value Gaps — bull: candle[i].low > candle[i-2].high, bear: opposite
detectBOS(candles, swings): Break of Structure — close crossing swing high/low
detectLiquidityGrabs(candles, levels): Spike beyond level + immediate reversal
calcLiquidityMap(candles): Returns {bsl, ssl, grabs, clusters} from swing highs/lows

══════════════════════════════════════════
SECTION 3: VOLUME PROFILE ENGINE
══════════════════════════════════════════

calcVolumeProfile(candles, numBuckets=100):
- Divide price range into 100 buckets
- Distribute volume proportionally across overlapping buckets per candle
- Find POC (max volume bucket)
- Expand from POC until 70% volume covered → VAH/VAL (Value Area)
- HVN: buckets > 2× average volume
- LVN: buckets < 0.3× average volume
- Returns {poc, vah, val, hvn, lvn, bSz, vols, minP, pocIdx, inVA}

Draw VP as Canvas overlay on the right 15% of chart area.
Colors: POC=red line, VAH/VAL=teal lines, in-VA bars=teal fill, out-VA=gray fill
Redraw VP canvas on EVERY viewport change:
  chartInst.timeScale().subscribeVisibleLogicalRangeChange(handler)
  + wheel, mouseup, touchend DOM events on chart element
Use priceToCoordinate() fresh on each redraw.

══════════════════════════════════════════
SECTION 4: 8 SCORING ENGINES
══════════════════════════════════════════

Each returns {score: -100 to +100, signals: {}}

Engine 1 — SMC/EMA Structure (weight 20%):
  bull_pct: price>EMA9(+25), price>EMA21(+20), price>EMA50(+15), RSI>50(+20), BOS bull(+20), VP(+5)
  smcRaw = (bull_pct - 50) × 2

Engine 2 — Wyckoff Phase (weight 10%):
  Markup(+30), Accumulation(+15), Distribution(-15), Markdown(-30)
  Detection: bullBias+RSI>55=Markup, price>EMA50+RSI>60=Distribution, bearBias+RSI<45=Markdown

Engine 3 — Volume Profile (weight 10%):
  Multi-TF VP (1w=40%, 1d=30%, 4h=20%, 1h=10%) weighted by price vs POC

Engine 4 — Momentum Score (weight 10%):
  calcMomentumScore: RSI + RSI divergence + MACD + StochRSI + CMF
  Combine sub-scores, return total -100 to +100

Engine 5 — Microstructure Score (weight 10%):
  calcMicrostructureScore: delta dominance (±20), cumulative delta divergence (±15), absorption (+5)

Engine 6 — OI/Funding Score (weight 10%, crypto only):
  FR > 0.001 → -20 (longs overleveraged), FR < -0.001 → +20
  FR > 0.0003 → -10, FR < -0.0003 → +10

Engine 7 — Sentiment Score (weight 5%):
  Fear&Greed ≤25 → +20, ≥75 → -20; news bull/bear ratio ±10

Engine 8 — Intermarket Score (weight 5%):
  DXY up → -10 for crypto; SPX up → +10 (risk-on)

calcMasterBias: Weighted sum of all 8 engines + 10% ML boost
Score > +60 = Strong Bullish, > +30 = Bullish, -30 to +30 = Neutral, etc.

══════════════════════════════════════════
SECTION 5: SIGNAL STABILITY
══════════════════════════════════════════

Apply TWO stability layers to Master Bias:

1. EMA Smoothing: biasEMA = Math.round(biasEMA × 0.65 + newScore × 0.35)
   Only 35% influence from new reading

2. Hold Lock: Signal only flips after 2 CONSECUTIVE readings in new direction
   Use useRef for biasEMA, lastSignal, signalHoldCount

══════════════════════════════════════════
SECTION 6: DEEP LEARNING NEURAL NETWORK
══════════════════════════════════════════

Implement class NeuralNet with pure JS:
- Architecture: [96, 64, 32, 3] layers
- Xavier initialization: scale = sqrt(2/inputSize)
- Sigmoid activation everywhere
- Standard backpropagation with configurable LR and epochs
- Methods: predict(input), train(data, lr, epochs), toJSON(), fromJSON()

Feature extraction per candle (12 features, all normalized 0-1):
1. RSI/100
2. MACD hist/(ATR×6) normalized
3. BB position (close-lower)/(upper-lower)
4. StochRSI K/100
5. Volume ratio vs 20-bar avg (capped 3×, then /3)
6. Body direction (0 or 1)
7. Body fraction |close-open|/(high-low)
8. Price > EMA9 (0 or 1)
9. Price > EMA21 (0 or 1)
10. Price > EMA50 (0 or 1)
11. 5-bar slope normalized ÷ (ATR×5)
12. CMF normalized (cmf+1)/2

SEQ_LEN=8 candles → flattened 96-input vector

Labels for next candle: [direction (0/1), magnitude (move/2ATR clamped), volatility (range/3ATR clamped)]

Training:
- Full: 20 epochs, LR=0.02
- Skip if candle count change < 2 AND no new corrections
- Include error-correction samples at weight=5

Self-correction:
- Log every prediction: {input, prediction[3], anchorPrice, anchorTime, symbol, tf}
- Validate when next candle known: compare predicted direction vs actual
- If wrong: immediate retrain (8 epochs, LR=0.03) with weight=5 samples
- Track: corrections count, recentCorrect/recentTotal (rolling-50 window)

localStorage keys: smc_deep_nn_v7, smc_deep_stats_v7, smc_deep_outcomes_v7, smc_deep_news_v7

══════════════════════════════════════════
SECTION 7: AI SUPERVISOR OVERRIDE
══════════════════════════════════════════

Blend NN into Master Bias score:
  maxWeight = liveAcc>=65 ? 0.55 : liveAcc>=60 ? 0.48 : 0.40
  nnWeight = min(maxWeight, max(0.05, nnConf×0.6 + max(0,(liveAcc-50)/80)))
  nnDirectional = (prob-0.5)×200
  aiAdjustedBias = masterBias×(1-nnWeight) + nnDirectional×nnWeight

Show badge: "⚠️ NN flipped signal" or "✓ NN confirms" on trade signal button

══════════════════════════════════════════
SECTION 8: GHOST CANDLE PREDICTION
══════════════════════════════════════════

futureCandlesSim(candles, signal, atr, count=12, ctx):
1. Run NN to get {nnProb, nnMag, nnVol}
2. Size floor: baseBody = max(avgBody×0.70, nnMag×2×atr)
              baseRange = max(avgRange×0.60, nnVol×3×atr)
3. Select structure sequence by NN confidence:
   nnConf > 0.35: impulse→continuation→continuation→pullback (repeat)
   nnConf > 0.15: impulse→inside→inside→breakout (repeat)
   else: range→inside alternating
4. Inject pinbar at VP/BB key levels
5. Build OHLC per type with body sizes:
   impulse=55-100%, continuation=55%, pullback=38%, inside=18%,
   breakout=88%, pinbar=7% body+72% wick, range=22%
6. Apply RSI dampening (×0.5 at extremes), BB dampening (×0.35 at bands)
7. Decay factor: max(0.22, 1-i×0.045) per candle

Colors: impulse/continuation=yellow, pullback=orange, inside=gray, breakout=blue, pinbar=purple
Markers: shape-only, size=0.5, NO text labels

══════════════════════════════════════════
SECTION 9: NEWS INTELLIGENCE
══════════════════════════════════════════

11 RSS feeds: Yahoo Finance, CNBC, CoinDesk, Google News (Fed, Geo, Crypto, Markets, Macro)
3 CORS proxy fallbacks: allorigins.win, corsproxy.io, codetabs.com
Parse with DOMParser, score each headline with bullish/bearish keyword matching
Impact scoring with HIGH-impact keyword multiplier (2×)
Only log HIGH-impact + directional (non-neutral) to NN news memory

Categories with colors: FED/RATES(orange), WAR/GEO(red), CRYPTO REG(blue),
EARNINGS(teal), MACRO(purple), CRYPTO MARKET(cyan), STOCK MARKET(green)

══════════════════════════════════════════
SECTION 10: FILE SYSTEM PERSISTENCE
══════════════════════════════════════════

Use File System Access API (showDirectoryPicker):
- Store directory handle in IndexedDB for auto-reconnect
- Files: model_weights.json, training_stats.json, prediction_outcomes.json,
         news_reactions.json, predictions_log.json, training_history.json
- Smart write guard: only write when sessions or corrections count changed
  (track _fsDirtySessionStamp and _fsDirtyCorStamp)
- Force write on folder connect

══════════════════════════════════════════
SECTION 11: DATA FETCHING
══════════════════════════════════════════

Crypto: Binance REST (spot first, futures fallback), WebSocket live feed
Stocks/ETFs: Yahoo Finance via CORS proxy
Fear & Greed: api.alternative.me/fng
Intermarket: DXY, SPX, Gold via Yahoo Finance; BTC dominance via CoinGecko

Quiet poll (every 30s, quiet=true):
- Fetch only 100 candles
- If last candle timestamp unchanged: return early, skip all TA
- Otherwise: fetch 600 candles, run full analysis

══════════════════════════════════════════
SECTION 12: SCANNER
══════════════════════════════════════════

Parallel scan of up to 500 symbols with configurable:
- Asset count (localStorage persisted)
- Auto-rescan interval (localStorage persisted)
- TF filter, strength threshold

Show results in 2-column grid (LONG/SHORT) with sparklines, bias bars, signal cards
Click card → navigate main chart to that symbol

══════════════════════════════════════════
SECTION 13: BACKGROUND VALIDATION
══════════════════════════════════════════

On every chart symbol/TF switch: backgroundValidateAll()
- Read all pending predictions
- Find up to 5 other symbol/TF combos with unvalidated predictions
- Fetch 80 candles each, 500ms stagger
- Validate and correct NN if wrong
- 5-minute cooldown between runs

══════════════════════════════════════════
UI REQUIREMENTS
══════════════════════════════════════════

Layout: Left sidebar (320px scrollable) + Main chart area
Tabs: Chart | Scanner | News

Sidebar sections (accordions):
- Asset price + live WebSocket feed
- Signal button (LONG/SHORT/NEUTRAL, pulsing animation) + AI supervisor badge
- Trade setup (entry zone, SL, TP1, TP2, RR ratios)
- Next candle prediction
- AI Supervisor stats card
- SMC Analysis (BOS, FVG, liquidity levels)
- Wyckoff phase + Volume Profile levels
- Multi-timeframe bias grid (1m/5m/15m/1h/4h/1d/1w)
- Momentum indicators (RSI, MACD, StochRSI, CMF, BB squeeze)
- Microstructure (delta dominance, absorption count, BVIX)
- OI/Funding (crypto only, from Binance FAPI)
- Intermarket (DXY, SPX, Gold, BTC dominance)
- Sentiment (Fear & Greed + news breakdown)

Chart overlays:
- EMA9 (blue), EMA21 (orange), EMA50 (purple)
- FVG zones (horizontal bands)
- BSL/SSL liquidity lines
- Volume Profile canvas (right 15%)
- Ghost candles (12 semi-transparent future candles)
- Ghost markers (shape-only, size 0.5, no text)

Symbol universe:
- 150+ crypto pairs (Binance USDT pairs)
- 30+ stocks (AAPL, MSFT, NVDA, TSLA, etc.)
- ETFs (SPY, QQQ, GLD, etc.)
- Commodities (Gold, Oil, Silver futures)
- Indices (S&P 500, NASDAQ, VIX)
- Support dynamic Binance symbol list fetch

Search: Ctrl+K modal with fuzzy search, type badges, exchange labels

Timeframes: 1m, 5m, 15m, 1h, 4h, 1d, 1w
```

---

## Quick Reference Card

| What | Signal | Action |
|------|--------|--------|
| Master Bias > +60 + NN confirms | Strong Bullish | LONG with 5-10× leverage |
| Master Bias +30 to +60 | Bullish | LONG with 3-5× leverage |
| Master Bias -8 to +8 | Neutral | Wait / no trade |
| Master Bias -30 to -60 | Bearish | SHORT with 3-5× leverage |
| Master Bias < -60 + NN confirms | Strong Bearish | SHORT with 5-10× leverage |
| NN flipped signal badge | NN override | Trust NN if recentAcc > 60% |
| RSI > 78 or < 22 | Exhaustion | Reduce size / wait |
| All ghost candles gray (inside) | Consolidation | Wait for breakout signal |
| Price at POC | Congestion | Tighter SL, smaller size |
| HIGH-impact bearish news | Risk-off | Don't go LONG regardless of signal |

---

*Documentation generated for SMC Pro v1.0 — April 2026*
