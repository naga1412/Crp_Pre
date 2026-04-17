# SMC Pro — Crypto & Stock Analyzer
## Complete Project Documentation & Recreation Guide

> **Single-file React 18 in-browser app** — no build step, no server, no npm.  
> Open `docs/index.html` in Chrome/Edge and it runs fully in the browser.
>
> **Current version:** ~5000 lines | Last updated: April 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technical Analysis (TA) Engine](#2-technical-analysis-ta-engine)
3. [Market Structure Detectors](#3-market-structure-detectors)
4. [The 12 Analysis Modules](#4-the-12-analysis-modules)
5. [Master Bias Score System](#5-master-bias-score-system)
6. [Deep Learning Supervisor (Neural Network)](#6-deep-learning-supervisor-neural-network)
7. [Cross-TF Training Pool & Multi-Window Sync](#7-cross-tf-training-pool--multi-window-sync)
8. [Next-Candle Auto-Validation System](#8-next-candle-auto-validation-system)
9. [Ghost Candle Prediction System](#9-ghost-candle-prediction-system)
10. [Volume Profile (VP) Engine](#10-volume-profile-vp-engine)
11. [News Intelligence System](#11-news-intelligence-system)
12. [Signal Stability System](#12-signal-stability-system)
13. [AI Supervisor Override (aiAdjustedBias)](#13-ai-supervisor-override-aiadjustedbias)
14. [DeepSeek-R1 Integration (Local Ollama)](#14-deepseek-r1-integration-local-ollama)
15. [Hybrid AI Decision System](#15-hybrid-ai-decision-system)
16. [AI Chat Tab (Streaming)](#16-ai-chat-tab-streaming)
17. [Data Sources & WebSocket Feed](#17-data-sources--websocket-feed)
18. [File System Persistence](#18-file-system-persistence)
19. [Scanner System — Hybrid Supervisor](#19-scanner-system--hybrid-supervisor)
20. [Background Validation](#20-background-validation)
21. [UI Layout & Tabs](#21-ui-layout--tabs)
22. [How to Use — Trading Guidelines](#22-how-to-use--trading-guidelines)
23. [Ollama Setup (Windows)](#23-ollama-setup-windows)
24. [Full Recreation Prompt](#24-full-recreation-prompt)

---

## 1. Architecture Overview

```
index.html (single file, ~5000 lines)
├── HTML/CSS — custom dark-theme styles (TradingView color palette)
├── JavaScript (Babel JSX — transpiled in-browser at runtime)
│   ├── TA Engine            — pure-JS indicators (EMA, RSI, ATR, MACD, etc.)
│   ├── Market Structure     — Swing detection, FVG, BOS, Liquidity Map
│   ├── 12 Analysis Modules  — weighted Master Bias Score
│   ├── Neural Network       — 96→64→32→3 feedforward NN (pure JS, self-correcting)
│   ├── Cross-TF Pool        — shared 3000-example training pool (all TFs, all windows)
│   ├── BroadcastChannel     — instant NN weight sync across browser tabs
│   ├── Next-Candle Timer    — auto-validates ghost predictions when real candle forms
│   ├── Ghost Candle Sim     — AI Supervisor + Hybrid NN-powered structural prediction
│   ├── Volume Profile       — 100-bucket histogram, POC/VAH/VAL/HVN/LVN
│   ├── News System          — 11 RSS feeds, keyword scoring, NN memory
│   ├── FS Storage           — File System Access API → user-chosen folder
│   ├── DeepSeek-R1          — local Ollama AI analysis (deepseek-r1:8b)
│   ├── Hybrid AI            — DeepSeek (35%) + Supervisor (65%) blended decision
│   ├── callOllamaJSON()     — global JSON-structured call to local Ollama
│   ├── Scanner              — 200+ symbol parallel scan with Hybrid Supervisor
│   ├── AI Chat Tab          — streaming DeepSeek chatbox with full market context
│   └── React Components     — App, Chart, Sidebar, Scanner, AIChatTab, _MsgBubble
└── CDN dependencies
    ├── React 18 (development UMD)
    ├── Babel Standalone (in-browser JSX transpile)
    └── LightweightCharts 4.1.1 (TradingView chart library)
```

**Key design principles:**
- Everything runs client-side. No backend. No build step.
- NN trains, validates, and self-corrects entirely in the browser using localStorage + File System Access API.
- DeepSeek-R1 runs locally via Ollama — no API key, no cloud, fully private.
- Multiple browser windows on different coins/TFs all share the same NN weights via BroadcastChannel.

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
- Signals: RSI > 70 → overbought (-15), RSI < 30 → oversold (+15), RSI > 55 → mild bullish (+8)
- Used in: StochRSI, RSI Divergence detection

### 2.3 ATR (Average True Range, period=14)
```
calcATR(highs, lows, closes, period=14)
```
- True Range = max(H-L, |H-prevC|, |L-prevC|)
- Used as: volatility unit for SL/TP levels, ghost candle sizing, signal thresholds

### 2.4 MACD (12/26/9)
```
calcMACD(closes, fast=12, slow=26, signal=9)
```
- Returns: `{macd, signal, hist, prevHist, cross, expansion}`
- `cross`: 'bull' (MACD crossed above signal), 'bear' (crossed below)
- `expansion`: |hist| > |prevHist| (momentum growing)

### 2.5 StochRSI (14/3/3)
```
calcStochRSI(closes, per=14, smoothK=3, smoothD=3)
```
- RSI → Stochastic of RSI → double-smoothed K and D lines
- Returns: `{k, d}` (0-100)

### 2.6 CMF (Chaikin Money Flow, period=20)
```
calcCMF(candles, period=20)
```
- Money Flow Multiplier = `(close-low - (high-close)) / (high-low)`
- Range: -1 to +1

### 2.7 Bollinger Bands (20/2)
```
calcBB(closes, period=20, mult=2)
```
- Returns: `{upper, middle, lower, bandwidth}`
- Used for BB squeeze detection, ghost candle dampening

### 2.8 Keltner Channel (20/1.5 ATR)
```
calcKeltner(candles, period=20, mult=1.5)
```
- Used for: BB Squeeze detection (BB inside Keltner = low volatility → big move coming)

### 2.9 RSI Divergence
```
detectRSIDivergence(closes, rsi)
```
- Compares last 12 bars in two 6-bar windows
- Detects: Bullish/Bearish Regular and Hidden divergence (±15 score each)

### 2.10 Candle Type Classifier
```
candleType(candle)
```
- Body/range > 75% → Marubozu | < 10% → Doji | Hammer/Shooting Star by wick ratios

### 2.11 Delta (Volume Imbalance)
```
calcDelta(candles)
```
- `buyFraction = (close-low)/(high-low)` — proxy for buy pressure
- Delta = buy - sell volume estimate (positive = buyers dominated)

### 2.12 VWAP
```
calcVWAP(candles)
```
- Cumulative (volume × typical price) / cumulative volume
- Used as key level for ghost candle pinbar injection

### 2.13 Ichimoku Cloud
```
calcIchimoku(highs, lows, closes)
```
- Tenkan-sen (9), Kijun-sen (26), Senkou Span A & B, Chikou
- Senkou A & B used as key levels in ghost candle system

### 2.14 ADX (Average Directional Index)
```
calcADX(highs, lows, closes, period=14)
```
- Returns: `{adx, plusDI, minusDI}` arrays
- ADX > 25 → strong trend; used to select ghost candle structure sequence

---

## 3. Market Structure Detectors

### 3.1 Swing Highs/Lows
```
detectSwings(highs, lows, lookback=5)
```
- A bar is a swing high if higher than 5 bars on each side
- Keeps last 5 swings in each direction
- Used for: BOS detection, liquidity level mapping, BSL/SSL

### 3.2 Fair Value Gaps (FVG)
```
detectFVG(candles)
```
- Bullish FVG: `candle[i].low > candle[i-2].high`
- Bearish FVG: `candle[i].high < candle[i-2].low`
- Marks imbalances where price typically returns to fill

### 3.3 Break of Structure (BOS)
```
detectBOS(candles, swings)
```
- Bullish BOS: close crosses above a recent swing high
- Bearish BOS: close crosses below a recent swing low

### 3.4 Liquidity Map
```
calcLiquidityMap(candles)
```
- BSL (Buy-Side Liquidity): above swing highs — stop-losses of shorts
- SSL (Sell-Side Liquidity): below swing lows — stop-losses of longs
- Liquidity grabs: spike beyond level + immediate reversal

### 3.5 Absorption Zones
```
detectAbsorption(candles, atr)
```
- High volume (>2× avg) + small range candle (<0.5×ATR) = smart money absorbing

---

## 4. The 12 Analysis Modules

Each module outputs a score (-100 to +100), a weight, and an accuracy tracker. All feed the Master Bias Score via `calcMasterBias`. Module accuracy is tracked live (`_modAccStats[name]`) and displayed in the AI Supervisor panel.

| # | Module | Weight | Description |
|---|--------|--------|-------------|
| 1 | **SMC/EMA Structure** | 20% | EMA alignment, BOS, VP context |
| 2 | **Wyckoff Phase** | 10% | Accumulation/Markup/Distribution/Markdown |
| 3 | **Volume Profile** | 10% | Multi-TF VP POC position (1w/1d/4h/1h blend) |
| 4 | **OI/Funding** | 10% | Binance FAPI funding rate → overcrowding signal |
| 5 | **Momentum** | 10% | RSI, MACD, StochRSI, CMF combined |
| 6 | **Microstructure** | 10% | Volume delta, cumulative delta divergence, absorption |
| 7 | **Sentiment** | 5% | Fear & Greed Index + news sentiment ratio |
| 8 | **Intermarket** | 5% | DXY, SPX, Gold, BTC dominance |
| 9 | **L/S Ratio** | 5% | Long/Short ratio from Binance → crowd positioning |
| 10 | **RS vs BTC** | 5% | Relative strength of coin vs BTC (outperformance) |
| 11 | **Previous Day H/L** | 5% | PDH/PDL sweeps → liquidity grabs + reversal signals |
| 12 | **HTF Bias** | 5% | Higher timeframe macro trend alignment |

### Module Accuracy Tracking
```javascript
_modAccStats[moduleName] = { correct: N, total: N }
// Accuracy = correct/total (minimum 10 predictions required to count)
// DeepSeek context shows live accuracy for each module
// Modules with <10 predictions flagged as "insufficient data"
```

### Module Descriptions (`_MOD_DESC`)
Each module has a plain-English description sent to DeepSeek so it understands the signal:
```javascript
_MOD_DESC = {
  'SMC/Structure': 'EMA alignment + BOS + VP context: are EMAs stacked bullish, has structure broken higher?',
  'Wyckoff':       'Market phase (Accumulation/Markup/Distribution/Markdown)',
  'VolumeProfile': 'Price vs POC/VAH/VAL across 4 timeframes...',
  // ... all 12 modules
}
```

---

## 5. Master Bias Score System

```
calcMasterBias(setup, momSig, microSig, oiSig, imSig, sentSig, mtfVP, price)
```

**Weighted blend of all 12 modules:**

| Module | Weight |
|--------|--------|
| SMC/Structure | 20% |
| Wyckoff | 10% |
| Volume Profile | 10% |
| OI/Funding | 10% |
| Momentum | 10% |
| Microstructure | 10% |
| L/S Ratio | 5% |
| RS vs BTC | 5% |
| Previous Day H/L | 5% |
| HTF Bias | 5% |
| Sentiment | 5% |
| Intermarket | 5% |

**Output:** Score -100 to +100
- > +60 → **Strong Bullish** (teal)
- > +30 → **Bullish**
- -30 to +30 → **Neutral** (gray)
- < -30 → **Bearish**
- < -60 → **Strong Bearish** (red)

**Signal thresholds:** Score ≥ +8 → LONG, Score ≤ -8 → SHORT, else NEUTRAL

### Adaptive Weight System
Module weights adjust based on live prediction accuracy:
```javascript
// Modules with >65% live accuracy get up to 1.4× weight boost
// Modules with <45% accuracy get up to 0.6× weight penalty
adaptedWeights[m] = baseWeight * (accuracyFactor)
// Normalised back to sum=100% after adjustment
```

---

## 6. Deep Learning Supervisor (Neural Network)

### Architecture
```
Input Layer:  96 neurons  (12 features × 8 candles)
Hidden Layer: 64 neurons  (sigmoid activation, Xavier init)
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
| 2 | MACD histogram / ATR normalized | 0-1 | TA Engine |
| 3 | Bollinger Band position | 0-1 | TA Engine |
| 4 | StochRSI K | 0-1 | TA Engine |
| 5 | Volume ratio vs 20-bar avg (capped 3×) | 0-1 | TA Engine |
| 6 | Body direction | 0 or 1 | Candle |
| 7 | Body fraction (body/range) | 0-1 | Candle |
| 8 | Price vs EMA9 | 0 or 1 | TA Engine |
| 9 | Price vs EMA21 | 0 or 1 | TA Engine |
| 10 | Price vs EMA50 | 0 or 1 | TA Engine |
| 11 | 5-bar slope normalized ÷ ATR | 0-1 | TA Engine |
| 12 | CMF (money flow normalized) | 0-1 | TA Engine |

**Sequence window:** Last 8 candles → flattened 96-element input vector

### Training Process
- **Full training:** 20 epochs, LR=0.02, on entire candle history + pool
- **Guard:** Skip if candle count hasn't grown by ≥2 since last training
- **Wrong-direction samples:** Weighted 5× (higher LR focus)
- **Wrong-size samples:** Weighted 3× (size correction)
- **Immediate self-correction:** 8 epochs, LR=0.03, triggered on validation miss

### Self-Correction Flow
```
Predict → log {input, prediction, anchorPrice, anchorTime} → wait for next candle
→ Validate: predicted direction vs actual candle direction
→ If WRONG direction: weight=5 retrain immediately (8 epochs)
→ If WRONG size (magErr > 60% or volErr > 60%): weight=3 retrain
→ _nnStats.corrections++ and _nnStats.sizeCorrections++ for tracking
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
  corrections: 23,     // total direction self-corrections
  sizeCorrections: 8,  // total size self-corrections
}
```

### Storage Keys (localStorage)
| Key | Contents |
|-----|----------|
| `smc_deep_nn_v7` | Neural network weights & biases (JSON) |
| `smc_deep_stats_v7` | Training stats, accuracy, session count |
| `smc_deep_outcomes_v7` | Pending predictions awaiting validation |
| `smc_deep_news_v7` | News→price reaction memory |
| `smc_nn_pool_v1` | Cross-TF training pool (max 3000 examples) |
| `smc_mod_acc_v1` | Per-module accuracy stats (all 12 modules) |

---

## 7. Cross-TF Training Pool & Multi-Window Sync

### Cross-TF Training Pool
Every timeframe you open contributes examples to a **shared pool** (`smc_nn_pool_v1`, max 3000 entries). This means:
- Opening BTC/1h trains on 1h patterns
- Then opening BTC/4h adds 4h patterns to the same model
- Opening ETH/1h adds ETH patterns too
- The NN learns from ALL timeframes and ALL coins simultaneously

```javascript
const _NN_POOL_KEY = 'smc_nn_pool_v1';
const _NN_POOL_MAX = 3000;

// On each TF load: sample up to 600 examples, append to pool
const sampleStep = Math.max(1, Math.floor(freshData.length / 600));
const sampledFresh = freshData.filter((_, i) => i % sampleStep === 0);
let pool = _loadPool();
pool.push(...sampledFresh);
if (pool.length > _NN_POOL_MAX) pool = pool.slice(pool.length - _NN_POOL_MAX);
_savePool(pool);
// Training uses: [...pool, ...freshData] (pool gives generalization, fresh gives recency)
```

### Multi-Window BroadcastChannel Sync
```javascript
const _nnChannel = new BroadcastChannel('smc_nn_sync');
const _WIN_REGISTRY_KEY = 'smc_win_reg_v1';
const _WINDOW_ID = Math.random().toString(36).slice(2); // unique per tab
```

When any window completes training:
1. Broadcasts `{type: 'nn_trained', weights, stats}` on `smc_nn_sync`
2. All other open windows receive it and immediately update `_nnInst` with new weights
3. Ghost candles in ALL tabs instantly improve after any one tab trains

**Fallback for Safari/Firefox (no BroadcastChannel):**
```javascript
window.addEventListener('storage', e => {
  if (e.key === 'smc_nn_sync_fb') {
    const msg = JSON.parse(e.newValue);
    if (msg.type === 'nn_trained') _getNN().fromJSON(msg.weights);
  }
});
```

### Window Registry
Each tab registers itself with `{symbol, tf, lastSeen}` in localStorage. The registry is used to:
- Know which symbol/TF combos are actively being watched
- Coordinate background validation (avoid duplicate work)

---

## 8. Next-Candle Auto-Validation System

### Problem Solved
When you have a ghost candle prediction for BTC/1h at 09:00, the app automatically knows when the next 1h candle closes (10:00) and validates whether the prediction was right.

### Timing Map
```javascript
const _TF_MS = {
  '1m':60000, '3m':180000, '5m':300000, '15m':900000, '30m':1800000,
  '1h':3600000, '2h':7200000, '4h':14400000, '6h':21600000,
  '8h':28800000, '12h':43200000, '1d':86400000, '3d':259200000,
  '1w':604800000
};
```

### Scheduler
```javascript
function scheduleNextCandleValidation(symbol, tf, candles) {
  const tfMs = _TF_MS[tf] || 0;
  const lastTs = candles[candles.length - 1].timestamp;
  const nextCandleAt = lastTs + tfMs;
  const waitMs = nextCandleAt - Date.now() + 4000; // +4s for candle to close fully
  if (waitMs > 0 && waitMs < 7 * 24 * 3600 * 1000) {
    _nextCandleTimers[key] = setTimeout(() => _validateAndReschedule(symbol, tf), waitMs);
  }
}
```

After each validation:
1. Fetches fresh candles
2. Validates all pending predictions for this symbol/TF
3. Retrains if any were wrong
4. **Reschedules** the next timer automatically → continuous forever

### Size Accuracy Validation
Beyond direction, the app tracks **candle size accuracy**:
```javascript
const magErr = Math.abs(predMag - actualMag) / (actualMag || 0.05);
const volErr = Math.abs(predVol - actualVol) / (actualVol || 0.05);
const sizeWrong = magErr > 0.60 || volErr > 0.60; // >60% size error = wrong
if (wrongDirection) weight = 5;   // direction wrong: highest penalty
else if (sizeWrong)  weight = 3;  // size wrong: medium penalty
```
`_nnStats.sizeCorrections` tracks how many times the NN has been corrected for size prediction errors.

### Page Visibility Resume
```javascript
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    // Tab just became visible — immediately validate any missed candles
    _validateAndReschedule(symbol, tf);
  }
});
```
Catches cases where the laptop was closed/screen locked during a candle close.

---

## 9. Ghost Candle Prediction System

Ghost candles are semi-transparent future candle predictions displayed after the live candle. The **AI Supervisor has full control** — ghost candles are always regenerated with complete Supervisor context.

### Context Object (`ctx2`)
Every `futureCandlesSim` call receives the full Supervisor context:
```javascript
ctx2 = {
  // TA indicators (freshly computed)
  momScore, microScore, vpData, macd, squeeze, rsiVal,
  bbData, cmf, stoch, adxVal, biasScore,
  lsLongPct, fundingRate,
  keyLevels: [ema200, vwap, ichiA, ichiB, pdh, pdl],
  // Pre-computed Supervisor NN outputs
  nnProb, nnMag, nnVol,
  aiConf, aiOverride,
  features,              // all 12 modules with scores, weights, accuracy
  accWeightedConviction, // accuracy-weighted total conviction
  // Hybrid AI fields (when DeepSeek has analyzed)
  hybridActive, hybridAgree, hybridConflict,
}
```

When `hybridBias` is available (DeepSeek has analyzed), `_activeBias = hybridBias` replaces `aiAdjustedBias` as the authority — ghost candle direction, size, and structure all reflect the combined verdict.

### Step 1 — Module Agreement Analysis
```javascript
// Each module votes on direction, weighted by live accuracy
agreeW += (module.weight × accuracy) for modules aligned with primary direction
disagreW += ... for opposing modules
agreementRatio = agreeW / (agreeW + disagreeW)  // 0=all disagree, 1=all agree

// Effective confidence: blend Supervisor mlConf with module agreement
effectiveConf = (supAiConf × 0.6) + (agreementRatio × 0.4)
// Accuracy-weighted conviction (normalized 0-1)
accConviction = |accWeightedConviction| / 30
```

### Step 2 — Body Size Multipliers
```javascript
rsiDamp      = RSI > 78 → 0.50 | RSI < 22 → 0.50 | extreme = dampen
biasMult     = max(0.65, min(1.6, 1.0 + accConviction × 0.8))   // accuracy-weighted
confMult     = max(0.70, min(1.4, 0.70 + effectiveConf × 0.70)) // consensus clarity
momMult      = max(0.65, min(1.4, 1.0 + (momScore × dir) / 200))
crowdMult    = crowdContra ? 1.25 : 1.0                           // contra-crowd boost
overridePenalty = supAiOverride ? 0.80 : 1.0                     // NN/indicator conflict
// NEW: Hybrid AI multipliers
hybridBoost  = hybridActive && hybridAgree    ? 1.12 : 1.0  // both AIs agree: +12% body
hybridDamp   = hybridActive && hybridConflict ? 0.72 : 1.0  // AIs split: -28% body, more wicks

baseBody  = max(avgBody×0.70, nnMag×2×atr) × biasMult × confMult × momMult × crowdMult × overridePenalty × hybridBoost × hybridDamp
baseRange = max(avgRange×0.60, nnVol×3×atr) × (aiOverride?1.3:1.0) × (hybridConflict?1.25:1.0)
```

### Step 3 — Structure Sequence Selection
| Condition | Pattern | Meaning |
|-----------|---------|---------|
| `hybridConflict` | deep range | DeepSeek and Supervisor disagree → uncertainty |
| `hybridAgree && effectiveConf > 0.45` | bold impulse | Both AIs agree → front-load impulse |
| BB squeeze active | coil → breakout | Squeeze release |
| `supAiOverride && agreementRatio < 0.55` | H&S ranging | NN/indicator internal conflict |
| `adxVal > 25 && agreementRatio > 0.60` | Elliott 5-wave | Strong trend consensus |
| `adxVal > 15` | impulse-consolidation | Moderate trend |
| Default | ranging | Low conviction |

### Step 4 — OHLC Construction Per Type
| Pattern | Body Size | Wick Behavior | Color |
|---------|-----------|---------------|-------|
| Impulse | 55-100% (NN confidence scales) | Small counter-wicks | Yellow |
| Continuation | 55% | Small counter-wicks | Yellow |
| Pullback | 38% (Fib retracement) | Equal wicks | Orange |
| Inside | 18% | Large equal wicks | Gray |
| Breakout | 88% | Minimal wicks | Blue |
| Pinbar | 7% body | 72% wick in rejection dir | Purple |
| Range | 22% | 22% equal wicks | Gray |

### Step 5 — Key Level Pinbar Injection
```javascript
// Key levels include: EMA200, VWAP, Ichimoku A&B, PDH, PDL, VP POC/VAH/VAL, BB bands
// If ghost candle would hit within 0.4×ATR of a key level → replace with pinbar
// Pullback depth: agreementRatio > 0.70 → shallow (0.24 Fib), aiOverride → deep (0.62)
```

### Step 6 — Decay
Each successive ghost candle shrinks: `decay = max(0.22, 1 - i × 0.045)`

---

## 10. Volume Profile (VP) Engine

```
calcVolumeProfile(candles, numBuckets=100)
```

### How It's Built
1. Find price range (min-max) across all candles
2. Divide into 100 equal price buckets
3. For each candle, distribute volume across overlapping buckets proportionally
4. Highest volume bucket → **POC** (Point of Control)
5. Expand from POC until 70% of total volume covered → **VAH/VAL** (Value Area)
6. Buckets >2× average → **HVN** | Buckets <0.3× average → **LVN**

### Key Levels
| Level | Description | Trading Use |
|-------|-------------|-------------|
| **POC** | Price of Control — highest traded volume | Strong magnet |
| **VAH** | Value Area High — top of 70% volume zone | Resistance above value area |
| **VAL** | Value Area Low — bottom of 70% volume zone | Support below value area |
| **HVN** | High Volume Node | Price stalls here (congestion) |
| **LVN** | Low Volume Node | Price moves quickly through |

### Canvas Rendering
- HTML5 Canvas overlay positioned exactly on chart (right 15% of chart width)
- Redrawn on every viewport change: `subscribeVisibleLogicalRangeChange`, `wheel`, `mouseup`, `touchend`
- `priceToCoordinate()` called fresh on each redraw

### Multi-TF VP
Computed for 1w, 1d, 4h, 1h. Each TF's POC feeds the Master Bias VP score:
- 1w=40%, 1d=30%, 4h=20%, 1h=10%

---

## 11. News Intelligence System

### RSS Feed Sources (11 feeds)
- Yahoo Finance (markets + crypto)
- CNBC (markets, world economy, crypto)
- CoinDesk
- Google News (Fed/FOMC, geopolitical, crypto, markets, macro/CPI)

### Processing Pipeline
1. Fetch all feeds in parallel via 3 CORS proxies (allorigins.win → corsproxy.io → codetabs.com)
2. Parse with DOMParser → extract title, link, pubDate
3. Score with bullish/bearish keyword matching
4. Deduplicate by first 65 chars of headline
5. Log HIGH-impact + directional news to NN memory

### Impact Levels
| Impact | Label | NN Memory? |
|--------|-------|-----------|
| ≥3 | HIGH | ✅ Logged |
| 1.5-3 | MEDIUM | ❌ |
| <1.5 | LOW | ❌ |

### News Reaction Memory
- On HIGH-impact arrival: log feature snapshot + news metadata
- After 5 candles: validate price reaction vs predicted direction
- Stored in `smc_deep_news_v7` + `news_reactions.json` (disk)

---

## 12. Signal Stability System

### Layer 1 — EMA Smoothing
```javascript
biasEMA = Math.round(biasEMA × 0.65 + newScore × 0.35)
```
New readings only have 35% influence — smooths micro-fluctuations from 30s polls.

### Layer 2 — Hold Lock
```javascript
if (newSignal !== currentSignal) {
  holdCount++;
  if (holdCount < 2 && currentSignal !== null) return currentSignal; // hold
}
// Flips only after 2 consecutive readings in new direction
```

### Result
Signal is stable for 15+ minutes under normal volatility. Only changes when the market has genuinely shifted.

---

## 13. AI Supervisor Override (aiAdjustedBias)

```javascript
const aiAdjustedBias = useMemo(() => {
  if (!masterBias) return null;
  if (!nnResult || nnResult.totalCandles < 150)
    return { ...masterBias, aiWeight: 0, aiOverride: false };

  const nnProb = nnResult.prob ?? 0.5;
  const nnBull = nnProb >= 0.58, nnBear = nnProb <= 0.42;
  const liveAcc = nnResult.recentAcc ?? 50;
  const nnConf = Math.abs(nnProb - 0.5) * 2;  // 0-1 directional confidence
  const maxWeight = liveAcc >= 65 ? 0.55 : liveAcc >= 60 ? 0.48 : 0.40;
  const nnWeight = Math.min(maxWeight, Math.max(0.05, nnConf * 0.6 + Math.max(0, (liveAcc-50)/80)));
  const nnContrib = Math.round((nnBull?60:nnBear?-60:0) * nnWeight);
  const aiScore = Math.max(-100, Math.min(100, Math.round(masterBias.score * (1-nnWeight)) + nnContrib));
  // ...
}, [masterBias, nnResult]);
```

| Live Accuracy | Max NN Weight |
|---------------|--------------|
| ≥ 65% | 55% |
| ≥ 60% | 48% |
| < 60% | 40% |
| < 150 candles | 5% (essentially inactive) |

### Status Badges
- **"⚠️ NN flipped signal"** — NN changed the final signal vs raw indicators
- **"✓ NN confirms"** — NN agrees with indicator consensus
- **"🧠 X% NN weight"** — shows current blend percentage

---

## 14. DeepSeek-R1 Integration (Local Ollama)

### Overview
DeepSeek-R1:8b runs **entirely on your local machine** via Ollama. No API key. No cloud. No cost. Fully private.

### `callOllamaJSON` (Global Helper)
```javascript
async function callOllamaJSON(systemPrompt, userMsg, maxTokens = 400) {
  const url = (localStorage.getItem('smc_ollama_url') || 'http://localhost:11434') + '/api/chat';
  const model = localStorage.getItem('smc_ollama_model') || 'deepseek-r1:8b';
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      model, stream: false,
      messages: [{role:'system',content:systemPrompt},{role:'user',content:userMsg}],
      options: {num_predict: maxTokens, temperature: 0.2}  // low temp for reliable JSON
    })
  });
  const d = await resp.json();
  // Ollama v0.7+ puts DeepSeek reasoning in message.thinking (NOT in content)
  const raw = (d?.message?.content || d?.message?.thinking || '').trim();
  const m = raw.match(/\{[\s\S]*\}/);  // extract first JSON object
  if (!m) throw new Error('No JSON in DeepSeek response');
  return JSON.parse(m[0]);
}
```

### What DeepSeek Receives
For each analysis, DeepSeek gets the **full 12-module context**:
```
Market: BTC/USDT / 1h | Price: 94520
AI Supervisor Score: +67 (Strong Bullish) | Signal: LONG
NN prob=72% acc=68% override=no

12 Modules:
[▲BULL] SMC/Structure: +78 | w=20% | liveAcc=71% → EMA alignment + BOS
[▲BULL] Wyckoff: +45 | w=10% | liveAcc=64% → Market phase (Markup)
[▼BEAR] OI/Funding: -22 | w=10% | liveAcc=58% → Funding rate extreme
...
RSI=58 L/S=61.2%L Funding=0.0008 FG=72 (Greed)
PDH=95000 PDL=92000
RS_vs_BTC=+1.24%
```

### Ollama v0.7+ Response Format
```javascript
// v0.7+ puts reasoning in separate field — NOT in <think> tags in content
const nativeThinking = d?.message?.thinking || '';
const nativeContent  = d?.message?.content  || '';
// Re-wrap so _parseDeepSeek works uniformly:
return nativeThinking
  ? `<think>${nativeThinking}</think>\n${nativeContent}`
  : nativeContent;
```

### Configurable Settings (localStorage)
| Key | Default | Description |
|-----|---------|-------------|
| `smc_ollama_url` | `http://localhost:11434` | Ollama server URL |
| `smc_ollama_model` | `deepseek-r1:8b` | Model to use |
| `smc_ai_provider` | `deepseek` | Active provider (deepseek/gemini/openai/anthropic) |

---

## 15. Hybrid AI Decision System

Both AI systems independently analyze the market, then their verdicts are **blended** into a single `hybridBias` that drives everything: ghost candles, signal display, scanner results.

### `runDeepSeekAnalysis` (Auto-triggered)
```javascript
const runDeepSeekAnalysis = useCallback(async (ctx) => {
  // Auto-runs on symbol/TF change, debounced 30s
  // Sends full 12-module scores + market data to DeepSeek
  // DeepSeek returns: {signal, dsScore:-60..+60, conviction:0-100,
  //                    reason, agree:[], disagree:[], candle}
  setDeepSeekAnalysis({...result, symbol, tf, timestamp});
}, []);
```

### `hybridBias` useMemo — The Blend Formula
```javascript
const hybridBias = useMemo(() => {
  if (!aiAdjustedBias) return null;
  if (!deepSeekAnalysis || deepSeekAnalysis.symbol !== symbol)
    return { ...aiAdjustedBias, source: 'Supervisor', hybridActive: false };

  const aiScore = aiAdjustedBias.score;
  const dsScore = Math.max(-60, Math.min(60, deepSeekAnalysis.dsScore));
  const aiDir = aiScore >= 8 ? 1 : aiScore <= -8 ? -1 : 0;
  const dsDir = dsScore >= 8 ? 1 : dsScore <= -8 ? -1 : 0;
  const agree   = (aiDir === dsDir) || (aiDir === 0) || (dsDir === 0);
  const conflict = !agree && aiDir !== 0 && dsDir !== 0;

  // Weighted blend: 65% Supervisor + 35% DeepSeek
  let hybrid = Math.round(aiScore * 0.65 + dsScore * 0.35);
  if (agree && aiDir !== 0)  hybrid = Math.round(hybrid * 1.10); // agree boost +10%
  if (conflict)               hybrid = Math.round(hybrid * 0.70); // conflict dampen -30%
  hybrid = Math.max(-100, Math.min(100, hybrid));

  return {
    ...aiAdjustedBias,
    score: hybrid, hybridSignal, hybridActive: true,
    agree, conflict,
    supervisorScore: aiScore, supervisorSignal: aiAdjustedBias.aiSignal,
    dsScore, dsSignal, dsConviction, dsReason, dsCandle,
  };
}, [aiAdjustedBias, deepSeekAnalysis, symbol, tf]);
```

### How It Affects Everything
| Component | Without DeepSeek | With DeepSeek |
|-----------|-----------------|---------------|
| Signal shown | Supervisor score | Hybrid blended score |
| Ghost candle direction | aiAdjustedBias.aiSignal | hybridBias.aiSignal |
| Ghost candle body size | Normal | +12% when agree / -28% when conflict |
| Ghost candle structure | Normal | "bold impulse" (agree) / "deep range" (conflict) |
| Chart re-render | On aiAdjustedBias change | Also on hybridBias change |

### Hybrid Decision UI Panel
Shown in sidebar below the main signal:
```
┌─────────────────────────────────────────────────┐
│ 🤝 Hybrid Decision                   [timestamp] │
├──────────────┬──────────────┬───────────────────┤
│ 🧠 Supervisor │  🤝/⚠️       │ 🐋 DeepSeek       │
│    LONG       │   agree      │    LONG           │
│    +67        │              │    Conv 82%       │
├─────────────────────────────────────────────────┤
│         Combined Signal: LONG +71               │
│   "Key level rejection at PDH, RSI divergence" │
│ ✓ Both AIs agree — higher conviction setup      │
└─────────────────────────────────────────────────┘
```

---

## 16. AI Chat Tab (Streaming)

### Architecture
- **`_MsgBubble` component** — standalone React component (extracted to fix hooks-in-map black screen bug)
- Each message has: `role`, `content`, `thinking` (DeepSeek reasoning), `streaming` (bool), `_sid` (unique stream ID)
- DeepSeek reasoning shown in collapsible "▼ Show DeepSeek reasoning" section

### Streaming Implementation
DeepSeek chat uses `stream: true` so tokens appear instantly:
```javascript
const resp = await fetch(`${ollamaUrl}/api/chat`, {
  method: 'POST',
  body: JSON.stringify({model, messages, stream: true, options: {num_predict: 650}})
});
const reader = resp.body.getReader();
const dec = new TextDecoder();
let buf = '';
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  buf += dec.decode(value, {stream: true});
  const lines = buf.split('\n');
  buf = lines.pop() || '';
  for (const line of lines) {
    const chunk = JSON.parse(line);
    accContent += chunk.message?.content || '';
    accThinking += chunk.message?.thinking || '';
    if (Date.now() - lastFlush > 80) flushMsg(); // update UI every 80ms
  }
}
```

### Smart Prompt Sizing
```javascript
// Detect simple/greeting messages → skip heavy market context for speed
const isSimple = /^(hi|hello|hey|thanks|ok|okay|yes|no|sure)[\s!?.]*$/i.test(q);
// Simple: minimal 1-line system prompt + num_predict:250 → responds in ~2s
// Analysis: full 12-module context + num_predict:650 → responds as tokens stream
```

### Blinking Cursor
```jsx
{msg.streaming && (
  <span style={{display:'inline-block', width:7, height:13, background:'#5b9bd5',
    animation:'blink 0.8s step-end infinite'}}>&#8203;</span>
)}
```

### Suggestion Chips
Pre-built quick-send chips:
- "Why is the signal showing LONG/SHORT?"
- "Which modules have highest accuracy?"
- "Is the NN prediction reliable?"
- "Explain the module disagreement"
- "What does L/S ratio mean here?"
- "Should I trust the ghost candles?"

### Context Passed to DeepSeek
```
symbol, tf, price, bias score, signal, AI override status
NN probability, accuracy, recent accuracy, pool size, corrections
All 12 module scores with weights + live accuracy %
RSI, L/S ratio, funding rate, Fear & Greed
PDH/PDL (with SWEPT flag if applicable)
RS vs BTC score
Entry/SL/TP levels from trade setup
```

### Providers Supported
| Provider | API Key? | Best For |
|----------|---------|---------|
| 🤖 DeepSeek-R1 (Local) | No | Full analysis, free, private |
| 🆓 Gemini (Google) | Yes (free tier) | Fast responses, large context |
| GPT-4o (OpenAI) | Yes | Code modifications |
| Claude (Anthropic) | Yes | Reasoning tasks |

---

## 17. Data Sources & WebSocket Feed

### Crypto (Binance)
- **REST Spot:** `https://api.binance.com/api/v3/klines` (up to 1000 candles)
- **REST Futures:** `https://fapi.binance.com/fapi/v1/klines` (fallback)
- **OI + Funding:** `https://fapi.binance.com/fapi/v1/openInterest` + `/fundingRate`
- **L/S Ratio:** `https://fapi.binance.com/futures/data/globalLongShortAccountRatio`
- **WebSocket live:** `wss://stream.binance.com:9443/ws/{symbol}@kline_{tf}`

### Stocks/ETFs/Indices (Yahoo Finance)
- Via CORS proxies: allorigins.win → corsproxy.io fallback
- Yahoo Finance chart API: `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`
- 4h candles: aggregated from 1h data (4 consecutive 1h candles)

### Fear & Greed Index
- `https://api.alternative.me/fng/?limit=2`

### Intermarket Data
- DXY, SPX, Gold via Yahoo Finance
- BTC dominance: CoinGecko global API

### Poll Frequency
- **Main poll:** Every 30 seconds (`quiet=true`)
- **Quiet poll optimization:** If last candle timestamp unchanged → skip all TA computation
- **Full load:** On symbol/TF switch → 600 candles + full TA + NN training

---

## 18. File System Persistence

### Setup
File System Access API (Chrome/Edge 86+).
1. User clicks "Connect Folder" → browser shows folder picker
2. Directory handle stored in IndexedDB for persistence across sessions
3. Auto-reconnect on next app start if permission still granted

### Files Written
| File | Contents | Trigger |
|------|----------|---------|
| `model_weights.json` | NN layers, weights, biases | Sessions or corrections changed |
| `training_stats.json` | Accuracy, sessions, corrections | Same |
| `prediction_outcomes.json` | Pending + validated predictions | Same |
| `news_reactions.json` | News → price reaction log | Same |
| `predictions_log.json` | Every prediction made (up to 2000) | Each prediction |
| `training_history.json` | Training batch history (up to 5000) | Each training run |

### Smart Write Guard
```javascript
async function fsSaveModel(force = false) {
  const sessChanged = _nnStats.sessions !== _fsDirtySessionStamp;
  const corChanged  = _nnStats.corrections !== _fsDirtyCorStamp;
  if (!force && !sessChanged && !corChanged) return; // skip unchanged
  // ... write files ...
}
```

---

## 19. Scanner System — Hybrid Supervisor

The scanner analyzes 200+ symbols in parallel, then runs the Hybrid Supervisor on the top picks.

### Phase 1 — Mass Scan (`scanAsset`)
For every symbol:
1. Fetch current TF candles + higher TF (HTF_MAP: 1m→5m, 1h→4h, 4h→1d, etc.) in parallel
2. Run `generateSetup` → all 12 module scores → `calcMasterBias` → raw `score`
3. **NN-adjusted score:** apply same `aiAdjustedBias` logic as main chart
```javascript
const nnWeight = Math.min(0.35, nnConf * nnAccFactor * 0.50);
const nnContrib = Math.round((nnBull?60:nnBear?-60:0) * nnWeight);
aiScore = Math.max(-100, Math.min(100, Math.round(score*(1-nnWeight)) + nnContrib));
```
4. MTF confirmation: current TF signal vs higher TF signal
5. Confirmation level: CONFIRMED / PROBABLE / WEAK / DIVERGING / NEUTRAL

### Phase 2 — Hybrid Supervisor (`runHybridAnalysis`)
Runs **after** scan completes, non-blocking. Only fires when DeepSeek provider selected.
1. Pick top 8 CONFIRMED/PROBABLE assets by `|aiScore|`
2. For each, send compact prompt to DeepSeek:
```
BTC/USDT 1h | Signal:LONG | AI Score:+72 | Conf:84% | Phase:Markup
HTF(4h):LONG | Confirmation:CONFIRMED | NN:74%bull conf:68% | 24h:+2.4%
Modules: SMC+78 Wyckoff+45 VP+32 | AIOverride:no | MTFagree:yes
Return ONLY JSON: {"signal","dsScore","conviction","reason","quality":"A/B/C/D"}
```
3. Blend into hybrid score: `hybrid = ai×0.65 + ds×0.35` with agree/conflict modifiers
4. Update scan cards live as each asset is analyzed (progressive streaming)
5. Final call: **market overview** summary of the entire scan

### Market Overview (DeepSeek)
```
Market scan — 1h: 68 bullish vs 42 bearish. Top hybrid longs: BTC, SOL, ETH.
Returns: {marketBias, strength:0-100, topLong[], topShort[], summary, watch}
```
Displayed as full-width banner above the scanner grid with clickable top picks.

### Scanner Card Display
Each card shows:
- **Symbol** + sparkline + 24h change %
- `[LONG]` signal badge + `[4h LONG]` HTF badge
- `[✓ CONFIRMED]` confirmation badge
- `🧠 LONG` NN badge (color: purple=agrees, orange=disagrees)
- `AI +72` badge (NN-adjusted score, shown if different from raw)
- `🤝 LONG +75` hybrid badge (green=agree, orange=conflict)
- `A` quality grade from DeepSeek (A=prime, B=good, C=fair, D=skip)
- DeepSeek reason: *"RSI divergence at key support, funding neutral"*
- Bias bar (uses hybridScore when available)
- `Conf 84%` indicator confidence + `DS 82%` DeepSeek conviction

### Filter Buttons
| Filter | Meaning |
|--------|---------|
| All | Show everything |
| ✓ Confirmed | All 4 pillars agree (safest trades) |
| ~ Probable | TFs agree, momentum/micro slightly off |
| ✗ Weak | Only bias shows direction (low conviction) |
| ⚡ Diverging | TF conflict — do NOT trade |
| 🤝 Hybrid | **Both DeepSeek and Supervisor agree** (highest conviction) |

### Sort Options
| Sort | Orders by |
|------|-----------|
| AI Score | `|aiScore|` (NN-adjusted, descending) |
| 🤝 Hybrid Score | `|hybridScore|` → `|aiScore|` → `|score|` |
| Most Bullish | hybridScore descending |
| Most Bearish | hybridScore ascending |
| 24h Change | absolute % change |
| Alphabetical | symbol name |

### Scanner Controls
- **Asset count:** 1–500 symbols (default: 220)
- **Rescan interval:** 0.5–60 minutes (default: 2 min)
- **Timeframe:** any of 1m/5m/15m/1h/4h/1d/1w
- **Watchlist:** add/remove favorites, shown at top of results
- **`🤝 Hybrid` button:** re-trigger DeepSeek analysis on current results without rescanning

---

## 20. Background Validation

### Purpose
When you switch from BTC/1h to ETH/4h, the app continues tracking whether BTC/1h predictions were correct.

### Implementation
```javascript
backgroundValidateAll(currentSym, currentTf)
// Triggered on every chart symbol/TF switch
```
1. Read all pending predictions from `smc_deep_outcomes_v7`
2. Find up to 5 other symbol/TF combos with unvalidated predictions
3. Fetch 80 candles each (500ms stagger to avoid rate limiting)
4. Validate and retrain if wrong
5. **5-minute cooldown** between runs

---

## 21. UI Layout & Tabs

### Left Sidebar (~320px, scrollable)
- Asset price + live WebSocket ticker
- Signal button (LONG/SHORT/NEUTRAL, pulsing animation) + AI supervisor badge
- **Hybrid Decision Panel** (DeepSeek + Supervisor combined verdict)
- **AI Supervisor Card** (NN score, raw score, NN weight%, mlConf bar)
- Trade setup (entry zone, SL, TP1, TP2, RR ratios)
- Next candle prediction (direction, type, body size, range size)
- SMC Analysis (BOS, FVG, liquidity)
- Wyckoff phase + Volume Profile levels
- Multi-timeframe bias grid
- Momentum indicators (RSI, MACD, StochRSI, CMF, BB squeeze)
- Microstructure (delta dominance, absorption count, BVIX)
- OI/Funding (crypto only — Binance FAPI)
- L/S Ratio (crowd positioning)
- RS vs BTC (relative strength)
- Intermarket (DXY, SPX, Gold, BTC dominance)
- Sentiment (Fear & Greed + news sentiment)

### Main Chart Area
- LightweightCharts 4.1.1 candlestick chart (dark TV theme)
- EMA9 (blue), EMA21 (orange), EMA50 (purple)
- FVG zones (horizontal bands — teal/red)
- Liquidity levels (BSL/SSL horizontal lines)
- Volume Profile canvas overlay (right 15%)
- Ghost candles (up to 50 semi-transparent future candles, AI Supervisor controlled)
- Ghost markers: shape-only dots (size=0.5, no text labels)

### Top Tab Bar
| Tab | Content |
|-----|---------|
| **Chart** | Main candlestick chart + full sidebar analysis |
| **Scanner** | 200+ symbol scan with Hybrid Supervisor analysis |
| **News** | RSS news feed with sentiment + NN memory logging |
| **AI Chat** | Streaming DeepSeek-R1 chatbox with full market context |

---

## 22. How to Use — Trading Guidelines

### Initial Setup
1. Open `docs/index.html` in Chrome or Edge (File System API required)
2. Run `start_ollama.bat` to start DeepSeek-R1 local AI (see Section 23)
3. Click **"📁 Connect Folder"** → select any folder for model persistence
4. Select AI provider: **DeepSeek-R1 (Local — FREE)** in the AI Chat tab
5. Search for an asset (Ctrl+K or click search bar)

### Reading the Signal
1. **Signal Button:** Green LONG (+8 score), Red SHORT (-8 score), Gray NEUTRAL
2. **AI Supervisor Badge:** "⚠️ NN flipped" = trust NN if recentAcc > 60%; "✓ NN confirms" = high conviction
3. **Hybrid Decision Panel:** If both AIs agree → ✓ higher conviction. If conflict → ⚠️ reduce size/wait
4. **Score Breakdown:** Raw score (indicators only) vs AI score (NN adjusted)

### Ghost Candle Reading
| Color | Pattern | Meaning |
|-------|---------|---------|
| 🟡 Yellow | Impulse/Continuation | Strong directional move expected |
| 🟠 Orange | Pullback | Brief counter-move before continuation |
| ⬜ Gray | Inside/Range | Consolidation — wait for breakout |
| 🔵 Blue | Breakout | Breakout from consolidation |
| 🟣 Purple | Pinbar | Rejection at key level (POC/BB/VWAP) |

**When ghost candles go ranging/choppy:** AIs are in conflict or low conviction — don't trade.

### Scanner Usage
1. Go to **Scanner** tab → auto-scans 200+ crypto pairs
2. Sort by **🤝 Hybrid Score** for highest-conviction picks
3. Filter by **🤝 Hybrid** to see only assets both AIs agree on
4. Check the **Market Overview panel** for overall market direction from DeepSeek
5. Grade A picks (🟡 A badge) = DeepSeek rated as prime setup
6. Click any card → opens that symbol on the main chart

### When NOT to Trade
- Master Bias -8 to +8 (NEUTRAL) — no edge
- Ghost candles all gray inside/range — consolidating
- RSI > 78 or < 22 — exhaustion, avoid momentum trades
- OI Funding > 0.001 or < -0.001 — squeeze risk
- ⚠️ Hybrid conflict — both AIs disagree, reduce size
- HIGH-impact bearish news + LONG signal — wait for confirmation

### NN Learning Period
| Stage | Candle Count | Trust Level |
|-------|-------------|-------------|
| Learning | < 150 | Ignore NN — use indicator signal only |
| Developing | 150-500 | NN has minor influence (5-15% weight) |
| Trained | 500+ | NN starts being meaningful |
| Reliable | recentAcc > 60% | Trust NN overrides |
| Expert | recentAcc > 65% | NN has 55% influence — can override indicators |

---

## 23. Ollama Setup (Windows)

### Prerequisites
1. Install Ollama: https://ollama.ai/download
2. Pull the model: open Command Prompt → `ollama pull deepseek-r1:8b`

### Starting Ollama with CORS (Required for GitHub Pages)
The app may be served from `file://` or a GitHub Pages domain. Ollama needs CORS open.

**Method 1 — Double-click the batch file:**
```
C:\Users\nagar\Downloads\claude\start_ollama.bat
```

**Method 2 — Manual PowerShell:**
```powershell
Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue
Start-Sleep 2
$p = New-Object System.Diagnostics.ProcessStartInfo
$p.FileName = 'C:\Users\nagar\AppData\Local\Programs\Ollama\ollama.exe'
$p.Arguments = 'serve'
$p.UseShellExecute = $false
$p.EnvironmentVariables['OLLAMA_ORIGINS'] = '*'
$p.EnvironmentVariables['OLLAMA_HOST'] = '0.0.0.0'
$proc = [System.Diagnostics.Process]::Start($p)
```

> ⚠️ **Critical:** Setting `OLLAMA_ORIGINS=*` via `set` in a batch file does NOT work because the env var doesn't inherit to child processes started with `Start-Process`. The PowerShell `ProcessStartInfo.EnvironmentVariables` approach injects the vars directly before the process starts.

### Verify It's Working
```
http://localhost:11434/api/tags
```
Response should include `deepseek-r1:8b` in the models list.

### Speed Notes
- **First response:** Slow (model loads into VRAM) — 15-30s
- **Subsequent:** Faster (model stays loaded) — 5-15s
- **Simple greetings:** The app detects these and uses a minimal prompt → ~2-5s
- **Analysis questions:** Full 12-module context → 15-40s (streams in real-time)
- **Scanner batch:** ~15-20s per asset for top 8 picks

---

## 24. Full Recreation Prompt

Copy this prompt into any AI assistant to recreate this exact project from scratch:

```
Build a single-file React 18 trading analyzer called "SMC Pro — Crypto & Stock Analyzer"
as docs/index.html with NO build step. Use CDN scripts only:
- React 18 UMD development
- Babel Standalone (in-browser JSX)
- LightweightCharts 4.1.1 (TradingView)

Dark TradingView theme: bg=#131722, panel=#1e222d, border=#2a2e39
Bull=#26a69a, Bear=#ef5350, Primary=#2962ff

════════════════════════════════════════
SECTION 1: TA ENGINE (pure-JS functions)
════════════════════════════════════════
calcEMA(arr, period), calcRSI(closes, 14), calcATR(H,L,C, 14),
calcMACD(closes, 12,26,9) → {macd,signal,hist,cross,expansion},
calcStochRSI(closes, 14,3,3), calcCMF(candles, 20),
calcBB(closes, 20,2), calcKeltner(candles, 20,1.5),
calcVWAP(candles), calcIchimoku(H,L,C), calcADX(H,L,C, 14),
detectRSIDivergence, candleType, calcDelta, detectAbsorption

════════════════════════════════════════
SECTION 2: MARKET STRUCTURE
════════════════════════════════════════
detectSwings(H,L, lookback=5), detectFVG(candles),
detectBOS(candles, swings), calcLiquidityMap(candles)

════════════════════════════════════════
SECTION 3: VOLUME PROFILE (100 buckets)
════════════════════════════════════════
calcVolumeProfile → {poc, vah, val, hvn[], lvn[]}
Draw as HTML5 Canvas overlay on right 15% of chart.
Redraw on every viewport change (subscribeVisibleLogicalRangeChange + wheel + mouseup)

════════════════════════════════════════
SECTION 4: 12 ANALYSIS MODULES
════════════════════════════════════════
Each outputs score -100 to +100:
1. SMC/EMA Structure (20%) — EMA alignment, BOS, VP context
2. Wyckoff Phase (10%) — Accumulation/Markup/Distribution/Markdown
3. Volume Profile (10%) — Multi-TF POC position (1w40%,1d30%,4h20%,1h10%)
4. OI/Funding (10%) — Binance FAPI funding rate overcrowding
5. Momentum (10%) — RSI+MACD+StochRSI+CMF
6. Microstructure (10%) — delta dominance, cumulative delta, absorption
7. Sentiment (5%) — Fear & Greed + news ratio
8. Intermarket (5%) — DXY, SPX, Gold, BTC dominance
9. L/S Ratio (5%) — long/short crowd positioning
10. RS vs BTC (5%) — relative strength vs BTC
11. Previous Day H/L (5%) — PDH/PDL sweeps
12. HTF Bias (5%) — higher timeframe macro trend

calcMasterBias: weighted sum of all 12 + adaptive weight system (accurate modules get 1.4× boost)
_modAccStats[name] = {correct, total} — live accuracy per module

════════════════════════════════════════
SECTION 5: SIGNAL STABILITY
════════════════════════════════════════
biasEMA = biasEMA×0.65 + newScore×0.35
Hold lock: flip only after 2 consecutive readings in new direction

════════════════════════════════════════
SECTION 6: NEURAL NETWORK (pure JS)
════════════════════════════════════════
Architecture: [96, 64, 32, 3] — Xavier init, sigmoid, backprop
12 features × 8 candle sequence = 96 inputs
Outputs: direction prob, magnitude, volatility
Training: 20 epochs, LR=0.02, wrong samples weight=5, size-wrong weight=3
Self-correction: immediate 8-epoch retrain on wrong direction/size prediction
smc_deep_nn_v7 / smc_deep_stats_v7 / smc_deep_outcomes_v7

Cross-TF pool (smc_nn_pool_v1, max 3000):
- Every TF opened samples 600 examples into shared pool
- Training uses pool + fresh data for generalization

BroadcastChannel('smc_nn_sync'):
- On training complete → broadcast weights to all open tabs
- All tabs update _nnInst immediately

Next-candle timers:
- _TF_MS = {1m:60000, 5m:300000, 1h:3600000, ...}
- Set timeout at lastTs + tfMs + 4s
- On fire: fetch fresh candles, validate predictions, retrain if wrong, reschedule

Size accuracy tracking:
- magErr = |predMag - actualMag| / actualMag
- volErr = |predVol - actualVol| / actualVol
- sizeWrong if magErr > 0.60 or volErr > 0.60

════════════════════════════════════════
SECTION 7: AI SUPERVISOR OVERRIDE
════════════════════════════════════════
aiAdjustedBias useMemo:
  maxWeight = liveAcc>=65?0.55:liveAcc>=60?0.48:0.40
  nnWeight = min(maxWeight, max(0.05, nnConf×0.6 + max(0,(liveAcc-50)/80)))
  aiScore = masterBias×(1-nnWeight) + (prob-0.5)×200×nnWeight

════════════════════════════════════════
SECTION 8: GHOST CANDLE SYSTEM
════════════════════════════════════════
futureCandlesSim(candles, signal, atr, count=50, ctx):
ctx includes: nnProb, nnMag, nnVol, aiConf, aiOverride, features[],
              accWeightedConviction, hybridActive, hybridAgree, hybridConflict,
              keyLevels (EMA200, VWAP, Ichimoku A&B, PDH, PDL)

Module agreement analysis → agreementRatio → effectiveConf
Body size: baseBody = max(avgBody×0.70, nnMag×2×atr) × biasMult × confMult × momMult × hybridBoost/Damp
hybridBoost = hybridAgree ? 1.12 : 1.0
hybridDamp = hybridConflict ? 0.72 : 1.0
baseRange × (hybridConflict ? 1.25 : 1.0)

Structure sequences (highest priority first):
1. hybridConflict → deep range pattern
2. hybridAgree + effectiveConf>0.45 → bold impulse
3. BB squeeze → coil→breakout
4. supAiOverride + agreementRatio<0.55 → H&S ranging
5. adxVal>25 + agreementRatio>0.60 → Elliott 5-wave
6. adxVal>15 → impulse-consolidation
7. default → ranging

Candle types: impulse(yellow), continuation(yellow), pullback(orange),
              inside(gray), breakout(blue), pinbar(purple), range(gray)
Inject pinbar near key levels (within 0.4×ATR)
Decay: max(0.22, 1 - i×0.045) per candle

════════════════════════════════════════
SECTION 9: DEEPSEEK-R1 INTEGRATION
════════════════════════════════════════
Ollama local: POST http://localhost:11434/api/chat
Model: deepseek-r1:8b (configurable via localStorage)

callOllamaJSON(sysPrompt, userMsg, maxTokens=400):
- stream:false for structured JSON calls
- Extract JSON via /\{[\s\S]*\}/ regex from response
- Ollama v0.7+: reasoning in message.thinking, answer in message.content

hybridBias useMemo:
  hybrid = aiScore×0.65 + dsScore×0.35
  if agree → hybrid×1.10
  if conflict → hybrid×0.70
  hybridActive: true when DeepSeek has analyzed this symbol/TF

runDeepSeekAnalysis: auto-triggers on symbol/TF change, debounced 30s
Sends all 12 module scores + market data, gets {signal, dsScore, conviction, reason, agree[], disagree[], candle}

Hybrid Decision UI: 3-column panel (Supervisor|agree/conflict|DeepSeek) + combined result box

════════════════════════════════════════
SECTION 10: AI CHAT TAB (STREAMING)
════════════════════════════════════════
_MsgBubble component (standalone — NOT inline in map):
- Shows collapsible DeepSeek thinking section
- Blinking cursor ▌ (@keyframes blink) while streaming
- msg.streaming flag, msg._sid unique stream ID

Stream path for DeepSeek (stream:true):
- ReadableStream + NDJSON parsing
- Accumulate content + thinking, flush to React state every 80ms
- Simple messages (hi/hello): minimal prompt + num_predict:250
- Analysis questions: full 12-module context + num_predict:650

════════════════════════════════════════
SECTION 11: SCANNER + HYBRID SUPERVISOR
════════════════════════════════════════
scanAsset: fetch current TF + HTF → run all 12 modules → compute aiScore (NN-adjusted)
MTF confirmation: CONFIRMED/PROBABLE/WEAK/DIVERGING

runHybridAnalysis (called after each scan):
- Pick top 8 CONFIRMED/PROBABLE by |aiScore|
- For each: send compact prompt to DeepSeek → get {signal, dsScore, conviction, reason, quality:A/B/C/D}
- Blend: hybrid = ai×0.65 + ds×0.35 + agree/conflict modifiers
- Update scan cards progressively as each is analyzed
- Final: market overview call → {marketBias, strength, topLong[], topShort[], summary, watch}

Market Overview panel: full-width banner with market bias, strength bar, clickable top picks

Scanner cards show: aiScore badge, 🤝hybrid badge, quality grade, DS reason, DS conviction
Filter: 🤝Hybrid = both AIs agree + |hybridScore|>=10
Sort: 🤝Hybrid Score = hybridScore > aiScore > raw score

════════════════════════════════════════
SECTION 12: DATA & STORAGE
════════════════════════════════════════
Crypto: Binance REST + WebSocket, FAPI for OI/Funding/L-S ratio
Stocks: Yahoo Finance via CORS proxy
Fear & Greed: api.alternative.me/fng
Intermarket: DXY, SPX, Gold (Yahoo); BTC dominance (CoinGecko)
Poll: every 30s, quiet mode skips TA if last candle unchanged

File System Access API:
- IndexedDB for directory handle auto-reconnect
- Files: model_weights, training_stats, prediction_outcomes, news_reactions,
         predictions_log, training_history
- Smart write guard: only write when sessions or corrections changed

localStorage keys: smc_deep_nn_v7, smc_deep_stats_v7, smc_deep_outcomes_v7,
                   smc_nn_pool_v1, smc_mod_acc_v1, smc_favorites, smc_scan_limit,
                   smc_ai_provider, smc_ollama_url, smc_ollama_model

════════════════════════════════════════
UI LAYOUT
════════════════════════════════════════
Left sidebar 320px scrollable + main chart + top tab bar (Chart|Scanner|News|AI Chat)

Sidebar includes Hybrid Decision Panel (when DeepSeek active):
3-column: Supervisor signal | 🤝/⚠️ | DeepSeek signal + combined result

Chart: LightweightCharts 4.1.1, EMA9/21/50, FVG zones, BSL/SSL lines,
       VP canvas overlay (right 15%), 50 ghost candles, shape-only markers

Scanner: 2-column grid, 200+ symbols, watchlist pinned at top,
         DeepSeek progress bar + Market Overview panel, hybrid badges,
         manual "🤝 Hybrid" re-analyze button
```

---

## Quick Reference Card

| Signal Condition | Action |
|-----------------|--------|
| Hybrid score > +60 + both AIs agree | LONG, higher confidence |
| Hybrid score > +30 | LONG, standard size |
| Hybrid -8 to +8 | Wait — no edge |
| ⚠️ AI conflict | Reduce size or wait for agreement |
| All gray ghost candles | Consolidation — don't trade |
| Scanner grade A + 🤝 Hybrid filter | Highest conviction picks |
| RSI > 78 or < 22 | Avoid momentum trades — exhaustion |
| Funding > 0.001 | Longs overcrowded — short squeeze risk |
| PDH/PDL swept | Liquidity grab — watch for reversal |
| DeepSeek streaming cursor ▌ | Analysis in progress — wait |

---

*SMC Pro Analyzer — Documentation v2.0 — April 2026*  
*GitHub: https://github.com/naga1412/Crp_Pre*
