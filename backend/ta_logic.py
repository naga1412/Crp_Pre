import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def smart_decimals(price: float) -> int:
    if price <= 0:        return 8
    if price < 0.000001: return 10
    if price < 0.00001:  return 9
    if price < 0.0001:   return 8
    if price < 0.001:    return 7
    if price < 0.01:     return 6
    if price < 0.1:      return 5
    if price < 1:        return 4
    if price < 100:      return 3
    if price < 10000:    return 2
    return 1


def fmt(value: float, decimals: int) -> float:
    return round(value, decimals)


# ─────────────────────────────────────────────
#  MARKET STRUCTURE
# ─────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 3:
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        return df
    df['bullish_fvg'] = df['low'] > df['high'].shift(2)
    df['bearish_fvg'] = df['high'] < df['low'].shift(2)
    return df


def identify_market_structure(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df['swing_high'] = False
    df['swing_low'] = False
    highs = df['high'].values
    lows  = df['low'].values
    sh_col = df.columns.get_loc('swing_high')
    sl_col = df.columns.get_loc('swing_low')
    for i in range(window, len(df) - window):
        if (highs[i] > highs[i-window:i]).all() and (highs[i] > highs[i+1:i+window+1]).all():
            df.iat[i, sh_col] = True
        if (lows[i] < lows[i-window:i]).all() and (lows[i] < lows[i+1:i+window+1]).all():
            df.iat[i, sl_col] = True
    df['last_swing_high'] = df['high'].where(df['swing_high']).ffill()
    df['last_swing_low']  = df['low'].where(df['swing_low']).ffill()
    df['bullish_break'] = (df['close'] > df['last_swing_high']) & \
                          (df['close'].shift(1) <= df['last_swing_high'])
    df['bearish_break'] = (df['close'] < df['last_swing_low']) & \
                          (df['close'].shift(1) >= df['last_swing_low'])
    return df


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bullish OB: last bearish candle before a significant bullish move (≥0.8 ATR).
    Bearish OB: last bullish candle before a significant bearish move (≥0.8 ATR).
    """
    if len(df) < 10:
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        return df
    atr = compute_atr(df, 14)
    df['bullish_ob'] = False
    df['bearish_ob'] = False
    boc = df.columns.get_loc('bullish_ob')
    brc = df.columns.get_loc('bearish_ob')
    for i in range(2, len(df) - 3):
        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v == 0:
            continue
        # Bullish OB
        if df['close'].iloc[i] < df['open'].iloc[i]:
            up = df['high'].iloc[i+1:i+4].max() - df['high'].iloc[i]
            if up >= atr_v * 0.8:
                df.iat[i, boc] = True
        # Bearish OB
        if df['close'].iloc[i] > df['open'].iloc[i]:
            dn = df['low'].iloc[i] - df['low'].iloc[i+1:i+4].min()
            if dn >= atr_v * 0.8:
                df.iat[i, brc] = True
    return df


def detect_liquidity_levels(df: pd.DataFrame, window: int = 10,
                             tolerance_pct: float = 0.0015) -> dict:
    """
    Equal highs  → Buy-Side Liquidity (BSL, above price, short stop cluster).
    Equal lows   → Sell-Side Liquidity (SSL, below price, long stop cluster).
    """
    if len(df) < window * 2:
        return {'bsl': [], 'ssl': []}
    current = float(df['close'].iloc[-1])
    d = smart_decimals(current)
    tail = min(150, len(df))
    highs = df['high'].values[-tail:]
    lows  = df['low'].values[-tail:]

    bsl, ssl = [], []
    for i in range(window, len(highs)):
        lvl = highs[i]
        near = [h for h in highs[max(0, i-window):i] if abs(h - lvl) / max(lvl, 1e-12) < tolerance_pct]
        if near:
            bsl.append(round(lvl, d))
    for i in range(window, len(lows)):
        lvl = lows[i]
        near = [l for l in lows[max(0, i-window):i] if abs(l - lvl) / max(lvl, 1e-12) < tolerance_pct]
        if near:
            ssl.append(round(lvl, d))

    bsl_u = sorted(set(bsl), reverse=True)
    ssl_u = sorted(set(ssl))
    return {
        'bsl': [x for x in bsl_u if x > current][:3],
        'ssl': [x for x in ssl_u if x < current][:3],
    }


def detect_advanced_patterns(df: pd.DataFrame, d: int = 2) -> dict:
    """
    Detect advanced chart patterns from swing highs/lows using Steve Nison /
    Edwards & Magee methodology:
    Rising/Falling Wedge, Ascending/Descending/Symmetrical Triangle,
    Triple Top/Bottom, Rounding Bottom/Top, Head & Shoulders (basic).
    Returns {"pattern": str|None, "bias": str, "confidence": str}.
    """
    if len(df) < 30:
        return {"pattern": None, "bias": None, "confidence": None}

    if 'swing_high' not in df.columns:
        df = identify_market_structure(df)

    sh_df = df[df['swing_high']].tail(6)
    sl_df = df[df['swing_low']].tail(6)

    if len(sh_df) < 3 or len(sl_df) < 3:
        return {"pattern": None, "bias": None, "confidence": None}

    sh_highs = sh_df['high'].values
    sl_lows  = sl_df['low'].values
    sh_idx   = sh_df.index.values.astype(float)
    sl_idx   = sl_df.index.values.astype(float)

    # Linear slope of swing highs / swing lows trendlines
    sh_slope = (sh_highs[-1] - sh_highs[0]) / max(sh_idx[-1] - sh_idx[0], 1)
    sl_slope = (sl_lows[-1]  - sl_lows[0])  / max(sl_idx[-1] - sl_idx[0], 1)

    # Normalise by price magnitude for relative comparison
    mid_price = float(df['close'].iloc[-1])
    sh_slope_r = sh_slope / max(mid_price, 1e-12)
    sl_slope_r = sl_slope / max(mid_price, 1e-12)
    flat_thresh = 0.0002   # < 0.02% per bar is "flat"

    both_rising  = sh_slope_r > flat_thresh and sl_slope_r > flat_thresh
    both_falling = sh_slope_r < -flat_thresh and sl_slope_r < -flat_thresh
    sh_flat      = abs(sh_slope_r) < flat_thresh
    sl_flat      = abs(sl_slope_r) < flat_thresh

    # ── Triple Top (bearish reversal) ────────────────────────────────
    if len(sh_highs) >= 3:
        h1, h2, h3 = sh_highs[-3], sh_highs[-2], sh_highs[-1]
        tol = mid_price * 0.013
        if abs(h1 - h2) < tol and abs(h2 - h3) < tol and abs(h1 - h3) < tol:
            return {"pattern": "Triple Top", "bias": "Bearish", "confidence": "High"}

    # ── Triple Bottom (bullish reversal) ─────────────────────────────
    if len(sl_lows) >= 3:
        l1, l2, l3 = sl_lows[-3], sl_lows[-2], sl_lows[-1]
        tol = mid_price * 0.013
        if abs(l1 - l2) < tol and abs(l2 - l3) < tol and abs(l1 - l3) < tol:
            return {"pattern": "Triple Bottom", "bias": "Bullish", "confidence": "High"}

    # ── Head & Shoulders (bearish) ───────────────────────────────────
    if len(sh_highs) >= 3:
        ls, head, rs = sh_highs[-3], sh_highs[-2], sh_highs[-1]
        if head > ls * 1.01 and head > rs * 1.01 and abs(ls - rs) / max(mid_price, 1e-12) < 0.015:
            return {"pattern": "Head & Shoulders", "bias": "Bearish", "confidence": "High"}

    # ── Inverse H&S (bullish) ────────────────────────────────────────
    if len(sl_lows) >= 3:
        ls, head, rs = sl_lows[-3], sl_lows[-2], sl_lows[-1]
        if head < ls * 0.99 and head < rs * 0.99 and abs(ls - rs) / max(mid_price, 1e-12) < 0.015:
            return {"pattern": "Inverse Head & Shoulders", "bias": "Bullish", "confidence": "High"}

    # ── Rising Wedge (bearish — converging upward) ───────────────────
    if both_rising and sh_slope_r < sl_slope_r:   # lows rising faster → converging
        return {"pattern": "Rising Wedge", "bias": "Bearish", "confidence": "Medium"}

    # ── Falling Wedge (bullish — converging downward) ────────────────
    if both_falling and sl_slope_r > sh_slope_r:  # highs falling faster → converging
        return {"pattern": "Falling Wedge", "bias": "Bullish", "confidence": "Medium"}

    # ── Ascending Triangle (bullish bias) ────────────────────────────
    if sh_flat and sl_slope_r > flat_thresh:
        return {"pattern": "Ascending Triangle", "bias": "Bullish", "confidence": "Medium"}

    # ── Descending Triangle (bearish bias) ───────────────────────────
    if sl_flat and sh_slope_r < -flat_thresh:
        return {"pattern": "Descending Triangle", "bias": "Bearish", "confidence": "Medium"}

    # ── Symmetrical Triangle (neutral → breakout imminent) ───────────
    if sh_slope_r < -flat_thresh and sl_slope_r > flat_thresh:
        return {"pattern": "Symmetrical Triangle", "bias": "Neutral", "confidence": "Low"}

    # ── Rounding Bottom / Cup (bullish) ──────────────────────────────
    if len(sl_lows) >= 4:
        mid = len(sl_lows) // 2
        first_down = (sl_lows[mid] - sl_lows[0])  / max(mid, 1)
        then_up    = (sl_lows[-1]  - sl_lows[mid]) / max(len(sl_lows) - mid, 1)
        if first_down < 0 and then_up > 0 and abs(then_up) > abs(first_down) * 0.5:
            return {"pattern": "Rounding Bottom (Cup)", "bias": "Bullish", "confidence": "Medium"}

    # ── Rounding Top (bearish) ────────────────────────────────────────
    if len(sh_highs) >= 4:
        mid = len(sh_highs) // 2
        first_up  = (sh_highs[mid] - sh_highs[0])  / max(mid, 1)
        then_down = (sh_highs[-1]  - sh_highs[mid]) / max(len(sh_highs) - mid, 1)
        if first_up > 0 and then_down < 0 and abs(then_down) > abs(first_up) * 0.5:
            return {"pattern": "Rounding Top", "bias": "Bearish", "confidence": "Medium"}

    # ── Bump and Run (bearish distribution) ──────────────────────────
    if len(sh_highs) >= 4:
        early_avg = sh_highs[:2].mean()
        late_avg  = sh_highs[-2:].mean()
        if late_avg > early_avg * 1.05 and sh_slope_r < 0:
            return {"pattern": "Bump and Run", "bias": "Bearish", "confidence": "Medium"}

    return {"pattern": None, "bias": None, "confidence": None}


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder's RMA (matches TA-Lib / TradingView)."""
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ─────────────────────────────────────────────
#  CANDLE TYPE DETECTION  (Steve Nison methodology)
# ─────────────────────────────────────────────

def detect_candle_type(o: float, h: float, l: float, c: float) -> str:
    """Classify a single candle by body/wick proportions.

    Note: without trend context, "Hammer" vs "Hanging Man" and "Inverted
    Hammer" vs "Shooting Star" cannot be disambiguated; we pick the
    common-case label based on body color.
    """
    full_range = h - l
    if full_range == 0:
        return "Doji"
    body = abs(c - o)
    body_ratio = body / full_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    bullish = c > o

    if body_ratio < 0.1:
        return "Doji"
    if body_ratio > 0.8:
        return "Marubozu (Bullish)" if bullish else "Marubozu (Bearish)"
    # Long lower wick, small upper wick → Hammer (bullish ctx) / Hanging Man (bearish ctx)
    if body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5:
        return "Hammer" if bullish else "Hanging Man"
    # Long upper wick, small lower wick → Inverted Hammer (bull) / Shooting Star (bear)
    if body > 0 and upper_wick > body * 2 and lower_wick < body * 0.5:
        return "Inverted Hammer" if bullish else "Shooting Star"
    if body_ratio < 0.3:
        return "Spinning Top"
    return "Bullish Candle" if bullish else "Bearish Candle"


# ─────────────────────────────────────────────
#  MTF ANALYSIS ENGINE
# ─────────────────────────────────────────────

# Higher timeframes carry more weight (HTF > LTF per SMC methodology)
TF_WEIGHTS = {
    '1m':  0.4, '3m':  0.5, '5m':  0.6, '15m': 1.0, '30m': 1.0,
    '1h':  1.5, '2h':  1.5, '4h':  2.0, '6h':  2.0, '8h':  2.0,
    '12h': 2.5, '1d':  3.0, '3d':  3.0, '1w':  4.0, '1M':  4.0,
}


def analyze_single_timeframe(df: pd.DataFrame, tf_label: str) -> dict:
    if df is None or len(df) < 20:
        return {"tf": tf_label, "error": "insufficient data"}

    close = df['close']
    current = float(close.iloc[-1])
    d = smart_decimals(current)

    # Trend via SMA
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(df) >= 50 else None
    trend = "Bullish" if current > sma20 else "Bearish"
    if sma50 is not None and not pd.isna(sma50):
        trend = "Strongly Bullish" if current > sma20 and current > sma50 else \
                "Strongly Bearish" if current < sma20 and current < sma50 else trend

    # RSI
    rsi_s = compute_rsi(close)
    rsi   = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

    # ATR
    atr_v = compute_atr(df).iloc[-1]
    atr   = float(atr_v) if not pd.isna(atr_v) else current * 0.02

    # Market structure
    if 'bullish_break' not in df.columns:
        df = identify_market_structure(df)
    r_bull = int(df['bullish_break'].iloc[-20:].sum())
    r_bear = int(df['bearish_break'].iloc[-20:].sum())
    structure = "Bullish BOS" if r_bull > r_bear else \
                "Bearish BOS" if r_bear > r_bull else "Ranging"

    # FVG
    if 'bullish_fvg' not in df.columns:
        df = detect_fvgs(df)
    has_bull_fvg = bool(df['bullish_fvg'].iloc[-5:].any())
    has_bear_fvg = bool(df['bearish_fvg'].iloc[-5:].any())

    # Order blocks
    if 'bullish_ob' not in df.columns:
        df = detect_order_blocks(df)
    bull_obs = df[df['bullish_ob']].iloc[-3:]
    bear_obs = df[df['bearish_ob']].iloc[-3:]
    last_bull_ob = round(float(bull_obs.iloc[-1]['low']),  d) if len(bull_obs) else None
    last_bear_ob = round(float(bear_obs.iloc[-1]['high']), d) if len(bear_obs) else None

    # Liquidity levels
    liq = detect_liquidity_levels(df)

    # Volume
    vol_sma    = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else None
    vol_cur    = float(df['volume'].iloc[-1])              if 'volume' in df.columns else None
    vol_signal = ("High"    if vol_sma and vol_cur and vol_cur > vol_sma * 1.5 else
                  "Low"     if vol_sma and vol_cur and vol_cur < vol_sma * 0.5 else "Average")

    # Last candle
    last = df.iloc[-1]
    candle_type = detect_candle_type(
        float(last['open']), float(last['high']),
        float(last['low']),  float(last['close'])
    )

    # Key S/R
    recent    = df.iloc[-50:]
    resistance = round(float(recent['high'].max()), d)
    support    = round(float(recent['low'].min()),  d)

    def _safe(col_name):
        if col_name not in df.columns:
            return None
        v = df[col_name].iloc[-1]
        try:
            f = float(v)
            return None if pd.isna(f) else round(f, d)
        except (TypeError, ValueError):
            return None

    return {
        "tf": tf_label,
        "trend": trend,
        "rsi": round(rsi, 1),
        "rsi_signal": rsi_signal,
        "atr": round(atr, d + 2),
        "structure": structure,
        "has_bullish_fvg": has_bull_fvg,
        "has_bearish_fvg": has_bear_fvg,
        "last_candle_type": candle_type,
        "resistance": resistance,
        "support": support,
        "last_swing_high": _safe('last_swing_high'),
        "last_swing_low":  _safe('last_swing_low'),
        "bullish_ob_level": last_bull_ob,
        "bearish_ob_level": last_bear_ob,
        "bsl": liq['bsl'],
        "ssl": liq['ssl'],
        "volume_signal": vol_signal,
        "current_price": round(current, d),
    }


def _predict_next_candle(tf_analysis: dict, htf_bias: str, confidence: str) -> dict:
    """Derive next-candle prediction for a given TF (Steve Nison candlestick logic)."""
    rsi    = tf_analysis.get('rsi', 50)
    struct = tf_analysis.get('structure', '')

    if htf_bias == 'Bullish' and rsi < 40:
        return {"type": "Bullish Engulfing", "direction": "Bullish",
                "body": "Large" if confidence == "High" else "Medium",
                "wicks": "Long lower wick (demand zone test)"}
    if htf_bias == 'Bearish' and rsi > 60:
        return {"type": "Bearish Engulfing", "direction": "Bearish",
                "body": "Large" if confidence == "High" else "Medium",
                "wicks": "Long upper wick (supply zone rejection)"}
    if rsi > 72:
        return {"type": "Shooting Star / Bearish Reversal", "direction": "Bearish",
                "body": "Small", "wicks": "Long upper wick"}
    if rsi < 28:
        return {"type": "Hammer / Bullish Reversal", "direction": "Bullish",
                "body": "Small", "wicks": "Long lower wick"}
    if htf_bias == 'Bullish' and 'Bullish BOS' in struct:
        return {"type": "Marubozu (Bullish momentum)", "direction": "Bullish",
                "body": "Large", "wicks": "Minimal"}
    if htf_bias == 'Bearish' and 'Bearish BOS' in struct:
        return {"type": "Marubozu (Bearish momentum)", "direction": "Bearish",
                "body": "Large", "wicks": "Minimal"}
    if htf_bias == 'Bullish':
        return {"type": "Inside Bar (Bullish continuation)", "direction": "Bullish",
                "body": "Medium", "wicks": "Balanced"}
    return {"type": "Inside Bar (Bearish continuation)", "direction": "Bearish",
            "body": "Medium", "wicks": "Balanced"}


def generate_mtf_prediction(analyses: list, primary_tf: str) -> dict:
    if not analyses:
        return {}

    primary = next((a for a in analyses if a.get('tf') == primary_tf), analyses[-1])

    # Weighted scoring — HTF has higher weight per SMC top-down analysis
    bull_score = 0.0
    bear_score = 0.0
    for a in analyses:
        if 'error' in a:
            continue
        tf_label = a.get('tf', '1h')
        w        = TF_WEIGHTS.get(tf_label, 1.0)
        trend    = a.get('trend', '')
        rsi      = a.get('rsi', 50)
        struct   = a.get('structure', '')
        if 'Strongly Bullish' in trend: bull_score += w * 2
        elif 'Bullish' in trend:        bull_score += w * 1
        if 'Strongly Bearish' in trend: bear_score += w * 2
        elif 'Bearish' in trend:        bear_score += w * 1
        if 'Bullish BOS' in struct:  bull_score += w * 1
        elif 'Bearish BOS' in struct: bear_score += w * 1
        if a.get('has_bullish_fvg'): bull_score += w * 0.5
        if a.get('has_bearish_fvg'): bear_score += w * 0.5
        if rsi < 35:  bull_score += w * 0.5
        elif rsi > 65: bear_score += w * 0.5

    total    = bull_score + bear_score
    bull_pct = round((bull_score / total * 100) if total > 0 else 50.0, 1)
    bear_pct = round(100 - bull_pct, 1)
    htf_bias = ("Bullish" if bull_score > bear_score else
                "Bearish" if bear_score > bull_score else "Neutral")
    agree    = max(bull_score, bear_score) / max(total, 1)
    confidence = "High" if agree > 0.7 else "Medium" if agree > 0.55 else "Low"

    # Wyckoff phase
    phase = ("Markup"       if htf_bias == "Bullish" and bull_pct > 65 else
             "Markdown"     if htf_bias == "Bearish" and bear_pct > 65 else
             "Accumulation" if htf_bias == "Bullish" else "Distribution")

    # Market condition
    primary_struct = primary.get('structure', 'Ranging')
    recent_breaks  = int(analyses[0].get('structure', '') != 'Ranging')
    condition = ("Breaking Out" if recent_breaks and confidence == "High" else
                 "Trending"     if primary_struct != 'Ranging' else "Ranging")

    # Primary TF next candle
    primary_nc = _predict_next_candle(primary, htf_bias, confidence)

    # 1H next candle (for cross-TF alignment check)
    a_1h = next((a for a in analyses if a.get('tf') == '1h'), None)
    nc_1h = _predict_next_candle(a_1h, htf_bias, confidence) if a_1h else None

    # Alignment check
    alignment = "YES"
    alignment_note = ""
    if nc_1h:
        if primary_nc['direction'] != nc_1h['direction']:
            alignment = "NO"
            alignment_note = f"1H signals {nc_1h['direction']} — HTF takes priority"

    # Cross-timeframe consistency (are any TFs contradicting the HTF bias?)
    conflict_tfs = []
    for a in analyses:
        if 'error' in a or a.get('tf') == primary_tf:
            continue
        tf_trend = a.get('trend', '')
        if htf_bias == 'Bullish' and 'Bearish' in tf_trend:
            conflict_tfs.append(f"{a['tf']} ({tf_trend})")
        elif htf_bias == 'Bearish' and 'Bullish' in tf_trend:
            conflict_tfs.append(f"{a['tf']} ({tf_trend})")
    mtf_consistency = "ALIGNED" if not conflict_tfs else f"MIXED — {', '.join(conflict_tfs[:2])}"

    # Aggregate key levels across all TFs
    all_bsl = sorted({x for a in analyses for x in a.get('bsl', [])}, reverse=True)[:5]
    all_ssl = sorted({x for a in analyses for x in a.get('ssl', [])})[:5]

    return {
        "htf_bias": htf_bias,
        "bullish_score_pct": bull_pct,
        "bearish_score_pct": bear_pct,
        "confidence": confidence,
        "phase": phase,
        "condition": condition,
        "next_candle_type":      primary_nc['type'],
        "next_candle_direction": primary_nc['direction'],
        "next_candle_body":      primary_nc['body'],
        "next_candle_wicks":     primary_nc['wicks'],
        "nc_1h": nc_1h,
        "alignment": alignment,
        "alignment_note": alignment_note,
        "mtf_consistency": mtf_consistency,
        "bsl": all_bsl,
        "ssl": all_ssl,
        "tf_analyses": analyses,
    }


# ─────────────────────────────────────────────
#  LIQUIDATION ZONES
# ─────────────────────────────────────────────

def approximate_liquidation_zones(current_price: float, atr: float,
                                  maintenance_margin: float = 0.005) -> list:
    """Approximate isolated-margin liquidation prices.

    Formula accounts for an initial margin = 1/leverage and a typical
    maintenance margin (~0.5% on most majors). Real exchange tiers vary —
    treat these as indicative levels, not exact thresholds.
    """
    d = smart_decimals(current_price)
    zones = []
    for lev in [100, 50, 25, 10]:
        # Long liq when equity loss = (initial_margin - maintenance) of position.
        loss_pct = (1.0 / lev) - maintenance_margin
        loss_pct = max(loss_pct, 0.0)
        zones.append({
            "leverage": lev,
            "short_liquidation": fmt(current_price * (1 + loss_pct), d),
            "long_liquidation":  fmt(current_price * (1 - loss_pct), d),
        })
    return zones


# ─────────────────────────────────────────────
#  TRADE SETUP GENERATOR
# ─────────────────────────────────────────────

# Real candle count per timeframe per trading day / period
TF_CANDLES_PER_DAY = {
    '1m':  60,   # show 1 hour of 1m predictions
    '3m':  20,
    '5m':  48,   # 4h window
    '15m': 96,   # 1 full day = 96 × 15m
    '30m': 48,
    '1h':  24,   # 1 full day
    '2h':  12,
    '4h':  6,    # 1 full day
    '6h':  4,
    '8h':  3,
    '12h': 2,
    '1d':  1,    # 1 future daily candle
    '3d':  1,
    '1w':  1,    # 1 future weekly candle
    '1M':  1,
}


def generate_trade_setup(df: pd.DataFrame, current_price: float,
                         mtf_prediction: dict = None,
                         tf_label: str = '1h') -> dict:
    d = smart_decimals(current_price)
    latest = df.iloc[-1]

    atr_v = compute_atr(df).iloc[-1]
    atr   = float(atr_v) if not pd.isna(atr_v) else current_price * 0.02

    # Confluence scoring
    bull_conf = bear_conf = 0
    if latest.get('bullish_break', False): bull_conf += 1
    if latest.get('bullish_fvg',   False): bull_conf += 1
    if latest.get('bearish_break', False): bear_conf += 1
    if latest.get('bearish_fvg',   False): bear_conf += 1
    if latest.get('bullish_ob',    False): bull_conf += 1
    if latest.get('bearish_ob',    False): bear_conf += 1

    if mtf_prediction:
        bias = mtf_prediction.get('htf_bias', 'Neutral')
        if bias == 'Bullish': bull_conf += 2
        elif bias == 'Bearish': bear_conf += 2

    sma20 = df['close'].rolling(20).mean().iloc[-1]
    baseline_bull = bool(current_price > sma20) if not pd.isna(sma20) else True

    if bull_conf > bear_conf or (bull_conf == bear_conf and baseline_bull):
        signal    = 'LONG'
        entry_low = fmt(current_price - atr * 0.5, d)
        entry_hi  = fmt(current_price, d)
        tp1       = fmt(current_price + atr * 2,   d)
        tp2       = fmt(current_price + atr * 4,   d)
        sl        = fmt(current_price - atr * 1.5, d)
        risk      = max(current_price - sl, 1e-12)
        rr1       = round((tp1 - current_price) / risk, 1)
        rr2       = round((tp2 - current_price) / risk, 1)
        invalidation = f"Close below {sl} (bearish structure break)"
    else:
        signal    = 'SHORT'
        entry_low = fmt(current_price, d)
        entry_hi  = fmt(current_price + atr * 0.5, d)
        tp1       = fmt(current_price - atr * 2,   d)
        tp2       = fmt(current_price - atr * 4,   d)
        sl        = fmt(current_price + atr * 1.5, d)
        risk      = max(sl - current_price, 1e-12)
        rr1       = round((current_price - tp1) / risk, 1)
        rr2       = round((current_price - tp2) / risk, 1)
        invalidation = f"Close above {sl} (bullish structure break)"

    entry_zone = [entry_low, entry_hi]
    leverage   = 5

    # Advanced pattern detection (Edwards & Magee + Steve Nison)
    adv = detect_advanced_patterns(df, d)
    if adv['pattern']:
        pattern = adv['pattern']
    else:
        # Fallback legacy logic
        def _get_swing(col):
            if col not in df.columns: return None
            v = df[col].iloc[-1]
            try:
                f = float(v)
                return None if pd.isna(f) else f
            except (TypeError, ValueError):
                return None

        last_sh = _get_swing('last_swing_high')
        last_sl = _get_swing('last_swing_low')
        pattern = "Standard Volatility Consolidation"

        if signal == 'LONG' and last_sl is not None:
            if df['low'].iloc[-1] > last_sl * 0.99 and df['low'].iloc[-1] < last_sl * 1.02:
                pattern = "Double Bottom (W-Shape)"
            elif bull_conf >= 2:
                pattern = "Cup & Handle Breakout"
        elif signal == 'SHORT' and last_sh is not None:
            if df['high'].iloc[-1] < last_sh * 1.01 and df['high'].iloc[-1] > last_sh * 0.98:
                pattern = "Double Top (M-Shape)"
            elif bear_conf >= 2:
                pattern = "Bear Flag Breakdown"

    # Future candle projection — steps matched to real candles-per-day for this timeframe.
    # Timestamps are computed client-side from the last real bar; we only emit OHLC.
    steps = TF_CANDLES_PER_DAY.get(tf_label, 12)

    if mtf_prediction:
        direction  = mtf_prediction.get('next_candle_direction', 'Bullish')
        conf_lvl   = mtf_prediction.get('confidence', 'Medium')
        mag        = 1.8 if conf_lvl == 'High' else 1.2 if conf_lvl == 'Medium' else 0.8
    else:
        direction = 'Bullish' if signal == 'LONG' else 'Bearish'
        mag       = 1.0

    price_step = (tp1 - current_price) / max(steps, 1)
    curr_p     = current_price
    rsi_series = compute_rsi(df['close'])
    rsi_last   = rsi_series.iloc[-1]
    rsi_now    = float(rsi_last) if not pd.isna(rsi_last) else 50.0
    future_candles = []
    for i in range(steps):
        noise = atr * 0.15 * (-1 if i % 3 == 1 else 0.5)
        o = curr_p
        c = o + price_step * mag + noise
        h = max(o, c) + abs(atr * 0.2)
        l = min(o, c) - abs(atr * 0.2)
        future_candles.append({
            "open":  fmt(o, d), "high": fmt(h, d),
            "low":   fmt(l, d), "close": fmt(c, d),
            "is_future": True,
        })
        curr_p = c

    liq_zones = approximate_liquidation_zones(current_price, atr)

    return {
        "signal":             signal,
        "entry_zone":         entry_zone,
        "take_profit_1":      tp1,
        "take_profit_2":      tp2,
        "stop_loss":          sl,
        "rr_ratio_tp1":       f"1:{rr1}",
        "rr_ratio_tp2":       f"1:{rr2}",
        "setup_invalidation": invalidation,
        "leverage_suggested": leverage,
        "pattern":            pattern,
        "pattern_bias":       adv.get('bias'),
        "pattern_confidence": adv.get('confidence'),
        "future_candles":     future_candles,
        "liquidation_zones":  liq_zones,
        "atr":                fmt(atr, d + 2),
        "rsi":                round(rsi_now, 1),
        "current_price":      fmt(current_price, d),
        "decimals":           d,
    }


# ─────────────────────────────────────────────
#  DATA PROCESSORS
# ─────────────────────────────────────────────

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = identify_market_structure(df)
    df = detect_fvgs(df)
    df = detect_order_blocks(df)
    return df


def process_ohlcv_data(ohlcv: list) -> pd.DataFrame:
    """Converts the [ts, o, h, l, c, v] tuples from Binance → enriched DataFrame."""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return _enrich(df)


def process_yfinance_data(hist) -> pd.DataFrame:
    """Converts a yfinance history DataFrame → enriched DataFrame (same schema)."""
    index = hist.index
    if getattr(index, 'tz', None) is not None:
        index = index.tz_convert('UTC').tz_localize(None)
    df = pd.DataFrame({
        'timestamp': index,
        'open':      hist['Open'].values.astype(float),
        'high':      hist['High'].values.astype(float),
        'low':       hist['Low'].values.astype(float),
        'close':     hist['Close'].values.astype(float),
        'volume':    hist['Volume'].values.astype(float),
    })
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return _enrich(df)
