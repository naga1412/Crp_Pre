"""Smoke tests for the TA pipeline. Pure-function checks — no network."""
import math

import pandas as pd
import pytest

from ta_logic import (
    analyze_single_timeframe,
    approximate_liquidation_zones,
    compute_atr,
    compute_rsi,
    detect_candle_type,
    generate_mtf_prediction,
    generate_trade_setup,
    process_ohlcv_data,
    smart_decimals,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def test_smart_decimals_buckets():
    assert smart_decimals(0) == 8
    assert smart_decimals(1e-7) == 10            # < 1e-6 → 10
    assert smart_decimals(5e-5) == 8             # < 1e-4 → 8
    assert smart_decimals(0.5) == 4
    assert smart_decimals(50) == 3
    assert smart_decimals(50_000) == 1


def test_detect_candle_type_table():
    assert detect_candle_type(100, 100, 100, 100) == "Doji"
    # body 0.001, range 0.1 → body_ratio < 0.1 → Doji
    assert detect_candle_type(100, 100.05, 99.95, 100.001) == "Doji"
    # Big bullish body
    assert "Marubozu" in detect_candle_type(100, 110, 100, 110)
    # body=2, range=12, lower_wick=10 → Hammer (bullish) / Hanging Man (bearish)
    assert detect_candle_type(100, 102, 90, 102) == "Hammer"
    assert detect_candle_type(102, 102, 90, 100) == "Hanging Man"
    # body=2, upper_wick=8 → Inverted Hammer (bullish) / Shooting Star (bearish)
    assert detect_candle_type(100, 110, 99.5, 102) == "Inverted Hammer"
    assert detect_candle_type(102, 110, 99.5, 100) == "Shooting Star"


# ── enrichment + indicators ────────────────────────────────────────────────

def test_process_ohlcv_data_enriches(synthetic_ohlcv):
    df = process_ohlcv_data(synthetic_ohlcv)
    expected = {
        "timestamp", "open", "high", "low", "close", "volume",
        "swing_high", "swing_low", "last_swing_high", "last_swing_low",
        "bullish_break", "bearish_break", "bullish_fvg", "bearish_fvg",
        "bullish_ob", "bearish_ob",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) == len(synthetic_ohlcv)
    assert df["close"].notna().all()


def test_compute_atr_is_finite_and_nonnegative(synthetic_ohlcv):
    df = process_ohlcv_data(synthetic_ohlcv)
    atr = compute_atr(df, period=14)
    # First (period-1) values may be NaN, but the rest must be positive & finite.
    tail = atr.dropna()
    assert len(tail) > 0
    assert (tail >= 0).all()
    assert tail.apply(lambda x: math.isfinite(x)).all()


def test_compute_rsi_bounds(synthetic_ohlcv):
    df = process_ohlcv_data(synthetic_ohlcv)
    rsi = compute_rsi(df["close"]).dropna()
    assert len(rsi) > 0
    assert ((rsi >= 0) & (rsi <= 100)).all()


# ── analysis pipeline ──────────────────────────────────────────────────────

def test_analyze_single_timeframe_returns_full_dict(synthetic_ohlcv):
    df = process_ohlcv_data(synthetic_ohlcv)
    a = analyze_single_timeframe(df, "1h")
    for key in ("tf", "trend", "rsi", "rsi_signal", "atr", "structure",
                "support", "resistance", "current_price", "bsl", "ssl"):
        assert key in a, f"missing {key}"
    assert a["tf"] == "1h"
    assert 0 <= a["rsi"] <= 100


def test_analyze_short_df_returns_error_marker():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "open":  [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low":   [1, 2, 3, 4, 5], "close":[1, 2, 3, 4, 5],
        "volume":[1, 1, 1, 1, 1],
    })
    a = analyze_single_timeframe(df, "1h")
    assert "error" in a


def test_generate_mtf_prediction_shape(synthetic_ohlcv):
    df = process_ohlcv_data(synthetic_ohlcv)
    primary = analyze_single_timeframe(df, "1h")
    mtf = generate_mtf_prediction([primary], "1h")
    for key in ("htf_bias", "bullish_score_pct", "bearish_score_pct",
                "confidence", "phase", "next_candle_type",
                "next_candle_direction", "alignment", "mtf_consistency"):
        assert key in mtf
    assert math.isclose(
        mtf["bullish_score_pct"] + mtf["bearish_score_pct"], 100.0, abs_tol=0.2)


def test_generate_trade_setup_invariants(synthetic_ohlcv):
    df = process_ohlcv_data(synthetic_ohlcv)
    primary = analyze_single_timeframe(df, "1h")
    mtf = generate_mtf_prediction([primary], "1h")
    price = float(df.iloc[-1]["close"])
    setup = generate_trade_setup(df, price, mtf_prediction=mtf, tf_label="1h")

    assert setup["signal"] in {"LONG", "SHORT"}
    assert setup["take_profit_1"] is not None and setup["stop_loss"] is not None
    if setup["signal"] == "LONG":
        assert setup["take_profit_1"] > price > setup["stop_loss"]
    else:
        assert setup["take_profit_1"] < price < setup["stop_loss"]

    assert "take_profit" not in setup, "dead alias must be removed"

    fc = setup["future_candles"]
    assert isinstance(fc, list) and len(fc) > 0
    # No timestamps emitted server-side anymore (frontend computes them).
    assert "timestamp" not in fc[0]
    for c in fc:
        assert c["high"] >= max(c["open"], c["close"])
        assert c["low"]  <= min(c["open"], c["close"])
        assert c["is_future"] is True


# ── liquidation zones ──────────────────────────────────────────────────────

def test_liquidation_zones_account_for_maintenance_margin():
    price = 50_000.0
    zones = approximate_liquidation_zones(price, atr=500.0, maintenance_margin=0.005)
    assert len(zones) == 4
    by_lev = {z["leverage"]: z for z in zones}
    # 100x: loss_pct = 1/100 - 0.005 = 0.005 → long liq at 50000*(1-0.005) = 49750
    assert math.isclose(by_lev[100]["long_liquidation"], 49_750.0, rel_tol=1e-4)
    assert math.isclose(by_lev[100]["short_liquidation"], 50_250.0, rel_tol=1e-4)
    # Naïve formula (no maintenance) would say 49500/50500 — confirm we're tighter.
    assert by_lev[100]["long_liquidation"] > 49_500.0


def test_liquidation_zones_clamp_when_maintenance_exceeds_margin():
    """At leverage 100x with 2% maintenance, loss_pct goes negative — must clamp to 0."""
    price = 50_000.0
    zones = approximate_liquidation_zones(price, atr=500.0, maintenance_margin=0.02)
    by_lev = {z["leverage"]: z for z in zones}
    assert by_lev[100]["long_liquidation"] == 50_000.0
    assert by_lev[100]["short_liquidation"] == 50_000.0
