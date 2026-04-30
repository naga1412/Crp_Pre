import os
import sys

import numpy as np
import pytest

# Allow `from ta_logic import ...` and `from main import ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def synthetic_ohlcv():
    """200 bars of seeded random-walk OHLCV in the Binance kline tuple format."""
    rng = np.random.default_rng(seed=42)
    n = 200
    base_ts = 1_700_000_000_000
    step_ms = 3_600_000
    closes = 50_000 + np.cumsum(rng.normal(0, 200, n))
    rows = []
    for i in range(n):
        o = float(closes[i])
        c = float(closes[i] + rng.normal(0, 50))
        h = max(o, c) + abs(float(rng.normal(0, 30)))
        l = min(o, c) - abs(float(rng.normal(0, 30)))
        v = abs(float(rng.normal(0, 1000))) + 100.0
        rows.append([base_ts + i * step_ms, o, h, l, c, v])
    return rows
