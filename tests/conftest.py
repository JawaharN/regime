"""Shared test fixtures: synthetic OHLCV, deterministic seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)
    yield


@pytest.fixture
def synthetic_ohlcv():
    """Two years of daily OHLCV with a bullish drift and one volatile patch."""
    n = 500
    rng = np.random.default_rng(42)
    # Two regimes glued together: calm bull, then volatile
    drift1 = rng.normal(0.0005, 0.008, n // 2)
    drift2 = rng.normal(-0.0002, 0.025, n - n // 2)
    rets = np.concatenate([drift1, drift2])
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)
    return df
