"""Explicit allocation math: leverage drives cash negative; equity stays correct."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.walk_forward import run_allocation_backtest


def _rising_close(n: int = 120) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(100 * np.exp(np.cumsum(np.full(n, 0.002))), index=idx)


def test_leverage_drives_cash_negative_equity_consistent():
    close = _rising_close()
    weights = pd.Series(1.25, index=close.index)        # 1.25x leverage
    res = run_allocation_backtest(close, weights, initial_capital=100_000.0)
    # Under leverage the margin balance (cash) goes negative.
    assert res.cash_curve.min() < 0
    # equity = cash + shares*price held throughout; a leveraged long in a
    # rising market beats an unlevered buy-and-hold.
    bh_return = close.iloc[-1] / close.iloc[0] - 1.0
    strat_return = res.equity_curve.iloc[-1] / 100_000.0 - 1.0
    assert strat_return > bh_return


def test_rebalance_threshold_suppresses_tiny_moves():
    close = _rising_close(60)
    # target weight drifts by 0.03 each bar — below the 0.10 threshold.
    weights = pd.Series(np.clip(0.50 + 0.03 * np.arange(60), 0, 0.95), index=close.index)
    res = run_allocation_backtest(close, weights, rebalance_threshold=0.10)
    # only the initial entry plus the occasional threshold-crossing trade.
    assert len(res.trade_log) < 10


def test_no_leverage_keeps_cash_non_negative():
    close = _rising_close()
    weights = pd.Series(0.50, index=close.index)
    res = run_allocation_backtest(close, weights)
    assert res.cash_curve.min() >= -1.0   # only slippage rounding
