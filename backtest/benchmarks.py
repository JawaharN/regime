"""Benchmark strategies for backtest comparison.

Each benchmark returns a dict with keys: equity, returns, trade_pnls
(pd.Series aligned to the input close index). Used behind `backtest --compare`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import metrics


def _from_weights(close: pd.Series, weights: pd.Series) -> dict:
    """Apply a target-weight series to a close-only equity simulation.

    weights[t] is the position held *into* bar t (causal: known at t-1).
    """
    rets = close.pct_change().fillna(0.0)
    strat_rets = rets * weights.shift(1).fillna(0.0)
    equity = (1 + strat_rets).cumprod()
    trade_pnls = strat_rets[weights.diff().fillna(0) != 0]
    return {"equity": equity, "returns": strat_rets, "trade_pnls": trade_pnls}


def buy_and_hold(close: pd.Series) -> dict:
    return _from_weights(close, pd.Series(1.0, index=close.index))


def sma_trend(close: pd.Series, window: int = 200) -> dict:
    sma = close.rolling(window).mean()
    return _from_weights(close, (close > sma).astype(float))


def random_baseline(close: pd.Series, seed: int = 0, p_invest: float = 0.5) -> dict:
    rng = np.random.default_rng(seed)
    w = pd.Series((rng.random(len(close)) < p_invest).astype(float), index=close.index)
    return _from_weights(close, w)


def random_baseline_ensemble(close: pd.Series, n_seeds: int = 100,
                             p_invest: float = 0.5) -> dict:
    """Run the random baseline across many seeds; report mean/std of outcomes."""
    rets, sharpes, dds = [], [], []
    for seed in range(n_seeds):
        b = random_baseline(close, seed=seed, p_invest=p_invest)
        rets.append(metrics.total_return(b["equity"]))
        sharpes.append(metrics.sharpe(b["returns"]))
        dds.append(metrics.max_drawdown(b["equity"]))
    return {
        "n_seeds": n_seeds,
        "total_return_mean": float(np.mean(rets)),
        "total_return_std": float(np.std(rets)),
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "max_drawdown_mean": float(np.mean(dds)),
        "max_drawdown_worst": float(np.min(dds)),
    }
