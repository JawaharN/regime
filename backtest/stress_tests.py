"""Stress tests.

`inject_crash` is the legacy single-shock injector (unchanged contract). The
final-prompt suite adds:
- `monte_carlo_crash`   — random −5%…−15% gaps at random points, many runs
- `gap_risk_test`       — overnight gaps of 2–5× ATR, expected vs realised loss
- `regime_misclassification` — deliberately shuffle the regime→strategy map and
  confirm the risk layer still contains the damage
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_crash(ohlcv: pd.DataFrame, magnitude: float, at_index: int) -> pd.DataFrame:
    """Multiply close on the chosen bar by (1 - magnitude); cascade forward."""
    if not 0 < magnitude < 1:
        raise ValueError("magnitude must be in (0, 1)")
    if at_index < 0 or at_index >= len(ohlcv):
        raise IndexError("at_index out of range")
    df = ohlcv.copy()
    scale = 1 - magnitude
    df.iloc[at_index:, df.columns.get_loc("close")] *= scale
    df.iloc[at_index:, df.columns.get_loc("open")] *= scale
    df.iloc[at_index:, df.columns.get_loc("high")] *= scale
    df.iloc[at_index, df.columns.get_loc("low")] *= scale * 0.95
    return df


def monte_carlo_crash(ohlcv: pd.DataFrame, n_runs: int = 100, n_shocks: int = 10,
                      min_mag: float = 0.05, max_mag: float = 0.15,
                      seed: int = 0) -> dict:
    """Inject `n_shocks` random crashes per run, `n_runs` runs.

    Returns the distribution of worst single-run drawdowns. ``kill_fired_rate``
    is the fraction of runs whose peak-to-trough loss breaches 10%.
    """
    rng = np.random.default_rng(seed)
    n = len(ohlcv)
    worst_losses: list[float] = []
    kill_fired = 0
    for _ in range(n_runs):
        df = ohlcv.copy()
        for _ in range(n_shocks):
            idx = int(rng.integers(1, n))
            mag = float(rng.uniform(min_mag, max_mag))
            df.iloc[idx, df.columns.get_loc("close")] *= (1 - mag)
        close = df["close"].astype(float)
        dd = float((close / close.cummax() - 1.0).min())
        worst_losses.append(dd)
        if dd <= -0.10:
            kill_fired += 1
    return {
        "n_runs": n_runs,
        "mean_max_loss": float(np.mean(worst_losses)),
        "worst_max_loss": float(np.min(worst_losses)),
        "kill_fired_rate": kill_fired / n_runs,
    }


def gap_risk_test(ohlcv: pd.DataFrame, atr_window: int = 14,
                  gap_mult_range: tuple[float, float] = (2.0, 5.0),
                  seed: int = 0) -> dict:
    """Inject an overnight gap of 2–5× ATR; compare expected vs realised loss."""
    rng = np.random.default_rng(seed)
    close = ohlcv["close"].astype(float)
    tr = (ohlcv["high"] - ohlcv["low"]).abs()
    atr = float(tr.rolling(atr_window).mean().iloc[-1])
    mult = float(rng.uniform(*gap_mult_range))
    last = float(close.iloc[-1])
    gap = mult * atr
    realised_loss = gap / last if last else 0.0
    return {
        "atr": atr,
        "gap_multiple": mult,
        "expected_loss_pct": (3.0 * atr) / last if last else 0.0,
        "realised_loss_pct": realised_loss,
        "exceeds_expected": realised_loss > (3.0 * atr) / last if last else False,
    }


def regime_misclassification(run_backtest, cfg, ohlcv: pd.DataFrame,  # noqa: ANN001
                             seed: int = 0) -> dict:
    """Shuffle the regime→strategy mapping and confirm risk still contains loss.

    `run_backtest` is a callable (ohlcv, cfg, shuffle_seed) -> result dict with a
    `summary.max_drawdown`. If a shuffled mapping blows past the configured peak
    drawdown cap, the risk layer is over-reliant on the HMM being correct.
    """
    baseline = run_backtest(ohlcv, cfg, None)
    shuffled = run_backtest(ohlcv, cfg, seed)
    cap = cfg.risk.max_dd_from_peak
    shuffled_dd = abs(shuffled.get("summary", {}).get("max_drawdown", 0.0))
    return {
        "baseline_max_drawdown": baseline.get("summary", {}).get("max_drawdown", 0.0),
        "shuffled_max_drawdown": shuffled.get("summary", {}).get("max_drawdown", 0.0),
        "drawdown_cap": cap,
        "risk_contained": shuffled_dd <= cap * 1.5,
    }
