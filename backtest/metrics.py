"""Backtest metrics.

`total_return` / `sharpe` / `max_drawdown` / `win_rate` and the `summarize`
five-key summary are the legacy contract (unchanged). `summarize_full` adds the
final-prompt metrics: CAGR, Sortino, Calmar, max-DD duration, profit factor,
avg win/loss, avg holding period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / periods_per_year
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=0))


def sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / periods_per_year
    downside = r[r < 0]
    if len(r) < 2 or downside.std(ddof=0) == 0 or downside.empty:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / downside.std(ddof=0))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    cummax = equity_curve.cummax()
    dd = equity_curve / cummax - 1.0
    return float(dd.min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Longest run (in bars) the curve spends below a prior peak."""
    if equity_curve.empty:
        return 0
    cummax = equity_curve.cummax()
    underwater = equity_curve < cummax
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2:
        return 0.0
    years = len(equity_curve) / periods_per_year
    if years <= 0:
        return 0.0
    growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(growth ** (1.0 / years) - 1.0) if growth > 0 else -1.0


def calmar(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity_curve))
    return float(cagr(equity_curve, periods_per_year) / mdd) if mdd > 0 else 0.0


def win_rate(trade_pnls: pd.Series) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = -trade_pnls[trade_pnls < 0].sum()
    return float(gains / losses) if losses > 0 else 0.0


def avg_win_loss(trade_pnls: pd.Series) -> tuple[float, float]:
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    return (float(wins.mean()) if not wins.empty else 0.0,
            float(losses.mean()) if not losses.empty else 0.0)


def slice_by_regime(returns: pd.Series, regimes: pd.Series) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in regimes.dropna().unique():
        mask = regimes == label
        r = returns[mask]
        if r.empty:
            continue
        eq = (1 + r.fillna(0)).cumprod()
        out[str(label)] = {
            "total_return": total_return(eq),
            "sharpe": sharpe(r),
            "max_drawdown": max_drawdown(eq),
            "n_bars": int(len(r)),
        }
    return out


def slice_by_confidence(returns: pd.Series, confidence: pd.Series,
                        buckets: list[float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    edges = list(buckets)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        r = returns[mask]
        if r.empty:
            continue
        eq = (1 + r.fillna(0)).cumprod()
        out[f"[{lo:.2f},{hi:.2f})"] = {
            "total_return": total_return(eq),
            "sharpe": sharpe(r),
            "max_drawdown": max_drawdown(eq),
            "n_bars": int(len(r)),
        }
    return out


def summarize(equity_curve: pd.Series, returns: pd.Series, trade_pnls: pd.Series) -> dict:
    """Legacy five-key summary — keys are a stable contract."""
    return {
        "total_return": total_return(equity_curve),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trade_pnls),
        "total_trades": int(len(trade_pnls)),
    }


def summarize_full(equity_curve: pd.Series, returns: pd.Series, trade_pnls: pd.Series,
                   rf: float = 0.0, periods_per_year: int = 252) -> dict:
    """Extended final-prompt metric set."""
    avg_w, avg_l = avg_win_loss(trade_pnls)
    return {
        **summarize(equity_curve, returns, trade_pnls),
        "cagr": cagr(equity_curve, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year, rf),
        "sortino": sortino(returns, periods_per_year, rf),
        "calmar": calmar(equity_curve, periods_per_year),
        "max_drawdown_duration": max_drawdown_duration(equity_curve),
        "profit_factor": profit_factor(trade_pnls),
        "avg_win": avg_w,
        "avg_loss": avg_l,
    }
