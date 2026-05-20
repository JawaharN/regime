"""Walk-forward backtest engine.

`run_walk_forward` keeps the legacy return-multiply simulation (stable output
contract used by the CLI and tests). `run_allocation_backtest` implements the
final-prompt **explicit allocation math**: it tracks cash and shares directly,
so leverage > 1.0 drives cash negative — that negative balance is the margin
loan, and ``equity = cash + shares*price`` stays correct throughout.

Both paths use the **filtered forward algorithm** and the stability filter, so
each OOS bar's regime is derived only from its own past.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtest import benchmarks, metrics
from core.hmm_engine import HMMEngine
from core.regime_stability import RegimeStabilityFilter
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import build_features, feature_spec_from_cfg

logger = logging.getLogger("regime_trader.backtest")


@dataclass
class WindowResult:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    total_return: float
    sharpe: float
    max_drawdown: float
    n_bars: int


@dataclass
class AllocationBacktestResult:
    equity_curve: pd.Series
    cash_curve: pd.Series
    trade_log: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ----------------------------------------------------- explicit allocation math

def run_allocation_backtest(close: pd.Series, target_weights: pd.Series,
                            initial_capital: float = 100_000.0,
                            slippage_pct: float = 0.0005,
                            rebalance_threshold: float = 0.10) -> AllocationBacktestResult:
    """Walk an allocation series with explicit cash/share accounting.

    `target_weights[t]` is the allocation decided from data up to bar t; it is
    filled at bar t+1 (1-bar delay). A rebalance only happens when the target
    differs from the current allocation by more than `rebalance_threshold`.
    """
    close = close.astype(float)
    weights = target_weights.reindex(close.index).ffill().fillna(0.0)

    cash = float(initial_capital)
    shares = 0.0
    equity_path: list[float] = []
    cash_path: list[float] = []
    trade_log: list[dict] = []

    for i, ts in enumerate(close.index):
        price = float(close.iloc[i])
        equity = cash + shares * price
        # 1-bar fill delay: act on the weight decided at the *previous* bar.
        target_alloc = float(weights.iloc[i - 1]) if i > 0 else 0.0
        current_alloc = (shares * price / equity) if equity > 0 else 0.0

        if abs(target_alloc - current_alloc) > rebalance_threshold and price > 0:
            target_shares = int(equity * target_alloc / price)
            delta = target_shares - shares
            if delta != 0:
                sign = 1.0 if delta > 0 else -1.0
                cash -= delta * price * (1 + slippage_pct * sign)
                shares = target_shares
                trade_log.append({
                    "ts": ts, "price": price, "delta_shares": delta,
                    "target_alloc": target_alloc, "equity": cash + shares * price,
                })
        equity_path.append(cash + shares * price)
        cash_path.append(cash)

    equity_curve = pd.Series(equity_path, index=close.index)
    cash_curve = pd.Series(cash_path, index=close.index)
    returns = equity_curve.pct_change().fillna(0.0)
    pnls = pd.Series([t["equity"] for t in trade_log]).pct_change().fillna(0.0)
    return AllocationBacktestResult(
        equity_curve=equity_curve,
        cash_curve=cash_curve,
        trade_log=trade_log,
        summary=metrics.summarize_full(equity_curve, returns, pnls),
    )


# --------------------------------------------------------- legacy walk-forward

def _apply_costs(weight_change: pd.Series, slippage_bps: float, commission_bps: float) -> pd.Series:
    bps = (slippage_bps + commission_bps) / 1e4
    return weight_change.abs() * bps


def _simulate(close: pd.Series, weights: pd.Series, slippage_bps: float,
              commission_bps: float) -> dict:
    rets = close.pct_change().fillna(0.0)
    pos = weights.shift(1).fillna(0.0)
    gross = rets * pos
    cost = _apply_costs(weights.diff().fillna(0.0), slippage_bps, commission_bps)
    strat_rets = gross - cost
    equity = (1 + strat_rets).cumprod()
    trade_pnls = strat_rets[weights.diff().fillna(0) != 0]
    return {"equity": equity, "returns": strat_rets, "trade_pnls": trade_pnls}


def run_walk_forward(symbol: str, ohlcv: pd.DataFrame, cfg) -> dict:  # noqa: ANN001
    """Rolling IS/OOS walk-forward backtest over the provided OHLCV history."""
    feature_cfg = cfg.features
    spec = feature_spec_from_cfg(feature_cfg)

    full_features = build_features(ohlcv, feature_cfg).dropna()
    aligned_close = ohlcv["close"].loc[full_features.index]

    is_days = cfg.backtest.in_sample_days
    oos_days = cfg.backtest.out_of_sample_days
    step = cfg.backtest.step_days

    window_results: list[WindowResult] = []
    all_returns = pd.Series(dtype=float)
    all_weights = pd.Series(dtype=float)
    all_regimes = pd.Series(dtype=object)
    all_confs = pd.Series(dtype=float)

    n = len(full_features)
    start = 0
    while start + is_days + oos_days <= n:
        is_slice = full_features.iloc[start: start + is_days]
        oos_slice = full_features.iloc[start + is_days: start + is_days + oos_days]
        oos_close = aligned_close.loc[oos_slice.index]

        engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, spec)
        try:
            engine.fit(is_slice)
        except Exception as e:  # noqa: BLE001
            logger.warning("HMM fit failed at window starting %s: %s", is_slice.index[0], e)
            start += step
            continue

        path = engine.infer_forward_path(oos_slice)
        stability = RegimeStabilityFilter(
            cfg.stability.min_persistence_bars, cfg.stability.flicker_window,
            cfg.stability.flicker_threshold, cfg.stability.unstable_confidence_decay,
            cfg.stability.transition_size_cut,
        )
        orch = StrategyOrchestrator(cfg.allocation, cfg.strategy)

        weights, regimes_out, confs_out = [], [], []
        close_history = ohlcv["close"].loc[: oos_close.index[-1]]
        for ts, inf in zip(oos_slice.index, path):
            stab = stability.update(inf.label, inf.confidence)
            actionable = stab.actionable_regime or "neutral"
            close_so_far = close_history.loc[: ts]
            sigs = orch.evaluate(symbol, actionable, stab.confidence, close_so_far)
            w = sigs[0].target_weight if sigs and sigs[0].side != "FLAT" else 0.0
            weights.append(w)
            regimes_out.append(actionable)
            confs_out.append(stab.confidence)

        weight_series = pd.Series(weights, index=oos_slice.index)
        regime_series = pd.Series(regimes_out, index=oos_slice.index)
        conf_series = pd.Series(confs_out, index=oos_slice.index)

        sim = _simulate(oos_close, weight_series,
                        cfg.backtest.slippage_bps, cfg.backtest.commission_bps)

        window_results.append(WindowResult(
            is_start=is_slice.index[0], is_end=is_slice.index[-1],
            oos_start=oos_slice.index[0], oos_end=oos_slice.index[-1],
            total_return=metrics.total_return(sim["equity"]),
            sharpe=metrics.sharpe(sim["returns"]),
            max_drawdown=metrics.max_drawdown(sim["equity"]),
            n_bars=int(len(sim["returns"])),
        ))

        all_returns = pd.concat([all_returns, sim["returns"]])
        all_weights = pd.concat([all_weights, weight_series])
        all_regimes = pd.concat([all_regimes, regime_series])
        all_confs = pd.concat([all_confs, conf_series])
        start += step

    if all_returns.empty:
        return {"windows": [], "summary": {}, "regime_breakdown": {},
                "confidence_breakdown": {}, "benchmarks": {}}

    equity = (1 + all_returns).cumprod()
    trade_pnls = all_returns[all_weights.diff().fillna(0) != 0]
    summary = metrics.summarize(equity, all_returns, trade_pnls)
    regime_breakdown = metrics.slice_by_regime(all_returns, all_regimes)
    confidence_breakdown = metrics.slice_by_confidence(
        all_returns, all_confs, cfg.backtest.confidence_buckets
    )

    bench_close = ohlcv["close"].loc[all_returns.index]
    benchmark_results = {
        "buy_and_hold": _bench_summary(benchmarks.buy_and_hold(bench_close)),
        "sma_trend_200": _bench_summary(benchmarks.sma_trend(bench_close, window=200)),
        "random_baseline": _bench_summary(benchmarks.random_baseline(bench_close)),
    }

    return {
        "windows": [w.__dict__ for w in window_results],
        "summary": summary,
        "regime_breakdown": regime_breakdown,
        "confidence_breakdown": confidence_breakdown,
        "benchmarks": benchmark_results,
    }


def _bench_summary(b: dict) -> dict:
    return {
        "total_return": metrics.total_return(b["equity"]),
        "sharpe": metrics.sharpe(b["returns"]),
        "max_drawdown": metrics.max_drawdown(b["equity"]),
    }
