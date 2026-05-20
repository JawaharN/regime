"""Strategy layer: legacy regime router + final-prompt vol-tier strategies.

Two coexisting designs:

1. **Legacy** — ``StrategyOrchestrator`` routes a regime *label* to either
   ``BaseCaseStrategy`` or ``LowVolumeBullStrategy``, which size off the
   allocation table. Kept intact for backward compatibility.

2. **Final prompt** — three volatility-tier strategies
   (``LowVolBullStrategy`` / ``MidVolCautiousStrategy`` /
   ``HighVolDefensiveStrategy``) with explicit ATR/EMA stop formulas, routed by
   ``RegimeOrchestrator`` strictly on *volatility rank* — never on the
   return-sorted label. Position = rank / (n_regimes - 1): ≤0.33 → low-vol,
   ≥0.67 → high-vol, else mid-vol.

The strategy is always-long / never-short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

from core.allocation import allocate


@dataclass
class Signal:
    """Trade intent. The first six fields are the legacy contract; the rest are
    the enriched final-prompt fields (all optional, so old callers are unaffected)."""

    symbol: str
    side: Literal["BUY", "SELL", "FLAT"] = "FLAT"
    target_weight: float = 0.0
    confidence: float = 0.0
    regime: str = ""
    reason: str = ""
    # enriched
    direction: Literal["LONG", "FLAT"] = "FLAT"
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size_pct: float = 0.0
    leverage: float = 1.0
    regime_id: int | None = None
    regime_name: str = ""
    regime_probability: float = 0.0
    strategy_name: str = ""
    timestamp: datetime | None = None
    metadata: dict = field(default_factory=dict)


# Regime aliases — user-friendly synonyms map back to canonical labels.
REGIME_ALIASES: dict[str, str] = {
    "calm": "bull",
    "low_vol_bull": "bull",
    "panic": "crash",
    "drawdown": "bear",
    "sideways": "neutral",
    "blowoff": "euphoria",
}


def canonical_regime(name: str) -> str:
    return REGIME_ALIASES.get(name, name)


# ---------------------------------------------------------------- legacy layer

class TrendIndicator:
    """Trend confirmation: fast SMA > slow SMA AND last close > slow SMA."""

    def __init__(self, fast: int, slow: int) -> None:
        self.fast = fast
        self.slow = slow

    def is_uptrend(self, close: pd.Series) -> bool:
        if len(close) < self.slow:
            return False
        fast_sma = close.rolling(self.fast).mean().iloc[-1]
        slow_sma = close.rolling(self.slow).mean().iloc[-1]
        if pd.isna(fast_sma) or pd.isna(slow_sma):
            return False
        return bool(fast_sma > slow_sma and close.iloc[-1] > slow_sma)


class BaseCaseStrategy:
    """Long-only: emit a BUY at the allocation target, else FLAT."""

    name = "base_case"

    def evaluate(self, symbol: str, regime: str, confidence: float,
                 trend_ok: bool, allocation_cfg) -> list[Signal]:  # noqa: ANN001
        decision = allocate(regime, confidence, trend_ok, allocation_cfg)
        side: Literal["BUY", "SELL", "FLAT"] = "BUY" if decision.target_exposure > 0 else "FLAT"
        return [Signal(
            symbol=symbol, side=side,
            target_weight=min(decision.target_exposure, decision.leverage_cap),
            confidence=confidence, regime=regime,
            reason=f"{self.name}: {decision.reason}",
            direction="LONG" if side == "BUY" else "FLAT", strategy_name=self.name,
        )]


class LowVolumeBullStrategy:
    """Aggressive in calm bull markets — uses up to the leverage cap."""

    name = "low_vol_bull"

    def evaluate(self, symbol: str, regime: str, confidence: float,
                 trend_ok: bool, allocation_cfg) -> list[Signal]:  # noqa: ANN001
        decision = allocate(regime, confidence, trend_ok, allocation_cfg)
        target = decision.target_exposure
        if regime == "bull" and confidence >= 0.8:
            target = decision.leverage_cap
        target = min(target, decision.leverage_cap)
        side: Literal["BUY", "SELL", "FLAT"] = "BUY" if target > 0 else "FLAT"
        return [Signal(
            symbol=symbol, side=side, target_weight=target,
            confidence=confidence, regime=regime,
            reason=f"{self.name}: {decision.reason}",
            direction="LONG" if side == "BUY" else "FLAT", strategy_name=self.name,
        )]


class StrategyOrchestrator:
    """Legacy router: bull regimes → LowVolumeBull, else BaseCase."""

    def __init__(self, allocation_cfg, strategy_cfg) -> None:  # noqa: ANN001
        self.allocation_cfg = allocation_cfg
        self.strategy_cfg = strategy_cfg
        self.trend = TrendIndicator(strategy_cfg.trend_confirmation.fast_sma,
                                    strategy_cfg.trend_confirmation.slow_sma)
        self._strategies = {
            "base_case": BaseCaseStrategy(),
            "low_vol_bull": LowVolumeBullStrategy(),
        }

    def evaluate(self, symbol: str, regime: str, confidence: float,
                 close_series: pd.Series) -> list[Signal]:
        regime = canonical_regime(regime)
        trend_ok = self.trend.is_uptrend(close_series)
        strategy_name = "low_vol_bull" if regime == "bull" else "base_case"
        return self._strategies[strategy_name].evaluate(
            symbol, regime, confidence, trend_ok, self.allocation_cfg
        )


# ------------------------------------------------------- final vol-tier layer

def _true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    return pd.concat([
        (bars["high"] - bars["low"]).abs(),
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _atr(bars: pd.DataFrame, window: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    val = _true_range(bars).rolling(window, min_periods=1).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _ema(series: pd.Series, span: int) -> float:
    val = series.ewm(span=span, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) else float(series.iloc[-1])


class BaseStrategy:
    """ABC for the vol-tier strategies."""

    name = "base"
    strategy_type = "mid_vol"

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state,
                        cfg) -> Signal | None:  # noqa: ANN001
        raise NotImplementedError


class LowVolBullStrategy(BaseStrategy):
    """Calm bull regime: full allocation, low-vol leverage, wide stop."""

    name = "low_vol_bull"
    strategy_type = "low_vol"

    def generate_signal(self, symbol, bars, regime_state, cfg) -> Signal | None:  # noqa: ANN001
        price = float(bars["close"].iloc[-1])
        atr, ema50 = _atr(bars), _ema(bars["close"], 50)
        stop = max(price - 3 * atr, ema50 - 0.5 * atr)
        pos, lev = cfg.low_vol_allocation, cfg.low_vol_leverage
        return _build(symbol, self.name, price, stop, pos, lev, regime_state,
                      "low-vol bull: full allocation with leverage")


class MidVolCautiousStrategy(BaseStrategy):
    """Mid-vol regime: full allocation only with price above EMA-50."""

    name = "mid_vol_cautious"
    strategy_type = "mid_vol"

    def generate_signal(self, symbol, bars, regime_state, cfg) -> Signal | None:  # noqa: ANN001
        price = float(bars["close"].iloc[-1])
        atr, ema50 = _atr(bars), _ema(bars["close"], 50)
        trend_ok = price > ema50
        pos = cfg.mid_vol_allocation_trend if trend_ok else cfg.mid_vol_allocation_no_trend
        stop = ema50 - 0.5 * atr
        reason = "mid-vol: " + ("trend confirmed" if trend_ok else "no trend → reduced")
        return _build(symbol, self.name, price, stop, pos, 1.0, regime_state, reason)


class HighVolDefensiveStrategy(BaseStrategy):
    """High-vol / crash regime: defensive allocation, tight stop, no leverage."""

    name = "high_vol_defensive"
    strategy_type = "high_vol"

    def generate_signal(self, symbol, bars, regime_state, cfg) -> Signal | None:  # noqa: ANN001
        price = float(bars["close"].iloc[-1])
        atr, ema50 = _atr(bars), _ema(bars["close"], 50)
        stop = ema50 - 1.0 * atr
        return _build(symbol, self.name, price, stop, cfg.high_vol_allocation, 1.0,
                      regime_state, "high-vol defensive: reduced exposure, tight stop")


def _build(symbol: str, strat: str, price: float, stop: float, pos: float,
           lev: float, regime_state, reason: str) -> Signal:  # noqa: ANN001
    return Signal(
        symbol=symbol, side="BUY", direction="LONG",
        target_weight=pos * lev, position_size_pct=pos, leverage=lev,
        entry_price=price, stop_loss=stop,
        confidence=getattr(regime_state, "probability", 0.0),
        regime=getattr(regime_state, "label", ""),
        regime_id=getattr(regime_state, "state_id", None),
        regime_name=getattr(regime_state, "label", ""),
        regime_probability=getattr(regime_state, "probability", 0.0),
        reason=reason, strategy_name=strat,
        timestamp=getattr(regime_state, "timestamp", None),
    )


# Final-prompt aliases.
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy

LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": HighVolDefensiveStrategy, "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR": HighVolDefensiveStrategy, "WEAK_BEAR": MidVolCautiousStrategy,
    "NEUTRAL": MidVolCautiousStrategy, "WEAK_BULL": MidVolCautiousStrategy,
    "BULL": LowVolBullStrategy, "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}

_TYPE_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "low_vol": LowVolBullStrategy,
    "mid_vol": MidVolCautiousStrategy,
    "high_vol": HighVolDefensiveStrategy,
}


class RegimeOrchestrator:
    """Vol-rank router for the final-prompt vol-tier strategies.

    On construction (and after every HMM retrain) the regime ids are sorted by
    *expected volatility* and mapped to a strategy tier via rank/(n-1) — this is
    independent of the return-sorted human label.
    """

    def __init__(self, strategy_cfg, regime_infos: list, min_confidence: float = 0.55) -> None:  # noqa: ANN001
        self.cfg = strategy_cfg
        self.min_confidence = min_confidence
        self._state_to_strategy: dict[int, BaseStrategy] = {}
        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: list) -> None:  # noqa: ANN001
        """Rebuild the state→strategy map after a retrain."""
        self._state_to_strategy = {}
        ordered = sorted(regime_infos, key=lambda r: r.expected_volatility)
        n = len(ordered)
        for rank, info in enumerate(ordered):
            pos = rank / (n - 1) if n > 1 else 0.0
            if pos <= 0.33:
                stype = "low_vol"
            elif pos >= 0.67:
                stype = "high_vol"
            else:
                stype = "mid_vol"
            self._state_to_strategy[info.regime_id] = _TYPE_TO_STRATEGY[stype]()

    def generate_signals(self, symbol: str, bars: pd.DataFrame, regime_state,
                         is_flickering: bool = False) -> list[Signal]:  # noqa: ANN001
        """One symbol → 0 or 1 signal. Applies uncertainty mode."""
        strat = self._state_to_strategy.get(getattr(regime_state, "state_id", -1))
        if strat is None or bars is None or bars.empty:
            return []
        sig = strat.generate_signal(symbol, bars, regime_state, self.cfg)
        if sig is None:
            return []

        uncertain = regime_state.probability < self.min_confidence or is_flickering
        if uncertain:
            sig.position_size_pct *= self.cfg.uncertainty_size_mult
            sig.leverage = 1.0
            sig.target_weight = sig.position_size_pct
            sig.reason += " [UNCERTAINTY — size halved]"
            sig.metadata["uncertainty"] = True
        return [sig]
