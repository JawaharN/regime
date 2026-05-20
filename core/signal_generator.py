"""Signal generator — ties the HMM regime read to the vol-tier orchestrator.

This is a thin coordinator: given fresh bars for a symbol it builds features,
runs forward-only inference, passes the result through the stability filter,
and asks the orchestrator for a signal. The main loop and the backtester both
reuse it so the live and historical paths stay identical.
"""

from __future__ import annotations

import logging

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_stability import RegimeStabilityFilter
from core.regime_strategies import RegimeOrchestrator, Signal
from data.feature_engineering import build_features

logger = logging.getLogger("regime_trader.signal_generator")


class SignalGenerator:
    def __init__(self, engine: HMMEngine, stability: RegimeStabilityFilter,
                 orchestrator: RegimeOrchestrator, features_cfg) -> None:  # noqa: ANN001
        self.engine = engine
        self.stability = stability
        self.orchestrator = orchestrator
        self.features_cfg = features_cfg
        self.last_state: RegimeState | None = None

    def generate(self, symbol: str, bars: pd.DataFrame) -> list[Signal]:
        """Bars → features → forward inference → stability → orchestrator."""
        features = build_features(bars, self.features_cfg).dropna()
        if features.empty:
            logger.warning("no usable features for %s — holding", symbol)
            return []

        ts = bars.index[-1] if len(bars.index) else None
        try:
            state = self.engine.infer_regime_state(features, timestamp=ts)
        except Exception:  # noqa: BLE001 — HMM failure → hold, emit nothing
            logger.exception("HMM inference failed for %s — holding regime", symbol)
            return []

        stab = self.stability.update(state.label, state.probability)
        state.is_confirmed = not stab.in_transition
        state.consecutive_bars = stab.persistence_count
        self.last_state = state

        return self.orchestrator.generate_signals(
            symbol, bars, state, is_flickering=stab.unstable,
        )
