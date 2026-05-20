"""Allocation: regime → exposure, leverage cap, trend gate, confidence floor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config
from core.allocation import allocate
from core.regime_strategies import (
    REGIME_ALIASES,
    StrategyOrchestrator,
    canonical_regime,
)


def _alloc_cfg():
    return load_config().allocation


def test_bull_uses_high_exposure():
    d = allocate("bull", confidence=0.9, trend_ok=True, cfg=_alloc_cfg())
    assert d.target_exposure == 0.95


def test_crash_goes_flat():
    d = allocate("crash", confidence=0.9, trend_ok=True, cfg=_alloc_cfg())
    assert d.target_exposure == 0.0


def test_neutral_flat_without_trend_confirmation():
    d = allocate("neutral", confidence=0.9, trend_ok=False, cfg=_alloc_cfg())
    assert d.target_exposure == 0.0
    assert "trend gate failed" in d.reason


def test_neutral_invested_with_trend_confirmation():
    d = allocate("neutral", confidence=0.9, trend_ok=True, cfg=_alloc_cfg())
    assert d.target_exposure == 0.5


def test_low_confidence_scales_exposure():
    cfg = _alloc_cfg()
    floor = cfg.confidence_floor   # default 0.5
    d = allocate("bull", confidence=floor / 2, trend_ok=True, cfg=cfg)
    # exposure = 0.95 * (0.25 / 0.5) = 0.475
    assert abs(d.target_exposure - 0.95 * 0.5) < 1e-9


def test_leverage_cap_respected_in_orchestrator():
    cfg = load_config()
    orch = StrategyOrchestrator(cfg.allocation, cfg.strategy)
    # Build a clear uptrend close series
    close = pd.Series(np.linspace(100, 200, 300))
    sigs = orch.evaluate("SPY", "bull", confidence=0.95, close_series=close)
    assert len(sigs) == 1
    s = sigs[0]
    assert s.side == "BUY"
    # LowVolumeBull pushes to leverage_cap (1.25) at high confidence in bull
    assert s.target_weight <= 1.25 + 1e-9
    assert s.target_weight >= 0.95 - 1e-9


def test_orchestrator_routes_crash_to_flat():
    cfg = load_config()
    orch = StrategyOrchestrator(cfg.allocation, cfg.strategy)
    close = pd.Series(np.linspace(100, 200, 300))  # uptrend
    sigs = orch.evaluate("SPY", "crash", confidence=0.9, close_series=close)
    assert sigs[0].side == "FLAT"
    assert sigs[0].target_weight == 0.0


def test_aliases_resolve_to_canonical():
    assert canonical_regime("calm") == "bull"
    assert canonical_regime("panic") == "crash"
    assert canonical_regime("bull") == "bull"  # already canonical
    # And the alias table covers each canonical name at least once via key→canonical
    for alias, target in REGIME_ALIASES.items():
        assert canonical_regime(alias) == target


def test_neutral_routes_to_base_case_and_requires_trend():
    cfg = load_config()
    orch = StrategyOrchestrator(cfg.allocation, cfg.strategy)
    # Flat series → no uptrend
    flat = pd.Series([100.0] * 300)
    sigs = orch.evaluate("SPY", "neutral", confidence=0.8, close_series=flat)
    assert sigs[0].side == "FLAT"
