"""Vol-tier orchestrator routes by volatility rank, never by the return label."""

from __future__ import annotations

from core.config import load_config
from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    RegimeOrchestrator,
)


def _info(rid: int, exp_return: float, exp_vol: float) -> RegimeInfo:
    return RegimeInfo(
        regime_id=rid, regime_name=f"r{rid}", expected_return=exp_return,
        expected_volatility=exp_vol, recommended_strategy_type="mid_vol",
        max_leverage_allowed=1.0, max_position_size_pct=0.95,
        min_confidence_to_act=0.55,
    )


def test_high_return_high_vol_regime_routes_to_defensive(synthetic_ohlcv):
    """A regime with the *best* return but the *worst* volatility must still
    route to the high-vol defensive strategy — label must not drive sizing."""
    cfg = load_config()
    # regime 4 has the highest expected return yet the highest volatility.
    infos = [
        _info(0, -0.02, 0.005),
        _info(1, -0.01, 0.010),
        _info(2, 0.00, 0.015),
        _info(3, 0.01, 0.020),
        _info(4, 0.05, 0.090),   # "euphoria"-like return, but most volatile
    ]
    orch = RegimeOrchestrator(cfg.strategy, infos, min_confidence=0.55)

    state = RegimeState(label="euphoria", state_id=4, probability=0.9,
                        state_probabilities=[0.0, 0.0, 0.0, 0.1, 0.9])
    sigs = orch.generate_signals("SPY", synthetic_ohlcv, state)
    assert len(sigs) == 1
    assert sigs[0].strategy_name == HighVolDefensiveStrategy.name
    assert sigs[0].position_size_pct == cfg.strategy.high_vol_allocation


def test_lowest_vol_regime_routes_to_low_vol_bull(synthetic_ohlcv):
    cfg = load_config()
    infos = [
        _info(0, 0.05, 0.005),   # calmest regime
        _info(1, 0.01, 0.020),
        _info(2, 0.00, 0.040),
        _info(3, -0.01, 0.070),
        _info(4, -0.05, 0.090),
    ]
    orch = RegimeOrchestrator(cfg.strategy, infos, min_confidence=0.55)
    state = RegimeState(label="bear", state_id=0, probability=0.9)
    sigs = orch.generate_signals("SPY", synthetic_ohlcv, state)
    assert sigs[0].strategy_name == LowVolBullStrategy.name
    assert sigs[0].leverage == cfg.strategy.low_vol_leverage


def test_uncertainty_mode_halves_size_and_drops_leverage(synthetic_ohlcv):
    cfg = load_config()
    infos = [_info(i, 0.0, 0.01 * (i + 1)) for i in range(3)]
    orch = RegimeOrchestrator(cfg.strategy, infos, min_confidence=0.55)
    # probability below min_confidence → uncertainty mode
    state = RegimeState(label="neutral", state_id=0, probability=0.40)
    sigs = orch.generate_signals("SPY", synthetic_ohlcv, state)
    assert sigs[0].leverage == 1.0
    assert "UNCERTAINTY" in sigs[0].reason
