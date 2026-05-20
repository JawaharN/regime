"""Final-prompt risk layer: validate_signal + two-tier circuit breakers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config import load_config
from core.regime_strategies import Signal
from core.risk_manager import CircuitBreaker, PortfolioState, RiskManager


def _cfg(tmp_path: Path):
    return load_config().risk.model_copy(update={
        "kill_switch_path": str(tmp_path / "kill.block"),
        "peak_equity_path": str(tmp_path / "peak.json"),
        "halt_lock_path": str(tmp_path / "trading_halted.lock"),
    })


@dataclass
class _Pos:
    symbol: str
    weight: float


def _signal(stop=98.0, entry=100.0, pos=0.10):
    return Signal(symbol="SPY", side="BUY", direction="LONG", target_weight=pos,
                  position_size_pct=pos, entry_price=entry, stop_loss=stop)


def _portfolio(equity=100_000.0, **kw):
    return PortfolioState(equity=equity, cash=equity, peak_equity=equity, **kw)


def test_signal_without_stop_is_blocked(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    sig = Signal(symbol="SPY", side="BUY", target_weight=0.1, position_size_pct=0.1,
                 entry_price=100.0)  # no stop_loss
    d = rm.validate_signal(sig, _portfolio())
    assert d.action == "BLOCK"
    assert not d.approved


def test_well_formed_signal_is_approved(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    d = rm.validate_signal(_signal(), _portfolio())
    assert d.approved
    assert d.action in {"ALLOW", "REDUCE"}


def test_single_position_cap_reduces(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    # full-size request with a tight 2% stop → risk sizing leaves it above the
    # 15% single-position cap, so the cap binds.
    d = rm.validate_signal(_signal(stop=98.0, pos=0.95), _portfolio())
    assert d.action == "REDUCE"
    assert any("single-position" in m for m in d.modifications)


def test_gap_risk_cap_applies_for_wide_stop(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    # stop 10% away → gap-risk overnight cap binds
    d = rm.validate_signal(_signal(stop=90.0, pos=0.95), _portfolio())
    assert "gap-risk" in d.reason


def test_duplicate_order_blocked(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    rm.record_trade("SPY", "LONG")
    d = rm.validate_signal(_signal(), _portfolio())
    assert d.action == "BLOCK"
    assert "duplicate" in (d.rejection_reason or "")


def test_correlation_reject(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    d = rm.validate_signal(_signal(), _portfolio(), correlations={"QQQ": 0.92})
    assert d.action == "BLOCK"
    assert "correlation" in (d.rejection_reason or "")


def test_max_concurrent_blocks(tmp_path):
    rm = RiskManager(_cfg(tmp_path))
    pf = _portfolio(positions=[_Pos(s, 0.1) for s in ("A", "B", "C", "D", "E")])
    d = rm.validate_signal(_signal(), pf)
    assert d.action == "BLOCK"


# ---------------------------------------------------- circuit breaker tiers

def test_daily_2pct_reduces(tmp_path):
    cb = CircuitBreaker(_cfg(tmp_path))
    r = cb.check(_portfolio(daily_pnl=-0.025))
    assert r.action == "REDUCE"


def test_daily_3pct_flattens(tmp_path):
    cb = CircuitBreaker(_cfg(tmp_path))
    r = cb.check(_portfolio(daily_pnl=-0.035))
    assert r.action == "FLATTEN"


def test_weekly_7pct_flattens(tmp_path):
    cb = CircuitBreaker(_cfg(tmp_path))
    r = cb.check(_portfolio(weekly_pnl=-0.075))
    assert r.action == "FLATTEN"


def test_peak_drawdown_arms_halt_lock(tmp_path):
    cfg = _cfg(tmp_path)
    cb = CircuitBreaker(cfg)
    r = cb.check(PortfolioState(equity=89_000.0, cash=89_000.0, peak_equity=100_000.0))
    assert r.action == "KILL"
    assert Path(cfg.halt_lock_path).exists()


def test_halt_lock_blocks_startup(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg.halt_lock_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.halt_lock_path).write_text("{}")
    rm = RiskManager(cfg)
    import pytest
    with pytest.raises(RuntimeError, match="halt lock"):
        rm.assert_safe_to_start()
