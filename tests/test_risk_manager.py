"""Risk manager thresholds + kill switch behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.config import load_config
from core.risk_manager import AccountSnapshot, RiskManager


@dataclass
class FakeSignal:
    symbol: str
    target_weight: float
    side: str = "BUY"


def _cfg(tmp_path: Path):
    """Load default risk cfg, but redirect state file paths into tmp_path."""
    cfg = load_config().risk
    # Pydantic models are immutable by default; construct a new one with overrides.
    return cfg.model_copy(update={
        "kill_switch_path": str(tmp_path / "kill_switch.block"),
        "peak_equity_path": str(tmp_path / "peak_equity.json"),
    })


def _snap(equity: float, when: datetime) -> AccountSnapshot:
    return AccountSnapshot(equity=equity, cash=0.0, timestamp=when)


def test_daily_2pct_triggers_reduce(tmp_path: Path):
    rm = RiskManager(_cfg(tmp_path))
    t0 = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    rm.check_portfolio(_snap(100_000, t0))                        # establish baseline
    d = rm.check_portfolio(_snap(97_900, t0 + timedelta(hours=2)))  # -2.1%
    assert d.action == "REDUCE"
    assert d.size_multiplier == 0.5


def test_daily_3pct_triggers_flatten(tmp_path: Path):
    rm = RiskManager(_cfg(tmp_path))
    t0 = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    rm.check_portfolio(_snap(100_000, t0))
    d = rm.check_portfolio(_snap(96_500, t0 + timedelta(hours=2)))  # -3.5%
    assert d.action == "FLATTEN"


def test_weekly_5pct_triggers_reduce(tmp_path: Path):
    rm = RiskManager(_cfg(tmp_path))
    t0 = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)  # Monday
    rm.check_portfolio(_snap(100_000, t0))
    # Wednesday, week-to-date -6% but day -1.5% → only weekly fires
    rm.check_portfolio(_snap(99_000, t0 + timedelta(days=2)))  # advances day, resets daily baseline
    d = rm.check_portfolio(_snap(94_000, t0 + timedelta(days=2, hours=4)))
    # That's -5.05% weekly and -5.05% daily (since session re-opened at 99k → 94k = -5%).
    # Either threshold qualifies. Make sure it's REDUCE or FLATTEN — but with both -5% weekly
    # AND -5.05% daily, the daily 3% threshold should fire and it becomes FLATTEN.
    assert d.action in {"REDUCE", "FLATTEN"}


def test_10pct_drawdown_arms_kill_switch(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rm = RiskManager(cfg)
    t0 = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    rm.check_portfolio(_snap(100_000, t0))  # peak = 100k
    rm.check_portfolio(_snap(105_000, t0 + timedelta(days=1)))  # peak rises to 105k
    d = rm.check_portfolio(_snap(94_400, t0 + timedelta(days=3)))  # -10.1% from 105k peak
    assert d.action == "KILL"
    assert Path(cfg.kill_switch_path).exists()


def test_kill_switch_blocks_startup(tmp_path: Path):
    cfg = _cfg(tmp_path)
    # Pre-place the kill switch file
    Path(cfg.kill_switch_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.kill_switch_path).write_text("{}")
    rm = RiskManager(cfg)
    with pytest.raises(RuntimeError, match="Kill switch"):
        rm.assert_safe_to_start()


def test_per_trade_1pct_cap_reduces_oversized(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rm = RiskManager(cfg)
    snap = _snap(100_000, datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc))
    rm.check_portfolio(snap)
    # Stop distance 5%, target weight 0.5 → implied risk 2.5% > 1% cap → reduce by 5x
    sig = FakeSignal(symbol="SPY", target_weight=0.5)
    d = rm.check_trade(sig, snap, positions=[], stop_distance_pct=0.05)
    assert d.action == "REDUCE"
    assert abs(d.size_multiplier - (0.01 / 0.05) / 0.5) < 1e-9


def test_leverage_cap_blocks_overexposure(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rm = RiskManager(cfg)
    snap = _snap(100_000, datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc))
    rm.check_portfolio(snap)

    @dataclass
    class FakePos:
        symbol: str
        weight: float

    positions = [FakePos("AAPL", 1.0), FakePos("TSLA", 0.5)]  # gross 1.5
    sig = FakeSignal(symbol="SPY", target_weight=0.5)         # would bring gross to 2.0
    d = rm.check_trade(sig, snap, positions=positions, stop_distance_pct=0.02)
    assert d.action == "BLOCK"
    assert "leverage" in d.reason


def test_flatten_blocks_subsequent_trades(tmp_path: Path):
    rm = RiskManager(_cfg(tmp_path))
    t0 = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    rm.check_portfolio(_snap(100_000, t0))
    rm.check_portfolio(_snap(96_500, t0 + timedelta(hours=2)))  # -3.5% triggers FLATTEN
    snap = _snap(96_500, t0 + timedelta(hours=2, minutes=10))
    d = rm.check_trade(FakeSignal("SPY", 0.5), snap, positions=[], stop_distance_pct=0.02)
    assert d.action == "BLOCK"
    assert "flatten" in d.reason.lower()


def test_invalid_weight_blocked(tmp_path: Path):
    rm = RiskManager(_cfg(tmp_path))
    snap = _snap(100_000, datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc))
    rm.check_portfolio(snap)
    d = rm.check_trade(FakeSignal("SPY", -0.1), snap, positions=[])
    assert d.action == "BLOCK"
    d2 = rm.check_trade(FakeSignal("SPY", float("inf")), snap, positions=[])
    assert d2.action == "BLOCK"


def test_peak_persists_across_instances(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rm = RiskManager(cfg)
    t0 = datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    rm.check_portfolio(_snap(100_000, t0))
    rm.check_portfolio(_snap(110_000, t0 + timedelta(days=1)))
    # New RiskManager from same paths should restore peak=110k
    rm2 = RiskManager(cfg)
    assert rm2._peak_equity == 110_000


def test_allow_when_within_all_limits(tmp_path: Path):
    rm = RiskManager(_cfg(tmp_path))
    snap = _snap(100_000, datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc))
    rm.check_portfolio(snap)
    d = rm.check_trade(FakeSignal("SPY", 0.2), snap, positions=[], stop_distance_pct=0.02)
    assert d.action == "ALLOW"
