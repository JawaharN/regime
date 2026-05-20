"""Regime stability: 3-bar persistence, flicker decay, unstable warning."""

from __future__ import annotations

import logging

from core.config import load_config
from core.regime_stability import RegimeStabilityFilter


def _build(**overrides):
    cfg = load_config().stability
    return RegimeStabilityFilter(
        min_persistence_bars=overrides.get("min_persistence_bars", cfg.min_persistence_bars),
        flicker_window=overrides.get("flicker_window", cfg.flicker_window),
        flicker_threshold=overrides.get("flicker_threshold", cfg.flicker_threshold),
        unstable_confidence_decay=overrides.get(
            "unstable_confidence_decay", cfg.unstable_confidence_decay
        ),
    )


def test_persistence_gate_holds_until_3_bars():
    f = _build()
    # First inference: not enough persistence yet → actionable still None
    r1 = f.update("bull", 0.9)
    assert r1.actionable_regime is None
    r2 = f.update("bull", 0.9)
    assert r2.actionable_regime is None
    r3 = f.update("bull", 0.9)
    assert r3.actionable_regime == "bull"


def test_single_flip_does_not_change_actionable():
    f = _build()
    # Establish bull
    for _ in range(3):
        f.update("bull", 0.9)
    # One bar of bear — must not flip actionable
    r = f.update("bear", 0.7)
    assert r.actionable_regime == "bull"
    # Two more bars of bear → still bull because streak is only 2
    r = f.update("bear", 0.7)
    assert r.actionable_regime == "bull"
    # Third consecutive bear → now actionable flips
    r = f.update("bear", 0.7)
    assert r.actionable_regime == "bear"


def test_flicker_count_marks_unstable_and_decays_confidence():
    # Use a small window for quick test
    f = _build(min_persistence_bars=2, flicker_window=10, flicker_threshold=4,
               unstable_confidence_decay=0.5)
    # 5 alternations → > 4 changes
    sequence = ["bull", "bear", "bull", "bear", "bull", "bear", "bull"]
    last = None
    for s in sequence:
        last = f.update(s, 1.0)
    assert last is not None
    assert last.unstable is True
    assert last.confidence == 0.5  # 1.0 * decay 0.5


def test_stable_regime_full_confidence():
    f = _build()
    last = None
    for _ in range(10):
        last = f.update("bull", 0.8)
    assert last is not None
    assert not last.unstable
    assert last.confidence == 0.8


def test_unstable_warning_logged(caplog):
    f = _build(min_persistence_bars=1, flicker_window=8, flicker_threshold=3,
               unstable_confidence_decay=0.5)
    with caplog.at_level(logging.WARNING, logger="regime_trader.regime_stability"):
        for s in ["bull", "bear", "bull", "bear", "bull", "bear"]:
            f.update(s, 1.0)
    assert any("regime unstable" in rec.message for rec in caplog.records)
