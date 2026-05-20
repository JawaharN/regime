"""Main loop integration: kill-switch gating + a stubbed single iteration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.config import load_config


def test_startup_aborts_if_kill_switch_present(tmp_path: Path, monkeypatch):
    """Risk manager must refuse to start when the block file exists."""
    cfg = load_config()
    risk_cfg = cfg.risk.model_copy(update={
        "kill_switch_path": str(tmp_path / "kill_switch.block"),
        "peak_equity_path": str(tmp_path / "peak.json"),
    })
    # Place a kill switch
    Path(risk_cfg.kill_switch_path).write_text("{}")

    from core.risk_manager import RiskManager
    rm = RiskManager(risk_cfg)
    with pytest.raises(RuntimeError, match="Kill switch"):
        rm.assert_safe_to_start()


def test_single_iteration_smoke(tmp_path: Path, synthetic_ohlcv):
    """End-to-end one iteration with mocked broker. Verifies the pipeline runs
    and produces ALLOW/REDUCE/BLOCK actions without raising."""
    from broker.broker_adapter import AccountInfo, BrokerAdapter
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from data.feature_engineering import build_features, feature_spec_from_cfg
    from core.hmm_engine import HMMEngine
    from core.regime_stability import RegimeStabilityFilter
    from core.risk_manager import AccountSnapshot, RiskManager
    from core.regime_strategies import StrategyOrchestrator

    cfg = load_config()
    risk_cfg = cfg.risk.model_copy(update={
        "kill_switch_path": str(tmp_path / "kill"),
        "peak_equity_path": str(tmp_path / "peak"),
    })
    risk = RiskManager(risk_cfg)

    # Train HMM on synthetic OHLCV
    features = build_features(synthetic_ohlcv, cfg.features).dropna()
    engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, feature_spec_from_cfg(cfg.features))
    engine.fit(features)

    # Mocked broker
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"; fake_t212_cfg.api_key = "k"; fake_t212_cfg.secret_key = "s"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"
    fake_client = MagicMock()
    fake_client.account.summary.return_value = MagicMock(
        total=100000.0, free=100000.0, invested=0.0, currencyCode="USD",
    )
    fake_client.positions.list.return_value = []
    fake_client.orders.place_market.return_value = MagicMock(id="ORD-1")

    with patch("trade212_bot.config.load_config", return_value=fake_t212_cfg), \
         patch("trade212_bot.client.Trade212Client", return_value=fake_client):
        broker = BrokerAdapter(cfg.broker).connect()
    tracker = PositionTracker(broker)
    executor = OrderExecutor(broker, risk, tracker)

    # Establish portfolio baseline so risk allows
    risk.check_portfolio(AccountSnapshot(100000.0, 100000.0,
                                          datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)))
    tracker.refresh(equity_hint=100000.0)

    stab = RegimeStabilityFilter(
        cfg.stability.min_persistence_bars, cfg.stability.flicker_window,
        cfg.stability.flicker_threshold, cfg.stability.unstable_confidence_decay,
    )
    orch = StrategyOrchestrator(cfg.allocation, cfg.strategy)

    # Feed several inferences so stability filter has an actionable regime
    result = None
    for _ in range(5):
        inf = engine.infer_forward(features)
        s = stab.update(inf.label, inf.confidence)
        sigs = orch.evaluate("SYN", s.actionable_regime or "neutral", s.confidence,
                              synthetic_ohlcv["close"])
        for sig in sigs:
            result = executor.submit(
                sig,
                AccountInfo(equity=100000.0, cash=100000.0, invested=0.0,
                            currency="USD", timestamp=datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)),
                current_price=float(synthetic_ohlcv["close"].iloc[-1]),
            )
    # Should have produced *some* result string (placed or "no change" / similar)
    assert result is not None
    assert isinstance(result.reason, str)


def test_clean_shutdown_on_signal(monkeypatch, capsys):
    """Verify the run loop respects the iterations cap (proxy for SIGINT handling)."""
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "run", "--paper", "--iterations", "1", "--poll-seconds", "1",
    ])
    # Patch broker + risk + market_data to no-op so we exercise the loop control.
    with patch("main._run_one_iteration") as one_iter, \
         patch("broker.broker_adapter.BrokerAdapter.connect") as fake_connect, \
         patch("broker.broker_adapter.BrokerAdapter.close") as fake_close:
        fake_adapter = MagicMock()
        fake_connect.return_value = fake_adapter
        # Stub the HMM model load so we don't need a real file.
        with patch("core.hmm_engine.HMMEngine.load") as fake_load:
            fake_load.return_value = MagicMock()
            rc = args.func(args)
    assert rc == 0
    one_iter.assert_called_once()
