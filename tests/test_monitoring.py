"""Phase 9: logger redaction, alerts, performance tracker."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pytest

from core.config import load_config
from core.risk_manager import AccountSnapshot, RiskManager
from monitoring.alerts import AlertBus, FileAlertSink
from monitoring.logger import RedactFilter, build_logger
from monitoring.performance import PerformanceTracker


def test_redact_filter_strips_secret(monkeypatch):
    monkeypatch.setenv("TRADING212_API_KEY", "SECRET_VAL_12345")
    f = RedactFilter(["TRADING212_API_KEY"])
    rec = logging.LogRecord(
        name="x", level=logging.INFO, pathname="", lineno=0,
        msg="connecting with key=SECRET_VAL_12345 to demo", args=(), exc_info=None,
    )
    f.filter(rec)
    assert "SECRET_VAL_12345" not in rec.getMessage()
    assert "***REDACTED***" in rec.getMessage()


def test_build_logger_idempotent():
    cfg = load_config().logging
    a = build_logger(cfg)
    b = build_logger(cfg)
    # Both calls return loggers, and the second call does not duplicate handlers.
    root_handlers = [h for h in logging.getLogger().handlers
                     if getattr(h, "tag", None) == "regime_trader_main_handler"]
    assert len(root_handlers) == 1
    assert a is b or a.name == b.name


def test_file_alert_sink_writes_jsonl(tmp_path: Path):
    sink = FileAlertSink(tmp_path / "alerts.jsonl")
    sink.emit("WARN", "kill_switch_armed", {"drawdown": -0.12})
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["event"] == "kill_switch_armed"
    assert obj["level"] == "WARN"
    assert "ts" in obj


def test_alert_bus_dispatches_to_all_sinks(tmp_path: Path):
    f = FileAlertSink(tmp_path / "a.jsonl")
    bus = AlertBus([f])
    bus.emit("INFO", "iteration", count=3)
    assert (tmp_path / "a.jsonl").exists()


def test_performance_tracker_round_trip(tmp_path: Path):
    pt = PerformanceTracker(tmp_path / "perf.parquet")
    pt.record({"equity": 100000.0, "cash": 50000.0, "regime": "bull", "confidence": 0.85})
    pt.record({"equity": 101000.0, "cash": 50000.0, "regime": "bull", "confidence": 0.90})
    df = pt.load()
    assert len(df) == 2
    assert set(df.columns) >= {"ts", "equity", "regime", "confidence"}
    assert df["equity"].iloc[-1] == 101000.0


def test_dashboard_module_imports():
    # Plain render check — the HTTP server is not launched here.
    from monitoring import dashboard
    assert "REGIME" in dashboard.render_text({"regime": {"label": "bull"}})


def test_dashboard_render_html_contains_sections():
    from monitoring.dashboard import render_html

    html_doc = render_html({
        "regime": {"label": "BULL", "probability": 0.72, "consecutive_bars": 14, "flicker_count": 1, "vol_rank": 0.2},
        "portfolio": {"equity": 105230, "daily_pnl": 0.0032, "allocation": 0.95, "leverage": 1.25, "cash": 5000, "buying_power": 12000},
        "positions": [{"symbol": "SPY", "side": "LONG", "current": 520.30, "unrealized_pnl_pct": 0.012, "stop": 508, "holding_bars": 3}],
        "recent_signals": [{"time": "14:30", "symbol": "SPY", "action": "Rebalance 60%→95%", "reason": "Low vol"}],
        "risk_status": {
            "daily_drawdown": {"value": "0.3%", "limit": "3%", "ok": True},
            "from_peak": {"value": "1.2%", "limit": "10%", "ok": True},
        },
        "system": {"data": "✅", "api": "✅ 23ms", "hmm": "2d ago", "mode": "PAPER"},
    }, refresh_seconds=5)

    assert "REGIME" in html_doc
    assert "PORTFOLIO" in html_doc
    assert "POSITIONS" in html_doc
    assert "RECENT SIGNALS" in html_doc
    assert "RISK STATUS" in html_doc
    assert "SYSTEM" in html_doc
    assert "BULL" in html_doc
    assert "SPY" in html_doc
    assert "Auto-refresh every 5 seconds." in html_doc



def test_dashboard_render_html_waiting_message():
    from monitoring.dashboard import WAITING_MESSAGE, render_html

    html_doc = render_html({}, refresh_seconds=5)
    assert WAITING_MESSAGE in html_doc



def test_dashboard_http_handler_serves_html_and_json(tmp_path: Path):
    from monitoring.dashboard import _make_handler
    from http.server import ThreadingHTTPServer
    from threading import Thread

    state_path = tmp_path / "dashboard.json"
    state = {"regime": {"label": "BULL"}, "portfolio": {"equity": 100000}, "positions": []}
    state_path.write_text(json.dumps(state))

    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state_path, refresh_seconds=5))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base_url + "/") as response:
            html_body = response.read().decode()
            assert response.status == 200
            assert "text/html" in response.headers["Content-Type"]
            assert "BULL" in html_body

        with urlopen(base_url + "/api/state") as response:
            payload = json.loads(response.read().decode())
            assert response.status == 200
            assert payload["regime"]["label"] == "BULL"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)



def test_risk_manager_dashboard_status(tmp_path: Path):
    cfg = load_config()
    risk_cfg = cfg.risk.model_copy(update={
        "kill_switch_path": str(tmp_path / "kill_switch.block"),
        "peak_equity_path": str(tmp_path / "peak_equity.json"),
    })
    risk = RiskManager(risk_cfg)
    baseline = AccountSnapshot(100000.0, 100000.0, datetime.fromisoformat("2026-05-20T09:30:00+00:00"))
    risk.check_portfolio(baseline)

    status = risk.dashboard_status(AccountSnapshot(98500.0, 98500.0, datetime.fromisoformat("2026-05-20T15:30:00+00:00")))
    assert status["daily_pnl"] == pytest.approx(-0.015)
    assert status["daily_drawdown"]["value"] == "1.5%"
    assert status["daily_drawdown"]["ok"] is True
    assert status["from_peak"]["value"] == "1.5%"
    assert status["kill_switch"] is False



def test_dashboard_state_builder_serializes_runtime_data(tmp_path: Path):
    from broker.broker_adapter import AccountInfo, PositionInfo
    from core.hmm_engine import RegimeState
    from core.regime_stability import StabilityResult
    from main import _build_dashboard_state, _write_dashboard_state

    cfg = load_config()
    cfg = cfg.model_copy(update={
        "monitoring": cfg.monitoring.model_copy(update={
            "state_dashboard_path": str(tmp_path / "dashboard.json"),
        }),
        "risk": cfg.risk.model_copy(update={
            "kill_switch_path": str(tmp_path / "kill_switch.block"),
            "peak_equity_path": str(tmp_path / "peak_equity.json"),
        }),
    })

    risk = RiskManager(cfg.risk)
    account = AccountInfo(
        equity=105230.0,
        cash=5000.0,
        invested=100230.0,
        currency="USD",
        timestamp=datetime.fromisoformat("2026-05-20T14:30:00+00:00"),
    )
    snap = AccountSnapshot(account.equity, account.cash, account.timestamp)
    risk.check_portfolio(AccountSnapshot(105000.0, 5000.0, datetime.fromisoformat("2026-05-20T09:30:00+00:00")))
    risk.check_portfolio(snap)

    tracker = type("Tracker", (), {
        "current": lambda self: [PositionInfo("SPY", 10.0, 520.0, 526.0, 60.0, 0.95)]
    })()
    executor = type("Executor", (), {
        "risk": risk,
        "brackets": {},
    })()
    generator = type("Generator", (), {
        "last_state": RegimeState(label="BULL", state_id=1, probability=0.72, timestamp=account.timestamp, is_confirmed=True, consecutive_bars=14),
        "last_stability": StabilityResult(actionable_regime="BULL", confidence=0.72, unstable=False, inferred_regime="BULL", persistence_count=14, flicker_count=1),
    })()

    payload = _build_dashboard_state(
        cfg,
        account=account,
        tracker=tracker,
        executor=executor,
        generators={"SPY": generator},
        account_snap=snap,
        recent_signals=[{"time": "14:30", "symbol": "SPY", "action": "BUY", "reason": "Low vol"}],
        data_ok=True,
        hmm_ok=True,
        dry_run=True,
    )
    assert payload["regime"]["label"] == "BULL"
    assert payload["regime"]["flicker_count"] == 1
    assert payload["portfolio"]["equity"] == 105230.0
    assert payload["positions"][0]["symbol"] == "SPY"
    assert payload["positions"][0]["side"] == "LONG"
    assert payload["recent_signals"][0]["symbol"] == "SPY"
    assert payload["system"]["mode"] == "DRY_RUN"

    _write_dashboard_state(
        cfg,
        account=account,
        tracker=tracker,
        executor=executor,
        generators={"SPY": generator},
        account_snap=snap,
        recent_signals=[{"time": "14:30", "symbol": "SPY", "action": "BUY", "reason": "Low vol"}],
        data_ok=True,
        hmm_ok=True,
        dry_run=False,
    )
    written = json.loads((tmp_path / "dashboard.json").read_text())
    assert written["regime"]["label"] == "BULL"
    assert written["portfolio"]["buying_power"] == 5000.0
