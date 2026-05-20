"""Phase 9: logger redaction, alerts, performance tracker."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import load_config
from monitoring.alerts import AlertBus, FileAlertSink, StdoutAlertSink
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
    # Plain render check — the Rich Live loop is not launched here.
    from monitoring import dashboard
    assert "REGIME" in dashboard.render_text({"regime": {"label": "bull"}})
