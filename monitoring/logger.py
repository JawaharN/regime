"""Structured JSON logging with credential redaction.

`build_logger` keeps the legacy single stdout handler (idempotent). The
final-prompt setup, `setup_rotating_logs`, adds four rotating JSON files —
``main.log`` / ``trades.log`` / ``alerts.log`` / ``regime.log`` — each 10 MB
with 30 backups. Any configured secret value appearing in a message is replaced
with ``***REDACTED***``.
"""

from __future__ import annotations

import json
import logging
import os
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RedactFilter(logging.Filter):
    """Replaces any occurrence of redacted env var values inside messages."""

    def __init__(self, env_keys: list[str]) -> None:
        super().__init__()
        self._secrets = [v for k in env_keys if (v := os.environ.get(k))]

    def filter(self, record: LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        for s in self._secrets:
            if s and s in msg:
                record.msg = msg.replace(s, "***REDACTED***")
                record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("regime", "regime_probability", "equity", "positions", "daily_pnl"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def build_logger(cfg) -> logging.Logger:  # noqa: ANN001 — cfg = LoggingCfg
    """Legacy single-handler stdout logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))

    handler_tag = "regime_trader_main_handler"
    if any(getattr(h, "tag", None) == handler_tag for h in root.handlers):
        return logging.getLogger("regime_trader")

    handler = logging.StreamHandler()
    handler.tag = handler_tag  # type: ignore[attr-defined]
    if cfg.json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactFilter(cfg.redact_env_keys))
    root.addHandler(handler)
    return logging.getLogger("regime_trader")


def setup_rotating_logs(monitoring_cfg, redact_env_keys: list[str]) -> dict[str, logging.Logger]:  # noqa: ANN001
    """Attach rotating JSON file handlers for the four log channels.

    Returns a dict {channel: Logger}. Each channel is a child logger of
    ``regime_trader`` so records also bubble to the stdout handler.
    """
    log_dir = Path(monitoring_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    redact = RedactFilter(redact_env_keys)
    fmt = JsonFormatter()
    channels = {"main": "main.log", "trades": "trades.log",
                "alerts": "alerts.log", "regime": "regime.log"}
    loggers: dict[str, logging.Logger] = {}
    for channel, filename in channels.items():
        lg = logging.getLogger(f"regime_trader.{channel}")
        tag = f"regime_trader_rotating_{channel}"
        if not any(getattr(h, "tag", None) == tag for h in lg.handlers):
            handler = RotatingFileHandler(
                log_dir / filename,
                maxBytes=monitoring_cfg.log_max_bytes,
                backupCount=monitoring_cfg.log_backup_count,
            )
            handler.tag = tag  # type: ignore[attr-defined]
            handler.setFormatter(fmt)
            handler.addFilter(redact)
            lg.addHandler(handler)
        loggers[channel] = lg
    return loggers
