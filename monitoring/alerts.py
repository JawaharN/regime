"""Critical-event alerts with pluggable sinks (stdout, file).

Hooked from risk circuit breakers, HMM failures, broker failures and the
kill-switch trigger. ``AlertBus`` can rate-limit per (level, event) key so a
flapping condition does not spam every sink.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("regime_trader.alerts")


class AlertSink(Protocol):
    def emit(self, level: str, event: str, payload: dict) -> None: ...


class StdoutAlertSink:
    def emit(self, level: str, event: str, payload: dict) -> None:
        print(f"[ALERT/{level}] {event} {json.dumps(payload, default=str)}")


class FileAlertSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, level: str, event: str, payload: dict) -> None:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level, "event": event, **payload,
        }, default=str)
        with self.path.open("a") as f:
            f.write(line + "\n")


class WebhookAlertSink:
    """Config-gated webhook sink (off by default). No-op unless a URL is set."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url

    def emit(self, level: str, event: str, payload: dict) -> None:
        if not self.url:
            return
        logger.info("webhook alert suppressed (delivery disabled): %s", event)


class AlertBus:
    def __init__(self, sinks: list[AlertSink], rate_limit_minutes: int = 0) -> None:
        self.sinks = sinks
        self.rate_limit_minutes = rate_limit_minutes
        self._last_emitted: dict[str, datetime] = {}

    def emit(self, level: str, event: str, **payload) -> None:
        if self._rate_limited(level, event):
            return
        for sink in self.sinks:
            try:
                sink.emit(level, event, payload)
            except Exception:  # noqa: BLE001
                logger.exception("alert sink failed")

    def _rate_limited(self, level: str, event: str) -> bool:
        if self.rate_limit_minutes <= 0:
            return False
        key = f"{level}:{event}"
        now = datetime.now(timezone.utc)
        last = self._last_emitted.get(key)
        if last and (now - last).total_seconds() < self.rate_limit_minutes * 60:
            return True
        self._last_emitted[key] = now
        return False


def build_default_bus(file_path: str | Path = "logs/alerts.jsonl",
                      rate_limit_minutes: int = 0) -> AlertBus:
    return AlertBus([StdoutAlertSink(), FileAlertSink(file_path)],
                    rate_limit_minutes=rate_limit_minutes)
