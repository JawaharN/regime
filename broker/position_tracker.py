"""Position tracker — caches open positions, reconciles, computes correlation.

Trade212 has no WebSocket trade-fill stream, so fills are detected by REST
polling. ``poll()`` diffs the latest broker positions against the cache and
fires the registered on-fill callbacks — the same contract a WebSocket version
would expose, so consumers don't care which transport is underneath.

Correlation uses a real rolling window of daily returns read from the parquet
cache under ``data/cache/``; if a symbol's history is missing it degrades to a
conservative "same symbol → 1.0, else 0.0" proxy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from broker.broker_adapter import BrokerAdapter, PositionInfo

logger = logging.getLogger("regime_trader.positions")


class PositionTracker:
    def __init__(self, broker: BrokerAdapter, correlation_window: int = 60) -> None:
        self.broker = broker
        self.correlation_window = correlation_window
        self._cache: list[PositionInfo] = []
        self._fill_callbacks: list[Callable[[PositionInfo, str], None]] = []

    # ------------------------------------------------------ polling / cache

    def refresh(self, equity_hint: float | None = None) -> list[PositionInfo]:
        self._cache = self.broker.positions(equity_hint=equity_hint)
        return self._cache

    def poll(self, equity_hint: float | None = None) -> list[PositionInfo]:
        """Refresh and fire on-fill callbacks for any detected position change."""
        before = {p.symbol: p.quantity for p in self._cache}
        new = self.broker.positions(equity_hint=equity_hint)
        for p in new:
            prev = before.get(p.symbol, 0.0)
            if abs(p.quantity - prev) > 1e-9:
                event = "opened" if prev == 0 else ("closed" if p.quantity == 0 else "changed")
                self._fire_fill(p, event)
        for sym, qty in before.items():
            if qty != 0 and sym not in {p.symbol for p in new}:
                self._fire_fill(PositionInfo(sym, 0.0, 0.0, 0.0, 0.0, 0.0), "closed")
        self._cache = new
        return self._cache

    def register_fill_callback(self, cb: Callable[[PositionInfo, str], None]) -> None:
        self._fill_callbacks.append(cb)

    def _fire_fill(self, pos: PositionInfo, event: str) -> None:
        for cb in self._fill_callbacks:
            try:
                cb(pos, event)
            except Exception:  # noqa: BLE001
                logger.exception("fill callback failed for %s", pos.symbol)

    def reconcile(self) -> list[PositionInfo]:
        """Sync the cache with the broker at startup."""
        self._cache = self.broker.positions()
        logger.info("reconciled %d open positions from broker", len(self._cache))
        return self._cache

    def current(self) -> list[PositionInfo]:
        return list(self._cache)

    def has(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self._cache)

    # ------------------------------------------------------ correlation

    def correlation_with_existing(self, symbol: str) -> float:
        """Worst-case correlation of `symbol` against held positions.

        Real rolling correlation when cached daily history exists; otherwise the
        conservative same-symbol proxy.
        """
        corrs = self.correlations(symbol)
        if corrs:
            return max(corrs.values(), default=0.0)
        return 1.0 if self.has(symbol) else 0.0

    def correlations(self, symbol: str) -> dict[str, float]:
        """Per-position rolling-return correlation against `symbol`."""
        held = [p.symbol for p in self._cache if p.symbol != symbol]
        if not held:
            return {}
        base = self._returns(symbol)
        if base is None:
            return {h: (1.0 if h == symbol else 0.0) for h in held}
        out: dict[str, float] = {}
        for h in held:
            other = self._returns(h)
            if other is None:
                continue
            joined = pd.concat([base, other], axis=1).dropna()
            if len(joined) >= 5:
                out[h] = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        return out

    def _returns(self, symbol: str) -> pd.Series | None:
        path = Path("data/cache") / f"{symbol}_1d.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            return df["close"].astype(float).pct_change().tail(self.correlation_window).dropna()
        except Exception:  # noqa: BLE001
            return None
