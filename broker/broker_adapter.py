"""Adapter from regime_trader's interface to the Trade212 API client.

The Trade212 wire client is `broker.trade212_api.Trade212Client` — vendored
in-repo, no external broker package.

Enforces demo-only operation: if TRADING212_ENV != "demo" and
broker.require_demo is True (the default), we refuse to connect.

Trade212 has no WebSocket trade-fill stream and no native OCO bracket orders;
those are emulated in `order_executor` / `position_tracker`. Every network call
goes through a 3-retry exponential-backoff wrapper.

Credentials load from env via `broker.trade212_api.load_config()`. They are
never logged — the logger redacts any configured secret value.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("regime_trader.broker")


@dataclass(frozen=True)
class AccountInfo:
    equity: float
    cash: float
    invested: float
    currency: str
    timestamp: datetime


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    quantity: float            # signed; positive = long
    average_price: float
    current_price: float
    unrealized_pnl: float
    weight: float              # |notional| / equity


class BrokerAdapter:
    """Lazy connection: instantiate, then call .connect() before use."""

    def __init__(self, broker_cfg, symbol_map: dict[str, str] | None = None) -> None:  # noqa: ANN001
        self.broker_cfg = broker_cfg
        self.symbol_map = symbol_map or {}
        self._t212_cfg = None
        self.client = None  # type: ignore[assignment]
        self._instrument_cache: list[object] | None = None
        self._market_order_timestamps: list[float] = []

    # ----------------------------------------------------- lifecycle

    def connect(self):
        from broker.trade212_api import Trade212Client
        from broker.trade212_api import load_config as load_t212_cfg

        self._t212_cfg = load_t212_cfg()
        if self.broker_cfg.require_demo and self._t212_cfg.env != "demo":
            raise RuntimeError(
                f"Refusing to connect: TRADING212_ENV={self._t212_cfg.env!r}, "
                "broker.require_demo=True. Paper trading only."
            )
        self.client = Trade212Client(
            base_url=self._t212_cfg.base_url,
            api_key=self._t212_cfg.api_key,
            secret_key=self._t212_cfg.secret_key,
        )
        logger.info("connected to Trade212 (env=%s)", self._t212_cfg.env)
        return self

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------- retry wrapper

    def _retry(self, fn, *args, **kwargs):
        """Call `fn` with exponential backoff on transient broker errors."""
        attempts = getattr(self.broker_cfg, "retry_attempts", 3)
        backoff = getattr(self.broker_cfg, "retry_backoff_seconds", 2.0)
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if i < attempts - 1:
                    delay = backoff * (2 ** i)
                    logger.warning("broker call failed (%s) — retry %d/%d in %.1fs",
                                   type(e).__name__, i + 1, attempts, delay)
                    time.sleep(delay)
        raise RuntimeError(f"broker call failed after {attempts} attempts") from last_exc

    def round_quantity(self, qty: float) -> float:
        """Truncate an order quantity to Trade212's accepted decimal precision.

        Trade212 rejects quantities with too many decimals ("invalid quantity
        precision"). We truncate toward zero — never round up — so a sized order
        can't drift above its intended notional and trip 'insufficient funds'.
        """
        precision = getattr(self.broker_cfg, "quantity_precision", 1)
        factor = 10 ** max(0, precision)
        return int(qty * factor) / factor

    def _metadata_instruments(self) -> list[object]:
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected")
        if self._instrument_cache is None:
            self._instrument_cache = self._retry(self.client.metadata.instruments)
        return self._instrument_cache

    def _to_t212_symbol(self, symbol: str) -> str:
        manual = self.symbol_map.get(symbol)
        if manual:
            logger.info("resolved %s via symbol_map override -> %s", symbol, manual)
            return manual

        symbol_upper = symbol.upper()
        exact_ticker_matches = []
        short_name_matches = []
        ticker_base_matches = []
        for inst in self._metadata_instruments():
            ticker = str(getattr(inst, "ticker", "") or "")
            short_name = str(getattr(inst, "shortName", "") or "")
            ticker_upper = ticker.upper()
            if ticker_upper == symbol_upper:
                exact_ticker_matches.append(ticker)
                continue
            if short_name.upper() == symbol_upper:
                short_name_matches.append(ticker)
                continue
            ticker_base = ticker.split("_", 1)[0].upper()
            if ticker_base == symbol_upper:
                ticker_base_matches.append(ticker)

        for matches in (exact_ticker_matches, short_name_matches, ticker_base_matches):
            unique = sorted(set(m for m in matches if m))
            if len(unique) == 1:
                logger.info("resolved %s via Trade212 metadata -> %s", symbol, unique[0])
                return unique[0]
            if len(unique) > 1:
                raise RuntimeError(
                    f"Ambiguous Trade212 instruments for {symbol}: {', '.join(unique)}. "
                    "Add universe.symbol_map override."
                )

        raise RuntimeError(
            f"No Trade212 instrument found for {symbol}. Add universe.symbol_map "
            "override or remove it from live trading."
        )

    # ----------------------------------------------------- account / positions

    def account(self) -> AccountInfo:
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected — call .connect() first")
        s = self._retry(self.client.account.summary)
        cash_block = _attr_or_key(s, "cash") or {}
        inv_block = _attr_or_key(s, "investments") or {}
        equity = _first_float(_attr_or_key(s, "totalValue"), _attr_or_key(s, "total"))
        cash = _first_float(_attr_or_key(cash_block, "availableToTrade"), _attr_or_key(s, "free"))
        invested = _first_float(_attr_or_key(inv_block, "currentValue"), _attr_or_key(s, "invested"))
        currency = (_attr_or_key(s, "currency") or _attr_or_key(s, "currencyCode") or "USD")
        return AccountInfo(
            equity=equity, cash=cash, invested=invested, currency=str(currency),
            timestamp=datetime.now(timezone.utc),
        )

    def positions(self, equity_hint: float | None = None) -> list[PositionInfo]:
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected")
        raw = self._retry(self.client.positions.list)
        equity = equity_hint if equity_hint is not None else self.account().equity
        out: list[PositionInfo] = []
        for p in raw:
            qty = float(getattr(p, "quantity", 0.0) or 0.0)
            curr = float(getattr(p, "currentPrice", 0.0) or 0.0)
            notional = abs(qty * curr)
            out.append(PositionInfo(
                symbol=str(getattr(p, "ticker", "")),
                quantity=qty,
                average_price=float(getattr(p, "averagePrice", 0.0) or 0.0),
                current_price=curr,
                unrealized_pnl=float(getattr(p, "ppl", 0.0) or 0.0),
                weight=(notional / equity) if equity > 0 else 0.0,
            ))
        return out

    def get_account(self) -> AccountInfo:
        """Alias matching the final-prompt interface."""
        return self.account()

    def get_positions(self, equity_hint: float | None = None) -> list[PositionInfo]:
        return self.positions(equity_hint=equity_hint)

    def get_order_history(self):
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected")
        try:
            return self._retry(self.client.orders.history)
        except Exception:  # noqa: BLE001 — endpoint optional across T212 accounts
            return []

    def get_available_margin(self) -> float:
        return self.account().cash

    def is_market_open(self, now: datetime | None = None) -> bool:
        """US-equity heuristic: Mon-Fri, 14:30–21:00 UTC.

        Trade212 has no public market-clock endpoint; this keeps the daily loop
        from acting on stale weekend/overnight data.
        """
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if now.weekday() >= 5:
            return False
        minutes = now.hour * 60 + now.minute
        return 14 * 60 + 30 <= minutes <= 21 * 60

    def get_clock(self) -> dict:
        return {"timestamp": datetime.now(timezone.utc).isoformat(),
                "is_open": self.is_market_open()}

    # ----------------------------------------------------- orders

    def _throttle_market_orders(self) -> None:
        limit = 50
        window_seconds = 60.0
        now = time.time()
        self._market_order_timestamps = [
            ts for ts in self._market_order_timestamps
            if now - ts < window_seconds
        ]
        if len(self._market_order_timestamps) >= limit:
            wait = max(0.0, window_seconds - (now - self._market_order_timestamps[0]))
            if wait > 0:
                logger.warning("market-order throttle hit (%d/%ss) — waiting %.1fs",
                               limit, int(window_seconds), wait)
                time.sleep(wait)
                now = time.time()
                self._market_order_timestamps = [
                    ts for ts in self._market_order_timestamps
                    if now - ts < window_seconds
                ]
        self._market_order_timestamps.append(time.time())

    def place_market(self, symbol: str, signed_qty: float):
        """Place a market order. signed_qty: positive = BUY, negative = SELL."""
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected")
        from broker.trade212_api import MarketOrderRequest
        t212_symbol = self._to_t212_symbol(symbol)
        qty = self.round_quantity(signed_qty)
        extended_hours = bool(getattr(self.broker_cfg, "market_extended_hours", True))
        self._throttle_market_orders()
        req = MarketOrderRequest(
            ticker=t212_symbol,
            quantity=qty,
            extendedHours=extended_hours,
        )
        order = self._retry(self.client.orders.place_market, req)
        logger.info("placed market: symbol=%s qty=%s extended_hours=%s order_id=%s",
                    t212_symbol, qty, extended_hours, getattr(order, "id", None))
        return order

    def place_limit(self, symbol: str, signed_qty: float, limit_price: float):
        """Place a limit order if the client supports it; else fall back to market."""
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected")
        t212_symbol = self._to_t212_symbol(symbol)
        try:
            from broker.trade212_api import LimitOrderRequest
            req = LimitOrderRequest(ticker=t212_symbol,
                                    quantity=self.round_quantity(signed_qty),
                                    limitPrice=limit_price)
            order = self._retry(self.client.orders.place_limit, req)
            logger.info("placed limit: symbol=%s qty=%s @ %.4f order_id=%s",
                        t212_symbol, signed_qty, limit_price, getattr(order, "id", None))
            return order
        except Exception:  # noqa: BLE001 — limit unsupported → market fallback
            logger.warning("limit order unsupported for %s — falling back to market", t212_symbol)
            return self.place_market(symbol, signed_qty)

    def cancel(self, order_id) -> None:  # noqa: ANN001
        if self.client is None:
            raise RuntimeError("BrokerAdapter not connected")
        self._retry(self.client.orders.cancel, order_id)
        logger.info("cancelled order_id=%s", order_id)


# Final-prompt name; same Trade212-backed adapter.
Trade212Client = BrokerAdapter


def _attr_or_key(obj, name):  # noqa: ANN001
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _first_float(*candidates) -> float:
    for c in candidates:
        if c is not None:
            try:
                return float(c)
            except (TypeError, ValueError):
                continue
    return 0.0
