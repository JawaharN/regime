"""Trade212 REST API client — self-contained, in-repo.

The whole Trade212 integration lives in the broker subsystem. This module is
the wire layer: credential loading, an httpx transport with Basic auth and
rate-limit handling, typed request/response models, and a ``Trade212Client``
facade exposing the three endpoint groups the bot uses (account, positions,
orders). `broker_adapter.BrokerAdapter` is the only consumer; it adds its own
retry/backoff and demo-mode enforcement on top.

Trade212 docs: https://docs.trading212.com/api  (equity API, /api/v0).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("regime_trader.broker.t212api")

Env = Literal["demo", "live"]

_BASE_URLS: dict[str, str] = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}


# --------------------------------------------------------------- config

@dataclass(frozen=True)
class Trade212Config:
    api_key: str
    secret_key: str
    env: Env
    base_url: str


def load_config() -> Trade212Config:
    """Load TRADING212_* credentials from the environment (.env supported).

    Base URL follows the env unless TRADING212_BASE_URL overrides it.
    """
    load_dotenv()

    api_key = os.environ.get("TRADING212_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TRADING212_API_KEY is not set (see .env.example).")

    secret_key = os.environ.get("TRADING212_SECRET_KEY", "").strip()
    if not secret_key:
        raise RuntimeError("TRADING212_SECRET_KEY is not set (see .env.example).")

    env = os.environ.get("TRADING212_ENV", "demo").strip().lower()
    if env not in _BASE_URLS:
        raise RuntimeError(f"TRADING212_ENV must be 'demo' or 'live', got {env!r}.")

    base_url = os.environ.get("TRADING212_BASE_URL", "").strip() or _BASE_URLS[env]
    return Trade212Config(api_key=api_key, secret_key=secret_key,
                          env=env, base_url=base_url)  # type: ignore[arg-type]


# --------------------------------------------------------------- errors

class Trade212Error(RuntimeError):
    """Any non-success response from the Trade212 API."""

    def __init__(self, message: str, *, status_code: int | None = None, body: object = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


# --------------------------------------------------------------- models

class AccountSummary(BaseModel):
    """`/equity/account/summary`. Permissive: T212 has shipped several shapes
    over time (flat `total`/`free` vs nested `totalValue`/`cash`), so unknown
    keys are kept and the adapter reads whichever set is present."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    currencyCode: str | None = None
    total: float | None = None
    free: float | None = None
    invested: float | None = None
    ppl: float | None = None
    result: float | None = None
    blocked: float | None = None


class Position(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str | None = None
    quantity: float
    averagePrice: float | None = None
    currentPrice: float | None = None
    ppl: float | None = None


class Order(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int | str | None = None
    ticker: str | None = None
    quantity: float | None = None
    filledQuantity: float | None = None
    limitPrice: float | None = None
    status: str | None = None
    type: str | None = None


class ExchangeMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    name: str
    workingSchedules: list[dict[str, Any]] = []


class InstrumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    name: str | None = None
    shortName: str | None = None
    isin: str | None = None
    type: str | None = None
    currencyCode: str | None = None
    extendedHours: bool | None = None
    maxOpenQuantity: float | None = None
    addedOn: str | None = None
    workingScheduleId: int | None = None


class MarketOrderRequest(BaseModel):
    """Signed quantity: positive = BUY, negative = SELL (Trade212 convention)."""

    model_config = ConfigDict(extra="forbid")
    ticker: str
    quantity: float
    extendedHours: bool = False


class LimitOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    quantity: float
    limitPrice: float
    timeValidity: str = "DAY"


# --------------------------------------------------------------- transport

class _Http:
    """httpx transport: Basic auth, JSON parsing, one 429 retry."""

    def __init__(self, base_url: str, api_key: str, secret_key: str,
                 timeout: float = 15.0) -> None:
        token = base64.b64encode(f"{api_key}:{secret_key}".encode()).decode()
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Basic {token}",
                     "Accept": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, path: str, *,
                json: dict[str, Any] | None = None) -> Any:
        for attempt in range(2):
            resp = self._client.request(method, path, json=json)
            if resp.status_code == 429 and attempt == 0:
                wait = self._retry_after(resp)
                logger.warning("Trade212 rate-limited on %s %s — waiting %.1fs",
                               method, path, wait)
                time.sleep(wait)
                continue
            return self._parse(resp, method, path)
        return None  # unreachable

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        raw = resp.headers.get("Retry-After")
        try:
            return max(0.5, float(raw)) if raw else 1.0
        except ValueError:
            return 1.0

    @staticmethod
    def _parse(resp: httpx.Response, method: str, path: str) -> Any:
        if resp.is_success:
            if not resp.content:
                return None
            try:
                return resp.json()
            except ValueError:
                return resp.text
        try:
            body: object = resp.json()
        except ValueError:
            body = resp.text
        # Surface the API's own detail — Trade212 returns useful messages
        # (insufficient funds, ticker not found, short-selling blocked, ...).
        detail = ""
        if isinstance(body, dict):
            detail = str(body.get("detail") or body.get("errorMessage")
                         or body.get("context") or body.get("title") or "")
        elif body:
            detail = str(body)[:200]
        suffix = f": {detail}" if detail else ""
        raise Trade212Error(f"{method} {path} -> HTTP {resp.status_code}{suffix}",
                            status_code=resp.status_code, body=body)


# --------------------------------------------------------------- endpoints

class _AccountEndpoint:
    def __init__(self, http: _Http) -> None:
        self._http = http

    def summary(self) -> AccountSummary:
        return AccountSummary.model_validate(
            self._http.request("GET", "/equity/account/summary"))


class _PositionsEndpoint:
    def __init__(self, http: _Http) -> None:
        self._http = http

    def list(self) -> list[Position]:
        data = self._http.request("GET", "/equity/positions") or []
        return [Position.model_validate(d) for d in data]


class _OrdersEndpoint:
    def __init__(self, http: _Http) -> None:
        self._http = http

    def list(self) -> list[Order]:
        data = self._http.request("GET", "/equity/orders") or []
        return [Order.model_validate(d) for d in data]

    def place_market(self, request: MarketOrderRequest) -> Order:
        return Order.model_validate(
            self._http.request("POST", "/equity/orders/market",
                               json=request.model_dump(exclude_none=True)))

    def place_limit(self, request: LimitOrderRequest) -> Order:
        return Order.model_validate(
            self._http.request("POST", "/equity/orders/limit",
                               json=request.model_dump(exclude_none=True)))

    def cancel(self, order_id: int | str) -> None:
        self._http.request("DELETE", f"/equity/orders/{order_id}")


class _MetadataEndpoint:
    def __init__(self, http: _Http) -> None:
        self._http = http

    def exchanges(self) -> list[ExchangeMetadata]:
        data = self._http.request("GET", "/equity/metadata/exchanges") or []
        return [ExchangeMetadata.model_validate(d) for d in data]

    def instruments(self) -> list[InstrumentMetadata]:
        data = self._http.request("GET", "/equity/metadata/instruments") or []
        return [InstrumentMetadata.model_validate(d) for d in data]


# --------------------------------------------------------------- facade

class Trade212Client:
    """Composes the endpoint groups over one authenticated transport."""

    def __init__(self, *, base_url: str, api_key: str, secret_key: str) -> None:
        self._http = _Http(base_url, api_key, secret_key)
        self.account = _AccountEndpoint(self._http)
        self.positions = _PositionsEndpoint(self._http)
        self.orders = _OrdersEndpoint(self._http)
        self.metadata = _MetadataEndpoint(self._http)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Trade212Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
