"""Broker adapter: mocked Trade212Client; signed quantity; demo-mode enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from broker.broker_adapter import AccountInfo, BrokerAdapter, PositionInfo
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.config import load_config
from core.regime_strategies import Signal
from core.risk_manager import RiskManager

# ---------- demo-mode enforcement ----------

def test_connect_refuses_live_env_when_require_demo():
    cfg = load_config().broker
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "live"
    fake_t212_cfg.base_url = "https://live.trading212.com/api/v0"
    fake_t212_cfg.api_key = "k"
    fake_t212_cfg.secret_key = "s"

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client") as fake_client:
        adapter = BrokerAdapter(cfg)
        with pytest.raises(RuntimeError, match="Refusing to connect"):
            adapter.connect()
        fake_client.assert_not_called()


def test_connect_succeeds_in_demo_env():
    cfg = load_config().broker
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"
    fake_t212_cfg.api_key = "k"
    fake_t212_cfg.secret_key = "s"

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client") as fake_client:
        adapter = BrokerAdapter(cfg)
        adapter.connect()
        fake_client.assert_called_once()


# ---------- credentials never logged ----------

def test_credentials_not_in_logs(caplog):
    import logging
    cfg = load_config().broker
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"
    fake_t212_cfg.api_key = "VERY_SECRET_KEY_VALUE"
    fake_t212_cfg.secret_key = "EVEN_MORE_SECRET"

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client"):
        adapter = BrokerAdapter(cfg)
        with caplog.at_level(logging.DEBUG, logger="regime_trader.broker"):
            adapter.connect()
        for rec in caplog.records:
            assert "VERY_SECRET_KEY_VALUE" not in rec.getMessage()
            assert "EVEN_MORE_SECRET" not in rec.getMessage()


# ---------- account / positions parsing ----------

def test_account_parses_legacy_schema():
    """Older Trade212 accounts return total / free / invested / currencyCode."""
    from broker.trade212_api import AccountSummary
    cfg = load_config().broker
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"
    fake_t212_cfg.api_key = "k"
    fake_t212_cfg.secret_key = "s"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"

    fake_client = MagicMock()
    fake_client.account.summary.return_value = AccountSummary(
        total=10000.0, free=8000.0, invested=2000.0, currencyCode="USD",
    )

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client", return_value=fake_client):
        adapter = BrokerAdapter(cfg).connect()
        info = adapter.account()
    assert info.equity == 10000.0
    assert info.cash == 8000.0
    assert info.currency == "USD"


def test_account_parses_new_schema():
    """Newer Trade212 accounts return totalValue / cash.availableToTrade / currency."""
    from broker.trade212_api import AccountSummary
    cfg = load_config().broker
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"
    fake_t212_cfg.api_key = "k"
    fake_t212_cfg.secret_key = "s"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"

    fake_client = MagicMock()
    # AccountSummary has extra="allow" so unknown fields land in extras.
    fake_client.account.summary.return_value = AccountSummary.model_validate({
        "currencyCode": None, "total": None, "free": None, "invested": None,
        "currency": "EUR",
        "totalValue": 1000.0,
        "cash": {"availableToTrade": 1000.0, "reservedForOrders": 0, "inPies": 0},
        "investments": {"currentValue": 0, "totalCost": 0,
                         "realizedProfitLoss": 0, "unrealizedProfitLoss": 0},
    })

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client", return_value=fake_client):
        adapter = BrokerAdapter(cfg).connect()
        info = adapter.account()
    assert info.equity == 1000.0
    assert info.cash == 1000.0
    assert info.invested == 0.0
    assert info.currency == "EUR"


# ---------- signed quantity from BUY signal ----------

@dataclass
class _Pos:
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    weight: float


def _make_adapter_and_executor(positions: list[_Pos], equity: float = 100_000.0,
                                cash: float = 100_000.0,
                                kill_path="/tmp/kill.block", peak_path="/tmp/peak.json",
                                tmp_path=None):
    cfg = load_config()
    risk_cfg = cfg.risk.model_copy(update={
        "kill_switch_path": str(tmp_path / "kill.block") if tmp_path else kill_path,
        "peak_equity_path": str(tmp_path / "peak.json") if tmp_path else peak_path,
    })
    risk = RiskManager(risk_cfg)

    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"
    fake_t212_cfg.api_key = "k"
    fake_t212_cfg.secret_key = "s"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"

    fake_client = MagicMock()
    fake_client.account.summary.return_value = MagicMock(
        total=equity, free=cash, invested=equity - cash, currencyCode="USD",
    )
    fake_client.positions.list.return_value = []  # we manually populate tracker cache
    fake_client.orders.place_market.return_value = MagicMock(id="ORD-1")

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client", return_value=fake_client):
        adapter = BrokerAdapter(cfg.broker).connect()
    tracker = PositionTracker(adapter)
    tracker._cache = [PositionInfo(**p.__dict__) for p in positions]
    executor = OrderExecutor(adapter, risk, tracker)
    return adapter, executor, fake_client, risk


def test_buy_signal_becomes_positive_quantity(tmp_path):
    adapter, executor, fake_client, risk = _make_adapter_and_executor([], tmp_path=tmp_path)
    # Establish a portfolio baseline so risk allows
    from core.risk_manager import AccountSnapshot
    risk.check_portfolio(AccountSnapshot(100_000.0, 100_000.0,
                                          datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)))

    sig = Signal(symbol="SPY", side="BUY", target_weight=0.5,
                 confidence=0.9, regime="bull", reason="test")
    acct = AccountInfo(equity=100_000.0, cash=100_000.0, invested=0.0,
                       currency="USD", timestamp=datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc))

    result = executor.submit(sig, acct, current_price=500.0)
    assert result.placed
    call = fake_client.orders.place_market.call_args
    req = call.args[0] if call.args else call.kwargs["request"]
    # quantity should be positive (BUY) and roughly 100k * 0.5 / 500 = 100 shares
    assert req.quantity > 0
    assert abs(req.quantity - 100.0) < 1e-6


def test_sell_signal_becomes_negative_quantity(tmp_path):
    # We already own 100 shares (long); strategy wants weight 0 → must sell 100
    existing = _Pos(symbol="SPY", quantity=100.0, average_price=400.0,
                    current_price=500.0, unrealized_pnl=0.0, weight=0.5)
    adapter, executor, fake_client, risk = _make_adapter_and_executor([existing], tmp_path=tmp_path)
    from core.risk_manager import AccountSnapshot
    risk.check_portfolio(AccountSnapshot(100_000.0, 50_000.0,
                                          datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)))

    sig = Signal(symbol="SPY", side="FLAT", target_weight=0.0,
                 confidence=0.9, regime="crash", reason="test")
    acct = AccountInfo(equity=100_000.0, cash=50_000.0, invested=50_000.0,
                       currency="USD", timestamp=datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc))

    result = executor.submit(sig, acct, current_price=500.0)
    assert result.placed
    call = fake_client.orders.place_market.call_args
    req = call.args[0] if call.args else call.kwargs["request"]
    assert req.quantity < 0
    assert abs(req.quantity - (-100.0)) < 1e-6


def test_symbol_map_translates_ticker():
    cfg = load_config()
    fake_t212_cfg = MagicMock()
    fake_t212_cfg.env = "demo"
    fake_t212_cfg.api_key = "k"
    fake_t212_cfg.secret_key = "s"
    fake_t212_cfg.base_url = "https://demo.trading212.com/api/v0"
    fake_client = MagicMock()
    fake_client.orders.place_market.return_value = MagicMock(id="x")
    fake_client.account.summary.return_value = MagicMock(total=100000, free=100000, invested=0, currencyCode="USD")
    fake_client.positions.list.return_value = []

    with patch("broker.trade212_api.load_config", return_value=fake_t212_cfg), \
         patch("broker.trade212_api.Trade212Client", return_value=fake_client):
        adapter = BrokerAdapter(cfg.broker, symbol_map={"SAP": "SAPd_EQ"}).connect()

    adapter.place_market("SAP", signed_qty=5.0)
    req = fake_client.orders.place_market.call_args.args[0]
    assert req.ticker == "SAPd_EQ"
