"""Order executor: signal → risk check → Trade212 order.

Signed-quantity convention: positive = BUY, negative = SELL (Trade212 rule).
``submit`` computes the delta between the signal's target weight and the
current position weight and converts it to a signed quantity.

Bracket orders: Trade212 has no native OCO. ``submit_bracket_order`` places the
entry, then registers a ``BracketRecord`` (entry + stop + take-profit) that the
position tracker watches; when the stop or target fills, the sibling is
cancelled. ``modify_stop`` only ever tightens a stop, never widens it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from broker.broker_adapter import AccountInfo, BrokerAdapter, PositionInfo
from broker.position_tracker import PositionTracker
from core.risk_manager import AccountSnapshot, RiskManager
from core.regime_strategies import Signal

logger = logging.getLogger("regime_trader.executor")


@dataclass
class ExecutionResult:
    placed: bool
    order_id: object | None
    reason: str
    multiplier_applied: float = 1.0


@dataclass
class BracketRecord:
    """Emulated OCO: entry + tracked stop + take-profit for one symbol."""

    symbol: str
    entry_order_id: object | None
    quantity: float
    stop_loss: float | None
    take_profit: float | None
    stop_order_id: object | None = None
    tp_order_id: object | None = None
    closed: bool = False
    metadata: dict = field(default_factory=dict)


class OrderExecutor:
    def __init__(self, broker: BrokerAdapter, risk: RiskManager,
                 positions: PositionTracker) -> None:
        self.broker = broker
        self.risk = risk
        self.positions = positions
        self.brackets: dict[str, BracketRecord] = {}

    def submit(self, signal: Signal, account: AccountInfo, current_price: float,
               stop_distance_pct: float | None = None) -> ExecutionResult:
        snap = AccountSnapshot(equity=account.equity, cash=account.cash,
                               timestamp=account.timestamp)

        decision = self.risk.check_trade(
            signal, snap, positions=self.positions.current(),
            stop_distance_pct=stop_distance_pct,
        )
        if decision.action == "BLOCK":
            return ExecutionResult(False, None, f"BLOCK: {decision.reason}")
        if decision.action == "KILL":
            return ExecutionResult(False, None, f"KILL: {decision.reason}")

        multiplier = decision.size_multiplier if decision.action == "REDUCE" else 1.0
        effective_weight = signal.target_weight * multiplier
        current = self._current_position(signal.symbol)
        current_weight = current.weight * (1 if current.quantity >= 0 else -1) if current else 0.0
        delta_weight = effective_weight - current_weight

        if math.isclose(delta_weight, 0.0, abs_tol=1e-6):
            return ExecutionResult(False, None, "no change", multiplier)
        if current_price <= 0:
            return ExecutionResult(False, None, "non-positive price", multiplier)

        signed_qty = (delta_weight * account.equity) / current_price
        if math.isclose(signed_qty, 0.0, abs_tol=1e-6):
            return ExecutionResult(False, None, "qty too small", multiplier)

        try:
            order = self.broker.place_market(signal.symbol, signed_qty=signed_qty)
        except Exception as e:  # noqa: BLE001
            logger.exception("order submission failed for %s", signal.symbol)
            return ExecutionResult(False, None, f"broker error: {type(e).__name__}", multiplier)

        return ExecutionResult(True, getattr(order, "id", None),
                               f"placed {signed_qty:+.4f} @ ~{current_price:.2f}", multiplier)

    def submit_bracket_order(self, signal: Signal, account: AccountInfo,
                             current_price: float) -> ExecutionResult:
        """Place the entry and register an emulated OCO bracket."""
        result = self.submit(signal, account, current_price)
        if not result.placed:
            return result
        signed_qty = (signal.target_weight * account.equity) / current_price
        self.brackets[signal.symbol] = BracketRecord(
            symbol=signal.symbol,
            entry_order_id=result.order_id,
            quantity=signed_qty,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        logger.info("bracket registered for %s: stop=%s tp=%s",
                    signal.symbol, signal.stop_loss, signal.take_profit)
        return result

    def modify_stop(self, symbol: str, new_stop: float) -> bool:
        """Tighten a tracked stop. A wider stop is rejected (never loosen risk)."""
        rec = self.brackets.get(symbol)
        if rec is None or rec.closed:
            return False
        long_pos = rec.quantity >= 0
        if rec.stop_loss is not None:
            tightens = new_stop > rec.stop_loss if long_pos else new_stop < rec.stop_loss
            if not tightens:
                logger.info("modify_stop(%s) rejected: %.4f does not tighten %.4f",
                            symbol, new_stop, rec.stop_loss)
                return False
        rec.stop_loss = new_stop
        logger.info("stop for %s tightened to %.4f", symbol, new_stop)
        return True

    def cancel_order(self, order_id) -> None:  # noqa: ANN001
        self.broker.cancel(order_id)

    def close_position(self, symbol: str, current_price: float) -> ExecutionResult:
        pos = self._current_position(symbol)
        if pos is None or pos.quantity == 0:
            return ExecutionResult(False, None, "no position to close")
        try:
            order = self.broker.place_market(symbol, signed_qty=-pos.quantity)
        except Exception as e:  # noqa: BLE001
            logger.exception("close_position failed for %s", symbol)
            return ExecutionResult(False, None, f"broker error: {type(e).__name__}")
        rec = self.brackets.get(symbol)
        if rec:
            rec.closed = True
        return ExecutionResult(True, getattr(order, "id", None), f"closed {symbol}")

    def close_all_positions(self) -> list[ExecutionResult]:
        out: list[ExecutionResult] = []
        for pos in self.positions.current():
            out.append(self.close_position(pos.symbol, pos.current_price))
        return out

    def _current_position(self, symbol: str) -> PositionInfo | None:
        for p in self.positions.current():
            if p.symbol == symbol:
                return p
        return None
