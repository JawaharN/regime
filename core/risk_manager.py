"""Hard risk layer with veto power — independent of strategy/HMM.

Legacy circuit breakers (``check_portfolio`` / ``check_trade``):
- daily P&L <= -2%  → size multiplier 0.5 (REDUCE)
- daily P&L <= -3%  → flatten, block new entries (FLATTEN)
- weekly P&L <= -5% → size multiplier 0.5 (REDUCE)
- drawdown from peak >= 10% → write kill_switch.block, halt (KILL)

Final-prompt additions (``validate_signal`` + ``CircuitBreaker``):
- two-tier daily (2%/3%) and weekly (5%/7%) breakers on realised P&L
- peak drawdown 10% → writes ``trading_halted.lock`` (separate from the kill
  switch; both require manual deletion)
- portfolio caps: 80% exposure, 15% single position, 30% sector, 5 concurrent,
  20 trades/day, 1.25x leverage
- every signal must carry a stop loss
- risk-based position sizing, gap-risk overnight cap
- correlation, bid-ask spread, and 60-second duplicate gates

State persists in ``state/peak_equity.json`` and ``state/kill_switch.block``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger("regime_trader.risk")


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    timestamp: datetime


@dataclass(frozen=True)
class RiskDecision:
    action: Literal["ALLOW", "REDUCE", "BLOCK", "FLATTEN", "KILL"]
    size_multiplier: float
    reason: str
    approved: bool = True
    rejection_reason: str | None = None
    modifications: tuple[str, ...] = ()


@dataclass
class PortfolioState:
    equity: float
    cash: float
    buying_power: float = 0.0
    positions: list = field(default_factory=list)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    circuit_breaker_status: str = "ok"
    flicker_rate: float = 0.0
    sectors: dict[str, str] = field(default_factory=dict)


class RiskManager:
    """Independent of strategy/HMM. Reads only account snapshots + trade intents."""

    def __init__(self, cfg) -> None:  # noqa: ANN001 — cfg = RiskCfg
        self.cfg = cfg
        self._peak_equity: float | None = None
        self._session_open_equity: float | None = None
        self._session_open_date: date | None = None
        self._week_open_equity: float | None = None
        self._week_open_iso_week: tuple[int, int] | None = None
        self._reduce_today = False
        self._flatten_today = False
        self._daily_trade_count = 0
        self._recent_orders: list[tuple[str, str, datetime]] = []
        self.breaker = CircuitBreaker(cfg)
        self._load_peak()

    # ------------------------------------------------------ public API

    def kill_switch_armed(self) -> bool:
        return Path(self.cfg.kill_switch_path).exists()

    def halt_lock_armed(self) -> bool:
        path = getattr(self.cfg, "halt_lock_path", None)
        return bool(path) and Path(path).exists()

    def assert_safe_to_start(self) -> None:
        """Called at startup. Raises if the kill switch or halt lock is armed."""
        if self.kill_switch_armed():
            raise RuntimeError(
                f"Kill switch is armed at {self.cfg.kill_switch_path}. "
                "Delete the file manually after reviewing risk before restarting."
            )
        if self.halt_lock_armed():
            raise RuntimeError(
                f"Trading halt lock is armed at {self.cfg.halt_lock_path}. "
                "Delete the file manually after reviewing risk before restarting."
            )

    def check_portfolio(self, snap: AccountSnapshot) -> RiskDecision:
        """Legacy portfolio-level circuit breakers (kept verbatim)."""
        self._update_rolling_baselines(snap)
        peak = self._peak_equity or snap.equity

        if snap.equity <= peak * (1 - self.cfg.total_drawdown_kill_pct):
            self._arm_kill_switch(snap, peak)
            return RiskDecision(
                "KILL", 0.0,
                f"drawdown {snap.equity / peak - 1:.2%} ≥ "
                f"{self.cfg.total_drawdown_kill_pct:.0%} kill threshold",
                approved=False,
            )

        daily_pnl = self._daily_pnl_pct(snap)
        if daily_pnl is not None and daily_pnl <= -self.cfg.daily_drawdown_halt_pct:
            self._flatten_today = True
            return RiskDecision("FLATTEN", 0.0,
                                f"daily P&L {daily_pnl:.2%} ≤ -{self.cfg.daily_drawdown_halt_pct:.0%}",
                                approved=False)

        weekly_pnl = self._weekly_pnl_pct(snap)
        reducers = []
        if daily_pnl is not None and daily_pnl <= -self.cfg.daily_drawdown_warn_pct:
            reducers.append(f"daily {daily_pnl:.2%}")
        if weekly_pnl is not None and weekly_pnl <= -self.cfg.weekly_drawdown_warn_pct:
            reducers.append(f"weekly {weekly_pnl:.2%}")
        if reducers:
            self._reduce_today = True
            return RiskDecision("REDUCE", 0.5, "soft drawdown: " + ", ".join(reducers))

        return RiskDecision("ALLOW", 1.0, "ok")

    def dashboard_status(self, snap: AccountSnapshot) -> dict:
        peak = self._peak_equity or snap.equity
        daily_pnl = self._daily_pnl_pct(snap)
        drawdown_from_peak = snap.equity / peak - 1.0 if peak else 0.0
        return {
            "daily_pnl": daily_pnl if daily_pnl is not None else 0.0,
            "daily_drawdown": {
                "value": f"{abs(min(daily_pnl or 0.0, 0.0)):.1%}",
                "limit": f"{self.cfg.daily_drawdown_halt_pct:.0%}",
                "ok": (daily_pnl or 0.0) > -self.cfg.daily_drawdown_halt_pct,
            },
            "from_peak": {
                "value": f"{abs(min(drawdown_from_peak, 0.0)):.1%}",
                "limit": f"{self.cfg.total_drawdown_kill_pct:.0%}",
                "ok": drawdown_from_peak > -self.cfg.total_drawdown_kill_pct,
            },
            "kill_switch": self.kill_switch_armed(),
            "halt_lock": self.halt_lock_armed(),
            "peak_equity": peak,
        }

    def check_trade(self, signal, snap: AccountSnapshot, positions: list,  # noqa: ANN001
                    stop_distance_pct: float | None = None) -> RiskDecision:
        """Legacy pre-trade validation. Returns ALLOW, REDUCE, or BLOCK."""
        if self._flatten_today:
            return RiskDecision("BLOCK", 0.0, "daily flatten in effect", approved=False)

        if not _finite(signal.target_weight) or signal.target_weight < 0:
            return RiskDecision("BLOCK", 0.0, f"invalid target_weight={signal.target_weight}",
                                approved=False)

        gross = sum(abs(getattr(p, "weight", 0.0)) for p in positions)
        proposed_gross = gross + abs(signal.target_weight)
        if proposed_gross > self.cfg.leverage_cap:
            return RiskDecision("BLOCK", 0.0,
                                f"leverage {proposed_gross:.2f} > cap {self.cfg.leverage_cap}",
                                approved=False)

        if stop_distance_pct is not None:
            implied_risk = signal.target_weight * stop_distance_pct
            if implied_risk > self.cfg.per_trade_risk_pct:
                if stop_distance_pct > 0:
                    new_weight = self.cfg.per_trade_risk_pct / stop_distance_pct
                    multiplier = new_weight / signal.target_weight if signal.target_weight > 0 else 0
                    return RiskDecision("REDUCE", multiplier,
                                        f"per-trade risk {implied_risk:.4f} > "
                                        f"cap {self.cfg.per_trade_risk_pct}")
                return RiskDecision("BLOCK", 0.0, "zero stop distance", approved=False)

        if self._reduce_today:
            return RiskDecision("REDUCE", 0.5, "soft drawdown in effect")

        return RiskDecision("ALLOW", 1.0, "ok")

    # ----------------------------------------- final-prompt validate_signal

    def validate_signal(self, signal, portfolio: PortfolioState,  # noqa: ANN001
                        spread_pct: float | None = None,
                        correlations: dict[str, float] | None = None) -> RiskDecision:
        """Full final-prompt validation: caps, breakers, sizing, gates.

        Returns a RiskDecision whose ``size_multiplier`` should be applied to the
        signal's ``position_size_pct``. ``action`` is BLOCK/REDUCE/ALLOW/KILL.
        """
        mods: list[str] = []

        # Circuit breakers on realised P&L.
        breaker = self.breaker.check(portfolio)
        if breaker.action in ("FLATTEN", "KILL"):
            return RiskDecision(breaker.action, 0.0, breaker.reason, approved=False,
                                rejection_reason=breaker.reason)

        # Every signal must carry a stop loss.
        if getattr(signal, "stop_loss", None) is None or getattr(signal, "entry_price", None) is None:
            return RiskDecision("BLOCK", 0.0, "signal has no stop loss", approved=False,
                                rejection_reason="missing stop loss")

        entry = float(signal.entry_price)
        stop = float(signal.stop_loss)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return RiskDecision("BLOCK", 0.0, "zero stop distance", approved=False,
                                rejection_reason="zero stop distance")

        # Concurrent / daily-trade caps.
        if len(portfolio.positions) >= self.cfg.max_concurrent and not _holds(portfolio, signal.symbol):
            return RiskDecision("BLOCK", 0.0,
                                f"max concurrent positions ({self.cfg.max_concurrent}) reached",
                                approved=False, rejection_reason="max concurrent")
        if self._daily_trade_count >= self.cfg.max_daily_trades:
            return RiskDecision("BLOCK", 0.0,
                                f"max daily trades ({self.cfg.max_daily_trades}) reached",
                                approved=False, rejection_reason="max daily trades")

        # Duplicate (same symbol+direction within the configured window).
        if self._is_duplicate(signal):
            return RiskDecision("BLOCK", 0.0, "duplicate order within window", approved=False,
                                rejection_reason="duplicate order")

        # Bid-ask spread gate.
        if spread_pct is not None and spread_pct > self.cfg.max_spread_pct:
            return RiskDecision("BLOCK", 0.0,
                                f"spread {spread_pct:.4f} > {self.cfg.max_spread_pct}",
                                approved=False, rejection_reason="spread too wide")

        # Risk-based position sizing.
        stop_distance_pct = stop_distance / entry
        risk_weight = self.cfg.max_risk_per_trade / stop_distance_pct
        target = float(getattr(signal, "position_size_pct", signal.target_weight))
        multiplier = 1.0
        if risk_weight < target:
            multiplier = risk_weight / target if target > 0 else 0.0
            mods.append(f"risk sizing → {risk_weight:.3f}")
        sized = target * multiplier

        # Single-position cap.
        if sized > self.cfg.max_single_position:
            multiplier *= self.cfg.max_single_position / sized
            sized = self.cfg.max_single_position
            mods.append(f"capped to single-position {self.cfg.max_single_position:.0%}")

        # Gap-risk overnight cap: gap_multiplier * stop_distance must stay
        # within gap_risk_budget of equity.
        gap_weight = self.cfg.gap_risk_budget / (self.cfg.gap_multiplier * stop_distance_pct)
        if sized > gap_weight:
            multiplier *= gap_weight / sized if sized > 0 else 0.0
            sized = gap_weight
            mods.append("gap-risk cap applied")

        # Portfolio exposure cap.
        gross = sum(abs(getattr(p, "weight", 0.0)) for p in portfolio.positions)
        if gross + sized > self.cfg.max_exposure:
            room = max(0.0, self.cfg.max_exposure - gross)
            multiplier *= room / sized if sized > 0 else 0.0
            sized = room
            mods.append(f"exposure cap → {room:.3f}")

        # Sector cap.
        sector = portfolio.sectors.get(signal.symbol)
        if sector:
            sector_gross = sum(
                abs(getattr(p, "weight", 0.0))
                for p in portfolio.positions
                if portfolio.sectors.get(getattr(p, "symbol", "")) == sector
            )
            if sector_gross + sized > self.cfg.max_sector_exposure:
                return RiskDecision("BLOCK", 0.0,
                                    f"sector '{sector}' exposure cap "
                                    f"{self.cfg.max_sector_exposure:.0%} breached",
                                    approved=False, rejection_reason="sector cap")

        # Correlation gate.
        if correlations:
            worst = max(correlations.values(), default=0.0)
            if worst > self.cfg.correlation_reject:
                return RiskDecision("BLOCK", 0.0,
                                    f"correlation {worst:.2f} > {self.cfg.correlation_reject}",
                                    approved=False, rejection_reason="correlation reject")
            if worst > self.cfg.correlation_reduce:
                multiplier *= 0.5
                mods.append(f"correlation {worst:.2f} → 0.5x size")

        action = "REDUCE" if multiplier < 1.0 - 1e-9 else "ALLOW"
        if breaker.action == "REDUCE":
            multiplier *= 0.5
            mods.append(breaker.reason)
            action = "REDUCE"
        reason = "; ".join(mods) if mods else "ok"
        return RiskDecision(action, multiplier, reason, approved=True,
                            modifications=tuple(mods))

    def record_trade(self, symbol: str, direction: str, when: datetime | None = None) -> None:
        """Register a placed order for duplicate detection + daily-count caps."""
        when = when or datetime.now(timezone.utc)
        self._recent_orders.append((symbol, direction, when))
        self._daily_trade_count += 1

    # ------------------------------------------------------ internals

    def _is_duplicate(self, signal) -> bool:  # noqa: ANN001
        window = self.cfg.duplicate_window_seconds
        now = datetime.now(timezone.utc)
        direction = getattr(signal, "direction", None) or getattr(signal, "side", "")
        for sym, d, when in self._recent_orders:
            if sym == signal.symbol and d == direction and (now - when).total_seconds() < window:
                return True
        return False

    def _update_rolling_baselines(self, snap: AccountSnapshot) -> None:
        today = snap.timestamp.astimezone(timezone.utc).date()
        if self._session_open_date != today:
            self._session_open_date = today
            self._session_open_equity = snap.equity
            self._reduce_today = False
            self._flatten_today = False
            self._daily_trade_count = 0

        iso_year, iso_week, _ = snap.timestamp.astimezone(timezone.utc).isocalendar()
        if self._week_open_iso_week != (iso_year, iso_week):
            self._week_open_iso_week = (iso_year, iso_week)
            self._week_open_equity = snap.equity

        if self._peak_equity is None or snap.equity > self._peak_equity:
            self._peak_equity = snap.equity
            self._persist_peak()

    def _daily_pnl_pct(self, snap: AccountSnapshot) -> float | None:
        if self._session_open_equity is None or self._session_open_equity == 0:
            return None
        return snap.equity / self._session_open_equity - 1.0

    def _weekly_pnl_pct(self, snap: AccountSnapshot) -> float | None:
        if self._week_open_equity is None or self._week_open_equity == 0:
            return None
        return snap.equity / self._week_open_equity - 1.0

    def _arm_kill_switch(self, snap: AccountSnapshot, peak: float) -> None:
        path = Path(self.cfg.kill_switch_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "armed_at": snap.timestamp.isoformat(),
            "peak_equity": peak,
            "equity_at_trigger": snap.equity,
            "drawdown_pct": snap.equity / peak - 1.0,
            "threshold_pct": self.cfg.total_drawdown_kill_pct,
            "note": "Delete this file manually after reviewing risk to restart.",
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.critical("KILL SWITCH ARMED: drawdown %.2f%% — wrote %s",
                        payload["drawdown_pct"] * 100, path)

    def _persist_peak(self) -> None:
        if self._peak_equity is None:
            return
        path = Path(self.cfg.peak_equity_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"peak_equity": self._peak_equity}))

    def _load_peak(self) -> None:
        path = Path(self.cfg.peak_equity_path)
        if not path.exists():
            return
        try:
            self._peak_equity = float(json.loads(path.read_text()).get("peak_equity"))
        except (ValueError, KeyError, json.JSONDecodeError):
            self._peak_equity = None


@dataclass
class _BreakerResult:
    action: Literal["ALLOW", "REDUCE", "FLATTEN", "KILL"]
    reason: str


class CircuitBreaker:
    """Four-tier breaker state machine on realised daily/weekly/peak P&L.

    Tiers: daily 2%/3%, weekly 5%/7%, peak 10%. The 10% peak breach writes the
    ``trading_halted.lock`` file, which must be removed manually.
    """

    def __init__(self, cfg) -> None:  # noqa: ANN001
        self.cfg = cfg
        self.history: list[_BreakerResult] = []

    def check(self, portfolio: PortfolioState) -> _BreakerResult:
        peak = portfolio.peak_equity or portfolio.equity
        dd = (portfolio.equity / peak - 1.0) if peak else 0.0

        if dd <= -self.cfg.max_dd_from_peak:
            self._write_halt_lock(dd)
            return self._record("KILL", f"peak drawdown {dd:.2%} — trading halted")
        if portfolio.daily_pnl <= -self.cfg.daily_dd_halt:
            return self._record("FLATTEN", f"daily P&L {portfolio.daily_pnl:.2%} ≤ -3%")
        if portfolio.weekly_pnl <= -self.cfg.weekly_dd_halt:
            return self._record("FLATTEN", f"weekly P&L {portfolio.weekly_pnl:.2%} ≤ -7%")
        if portfolio.daily_pnl <= -self.cfg.daily_dd_reduce:
            return self._record("REDUCE", f"daily P&L {portfolio.daily_pnl:.2%} ≤ -2%")
        if portfolio.weekly_pnl <= -self.cfg.weekly_dd_reduce:
            return self._record("REDUCE", f"weekly P&L {portfolio.weekly_pnl:.2%} ≤ -5%")
        return self._record("ALLOW", "ok")

    def get_history(self) -> list[_BreakerResult]:
        return list(self.history)

    def _record(self, action: str, reason: str) -> _BreakerResult:
        r = _BreakerResult(action, reason)  # type: ignore[arg-type]
        self.history.append(r)
        return r

    def _write_halt_lock(self, dd: float) -> None:
        path = getattr(self.cfg, "halt_lock_path", None)
        if not path:
            return
        p = Path(path)
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "armed_at": datetime.now(timezone.utc).isoformat(),
            "drawdown_pct": dd,
            "note": "Peak drawdown breach. Delete manually after reviewing risk.",
        }, indent=2))
        logger.critical("TRADING HALT LOCK armed: drawdown %.2f%% — wrote %s", dd * 100, p)


def _holds(portfolio: PortfolioState, symbol: str) -> bool:
    return any(getattr(p, "symbol", "") == symbol for p in portfolio.positions)


def _finite(x: float) -> bool:
    try:
        return x == x and x not in (float("inf"), float("-inf"))
    except TypeError:
        return False
