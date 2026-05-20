"""regime_trader CLI entry point + main loop.

Subcommands:
  scaffold     — verify install + config
  train        — fit the HMM on daily history and persist the model
  backtest     — walk-forward backtest (--compare benchmarks, --stress-test)
  broker-test  — Trade212 demo connection smoke test
  run          — start the daily main loop (--dry-run, --paper, --dashboard)
  dashboard    — launch the HTML dashboard server

The loop runs on a 1-Day cadence: per bar it reads the regime (forward-only),
filters for stability, generates vol-tier signals, validates them through the
risk layer, and places orders on Trade212 demo. The HMM retrains weekly (or when
the saved model is older than `hmm.retrain_days`). SIGINT/SIGTERM trigger a
clean shutdown that writes `state/state_snapshot.json` and never closes
positions.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from core.config import load_config, project_root

logger = logging.getLogger("regime_trader.main")


# -------------------------------------------------- subcommands

def _cmd_scaffold(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"[scaffold] universe = {cfg.universe.symbols}")
    print(f"[scaffold] runtime bars = {cfg.bars.runtime_interval}")
    print(f"[scaffold] regime labels = {cfg.regime_labels.names}")
    print(f"[scaffold] hmm candidates = {cfg.hmm.n_candidates} (BIC selection)")
    print("[scaffold] config loaded OK — verification complete.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import build_features, feature_spec_from_cfg
    from data.market_data import load_history

    symbols = args.symbols or [args.symbol] if getattr(args, "symbol", None) else args.symbols
    symbols = symbols or cfg.universe.symbols
    spec = feature_spec_from_cfg(cfg.features)
    for sym in symbols:
        print(f"[train] loading {sym} daily history ({cfg.bars.training_years}y)")
        ohlcv = load_history(sym, interval=cfg.bars.training_interval,
                             years=cfg.bars.training_years)
        features = build_features(ohlcv, cfg.features).dropna()
        engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, spec)
        engine.fit(features)
        out = project_root() / "state" / "trained_models" / f"{sym}.pkl"
        engine.save(out)
        print(f"[train] {sym}: n_components={engine.n_components} "
              f"BIC={engine.bic_scores_} → {out}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from backtest import benchmarks, stress_tests
    from backtest.walk_forward import run_walk_forward
    from data.market_data import load_history

    symbols = args.symbols or [args.symbol]
    for sym in symbols:
        ohlcv = load_history(sym, interval=cfg.bars.training_interval, years=args.years)
        result = run_walk_forward(sym, ohlcv, cfg)
        print(f"\n=== {sym} ===")
        print(json.dumps(_jsonify(result), indent=2, default=str))
        if args.compare and not result.get("summary"):
            print("[backtest] not enough history for benchmark comparison")
        elif args.compare:
            ens = benchmarks.random_baseline_ensemble(
                ohlcv["close"], n_seeds=cfg.backtest.random_seeds)
            print("[compare] random-baseline ensemble:", json.dumps(ens, indent=2))
        if args.stress_test:
            mc = stress_tests.monte_carlo_crash(ohlcv)
            print("[stress] monte-carlo crash:", json.dumps(mc, indent=2))
            print("[stress] gap risk:", json.dumps(stress_tests.gap_risk_test(ohlcv), indent=2))
    return 0


def _cmd_broker_test(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from broker.broker_adapter import BrokerAdapter
    with BrokerAdapter(cfg.broker, symbol_map=cfg.universe.symbol_map) as broker:
        info = broker.account()
        print(f"[broker-test] equity={info.equity:.2f} {info.currency}  "
              f"cash={info.cash:.2f}  invested={info.invested:.2f}")
        positions = broker.positions(equity_hint=info.equity)
        print(f"[broker-test] {len(positions)} open positions  "
              f"market_open={broker.is_market_open()}")
        for p in positions:
            print(f"  - {p.symbol}: qty={p.quantity} avg={p.average_price} weight={p.weight:.2%}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    from monitoring.dashboard import run_dashboard_server
    host = args.host or cfg.monitoring.dashboard_host
    port = args.port or cfg.monitoring.dashboard_port
    return run_dashboard_server(cfg.monitoring.state_dashboard_path,
                                host=host,
                                port=port,
                                refresh_seconds=cfg.monitoring.dashboard_refresh_seconds)


def _cmd_run(args: argparse.Namespace) -> int:
    from broker.broker_adapter import BrokerAdapter
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.regime_stability import RegimeStabilityFilter
    from core.regime_strategies import RegimeOrchestrator
    from core.risk_manager import RiskManager
    from core.signal_generator import SignalGenerator
    from data.feature_engineering import build_features, feature_spec_from_cfg
    from data.market_data import latest_bars

    cfg = load_config(args.config)
    _configure_logging(cfg.logging, cfg.monitoring)

    risk = RiskManager(cfg.risk)
    risk.assert_safe_to_start()
    logger.info("kill switch + halt lock clear — proceeding")

    broker = BrokerAdapter(cfg.broker, symbol_map=cfg.universe.symbol_map).connect()
    tracker = PositionTracker(broker, correlation_window=cfg.risk.correlation_window)
    executor = OrderExecutor(broker, risk, tracker)

    spec = feature_spec_from_cfg(cfg.features)
    generators: dict[str, SignalGenerator] = {}
    for sym in cfg.universe.symbols:
        model_path = project_root() / "state" / "trained_models" / f"{sym}.pkl"
        engine = _load_or_train(sym, model_path, cfg, spec, build_features, latest_bars)
        stability = RegimeStabilityFilter(
            cfg.stability.min_persistence_bars, cfg.stability.flicker_window,
            cfg.stability.flicker_threshold, cfg.stability.unstable_confidence_decay,
            cfg.stability.transition_size_cut,
        )
        orch = RegimeOrchestrator(cfg.strategy, engine.regime_infos, cfg.hmm.min_confidence)
        generators[sym] = SignalGenerator(engine, stability, orch, cfg.features)

    stop_flag = {"stop": False}

    def _on_signal(signum, frame):  # noqa: ANN001
        logger.warning("received signal %s — shutting down", signum)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    iterations = args.iterations
    i = 0
    try:
        while not stop_flag["stop"] and (iterations <= 0 or i < iterations):
            _run_one_iteration(cfg, broker, tracker, executor, generators,
                               dry_run=args.dry_run)
            i += 1
            if iterations > 0 and i >= iterations:
                break
            for _ in range(args.poll_seconds):
                if stop_flag["stop"]:
                    break
                time.sleep(1)
    finally:
        _write_state_snapshot(cfg, tracker, risk)
        broker.close()
        logger.info("clean shutdown — state snapshot written")
    return 0


def _load_or_train(sym, model_path, cfg, spec, build_features, latest_bars):  # noqa: ANN001
    from core.hmm_engine import HMMEngine

    stale = False
    if model_path.exists():
        age_days = (time.time() - model_path.stat().st_mtime) / 86400
        stale = age_days > cfg.hmm.retrain_days
    try:
        if stale:
            raise RuntimeError(f"model older than {cfg.hmm.retrain_days}d — retraining")
        engine = HMMEngine.load(model_path, cfg.hmm, spec)
        logger.info("loaded HMM for %s from %s", sym, model_path)
        return engine
    except (FileNotFoundError, RuntimeError) as e:
        logger.info("training HMM for %s (%s)", sym, e)
        ohlcv = latest_bars(sym, interval=cfg.bars.training_interval,
                            n_bars=int(cfg.bars.training_years * 252))
        features = build_features(ohlcv, cfg.features).dropna()
        engine = HMMEngine(cfg.hmm, cfg.regime_labels.names, spec)
        engine.fit(features)
        engine.save(model_path)
        return engine


def _run_one_iteration(cfg, broker, tracker, executor, generators,  # noqa: ANN001
                       dry_run: bool = False) -> None:
    from core.risk_manager import AccountSnapshot
    from data.market_data import latest_bars

    try:
        account = broker.account()
    except Exception as e:  # noqa: BLE001
        logger.exception("broker.account() failed: %s", e)
        return

    account_snap = AccountSnapshot(account.equity, account.cash, account.timestamp)
    risk_dec = executor.risk.check_portfolio(account_snap)
    logger.info("portfolio check: %s (%s)", risk_dec.action, risk_dec.reason)
    if risk_dec.action == "KILL":
        logger.critical("kill switch armed during loop — exiting")
        raise SystemExit(2)

    tracker.refresh(equity_hint=account.equity)
    remaining_cash = account.cash
    recent_signals: list[dict] = []
    data_ok = True
    hmm_ok = False

    for sym, gen in generators.items():
        try:
            bars = latest_bars(sym, interval=cfg.bars.runtime_interval, n_bars=400)
            for sig in gen.generate(sym, bars):
                hmm_ok = True
                price = float(bars["close"].iloc[-1])
                signal_row = {
                    "time": account.timestamp.astimezone(timezone.utc).strftime("%H:%M"),
                    "symbol": sym,
                    "action": sig.side,
                    "reason": sig.reason,
                }
                if dry_run:
                    logger.info("[dry-run] %s %s w=%.3f stop=%s — not submitted",
                                sym, sig.side, sig.target_weight, sig.stop_loss)
                    recent_signals.append(signal_row)
                    continue
                iteration_account = replace(account, cash=remaining_cash)
                result = executor.submit(sig, iteration_account, current_price=price)
                signal_row["action"] = result.reason
                recent_signals.append(signal_row)
                if result.placed and result.signed_qty > 0:
                    remaining_cash = max(0.0, remaining_cash - result.estimated_notional)
                logger.info("[%s] regime=%s conf=%.2f → %s",
                            sym, sig.regime, sig.confidence, result.reason)
        except Exception:  # noqa: BLE001
            data_ok = False
            logger.exception("iteration failed for %s", sym)

    _write_dashboard_state(
        cfg,
        account=account,
        tracker=tracker,
        executor=executor,
        generators=generators,
        account_snap=account_snap,
        recent_signals=recent_signals,
        data_ok=data_ok,
        hmm_ok=hmm_ok,
        dry_run=dry_run,
    )


# -------------------------------------------------- helpers

def _configure_logging(log_cfg, monitoring_cfg=None) -> None:  # noqa: ANN001
    from monitoring.logger import build_logger

    build_logger(log_cfg)
    if monitoring_cfg is not None:
        from monitoring.logger import setup_rotating_logs
        setup_rotating_logs(monitoring_cfg, log_cfg.redact_env_keys)


def _select_dashboard_regime(generators: dict) -> dict:
    states = []
    for sym, gen in generators.items():
        state = getattr(gen, "last_state", None)
        if state is None:
            continue
        stability = getattr(gen, "last_stability", None)
        states.append((sym, state, stability))
    if not states:
        return {}
    states.sort(key=lambda item: (bool(item[1].is_confirmed), item[1].probability), reverse=True)
    sym, state, stability = states[0]
    return {
        "symbol": sym,
        "label": state.label,
        "probability": state.probability,
        "consecutive_bars": state.consecutive_bars,
        "flicker_count": getattr(stability, "flicker_count", 0),
        "vol_rank": state.state_id,
    }



def _serialize_dashboard_positions(tracker, executor) -> list[dict]:  # noqa: ANN001
    rows = []
    for pos in tracker.current():
        denominator = abs(pos.quantity * pos.average_price)
        upnl_pct = (pos.unrealized_pnl / denominator) if denominator > 0 else 0.0
        bracket = executor.brackets.get(pos.symbol)
        rows.append({
            "symbol": pos.symbol,
            "side": "LONG" if pos.quantity > 0 else "FLAT",
            "entry": pos.average_price,
            "current": pos.current_price,
            "unrealized_pnl": pos.unrealized_pnl,
            "unrealized_pnl_pct": upnl_pct,
            "stop": bracket.stop_loss if bracket and bracket.stop_loss is not None else 0,
            "holding_bars": 0,
        })
    return rows



def _build_dashboard_state(cfg, account, tracker, executor, generators, account_snap,  # noqa: ANN001
                           recent_signals: list[dict], data_ok: bool,
                           hmm_ok: bool, dry_run: bool) -> dict:
    risk_status = executor.risk.dashboard_status(account_snap)
    positions = _serialize_dashboard_positions(tracker, executor)
    allocation = sum(abs(pos.weight) for pos in tracker.current())
    return {
        "updated_at": account.timestamp.isoformat(),
        "regime": _select_dashboard_regime(generators),
        "portfolio": {
            "equity": account.equity,
            "daily_pnl": risk_status["daily_pnl"],
            "allocation": allocation,
            "leverage": allocation,
            "cash": account.cash,
            "buying_power": account.cash,
        },
        "positions": positions,
        "recent_signals": recent_signals[-10:],
        "risk_status": {
            "daily_drawdown": risk_status["daily_drawdown"],
            "from_peak": risk_status["from_peak"],
        },
        "system": {
            "data": "ok" if data_ok else "degraded",
            "api": "ok",
            "hmm": "ok" if hmm_ok else "idle",
            "mode": "DRY_RUN" if dry_run else "PAPER",
            "halt_lock": "armed" if risk_status["halt_lock"] else "clear",
        },
        "kill_switch": risk_status["kill_switch"],
    }



def _write_dashboard_state(cfg, **kwargs) -> None:  # noqa: ANN001
    try:
        payload = _build_dashboard_state(cfg, **kwargs)
        path = Path(cfg.monitoring.state_dashboard_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:  # noqa: BLE001
        logger.exception("failed to write dashboard state")



def _write_state_snapshot(cfg, tracker, risk) -> None:  # noqa: ANN001
    try:
        path = project_root() / "state" / "state_snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "peak_equity": risk._peak_equity,
            "positions": [
                {"symbol": p.symbol, "quantity": p.quantity,
                 "average_price": p.average_price}
                for p in tracker.current()
            ],
        }, indent=2, default=str))
    except Exception:  # noqa: BLE001
        logger.exception("failed to write state snapshot")


def _jsonify(obj):  # noqa: ANN001
    import pandas as pd
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# -------------------------------------------------- arg parsing

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="regime_trader", description="HMM regime trading bot")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scaffold").set_defaults(func=_cmd_scaffold)

    t = sub.add_parser("train")
    t.add_argument("--symbol", default=None)
    t.add_argument("--symbols", nargs="*", default=None)
    t.set_defaults(func=_cmd_train)

    b = sub.add_parser("backtest")
    b.add_argument("--symbol", default="NVDA")
    b.add_argument("--symbols", nargs="*", default=None)
    b.add_argument("--years", type=int, default=6)
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--compare", action="store_true")
    b.add_argument("--stress-test", dest="stress_test", action="store_true")
    b.set_defaults(func=_cmd_backtest)

    sub.add_parser("broker-test").set_defaults(func=_cmd_broker_test)

    r = sub.add_parser("run")
    r.add_argument("--paper", action="store_true",
                   help="(reserved) paper mode — the only supported mode")
    r.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="walk the loop without submitting orders")
    r.add_argument("--dashboard", action="store_true",
                   help="(reserved) refresh the dashboard state file")
    r.add_argument("--wait-open", dest="wait_open", action="store_true")
    r.add_argument("--iterations", type=int, default=0,
                   help="iterations before exit (0 = infinite)")
    r.add_argument("--poll-seconds", type=int, default=300)
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("dashboard")
    d.add_argument("--host", default=None,
                   help="HTTP bind host (default: monitoring.dashboard_host or 0.0.0.0)")
    d.add_argument("--port", type=int, default=None,
                   help="HTTP port (default: monitoring.dashboard_port or 8000)")
    d.set_defaults(func=_cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
