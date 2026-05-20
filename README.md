# regime_trader

HMM regime-based, paper-first trading framework. Detects market regime with a Hidden Markov Model (a **volatility classifier**, not a price predictor), routes allocation/strategy by **volatility rank**, enforces hard risk controls independent of the model, supports walk-forward backtesting with explicit allocation math, and integrates with **Trade212 paper trading** via a vendored, in-repo API client.

This is a **self-contained project** — it has no path dependency on any sibling repo.

> **Not a toy.** Risk layer overrides the model. Paper-only. No look-ahead bias anywhere.

## Quickstart

```bash
cd /home/jawahar/regime
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env:
#   - TRADING212_API_KEY / TRADING212_SECRET_KEY  (demo account)
#   - TV_SESSIONID  (optional: tradingview.com sessionid cookie for un-throttled data)

pytest -q
```

## Layout (flat top-level packages)

1. **`core/`** — HMM engine, regime labeling/stability, vol-tier strategies, allocation, risk manager, signal generator
2. **`data/`** — feature engineering (causal, z-scored) + market data loader
3. **`broker/`** — Trade212 adapter, order executor (bracket emulation), position tracker
4. **`backtest/`** — walk-forward + allocation-math backtester, metrics, benchmarks, stress tests
5. **`monitoring/`** — rotating JSON logging, alerts, performance tracker, Rich terminal dashboard
6. **`main.py`** — CLI entry point + daily main loop

## How it works

- **HMM**: Gaussian HMM, **pure-BIC** model selection across `n ∈ [3,4,5,6,7]` with `n_init` restarts. Live inference is **forward-only** (filtered) — never Viterbi — so the regime at bar *t* uses only bars `0..t`.
- **Strategies**: three volatility tiers — `LowVolBullStrategy` / `MidVolCautiousStrategy` / `HighVolDefensiveStrategy`. The orchestrator routes by **volatility rank** (`rank/(n-1)`), independent of the return-sorted label.
- **Risk**: two-tier daily (2%/3%) and weekly (5%/7%) circuit breakers, 10% peak-drawdown halt, exposure/sector/concurrency caps, risk-based sizing, gap-risk and correlation gates. Every signal must carry a stop loss.

## CLI

All commands are subcommands of `python main.py`. A global `--config PATH`
overrides the config file (defaults to `config/settings.yaml`, or `$REGIME_TRADER_CONFIG`).

### `scaffold` — verify install + config

```bash
python main.py scaffold
```

### `train` — fit + persist the HMM

```bash
python main.py train --symbol SPY                    # one symbol
python main.py train --symbols SPY AAPL MSFT         # several
python main.py train                                 # whole config universe
```

Models are written to `state/trained_models/<SYMBOL>.pkl`.

### `backtest` — walk-forward backtest

```bash
python main.py backtest --symbol SPY                          # basic
python main.py backtest --symbol SPY --years 6                # history window
python main.py backtest --symbol SPY --years 6 --compare      # vs buy&hold / SMA / random
python main.py backtest --symbol SPY --stress-test            # monte-carlo crash + gap risk
python main.py backtest --symbol SPY --years 6 --compare --stress-test
python main.py backtest --symbols SPY NVDA AMD                # multiple symbols
```

Flags: `--years N` (default 6), `--start / --end` (date window), `--compare`, `--stress-test`.

### `broker-test` — Trade212 demo smoke test

```bash
python main.py broker-test                           # prints equity, cash, open positions
```

### `run` — the daily paper-trading loop

```bash
python main.py run --paper --dry-run                 # walk the loop, submit no orders
python main.py run --paper                           # live paper loop (infinite)
python main.py run --paper --iterations 1            # single pass, then exit
python main.py run --paper --poll-seconds 300        # seconds between iterations (default 300)
```

Flags: `--paper` (the only supported mode), `--dry-run` (no orders), `--iterations N`
(0 = run forever), `--poll-seconds N`. SIGINT/SIGTERM trigger a clean shutdown that
writes `state/state_snapshot.json` and never closes positions.

### `dashboard` — Rich terminal dashboard

```bash
python main.py dashboard                             # refresh until interrupted
python main.py dashboard --iterations 1              # render once, then exit
```


## Non-negotiable rules

- No look-ahead bias — HMM uses **forward-only** inference.
- Secrets in `.env` only. `.env` is git-ignored.
- Paper trading only. `TRADING212_ENV` must be `demo`.
- Risk layer overrides the model — always.
- Modular, testable, deterministic, configurable.

## Kill switch & halt lock

A ≥10% drawdown from peak writes `state/kill_switch.block` (legacy breaker) /
`state/trading_halted.lock` (circuit breaker); the main loop **refuses to
start** until the file is manually deleted. This is intentional friction.

## Configuration

All knobs live in `config/settings.yaml` — universe (10 symbols), bar interval,
HMM/feature settings, vol-tier strategy parameters, risk thresholds, backtest
windows, broker settings, monitoring/logging.

## Broker: self-contained Trade212 client

The entire Trade212 integration lives in `broker/`. `broker/trade212_api.py` is
the wire client — authenticated httpx transport, rate-limit handling, typed
order/account/position models, and a `Trade212Client` facade. `broker/broker_adapter.py`
wraps it with retry/backoff and enforces demo-only (paper) operation. Trade212
has no native OCO or trade-fill WebSocket — brackets are emulated and fills are
detected by REST polling.
