# regime_trader

HMM regime-based, paper-first trading framework. Detects market regime with a Hidden Markov Model (a **volatility classifier**, not a price predictor), routes allocation/strategy by **volatility rank**, enforces hard risk controls independent of the model, supports walk-forward backtesting with explicit allocation math, and integrates with **Trade212 paper trading** via the local `trade212_bot` package.

> **Not a toy.** Risk layer overrides the model. Paper-only. No look-ahead bias anywhere.

## Quickstart

```bash
cd /home/jawahar/regime
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in TRADING212_API_KEY / TRADING212_SECRET_KEY (demo account)

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

```bash
python main.py scaffold                              # verify install + config
python main.py train --symbol SPY                    # fit + persist the HMM
python main.py backtest --symbol SPY --years 6 --compare --stress-test
python main.py broker-test                           # Trade212 demo account smoke test
python main.py run --paper --dry-run                 # walk the loop, no orders
python main.py run --paper                           # live paper main loop
python main.py dashboard                             # Rich terminal dashboard
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

## Reused: trade212_bot

The broker layer wraps `/home/jawahar/trading/trade212` (installed editable):
authenticated `Trade212Client`, signed-quantity order placement, rate-limit
handling, paper-mode enforcement. Trade212 has no native OCO or trade-fill
WebSocket — brackets are emulated and fills are detected by REST polling.
