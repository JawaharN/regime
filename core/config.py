"""Typed config loader.

Reads `config/settings.yaml` (path overridable via REGIME_TRADER_CONFIG) and
exposes a pydantic-validated `Config`. Broker credentials are NOT loaded here —
they live in broker.trade212_api.load_config() and are pulled lazily by the
broker adapter, so this module stays usable in tests that never touch the
network.

Schema note: every field the final build prompt added carries a default, so a
config that only sets the earlier-draft keys still validates. The shipped
`settings.yaml` sets all of them explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = "config/settings.yaml"


class UniverseCfg(BaseModel):
    symbols: list[str]
    symbol_map: dict[str, str] = Field(default_factory=dict)
    sectors: dict[str, str] = Field(default_factory=dict)


class BarsCfg(BaseModel):
    runtime_interval: str
    training_interval: str
    training_years: int


class FeaturesCfg(BaseModel):
    # legacy windows
    return_window: int
    vol_window: int
    volume_zscore_window: int
    atr_window: int
    range_expansion_window: int
    # final-prompt feature windows
    ret_windows: list[int] = [1, 5, 20]
    realized_vol_window: int = 20
    vol_ratio_fast: int = 5
    vol_ratio_slow: int = 20
    volume_trend_window: int = 10
    adx_window: int = 14
    sma_slope_window: int = 50
    sma_long_window: int = 200
    rsi_window: int = 14
    roc_windows: list[int] = [10, 20]
    zscore_window: int = 252
    zscore_min_periods: int = 60


class HMMCfg(BaseModel):
    n_components_min: int
    n_components_max: int
    covariance_type: str
    n_iter: int
    tol: float
    random_state: int
    validation_fraction: float
    n_candidates: list[int] = [3, 4, 5, 6, 7]
    n_init: int = 10
    bic_selection: bool = True
    min_train_bars: int = 252
    min_confidence: float = 0.55
    retrain_days: int = 7


class RegimeLabelsCfg(BaseModel):
    names: list[str]


class StabilityCfg(BaseModel):
    min_persistence_bars: int
    flicker_window: int
    flicker_threshold: int
    unstable_confidence_decay: float
    transition_size_cut: float = 0.25


class AllocationEntry(BaseModel):
    target_exposure: float
    leverage_cap: float
    requires_trend_confirmation: bool


class AllocationCfg(BaseModel):
    by_regime: dict[str, AllocationEntry]
    confidence_floor: float


class TrendConfirmationCfg(BaseModel):
    fast_sma: int
    slow_sma: int


class StrategyCfg(BaseModel):
    trend_confirmation: TrendConfirmationCfg
    low_vol_allocation: float = 0.95
    mid_vol_allocation_trend: float = 0.95
    mid_vol_allocation_no_trend: float = 0.60
    high_vol_allocation: float = 0.60
    low_vol_leverage: float = 1.25
    rebalance_threshold: float = 0.10
    uncertainty_size_mult: float = 0.50


class RiskCfg(BaseModel):
    # legacy
    per_trade_risk_pct: float
    leverage_cap: float
    daily_drawdown_warn_pct: float
    daily_drawdown_halt_pct: float
    weekly_drawdown_warn_pct: float
    total_drawdown_kill_pct: float
    kill_switch_path: str
    peak_equity_path: str
    max_position_correlation: float
    # final-prompt
    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.80
    max_leverage: float = 1.25
    max_single_position: float = 0.15
    max_sector_exposure: float = 0.30
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = 0.02
    daily_dd_halt: float = 0.03
    weekly_dd_reduce: float = 0.05
    weekly_dd_halt: float = 0.07
    max_dd_from_peak: float = 0.10
    gap_multiplier: float = 3.0
    gap_risk_budget: float = 0.02
    correlation_window: int = 60
    correlation_reduce: float = 0.70
    correlation_reject: float = 0.85
    max_spread_pct: float = 0.005
    duplicate_window_seconds: int = 60
    min_position_value: float = 100.0
    halt_lock_path: str = "state/trading_halted.lock"


class BacktestCfg(BaseModel):
    in_sample_days: int
    out_of_sample_days: int
    step_days: int
    slippage_bps: float
    commission_bps: float
    confidence_buckets: list[float]
    slippage_pct: float = 0.0005
    initial_capital: float = 100000.0
    train_window: int = 252
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = 0.045
    rebalance_threshold: float = 0.10
    random_seeds: int = 100


class BrokerCfg(BaseModel):
    require_demo: bool
    poll_account_seconds: int
    poll_positions_seconds: int
    paper_trading: bool = True
    timeframe: str = "1Day"
    order_type: str = "limit"
    limit_offset_pct: float = 0.001
    unfilled_cancel_seconds: int = 30
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    market_extended_hours: bool = True
    # Trade212 rejects order quantities with too many decimal places
    # ("invalid quantity precision"). Quantities are truncated to this many
    # decimals before submission. 1 is universally accepted.
    quantity_precision: int = 1
    cash_buffer_pct: float = 0.02


class DashboardCfg(BaseModel):
    refresh_seconds: int
    history_window_bars: int


class MonitoringCfg(BaseModel):
    dashboard_refresh_seconds: int = 5
    alert_rate_limit_minutes: int = 15
    log_dir: str = "logs"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 30
    state_dashboard_path: str = "state/dashboard.json"


class LoggingCfg(BaseModel):
    level: str
    json_format: bool = Field(alias="json")
    redact_env_keys: list[str]

    model_config = {"populate_by_name": True}


class Config(BaseModel):
    universe: UniverseCfg
    bars: BarsCfg
    features: FeaturesCfg
    hmm: HMMCfg
    regime_labels: RegimeLabelsCfg
    stability: StabilityCfg
    allocation: AllocationCfg
    strategy: StrategyCfg
    risk: RiskCfg
    backtest: BacktestCfg
    broker: BrokerCfg
    dashboard: DashboardCfg
    monitoring: MonitoringCfg = MonitoringCfg()
    logging: LoggingCfg


def project_root() -> Path:
    """Walk up from this file to find the repo root (contains `config/`)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "config" / "settings.yaml").exists():
            return parent
    return Path.cwd()


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("REGIME_TRADER_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with cfg_path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Config.model_validate(raw)
