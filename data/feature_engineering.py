"""Feature engineering for the HMM regime engine.

All transforms are **causal**: only bars at or before time t feed feature[t].
Nothing is back-filled or forward-shifted. The final-prompt feature set:

  ret_1 / ret_5 / ret_20   log returns over 1 / 5 / 20 bars
  realized_vol             rolling std of 1-bar log returns
  vol_ratio                short-window vol / long-window vol
  volume_z                 volume z-score over volume_zscore_window
  volume_trend             rolling linear slope of volume
  adx                      Average Directional Index (trend strength)
  sma_slope                slope of the 50-bar SMA, normalised by price
  rsi                      Relative Strength Index
  dist_sma_long            % distance of price from the 200-bar SMA
  roc_10 / roc_20          rate of change over 10 / 20 bars
  atr_norm                 ATR normalised by price

Every column is then z-scored over a long rolling window (default 252 bars,
``min_periods`` kept low so short histories stay usable). Z-scoring is causal —
the mean/std at t use only bars ≤ t. Warmup rows are NaN; callers drop them.

ADX / RSI / ROC are implemented in pure pandas so the package has no hard
dependency on the optional ``ta`` library.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical, ordered feature set the HMM consumes. build_features always emits
# exactly these columns.
FEATURE_COLUMNS = (
    "ret_1", "ret_5", "ret_20",
    "realized_vol", "vol_ratio",
    "volume_z", "volume_trend",
    "adx", "sma_slope", "rsi",
    "dist_sma_long", "roc_10", "roc_20", "atr_norm",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Identifies a feature configuration so trained models invalidate when it changes."""

    return_window: int
    vol_window: int
    volume_zscore_window: int
    atr_window: int
    range_expansion_window: int

    def hash(self) -> str:
        s = (f"rw={self.return_window};vw={self.vol_window};vzw={self.volume_zscore_window};"
             f"aw={self.atr_window};rew={self.range_expansion_window}")
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def feature_spec_from_cfg(cfg) -> FeatureSpec:  # noqa: ANN001
    return FeatureSpec(
        return_window=cfg.return_window,
        vol_window=cfg.vol_window,
        volume_zscore_window=cfg.volume_zscore_window,
        atr_window=cfg.atr_window,
        range_expansion_window=cfg.range_expansion_window,
    )


# --------------------------------------------------------------------- helpers

def _true_range(ohlcv: pd.DataFrame) -> pd.Series:
    prev_close = ohlcv["close"].shift(1)
    return pd.concat([
        (ohlcv["high"] - ohlcv["low"]).abs(),
        (ohlcv["high"] - prev_close).abs(),
        (ohlcv["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _adx(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    high, low = ohlcv["high"], ohlcv["low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = _true_range(ohlcv)
    atr = tr.rolling(window).mean()
    plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling OLS slope of `series` against an index 0..window-1."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = float((x_dev ** 2).sum())

    def _slope(vals: np.ndarray) -> float:
        y_dev = vals - vals.mean()
        return float((x_dev * y_dev).sum() / denom) if denom else 0.0

    return series.rolling(window).apply(_slope, raw=True)


def _zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


# ------------------------------------------------------------------ public API

def build_features(ohlcv: pd.DataFrame, cfg) -> pd.DataFrame:  # noqa: ANN001 (cfg = FeaturesCfg)
    """Compute the causal, z-scored feature matrix.

    Returns a DataFrame indexed identically to `ohlcv` with columns
    FEATURE_COLUMNS. Warmup rows contain NaN — callers drop them.
    """
    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float)

    ret_w = list(cfg.ret_windows) if len(cfg.ret_windows) >= 3 else [1, 5, 20]
    roc_w = list(cfg.roc_windows) if len(cfg.roc_windows) >= 2 else [10, 20]

    log_close = np.log(close)
    raw = pd.DataFrame(index=ohlcv.index)
    raw["ret_1"] = log_close.diff(ret_w[0])
    raw["ret_5"] = log_close.diff(ret_w[1])
    raw["ret_20"] = log_close.diff(ret_w[2])

    one_bar_ret = log_close.diff(1)
    raw["realized_vol"] = one_bar_ret.rolling(cfg.realized_vol_window).std()
    vol_fast = one_bar_ret.rolling(cfg.vol_ratio_fast).std()
    vol_slow = one_bar_ret.rolling(cfg.vol_ratio_slow).std()
    raw["vol_ratio"] = vol_fast / vol_slow.replace(0, np.nan)

    v_mean = volume.rolling(cfg.volume_zscore_window).mean()
    v_std = volume.rolling(cfg.volume_zscore_window).std()
    raw["volume_z"] = (volume - v_mean) / v_std.replace(0, np.nan)
    raw["volume_trend"] = _rolling_slope(volume, cfg.volume_trend_window) / v_mean.replace(0, np.nan)

    raw["adx"] = _adx(ohlcv, cfg.adx_window)

    sma_fast = close.rolling(cfg.sma_slope_window).mean()
    raw["sma_slope"] = sma_fast.diff(5) / close
    raw["rsi"] = _rsi(close, cfg.rsi_window)

    sma_long = close.rolling(cfg.sma_long_window).mean()
    raw["dist_sma_long"] = (close - sma_long) / sma_long.replace(0, np.nan)

    raw["roc_10"] = close.pct_change(roc_w[0])
    raw["roc_20"] = close.pct_change(roc_w[1])

    atr = _true_range(ohlcv).rolling(cfg.atr_window).mean()
    raw["atr_norm"] = atr / close

    out = pd.DataFrame(index=ohlcv.index)
    for col in FEATURE_COLUMNS:
        out[col] = _zscore(raw[col], cfg.zscore_window, cfg.zscore_min_periods)
    return out[list(FEATURE_COLUMNS)]
