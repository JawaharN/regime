"""Market data loader.

Tries tvdatafeed for OHLCV; if unavailable falls back to a parquet cache under
``data/cache/``. Both paths return a DataFrame indexed by UTC timestamp with
columns: open, high, low, close, volume.

The cache fallback lets paper-trade rehearsals and backtests run on whatever
history the user already fetched, without a hard network dependency at import
time. Gaps (weekends / holidays / halts) are tolerated — nothing is back-filled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("regime_trader.market_data")

REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _cache_path(symbol: str, interval: str) -> Path:
    return Path("data/cache") / f"{symbol}_{interval}.parquet"


def load_history(symbol: str, interval: str = "1d", years: int = 2) -> pd.DataFrame:
    """Return OHLCV history. Prefers tvdatafeed; falls back to parquet cache."""
    cache = _cache_path(symbol, interval)
    try:
        from tvDatafeed import TvDatafeed
        tv = TvDatafeed()
        tv_interval = _to_tv_interval(interval)
        n_bars = int(years * _bars_per_year(interval))
        df = tv.get_hist(symbol=symbol, exchange="NASDAQ", interval=tv_interval, n_bars=n_bars)
        if df is None or df.empty:
            raise RuntimeError("tvdatafeed returned empty frame")
        df = df.rename(columns=str.lower)[list(REQUIRED_COLS)]
        df.index = pd.to_datetime(df.index, utc=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        return df
    except Exception as e:
        logger.warning("tvdatafeed unavailable (%s); trying cache at %s", e, cache)
        if cache.exists():
            df = pd.read_parquet(cache)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise RuntimeError(f"cached file {cache} is missing columns {missing}") from e
            return df
        raise RuntimeError(
            f"No market data available for {symbol} @ {interval}. "
            f"Install tvdatafeed or pre-populate {cache}."
        ) from e


def latest_bars(symbol: str, interval: str, n_bars: int) -> pd.DataFrame:
    """Last n_bars for the runtime loop. Loads enough history for feature warmup."""
    needed = max(n_bars, 500)
    df = load_history(symbol, interval, years=_years_for(needed, interval))
    return df.iloc[-n_bars:]


def get_latest_bar(symbol: str, interval: str = "1d") -> pd.Series | None:
    """Most recent completed OHLCV bar, or None when no data is available."""
    try:
        df = latest_bars(symbol, interval, n_bars=1)
    except Exception as e:  # noqa: BLE001
        logger.warning("get_latest_bar(%s) failed: %s", symbol, e)
        return None
    return None if df.empty else df.iloc[-1]


def get_latest_quote(symbol: str, interval: str = "1d") -> float | None:
    """Best-effort latest price (close of the latest bar)."""
    bar = get_latest_bar(symbol, interval)
    return None if bar is None else float(bar["close"])


def get_snapshot(symbols: list[str], interval: str = "1d") -> dict[str, float]:
    """Latest price per symbol; symbols with no data are omitted."""
    out: dict[str, float] = {}
    for sym in symbols:
        q = get_latest_quote(sym, interval)
        if q is not None:
            out[sym] = q
    return out


def _bars_per_year(interval: str) -> float:
    return {
        "1d": 252,
        "5m": 252 * 78,
        "15m": 252 * 26,
        "1h": 252 * 6.5,
    }.get(interval, 252)


def _years_for(n_bars: int, interval: str) -> int:
    return max(1, int(n_bars / _bars_per_year(interval)) + 1)


def _to_tv_interval(interval: str):
    from tvDatafeed import Interval
    return {
        "1d": Interval.in_daily,
        "1h": Interval.in_1_hour,
        "15m": Interval.in_15_minute,
        "5m": Interval.in_5_minute,
    }[interval]
