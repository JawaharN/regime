"""Market data loader.

Tries tvdatafeed for OHLCV; if unavailable falls back to a parquet cache under
``data/cache/``. Both paths return a DataFrame indexed by UTC timestamp with
columns: open, high, low, close, volume.

The cache fallback lets paper-trade rehearsals and backtests run on whatever
history the user already fetched, without a hard network dependency at import
time. Gaps (weekends / holidays / halts) are tolerated — nothing is back-filled.

Authentication: anonymous tvdatafeed access is heavily rate-capped and returns
limited history. Set ``TV_SESSIONID`` in ``.env`` (the `sessionid` cookie from a
logged-in tradingview.com tab) and we exchange it for the websocket auth token
to lift those limits.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("regime_trader.market_data")

REQUIRED_COLS = ("open", "high", "low", "close", "volume")

# tvdatafeed needs the listing exchange, which varies across the universe
# (e.g. SPY is AMEX-listed, AAPL is NASDAQ). We probe in order and use the
# first exchange that returns data, so callers never have to know the venue.
_PROBE_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")

# Resolved TradingView auth tokens, keyed by sessionid (avoids re-scraping).
_TV_TOKEN_CACHE: dict[str, str] = {}


def _cache_path(symbol: str, interval: str) -> Path:
    return Path("data/cache") / f"{symbol}_{interval}.parquet"


def _resolve_tv_token(sessionid: str) -> str | None:
    """Exchange a TradingView ``sessionid`` cookie for the websocket auth token.

    tvdatafeed's username/password sign-in is broken upstream, so we scrape the
    JWT off the logged-in /chart/ page instead. Returns None if it can't.
    """
    if sessionid in _TV_TOKEN_CACHE:
        return _TV_TOKEN_CACHE[sessionid]
    try:
        import requests
        resp = requests.get(
            "https://www.tradingview.com/chart/",
            cookies={"sessionid": sessionid},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        match = re.search(r'"auth_token":"([^"]+)"', resp.text)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not resolve TV_SESSIONID to an auth token: %s", e)
        return None
    if not match:
        logger.warning("TV_SESSIONID did not yield an auth token — it may be expired")
        return None
    _TV_TOKEN_CACHE[sessionid] = match.group(1)
    return match.group(1)


def _apply_tv_session(tv) -> None:  # noqa: ANN001
    """Authenticate a TvDatafeed instance from TV_SESSIONID, if it is set."""
    sessionid = os.environ.get("TV_SESSIONID", "").strip()
    if not sessionid:
        logger.info("TV_SESSIONID not set — using anonymous tvdatafeed (rate-limited)")
        return
    token = _resolve_tv_token(sessionid)
    if token:
        tv.token = token
        logger.info("tvdatafeed authenticated via TV_SESSIONID")


def load_history(symbol: str, interval: str = "1d", years: int = 2) -> pd.DataFrame:
    """Return OHLCV history. Prefers tvdatafeed; falls back to parquet cache."""
    cache = _cache_path(symbol, interval)
    try:
        from tvDatafeed import TvDatafeed
        tv = TvDatafeed()
        _apply_tv_session(tv)
        tv_interval = _to_tv_interval(interval)
        n_bars = int(years * _bars_per_year(interval))
        df = None
        for exchange in _PROBE_EXCHANGES:
            candidate = tv.get_hist(symbol=symbol, exchange=exchange,
                                    interval=tv_interval, n_bars=n_bars)
            if candidate is not None and not candidate.empty:
                logger.info("tvdatafeed: %s found on %s (%d bars)",
                            symbol, exchange, len(candidate))
                df = candidate
                break
        if df is None:
            raise RuntimeError(
                f"tvdatafeed returned no data for {symbol} on any of {_PROBE_EXCHANGES}")
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
