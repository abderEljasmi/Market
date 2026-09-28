"""Price data layer.

Returns a dict {ticker: DataFrame[open, high, low, close, volume]} indexed by
tz-aware bar start time (America/New_York). Providers:
  - yfinance         : real hourly US stock bars
  - binance          : real hourly crypto bars (public market-data API, no key needed)
  - synthetic        : random-walk stocks with market-hour timestamps, for offline tests
  - synthetic_crypto : random-walk coins, 24/7 timestamps, for offline tests
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
FIELDS = ["open", "high", "low", "close", "volume"]
TZ = "America/New_York"
BINANCE_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]


def fetch_prices(cfg: dict, tickers: list[str], days: int, now: datetime | None = None) -> dict[str, pd.DataFrame]:
    provider = cfg["data"]["provider"]
    if provider == "yfinance":
        data = _fetch_yfinance(tickers, days, cfg["data"]["interval"])
    elif provider == "synthetic":
        data = _fetch_synthetic(tickers, days)
    elif provider == "binance":
        data = _fetch_binance(tickers, days)
    elif provider == "synthetic_crypto":
        data = _fetch_synthetic_crypto(tickers, days)
    else:
        raise ValueError(f"Unknown data provider: {provider}")
    crypto = provider in ("binance", "synthetic_crypto")
    return {t: drop_incomplete_bar(df, now, crypto=crypto) for t, df in data.items() if len(df)}


def drop_incomplete_bar(df: pd.DataFrame, now: datetime | None = None, bar=timedelta(hours=1),
                        crypto: bool = False) -> pd.DataFrame:
    """Only use completed bars, so live decisions match the backtest exactly."""
    if now is None:
        now = pd.Timestamp.now(tz=TZ)
    now = pd.Timestamp(now).tz_convert(TZ) if pd.Timestamp(now).tzinfo else pd.Timestamp(now).tz_localize(TZ)
    if crypto:
        return df[[ts + bar <= now for ts in df.index]]
    # the last session bar (15:30) only lasts 30 min, closing at 16:00
    ends = [min(ts + bar, ts.normalize() + pd.Timedelta(hours=16)) for ts in df.index]
    return df[[e <= now for e in ends]]


# ---------------------------------------------------------------- stocks (yfinance)
def _fetch_yfinance(tickers: list[str], days: int, interval: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers, period=f"{days}d", interval=interval, group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.rename(columns=str.lower)[FIELDS].dropna(subset=["close"])
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(TZ)
            out[t] = df
        except (KeyError, ValueError) as e:
            log.warning("No price data for %s (%s)", t, e)
    return out


# ---------------------------------------------------------------- crypto (Binance public API)
def _binance_get(path: str, params: dict | None = None, retries: int = 4):
    """GET with host fallback and back-off on rate limits (429/418)."""
    last = None
    for attempt in range(retries):
        for host in BINANCE_HOSTS:
            try:
                r = requests.get(host + path, params=params, timeout=20)
                if r.status_code in (429, 418):
                    time.sleep(2 ** attempt * 2)
                    continue
                if r.status_code in (451, 403):      # geo-blocked host, try the next one
                    last = RuntimeError(f"{host} returned {r.status_code}")
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = e
    raise RuntimeError(f"Binance request failed for {path}: {last}")


def _klines(symbol: str, days: int) -> pd.DataFrame:
    n_bars = days * 24
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - n_bars * 3600 * 1000
    rows: list = []
    while start_ms < end_ms:
        chunk = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": "1h",
                                                "startTime": start_ms, "limit": 1000})
        if not chunk:
            break
        rows += chunk
        start_ms = chunk[-1][0] + 3600 * 1000
        if len(chunk) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=FIELDS)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["ts"] + FIELDS
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True).dt.tz_convert(TZ)
    return df.astype(float)[~df.index.duplicated()]


def _fetch_binance(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    def one(sym: str):
        try:
            return sym, _klines(sym, days)
        except Exception as e:                       # one bad coin must not kill the run
            log.warning("No price data for %s (%s)", sym, e)
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, tickers))


# ---------------------------------------------------------------- synthetic (offline)
def market_hour_index(days: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Hourly bar starts 09:30..15:30 on weekdays, like yfinance."""
    end = (pd.Timestamp.now(tz=TZ) if end is None else end).normalize()
    sessions = pd.bdate_range(end - pd.Timedelta(days=days), end, tz=TZ)
    starts = [s + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h) for s in sessions for h in range(7)]
    return pd.DatetimeIndex(starts)


def _synthetic_frame(rng, idx, close, vol_lo=200_000, vol_hi=3_000_000) -> pd.DataFrame:
    noise = np.abs(rng.normal(0, 0.002, len(idx)))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, len(idx))),
        "high": close * (1 + noise),
        "low": close * (1 - noise),
        "close": close,
        "volume": rng.integers(vol_lo, vol_hi, len(idx)).astype(float),
    }, index=idx)


def _fetch_synthetic(tickers: list[str], days: int, seed: int = 7) -> dict[str, pd.DataFrame]:
    idx = market_hour_index(days, end=pd.Timestamp("2026-09-25", tz=TZ))
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0001, 0.004, len(idx))           # common market factor
    out = {}
    for t in tickers:
        if t.startswith("^"):                                # volatility index
            close = 18 + np.cumsum(rng.normal(0, 0.3, len(idx)))
            close = np.clip(close, 10, 60)
        else:
            beta = 0.6 + 0.8 * rng.random()
            drift = rng.normal(0, 0.0003)                      # some names trend
            rets = beta * market + drift + rng.normal(0, 0.006, len(idx))
            close = (50 + 400 * rng.random()) * np.exp(np.cumsum(rets))
        out[t] = _synthetic_frame(rng, idx, close)
    return out


def _fetch_synthetic_crypto(tickers: list[str], days: int, seed: int = 11) -> dict[str, pd.DataFrame]:
    idx = pd.date_range(end=pd.Timestamp("2026-09-25 12:00", tz=TZ), periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.006, len(idx))
    out = {}
    for t in tickers:
        beta = 0.7 + 1.0 * rng.random()
        drift = rng.normal(0, 0.0005)
        rets = beta * market + drift + rng.normal(0, 0.010, len(idx))
        close = (0.5 + 200 * rng.random()) * np.exp(np.cumsum(rets))
        out[t] = _synthetic_frame(rng, idx, close, 50_000, 5_000_000)
    return out
