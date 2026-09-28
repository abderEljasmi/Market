"""Technical indicators and factor panels.

Everything here is CAUSAL: the value at bar t only uses bars <= t.
That lets the live agent (last row) and the backtest (every row) share
exactly the same code, which is the main defence against backtest/live drift.

Panels are DataFrames indexed by bar time with one column per ticker.

Indicators are grouped into factors, each scaled to [-1, 1]:
  trend        EMA20/EMA50 gap + slope (ATR units)
  momentum     1-day return + MACD histogram, ranked across the universe
  rel_strength 5-day return minus benchmark, ranked across the universe
  volume       relative volume signed by bar direction
  trend_ext    ADX/DI direction, Aroon oscillator, Ichimoku cloud, distance to EMA200
  momentum_ext short/1-day rate of change, TRIX
  breakout     Donchian channel position, Bollinger %B (continuation reading)
  flow         OBV slope, Chaikin money flow, MFI, rolling VWAP deviation
  reversion    contrarian brake: Bollinger/CCI/Williams %R/StochRSI extremes
Only factors listed in config `weights` influence the score, so old configs
keep their exact behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BARS_PER_DAY = 7      # default (US stocks); crypto passes 24 through build_panels(bpd=...)


# ---------- single-series indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd_hist(close: pd.Series) -> pd.Series:
    line = ema(close, 12) - ema(close, 26)
    return line - ema(line, 9)


def bollinger(close: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """(%B scaled so +-1 = the bands, bandwidth as fraction of price)."""
    ma, sd = close.rolling(n).mean(), close.rolling(n).std()
    return (close - ma) / (2 * sd.replace(0, np.nan)), 4 * sd / ma


def stoch_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    ll, hh = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def stoch_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    r = rsi(close, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    return 100 * (r - lo) / (hi - lo).replace(0, np.nan)


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    ll, hh = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def _rolling_mad(s: pd.Series, n: int) -> pd.Series:
    """Mean absolute deviation over a rolling window (vectorised, causal)."""
    out = np.full(len(s), np.nan)
    if len(s) >= n:
        w = sliding_window_view(s.to_numpy(dtype=float), n)
        out[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=s.index)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp - tp.rolling(n).mean()) / (0.015 * _rolling_mad(tp, n).replace(0, np.nan))


def adx_di(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(ADX, +DI, -DI)."""
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def aroon_osc(df: pd.DataFrame, n: int = 25) -> pd.Series:
    """Aroon up minus down, -100..100: how recent the highest high / lowest low is."""
    out = np.full(len(df), np.nan)
    if len(df) > n:
        hi = sliding_window_view(df["high"].to_numpy(dtype=float), n + 1)
        lo = sliding_window_view(df["low"].to_numpy(dtype=float), n + 1)
        up = 100 * hi.argmax(axis=1) / n
        dn = 100 * lo.argmin(axis=1) / n
        out[n:] = up - dn
    return pd.Series(out, index=df.index)


def obv_slope(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Net signed volume over n bars as a fraction of total volume, -1..1."""
    obv = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()
    return (obv - obv.shift(n)) / df["volume"].rolling(n).sum().replace(0, np.nan)


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (mfm.fillna(0) * df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    pos = flow.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def trix(close: pd.Series, n: int = 15) -> pd.Series:
    e3 = close.ewm(span=n, adjust=False).mean().ewm(span=n, adjust=False).mean().ewm(span=n, adjust=False).mean()
    return e3.pct_change()


def donchian_pos(df: pd.DataFrame, n: int = 48) -> pd.Series:
    """Where the close sits inside the n-bar high/low channel, -1 (low) to +1 (high)."""
    ll, hh = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    return 2 * (df["close"] - ll) / (hh - ll).replace(0, np.nan) - 1


def ichimoku_raw(df: pd.DataFrame, a: pd.Series) -> pd.Series:
    """Cloud position + tenkan/kijun gap in ATR units (cloud not shifted forward: causal)."""
    h, l, c = df["high"], df["low"], df["close"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    top, bot = pd.concat([span_a, span_b], axis=1).max(axis=1), pd.concat([span_a, span_b], axis=1).min(axis=1)
    cloud = np.where(c > top, c - top, np.where(c < bot, c - bot, 0.0))
    return (pd.Series(cloud, index=c.index) + (tenkan - kijun)) / a


def rolling_vwap_dev(df: pd.DataFrame, n: int, a: pd.Series) -> pd.Series:
    vwap = (df["close"] * df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum().replace(0, np.nan)
    return (df["close"] - vwap) / a


def _mean_of(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """Row/column-wise mean ignoring NaN (a young coin may miss the slow indicators)."""
    vals = sum(p.fillna(0) for p in panels)
    cnt = sum(p.notna().astype(float) for p in panels)
    return (vals / cnt.replace(0, np.nan)).fillna(0)


# ---------- panels ----------
RAW = ("trend_raw", "mom_raw", "rs_raw", "vol_raw", "rsi", "atr_pct", "dollar_vol",
       "adx", "di_diff", "aroon", "ema200_dist", "ichimoku", "roc_short", "roc_day", "trix",
       "donchian", "pctb", "bb_width", "obv_slope", "cmf", "mfi", "vwap_dev",
       "stoch_k", "stoch_rsi", "cci", "willr")


def build_panels(prices: dict[str, pd.DataFrame], benchmark: str, bpd: int = BARS_PER_DAY,
                 universe: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Raw per-ticker features aligned on a common time index.

    The benchmark is excluded from the scored tickers unless it is in `universe`
    (BTC is both the crypto benchmark and a coin we want to score).
    """
    close = pd.DataFrame({t: df["close"] for t, df in prices.items()}).sort_index().ffill()
    if universe is None:
        tickers = [t for t in close.columns if t != benchmark and not t.startswith("^")]
    else:
        tickers = [t for t in close.columns if t in universe]
    bench = close[benchmark] if benchmark in close else close[tickers].mean(axis=1)

    feats = {k: {} for k in RAW}
    bench_ret_5d = bench.pct_change(5 * bpd)
    for t in tickers:
        df = prices[t].reindex(close.index).ffill()
        c = df["close"]
        a = atr(df)
        # trend: distance between fast and slow EMA in ATR units, plus slope of slow EMA
        e20, e50 = ema(c, 20), ema(c, 50)
        feats["trend_raw"][t] = (e20 - e50) / a + 0.5 * (e50 - e50.shift(bpd)) / a
        # momentum: 1-day return in ATR units + MACD histogram in ATR units
        feats["mom_raw"][t] = (c - c.shift(bpd)) / a + macd_hist(c) / a
        # relative strength vs benchmark over 5 days
        feats["rs_raw"][t] = c.pct_change(5 * bpd) - bench_ret_5d
        # volume: relative volume signed by the direction of the bar
        rvol = df["volume"] / df["volume"].rolling(20 * bpd, min_periods=20).mean()
        feats["vol_raw"][t] = np.sign(c.diff()) * (rvol - 1).clip(-1, 3)
        feats["rsi"][t] = rsi(c)
        feats["atr_pct"][t] = a / c
        feats["dollar_vol"][t] = (df["volume"] * c).rolling(20 * bpd, min_periods=20).mean()
        # extended indicators
        adx, pdi, mdi = adx_di(df)
        feats["adx"][t], feats["di_diff"][t] = adx, pdi - mdi
        feats["aroon"][t] = aroon_osc(df)
        feats["ema200_dist"][t] = (c - ema(c, 200)) / a
        feats["ichimoku"][t] = ichimoku_raw(df, a)
        ap = a / c
        feats["roc_short"][t] = c.pct_change(6) / (ap * np.sqrt(6))
        feats["roc_day"][t] = c.pct_change(bpd) / (ap * np.sqrt(bpd))
        feats["trix"][t] = trix(c)
        feats["donchian"][t] = donchian_pos(df)
        feats["pctb"][t], feats["bb_width"][t] = bollinger(c)
        feats["obv_slope"][t] = obv_slope(df)
        feats["cmf"][t] = cmf(df)
        feats["mfi"][t] = mfi(df)
        feats["vwap_dev"][t] = rolling_vwap_dev(df, bpd, a)
        feats["stoch_k"][t] = stoch_k(df)
        feats["stoch_rsi"][t] = stoch_rsi(c)
        feats["cci"][t] = cci(df)
        feats["willr"][t] = williams_r(df)

    panels = {k: pd.DataFrame(v) for k, v in feats.items()}
    panels["close"] = close[tickers]
    panels["bench_close"] = bench.to_frame("bench")
    panels["_high"] = pd.DataFrame({t: prices[t]["high"].reindex(close.index).ffill() for t in tickers})
    panels["_low"] = pd.DataFrame({t: prices[t]["low"].reindex(close.index).ffill() for t in tickers})
    return panels


def xs_zscore(panel: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per row, scaled to [-1, 1]."""
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    z = panel.sub(mu, axis=0).div(sd, axis=0)
    return (z.clip(-clip, clip) / clip).fillna(0)


def _stretch(x: pd.DataFrame, thr: float, scale: float) -> pd.DataFrame:
    """Contrarian signal: 0 inside +-thr, ramps to -+1 as x runs beyond it."""
    return -np.sign(x) * ((x.abs() - thr) / scale).clip(0, 1)


def technical_factors(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Map raw features to factor scores in [-1, 1]."""
    p = panels
    adx_dir = np.sign(p["di_diff"]) * ((p["adx"] - 15) / 25).clip(0, 1)
    trend_ext = _mean_of([adx_dir, p["aroon"] / 100, np.tanh(p["ichimoku"] / 2), np.tanh(p["ema200_dist"] / 3)])
    momentum_ext = _mean_of([np.tanh(p["roc_short"] / 2), np.tanh(p["roc_day"] / 2), xs_zscore(p["trix"])])
    breakout = _mean_of([p["donchian"], p["pctb"].clip(-1, 1)])
    flow = _mean_of([p["obv_slope"].clip(-1, 1), (2 * p["cmf"]).clip(-1, 1), (p["mfi"] - 50) / 50,
                     np.tanh(p["vwap_dev"] / 2)])
    reversion = _mean_of([_stretch(p["pctb"], 1.0, 0.5), _stretch(p["cci"], 100, 100),
                          _stretch((p["willr"] + 50) / 50, 0.6, 0.4), _stretch((p["stoch_rsi"] - 50) / 50, 0.6, 0.4)])
    return {
        "trend": np.tanh(p["trend_raw"] / 2).fillna(0),
        "momentum": xs_zscore(p["mom_raw"]),
        "rel_strength": xs_zscore(p["rs_raw"]),
        "volume": np.tanh(p["vol_raw"]).fillna(0),
        "trend_ext": trend_ext,
        "momentum_ext": momentum_ext,
        "breakout": breakout,
        "flow": flow,
        "reversion": reversion,
    }


def snapshot(panels: dict[str, pd.DataFrame], ticker: str, loc=-1) -> dict[str, float]:
    """Raw indicator values for one ticker at one bar, for the post-mortem log."""
    out = {}
    for k in RAW:
        v = panels[k][ticker].iloc[loc] if isinstance(loc, int) else panels[k][ticker].loc[loc]
        if pd.notna(v):
            out[k] = round(float(v), 4)
    return out


def regime_series(panels: dict[str, pd.DataFrame], vix: pd.Series | None, vix_risk_off: float) -> pd.Series:
    """risk_on / neutral / risk_off per bar, from benchmark trend + VIX."""
    b = panels["bench_close"]["bench"]
    up = (b > ema(b, 50)) & (ema(b, 20) > ema(b, 50))
    down = (b < ema(b, 50)) & (ema(b, 20) < ema(b, 50))
    reg = pd.Series("neutral", index=b.index)
    reg[up] = "risk_on"
    reg[down] = "risk_off"
    if vix is not None and len(vix):
        v = vix.reindex(b.index).ffill()
        reg[v >= vix_risk_off] = "risk_off"
    return reg
