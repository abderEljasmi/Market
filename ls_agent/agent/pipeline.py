"""One hourly run: data -> factors -> sentiment -> gate -> log -> report."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

from . import indicators as ind
from . import news, notify, sentiment, store
from .config import bars_per_day, config_hash, is_crypto
from .data_prices import TZ, fetch_prices
from .scoring import decide
from .universe import resolve_universe

log = logging.getLogger(__name__)


def earnings_blackout(cfg: dict, tickers: list[str], today: date) -> set[str]:
    """Tickers with earnings within N days. Cached once per day (yfinance is slow)."""
    if cfg["data"]["provider"] != "yfinance" or cfg["gate"]["earnings_blackout_days"] <= 0:
        return set()
    cache = Path(cfg["paths"]["db"]).parent / f"earnings_{today.isoformat()}.json"
    if cache.exists():
        dates = json.loads(cache.read_text())
    else:
        import yfinance as yf
        dates = {}
        for t in tickers:
            try:
                cal = yf.Ticker(t).calendar or {}
                ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
                if ed:
                    dates[t] = [str(d) for d in (ed if isinstance(ed, list) else [ed])]
            except Exception as e:
                log.debug("calendar failed for %s: %s", t, e)
        cache.write_text(json.dumps(dates))
    n = cfg["gate"]["earnings_blackout_days"]
    out = set()
    for t, ds in dates.items():
        for d in ds:
            try:
                if abs((date.fromisoformat(d[:10]) - today).days) <= n:
                    out.add(t)
            except ValueError:
                pass
    return out


def bar_end(cfg: dict, bar_ts: pd.Timestamp) -> pd.Timestamp:
    """When the bar closed: +1h for crypto, capped at the 16:00 close for stocks."""
    if is_crypto(cfg):
        return bar_ts + pd.Timedelta(hours=1)
    return min(bar_ts + pd.Timedelta(hours=1), bar_ts.normalize() + pd.Timedelta(hours=16))


def in_event_blackout(cfg: dict, bar_end_ts: pd.Timestamp) -> bool:
    hours = cfg["gate"]["event_blackout_hours"]
    for e in cfg["gate"]["events_et"]:
        ev = pd.Timestamp(e, tz=TZ)
        if abs((ev - bar_end_ts).total_seconds()) <= hours * 3600:
            return True
    return False


def format_side(name: str, picks: pd.DataFrame, table: pd.DataFrame, candidate: str) -> list[str]:
    lines = [name]
    if picks.empty:
        near = table[table.candidate_side == candidate]
        if near.empty:
            near = table.sort_values("score", ascending=(candidate == "SHORT")).head(1)
        else:
            near = near.reindex(near.score.abs().sort_values(ascending=False).index).head(1)
        t = near.index[0]
        lines.append(f"  NEUTRAL - no conviction (closest: {t} {near.score.iloc[0]:+.2f}, blocked by: {near.blocked_by.iloc[0]})")
        return lines
    for i, (t, r) in enumerate(picks.iterrows(), 1):
        lines.append(f"  {i}. {t:<10} entry {r.price:<11.6g} conviction {r.conviction:>3}  score {r.score:+.2f}  [{r.why}]")
    return lines


def run_once(cfg: dict, now: pd.Timestamp | None = None) -> dict | None:
    now = now or pd.Timestamp.now(tz=TZ)
    universe = resolve_universe(cfg)
    vix_t = cfg.get("volatility_index")
    all_tickers = list(dict.fromkeys(universe + [cfg["benchmark"]] + ([vix_t] if vix_t else [])))

    prices = fetch_prices(cfg, all_tickers, cfg["data"]["live_lookback_days"], now=now)
    vix_df = prices.pop(vix_t, None) if vix_t else None
    missing = [t for t in universe if t not in prices]
    if missing:
        log.warning("missing price data for %d tickers: %s", len(missing), missing[:20])
    if not [t for t in universe if t in prices]:
        log.error("no price data at all (network blocked? wrong provider?); nothing to score")
        return None

    panels = ind.build_panels(prices, cfg["benchmark"], bars_per_day(cfg), universe)
    factors_panel = ind.technical_factors(panels)
    vix = vix_df["close"] if vix_df is not None else None
    regime = ind.regime_series(panels, vix, cfg["gate"]["vix_risk_off"])

    bar_ts = panels["close"].index[-1]
    version = cfg["strategy_version"]
    con = store.connect(cfg["paths"]["db"])
    if store.already_logged(con, str(bar_ts), version):
        log.info("bar %s already processed for %s; skipping (market closed or duplicate run)", bar_ts, version)
        return None

    factors = {k: v.iloc[-1] for k, v in factors_panel.items()}
    items = news.gather(cfg, universe)
    sent = sentiment.score_sentiment(cfg, items)
    if sent:
        factors["sentiment"] = pd.Series({t: s["value"] for t, s in sent.items()}).reindex(factors["trend"].index)

    decision = decide(
        factors, panels["rsi"].iloc[-1], panels["dollar_vol"].iloc[-1], regime.iloc[-1], cfg,
        blackout=earnings_blackout(cfg, universe, bar_ts.date()),
        event_blackout=in_event_blackout(cfg, bar_end(cfg, bar_ts)),
        sentiment_meta=sent,
    )

    vix_now = float(vix.reindex(panels["close"].index).ffill().iloc[-1]) if vix is not None else None
    run = {
        "run_id": uuid.uuid4().hex[:12], "run_ts": now.isoformat(), "bar_ts": str(bar_ts),
        "strategy_version": version, "config_hash": config_hash(cfg), "regime": regime.iloc[-1],
        "vix": vix_now, "llm_used": bool(sent), "n_news": sum(len(v) for v in items.values()),
        "status": "ok", "thresholds": decision["thresholds"],
    }
    factor_rows = pd.DataFrame(factors).round(3).to_dict(orient="index")
    closes = panels["close"].iloc[-1]
    store.log_run(con, run, decision["table"], closes, factor_rows)
    # entry reference = close of the last completed bar (what the 6-bar check measures from)
    for side in ("longs", "shorts"):
        decision[side] = decision[side].assign(price=closes.reindex(decision[side].index))
    picked = list(decision["longs"].index) + list(decision["shorts"].index)
    store.log_features(con, run["run_id"], {t: ind.snapshot(panels, t) for t in picked})

    vix_txt = f" (VIX {vix_now:.1f})" if vix_now and vix_now == vix_now else ""
    header = (f"=== Long/short agent | bar {bar_ts:%Y-%m-%d %H:%M} ET | {version} | {len(panels['close'].columns)} assets ===\n"
              f"Regime: {run['regime']}{vix_txt}"
              f" | sentiment: {'on, ' + str(len(sent)) + ' tickers' if sent else 'off'}")
    lines = [header]
    lines += format_side("LONG", decision["longs"], decision["table"], "LONG")
    lines += format_side("SHORT", decision["shorts"], decision["table"], "SHORT")
    lines.append(f"Entry = close of the {bar_ts:%H:%M} bar (the run happens a few minutes later, so live fills differ). "
                 f"Each pick is re-checked after {cfg['verify']['horizon_bars']} bars. "
                 "Research output, not financial advice.")
    report = "\n".join(lines)
    print(report)
    if picked:
        notify.send(report)

    out_dir = Path(cfg["paths"]["out_dir"])
    payload = {
        "run": run,
        "longs": decision["longs"].reset_index().to_dict(orient="records"),
        "shorts": decision["shorts"].reset_index().to_dict(orient="records"),
        "long_neutral": decision["longs"].empty, "short_neutral": decision["shorts"].empty,
    }
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "latest.txt").write_text(report)
    return payload
