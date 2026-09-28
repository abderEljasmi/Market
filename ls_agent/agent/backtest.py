"""Walk-forward backtest of the TECHNICAL core with the exact live gate.

Sentiment is not backtested: historical news/X data is not available point-in-time
here, and using today's news feed on old bars would leak the future. So the
backtest answers one question: does the technical core + gate have an edge by
itself? Sentiment has to prove its value in paper trading (compare versions).

Data is split by time: the first part is for tuning, the last part (holdout)
stays untouched until you are done tuning.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import evaluate as ev
from . import indicators as ind
from .config import bars_per_day
from .data_prices import fetch_prices
from .pipeline import bar_end, in_event_blackout
from .scoring import decide
from .universe import resolve_universe

log = logging.getLogger(__name__)


def run_backtest(cfg: dict, holdout_frac: float = 0.3, reveal_holdout: bool = False, step: int = 1) -> dict:
    universe = resolve_universe(cfg)
    vix_t = cfg.get("volatility_index")
    tickers = list(dict.fromkeys(universe + [cfg["benchmark"]] + ([vix_t] if vix_t else [])))
    bpd = bars_per_day(cfg)
    prices = fetch_prices(cfg, tickers, cfg["data"]["backtest_lookback_days"])
    vix_df = prices.pop(vix_t, None) if vix_t else None
    panels = ind.build_panels(prices, cfg["benchmark"], bpd, universe)
    fpanel = ind.technical_factors(panels)
    regime = ind.regime_series(panels, vix_df["close"] if vix_df is not None else None, cfg["gate"]["vix_risk_off"])

    h = cfg["evaluation"]["primary_horizon_bars"]
    idx = panels["close"].index
    warmup = 20 * bpd
    decision_bars = idx[warmup:len(idx) - max(cfg["evaluation"]["horizons_bars"])][::step]
    log.info("backtest: %d decision bars, %d tickers", len(decision_bars), panels["close"].shape[1])

    rows = []
    for ts in decision_bars:
        factors = {k: v.loc[ts] for k, v in fpanel.items()}
        d = decide(factors, panels["rsi"].loc[ts], panels["dollar_vol"].loc[ts], regime.loc[ts], cfg,
                   event_blackout=in_event_blackout(cfg, bar_end(cfg, ts)))
        for side_df in (d["longs"], d["shorts"]):
            for t, r in side_df.iterrows():
                rows.append({"bar_ts": ts, "ticker": t, "side": r.side, "conviction": r.conviction, "score": r.score})

    picks = pd.DataFrame(rows, columns=["bar_ts", "ticker", "side", "conviction", "score"])
    split = decision_bars[int(len(decision_bars) * (1 - holdout_frac))] if len(decision_bars) else None
    cost = cfg["evaluation"]["cost_bps_round_trip"]
    results = {}

    for name, sel, n_runs in (
        ("train", picks.bar_ts < split, int((decision_bars < split).sum())),
        ("holdout", picks.bar_ts >= split, int((decision_bars >= split).sum())),
    ):
        sub = ev.attach_returns(picks[sel], panels["close"], h, cost)
        results[name] = ev.scorecard(sub, h, n_runs=n_runs)
        # other horizons, for information only
        results[name]["edge_by_horizon_bps"] = {
            hh: round(float(ev.attach_returns(picks[sel], panels["close"], hh, cost)["excess"].mean() * 1e4), 1)
            for hh in cfg["evaluation"]["horizons_bars"]} if sel.any() else {}

    ev.print_card(f"BACKTEST train (technical only, {cfg['strategy_version']}, horizon {h} bars)",
                  results["train"], cfg)
    if reveal_holdout:
        ev.print_card("BACKTEST holdout - look only when tuning is finished", results["holdout"], cfg)
    else:
        print("\n(holdout hidden: rerun with --reveal-holdout once you stop tuning)")
    return results
