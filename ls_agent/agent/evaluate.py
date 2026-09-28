"""Scorecard + reliability verdict, shared by the backtest and the live log.

Input: one row per pick with columns
    bar_ts, side (LONG/SHORT), conviction, net (side-adjusted return after costs),
    excess (side-adjusted return vs universe average, after costs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return from close of bar t to close of bar t+h (NaN when not yet known)."""
    return close.shift(-h) / close - 1


def attach_returns(picks: pd.DataFrame, close: pd.DataFrame, h: int, cost_bps: float) -> pd.DataFrame:
    """Add net and excess returns to picks (needs bar_ts as Timestamp and ticker)."""
    fwd = forward_returns(close, h)
    univ = fwd.mean(axis=1)
    rets, excess = [], []
    for _, p in picks.iterrows():
        ts = p["bar_ts"]
        if ts not in fwd.index or p["ticker"] not in fwd.columns:
            rets.append(np.nan); excess.append(np.nan); continue
        r, u = fwd.at[ts, p["ticker"]], univ.at[ts]
        sign = 1 if p["side"] == "LONG" else -1
        rets.append(sign * r - cost_bps / 1e4)
        excess.append(sign * (r - u) - cost_bps / 1e4)
    out = picks.copy()
    out["net"], out["excess"] = rets, excess
    return out.dropna(subset=["net"])


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    return float((1 - equity / equity.cummax()).max())


def scorecard(picks: pd.DataFrame, h: int, n_runs: int | None = None) -> dict:
    if picks.empty:
        return {"n_picks": 0}
    p = picks.copy()
    p["bar_ts"] = pd.to_datetime(p["bar_ts"], utc=True)
    span_weeks = (p["bar_ts"].max() - p["bar_ts"].min()).days / 7
    card = {
        "n_picks": len(p),
        "weeks": round(span_weeks, 1),
        "hit_rate": round(float((p["net"] > 0).mean()), 3),
        "mean_net_bps": round(float(p["net"].mean() * 1e4), 1),
        "mean_edge_bps": round(float(p["excess"].mean() * 1e4), 1),
        "long_edge_bps": round(float(p.loc[p.side == "LONG", "excess"].mean() * 1e4), 1) if (p.side == "LONG").any() else None,
        "short_edge_bps": round(float(p.loc[p.side == "SHORT", "excess"].mean() * 1e4), 1) if (p.side == "SHORT").any() else None,
    }
    # calibration: do higher-conviction picks actually do better?
    if p["conviction"].nunique() >= 3 and len(p) >= 30:
        buckets = pd.qcut(p["conviction"].rank(method="first"), 3, labels=["low", "mid", "high"])
        by = p.groupby(buckets, observed=True)["excess"].mean() * 1e4
        card["calibration_bps"] = {k: round(float(v), 1) for k, v in by.items()}
        card["calibrated"] = bool(by["high"] > by["low"])
    # drawdown on non-overlapping runs (every h-th decision bar), equal weight per run
    per_run = p.groupby("bar_ts")["excess"].mean().sort_index()
    card["max_drawdown"] = round(max_drawdown(per_run.iloc[::h]), 3)
    if n_runs:
        card["neutral_rate"] = round(1 - per_run.size / n_runs, 3)
    return card


def verdict(card: dict, cfg: dict) -> tuple[bool, list[str]]:
    r = cfg["reliability"]
    checks = [
        (card.get("n_picks", 0) >= r["min_picks"], f"picks {card.get('n_picks', 0)} / {r['min_picks']}"),
        (card.get("weeks", 0) >= r["min_weeks"], f"weeks {card.get('weeks', 0)} / {r['min_weeks']}"),
        (card.get("hit_rate", 0) >= r["min_hit_rate"], f"hit rate {card.get('hit_rate')} / {r['min_hit_rate']}"),
        (card.get("mean_edge_bps", -1e9) >= r["min_edge_bps"], f"edge {card.get('mean_edge_bps')} bps / {r['min_edge_bps']}"),
        (card.get("max_drawdown", 1) <= r["max_drawdown"], f"drawdown {card.get('max_drawdown')} / {r['max_drawdown']}"),
        (card.get("calibrated", False), "conviction is calibrated (high > low)"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return (not failed), failed


def print_card(title: str, card: dict, cfg: dict) -> None:
    print(f"\n--- {title} ---")
    for k, v in card.items():
        print(f"  {k:<16} {v}")
    ok, failed = verdict(card, cfg)
    print(f"  VERDICT          {'RELIABLE (by your thresholds)' if ok else 'NOT YET'}")
    for f in failed:
        print(f"    x {f}")
