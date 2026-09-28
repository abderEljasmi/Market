"""Composite score, factor agreement and the conviction gate.

The gate is deliberately strict: a ticker needs a strong composite score AND
several independent factors pointing the same way. When nothing clears the
bar on a side, that side is NEUTRAL. Neutral is a valid, frequent answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = ["trend", "momentum", "rel_strength", "volume", "trend_ext", "momentum_ext",
           "breakout", "flow", "reversion", "sentiment"]


def composite(factors: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted mean of available factors per ticker (weights renormalised if a factor is missing)."""
    df = pd.DataFrame({k: v for k, v in factors.items() if k in weights})
    w = pd.Series(weights)[df.columns]
    mask = df.notna()
    num = df.fillna(0).mul(w, axis=1).sum(axis=1)
    den = mask.mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    return (num / den).fillna(0)


def agreement(factors: dict[str, pd.Series], comp: pd.Series, min_abs: float) -> pd.Series:
    df = pd.DataFrame(factors)
    sign = np.sign(comp)
    agrees = (np.sign(df).eq(sign, axis=0)) & (df.abs() >= min_abs)
    return agrees.sum(axis=1)


def conviction(score: float, agree: int, n_avail: int) -> int:
    """0-100 label used for ranking and for the calibration check."""
    return int(round(100 * min(1.0, abs(score) / 0.7) * (agree / max(n_avail, 1))))


def explain(ticker: str, factors: dict[str, pd.Series], sentiment_meta: dict) -> str:
    parts = []
    for k in FACTORS:
        s = factors.get(k)
        if s is not None and ticker in s and pd.notna(s[ticker]):
            parts.append(f"{k} {s[ticker]:+.2f}")
    cat = sentiment_meta.get(ticker, {}).get("catalyst")
    if cat and cat.lower() != "none":
        parts.append(f"catalyst: {cat}")
    return ", ".join(parts)


def decide(factors: dict[str, pd.Series], rsi: pd.Series, dollar_vol: pd.Series, regime: str,
           cfg: dict, blackout: set[str] | None = None, event_blackout: bool = False,
           sentiment_meta: dict | None = None) -> dict:
    """Apply the conviction gate for one timestamp. Returns longs, shorts and every score."""
    g = cfg["gate"]
    blackout = blackout or set()
    sentiment_meta = sentiment_meta or {}
    comp = composite(factors, cfg["weights"])
    # only factors that carry weight count as votes, so unused indicators never change a decision
    voting = {k: v for k, v in factors.items() if k in cfg["weights"]}
    agree = agreement(voting, comp, g["factor_agree_min"])
    n_avail = pd.DataFrame(voting).notna().sum(axis=1)

    thr_long = g["min_abs_score"] * (g["regime_penalty"] if regime == "risk_off" else 1)
    thr_short = g["min_abs_score"] * (g["regime_penalty"] if regime == "risk_on" else 1)

    rows = []
    for t in comp.index:
        s, a = float(comp[t]), int(agree[t])
        reasons = []
        if event_blackout:
            reasons.append("event blackout")
        if t in blackout:
            reasons.append("earnings blackout")
        if pd.isna(dollar_vol.get(t)) or dollar_vol.get(t, 0) < g["min_dollar_volume"]:
            reasons.append("illiquid")
        if a < g["min_agreeing_factors"]:
            reasons.append(f"only {a} factors agree")
        side = None
        if s >= thr_long:
            side = "LONG"
            if rsi.get(t, 50) > g["max_rsi_long"]:
                reasons.append("overbought")
        elif s <= -thr_short:
            side = "SHORT"
            if rsi.get(t, 50) < g["min_rsi_short"]:
                reasons.append("oversold")
        else:
            reasons.append("score below threshold")
        rows.append({"ticker": t, "score": s, "agree": a, "n_factors": int(n_avail[t]),
                     "side": side if not reasons else None, "candidate_side": side,
                     "conviction": conviction(s, a, int(n_avail[t])),
                     "blocked_by": "; ".join(reasons), "why": explain(t, voting, sentiment_meta)})

    table = pd.DataFrame(rows).set_index("ticker")
    k = g["max_picks_per_side"]
    longs = table[table.side == "LONG"].sort_values("score", ascending=False).head(k)
    shorts = table[table.side == "SHORT"].sort_values("score").head(k)
    # Only the reported picks count as picks. Gate-passers ranked below the top k stay logged
    # (candidate_side + blocked_by) but must not inflate the scorecard or the outcome check.
    extra = table.index[table.side.notna() & ~table.index.isin(longs.index.union(shorts.index))]
    table.loc[extra, "blocked_by"] = f"ranked below top {k}"
    table.loc[extra, "side"] = None
    return {"longs": longs, "shorts": shorts, "table": table,
            "thresholds": {"long": thr_long, "short": thr_short}}
