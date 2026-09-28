"""Outcome check: N bars after every pick, did the call work, and if not, why?

`verify_due` runs after each hourly run (and via `python main.py verify`):
  1. finds logged picks whose horizon (default 6 bars ~ 6 hours) has completed,
  2. compares entry close with the close N bars later, side-adjusted, after costs,
  3. records max favourable / adverse excursion inside the window,
  4. for failures, tags the likely cause (see `diagnose`) from what was logged at pick time.

`improve` aggregates all resolved outcomes into a post-mortem report and a SUGGESTED config.
It never edits your live config: automatically re-tuning on a few dozen trades mostly
fits noise, and it would also break the "one change per version" comparison that
tells you whether a change helped. You (or Claude in a session) review and apply it.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import notify, store
from .data_prices import TZ, fetch_prices

log = logging.getLogger(__name__)

REASONS = {
    "cost_drag": "direction was right but the move was smaller than trading costs",
    "reversal_after_favorable_move": "was in profit inside the window, then reversed (horizon too long / no exit rule)",
    "market_headwind": "the pick beat the benchmark but the market moved against the side",
    "counter_regime": "traded against the market regime",
    "stretched_entry": "entered an already stretched move (chasing)",
    "choppy_market": "ADX < 20: no real trend, trend-following signals are unreliable",
    "noise": "move stayed within a normal 1-ATR wiggle; no real signal either way",
    "signal_wrong": "price moved decisively against the call",
}


def _sign(side: str) -> int:
    return 1 if side == "LONG" else -1


def diagnose(side: str, o: dict, feats: dict, factors: dict, regime: str, bench_ret: float,
             cost: float, horizon: int) -> tuple[str, dict]:
    """Return (primary reason, detail). Rules are transparent on purpose: no black box."""
    sg = _sign(side)
    flags = []
    atr_pct = feats.get("atr_pct")
    expected_move = atr_pct * math.sqrt(horizon) if atr_pct else None    # typical move over the window

    if o["direction_ok"] and o["net"] <= 0:
        flags.append("cost_drag")
    if o["mfe"] >= max(2 * cost, 0.0005) and o["net"] <= 0:
        flags.append("reversal_after_favorable_move")
    if o["excess"] > 0 and o["net"] <= 0:
        flags.append("market_headwind")
    if (regime == "risk_on" and side == "SHORT") or (regime == "risk_off" and side == "LONG"):
        flags.append("counter_regime")
    rsi = feats.get("rsi", 50)
    pctb = feats.get("pctb", 0)
    if (sg == 1 and (rsi >= 68 or pctb > 1)) or (sg == -1 and (rsi <= 32 or pctb < -1)):
        flags.append("stretched_entry")
    if feats.get("adx", 99) < 20:
        flags.append("choppy_market")
    if expected_move and abs(o["ret"]) < expected_move:
        flags.append("noise")
    if not flags:
        flags.append("signal_wrong")

    misleading = sorted(((k, v) for k, v in factors.items() if v is not None and np.sign(v) == sg and abs(v) >= 0.1),
                        key=lambda kv: -abs(kv[1]))
    detail = {"flags": flags, "bench_ret": round(bench_ret, 5),
              "move_in_atr": round(abs(o["ret"]) / expected_move, 2) if expected_move else None,
              "factors_that_agreed": {k: round(v, 2) for k, v in misleading}}
    return flags[0], detail


def _resolve(cfg: dict, pend: pd.DataFrame, prices: dict[str, pd.DataFrame], h: int, cost: float,
             now: pd.Timestamp) -> list[dict]:
    bench_t = cfg["benchmark"]
    close = pd.DataFrame({t: d["close"] for t, d in prices.items()}).sort_index().ffill()
    hi = pd.DataFrame({t: d["high"] for t, d in prices.items()}).sort_index().ffill()
    lo = pd.DataFrame({t: d["low"] for t, d in prices.items()}).sort_index().ffill()
    out = []
    for _, p in pend.iterrows():
        ts = pd.Timestamp(p["bar_ts"]).tz_convert(TZ)
        if p["ticker"] not in close or ts not in close.index:
            continue
        pos = close.index.get_loc(ts)
        if pos + h > len(close.index) - 1:
            continue                                          # horizon not complete yet
        t = p["ticker"]
        sg = _sign(p["side"])
        entry, exit_ = float(close[t].iloc[pos]), float(close[t].iloc[pos + h])
        win_hi, win_lo = float(hi[t].iloc[pos + 1:pos + h + 1].max()), float(lo[t].iloc[pos + 1:pos + h + 1].min())
        ret = exit_ / entry - 1
        bench_ret = float(close[bench_t].iloc[pos + h] / close[bench_t].iloc[pos] - 1) if bench_t in close else 0.0
        mfe = (win_hi / entry - 1) if sg == 1 else (1 - win_lo / entry)
        mae = (win_lo / entry - 1) if sg == 1 else (1 - win_hi / entry)
        o = {"run_id": p["run_id"], "ticker": t, "horizon": h, "side": p["side"], "bar_ts": p["bar_ts"],
             "strategy_version": p["strategy_version"], "entry": entry, "exit": exit_, "ret": ret,
             "net": sg * ret - cost, "excess": sg * (ret - bench_ret) - cost, "mfe": mfe, "mae": mae,
             "direction_ok": sg * ret > 0, "success": sg * ret - cost > 0, "checked_ts": now.isoformat()}
        if o["success"]:
            o["reason"], o["detail"] = "success", {}
        else:
            feats = json.loads(p["features"]) if isinstance(p["features"], str) else {}
            factors = json.loads(p["factors"]) if isinstance(p["factors"], str) else {}
            o["reason"], o["detail"] = diagnose(p["side"], o, feats, factors, p["regime"], bench_ret, cost, h)
        out.append(o)
    return out


def format_outcome(o: dict) -> str:
    mark = "OK  " if o["success"] else "MISS"
    line = (f"{mark} {o['side']:<5} {o['ticker']:<12} {o['horizon']}h: {o['ret'] * _sign(o['side']):+.2%} for the side "
            f"(net {o['net']:+.2%}, best {o['mfe']:+.2%}, worst {o['mae']:+.2%})")
    if not o["success"]:
        line += f"\n      why: {o['reason']} - {REASONS.get(o['reason'], '')}"
        agreed = o["detail"].get("factors_that_agreed")
        if agreed:
            line += "\n      factors that had agreed with the call: " + ", ".join(f"{k} {v:+.2f}" for k, v in agreed.items())
    return line


def verify_due(cfg: dict, now: pd.Timestamp | None = None) -> list[dict]:
    """Resolve every pick whose horizon has completed. Returns the new outcomes."""
    now = now or pd.Timestamp.now(tz=TZ)
    h = cfg["verify"]["horizon_bars"]
    cost = cfg["evaluation"]["cost_bps_round_trip"] / 1e4
    con = store.connect(cfg["paths"]["db"])
    pend = store.load_unverified(con, h)
    if pend.empty:
        return []
    age_h = (now - pd.to_datetime(pend["bar_ts"], utc=True).dt.tz_convert(TZ)).dt.total_seconds() / 3600
    pend = pend[age_h >= h]
    if pend.empty:
        return []

    days = int((now - pd.to_datetime(pend["bar_ts"], utc=True).min()).days) + 3
    tickers = sorted(set(pend["ticker"]) | {cfg["benchmark"]})
    prices = fetch_prices(cfg, tickers, max(days, 3), now=now)
    outs = _resolve(cfg, pend, prices, h, cost, now)
    for o in outs:
        store.save_outcome(con, o)
    if outs:
        ok = sum(o["success"] for o in outs)
        text = "\n".join([f"=== {h}-bar outcome check | {len(outs)} picks resolved, {ok} worked ==="] +
                         [format_outcome(o) for o in sorted(outs, key=lambda o: o["success"])])
        print(text)
        notify.send(text)
        out_dir = Path(cfg["paths"]["out_dir"])
        (out_dir / "latest_outcomes.txt").write_text(text)
    return outs


# ---------------------------------------------------------------- learning
def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def improve(cfg: dict, config_path: str, version: str | None = None) -> str:
    """Post-mortem over all resolved outcomes; writes out/postmortem.md and config.suggested.yaml."""
    h = cfg["verify"]["horizon_bars"]
    v = cfg["verify"]
    version = version or cfg["strategy_version"]
    con = store.connect(cfg["paths"]["db"])
    df = store.load_outcomes(con, h, version)
    lines = [f"# Post-mortem - strategy {version}, {h}-bar horizon", ""]
    if len(df) < v["min_outcomes_report"]:
        msg = f"Only {len(df)} resolved picks for {version}; need {v['min_outcomes_report']} before any conclusion."
        print(msg)
        return msg

    lines += [f"Resolved picks: **{len(df)}**  |  success rate (net of costs): **{df.success.mean():.1%}**  |  "
              f"mean net: **{df.net.mean() * 1e4:+.1f} bps**  |  direction right: **{df.direction_ok.mean():.1%}**", ""]
    for side, g in df.groupby("side"):
        lines.append(f"- {side}: n={len(g)}, success {g.success.mean():.1%}, mean net {g.net.mean() * 1e4:+.1f} bps")
    lines.append("")

    fails = df[~df.success.astype(bool)]
    lines += ["## Why picks failed (primary reason)", ""]
    for reason, n in Counter(fails.reason).most_common():
        lines.append(f"- **{reason}**: {n} ({n / len(fails):.0%}) - {REASONS.get(reason, '')}")
    flag_counts = Counter(f for d in fails.detail for f in json.loads(d).get("flags", []))
    lines += ["", "Contributing flags (a failure can have several): " +
              ", ".join(f"{k} {n}" for k, n in flag_counts.most_common()), ""]

    # per-factor: did picks where the factor agreed with the side do better than those where it did not?
    weights = dict(cfg["weights"])
    rows, suggested = [], dict(weights)
    fac = df["factors"].map(json.loads)
    sg = df["side"].map(_sign)
    for k in weights:
        vals = fac.map(lambda d, k=k: d.get(k))
        agree = np.array([(x is not None and np.sign(x) == s and abs(x) >= cfg["gate"]["factor_agree_min"])
                          for x, s in zip(vals, sg)])
        if agree.sum() < 5 or (~agree).sum() < 5:
            continue
        a, b = df.net.to_numpy()[agree], df.net.to_numpy()[~agree]
        t = _welch_t(a, b)
        rows.append((k, int(agree.sum()), float(a.mean() * 1e4), float(b.mean() * 1e4), t))
        if agree.sum() >= v["min_outcomes_tuning"] // 2 and (~agree).sum() >= 20:
            if t <= -v["t_threshold"]:
                suggested[k] = round(weights[k] * 0.7, 4)
            elif t >= v["t_threshold"]:
                suggested[k] = round(weights[k] * 1.2, 4)
    lines += ["## Factor check (net bps when the factor agreed with the side vs when it did not)", "",
              "| factor | n agreed | net bps agreed | net bps not | t-stat |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {n} | {a:+.1f} | {b:+.1f} | {t:+.1f} |" for k, n, a, b, t in rows]
    lines.append("")

    # regime check
    cr = df[[(r == "risk_on" and s == "SHORT") or (r == "risk_off" and s == "LONG")
             for r, s in zip(df.regime, df.side)]]
    if len(cr) >= 10:
        lines.append(f"Counter-regime picks: n={len(cr)}, mean net {cr.net.mean() * 1e4:+.1f} bps vs "
                     f"{df.drop(cr.index).net.mean() * 1e4:+.1f} bps with the regime.")
    rev = (fails.reason == "reversal_after_favorable_move").mean() if len(fails) else 0
    if rev >= 0.3:
        lines.append(f"{rev:.0%} of failures were in profit first and then reversed: consider a shorter horizon "
                     f"or a take-profit/trailing-stop rule (average best excursion of failures: "
                     f"{fails.mfe.mean() * 1e4:+.0f} bps).")

    changed = {k: (weights[k], suggested[k]) for k in weights if suggested[k] != weights[k]}
    lines += ["", "## Suggested change", ""]
    if len(df) < v["min_outcomes_tuning"]:
        lines.append(f"Not enough data to tune yet ({len(df)} / {v['min_outcomes_tuning']} resolved picks). "
                     "Everything above is descriptive only.")
    elif not changed:
        lines.append("No factor is significantly better or worse than the rest. Do not change anything.")
    else:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        raw["weights"] = suggested
        base = str(raw["strategy_version"]).split("+")[0]
        raw["strategy_version"] = f"{base}+tune{len(df)}"
        out = Path(config_path).with_name("config.suggested.yaml")
        out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        lines += [f"Wrote `{out.name}` with new version `{raw['strategy_version']}`:", ""]
        lines += [f"- {k}: {a} -> {b}" for k, (a, b) in changed.items()]
        lines += ["", "Caveats: these are many simultaneous tests on the same data, so some will look significant by "
                  "chance. Apply at most ONE change, run it in paper trading next to the old version, and keep it only "
                  "if `evaluate` shows it beating the old version out of sample."]
    text = "\n".join(lines)
    (Path(cfg["paths"]["out_dir"]) / "postmortem.md").write_text(text, encoding="utf-8")
    print(text)
    return text
