"""CLI entry point.

  python main.py run                 # one run now
  python main.py loop                # run at every scheduled time; re-checks past picks after each run
  python main.py verify              # re-check picks whose 6-bar horizon has passed (+ why-failed diagnosis)
  python main.py improve             # post-mortem over all checked picks + suggested config
  python main.py backtest            # walk-forward backtest of the technical core
  python main.py backtest --reveal-holdout
  python main.py evaluate            # scorecard of live/paper picks + reliability verdict
  python main.py evaluate --version v1.0

Pick the market with --config: config.yaml (US stocks), config.crypto.yaml (top-200 coins).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from agent import evaluate as ev
from agent import notify, store
from agent.backtest import run_backtest
from agent.config import load_config
from agent.data_prices import TZ, fetch_prices
from agent.outcomes import improve, verify_due
from agent.pipeline import run_once
from agent.universe import resolve_universe

log = logging.getLogger("agent")


def next_run_time(cfg: dict, now: datetime) -> datetime:
    sch = cfg["schedule"]
    tz = ZoneInfo(sch["timezone"])
    now = now.astimezone(tz)
    if sch.get("hourly"):                       # 24/7 markets: every hour at :minute
        t = now.replace(minute=sch.get("minute", 5), second=0, microsecond=0)
        return t if t > now else t + timedelta(hours=1)
    for day in range(8):
        d = (now + timedelta(days=day)).date()
        if sch["weekdays_only"] and d.weekday() >= 5:
            continue
        for hhmm in sorted(sch["run_times"]):
            h, m = map(int, hhmm.split(":"))
            t = datetime(d.year, d.month, d.day, h, m, tzinfo=tz)
            if t > now:
                return t
    raise RuntimeError("no run time found in schedule")


def cycle(cfg: dict, state: dict) -> None:
    """One scheduled tick: new picks, then re-check old ones. Errors never stop the loop."""
    try:
        run_once(cfg)
    except Exception:
        log.exception("run failed; will retry at next scheduled time")
    try:
        verify_due(cfg)
    except Exception:
        log.exception("outcome check failed")
    today = datetime.now(ZoneInfo("UTC")).date()
    if cfg["verify"]["daily_postmortem"] and state.get("last_postmortem") != today:
        state["last_postmortem"] = today
        try:
            text = improve(cfg, cfg["_path"])
            if text and not text.startswith("Only"):
                notify.send(text[:3500])
        except Exception:
            log.exception("post-mortem failed")


def loop(cfg: dict) -> None:
    log.info("agent loop started, schedule %s", cfg["schedule"])
    state: dict = {}
    while True:
        nxt = next_run_time(cfg, datetime.now(ZoneInfo(cfg["schedule"]["timezone"])))
        log.info("next run at %s", nxt.isoformat())
        time.sleep(max(0, (nxt - datetime.now(nxt.tzinfo)).total_seconds()))
        cycle(cfg, state)


def evaluate_live(cfg: dict, version: str | None) -> dict:
    con = store.connect(cfg["paths"]["db"])
    picks = store.load_picks(con, version)
    runs = store.load_runs(con, version)
    if picks.empty:
        print("No logged picks yet. Let the agent run (paper trading) and come back.")
        return {}
    picks["bar_ts"] = pd.to_datetime(picks["bar_ts"], utc=True).dt.tz_convert(TZ)
    days = (pd.Timestamp.now(tz=TZ) - picks["bar_ts"].min()).days + 10
    tickers = list(dict.fromkeys(resolve_universe(cfg) + list(picks["ticker"].unique()) + [cfg["benchmark"]]))
    prices = fetch_prices(cfg, tickers, days)
    close = pd.DataFrame({t: df["close"] for t, df in prices.items()}).sort_index().ffill()
    cost = cfg["evaluation"]["cost_bps_round_trip"]
    h = cfg["evaluation"]["primary_horizon_bars"]
    scored = ev.attach_returns(picks, close, h, cost)
    pending = len(picks) - len(scored)
    card = ev.scorecard(scored, h, n_runs=len(runs))
    card["pending_picks"] = pending
    ev.print_card(f"LIVE / PAPER picks ({version or 'all versions'}, horizon {h} bars)", card, cfg)
    if version is None and picks["strategy_version"].nunique() > 1:
        print("\nBy strategy version (edge bps, net of costs):")
        for v, g in scored.groupby("strategy_version"):
            print(f"  {v:<10} n={len(g):<5} edge={g['excess'].mean() * 1e4:+.1f} bps  hit={(g['net'] > 0).mean():.2f}")
    return card


def main() -> None:
    p = argparse.ArgumentParser(description="Hourly long/short recommendation agent (stocks + crypto)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("loop")
    sub.add_parser("verify")
    im = sub.add_parser("improve")
    im.add_argument("--version", default=None)
    b = sub.add_parser("backtest")
    b.add_argument("--reveal-holdout", action="store_true")
    b.add_argument("--step", type=int, default=1, help="evaluate every Nth bar (faster)")
    e = sub.add_parser("evaluate")
    e.add_argument("--version", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = load_config(args.config)

    if args.cmd == "run":
        run_once(cfg)
    elif args.cmd == "loop":
        loop(cfg)
    elif args.cmd == "verify":
        if not verify_due(cfg):
            print("No picks are due for an outcome check yet.")
    elif args.cmd == "improve":
        improve(cfg, args.config, args.version)
    elif args.cmd == "backtest":
        run_backtest(cfg, reveal_holdout=args.reveal_holdout, step=args.step)
    elif args.cmd == "evaluate":
        evaluate_live(cfg, args.version)


if __name__ == "__main__":
    main()
