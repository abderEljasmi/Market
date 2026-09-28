"""Unit tests for the parts that must never silently break."""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import data_prices, outcomes, store, universe
from agent import evaluate as ev
from agent import indicators as ind
from agent.config import bars_per_day, config_hash, load_config
from agent.data_prices import TZ, _fetch_synthetic, _fetch_synthetic_crypto
from agent.pipeline import run_once
from agent.scoring import decide
from agent.sentiment import _extract_json
from main import next_run_time

ROOT = Path(__file__).resolve().parents[1]
STOCK_FACTORS = ("trend", "momentum", "rel_strength", "volume")


@pytest.fixture
def cfg():
    return load_config(str(ROOT / "config.yaml"))


@pytest.fixture
def prices(cfg):
    return _fetch_synthetic(cfg["universe"][:10] + [cfg["benchmark"]], 120)


@pytest.fixture
def crypto_cfg(tmp_path):
    c = load_config(str(ROOT / "config.synthetic_crypto.yaml"))
    c["paths"] = {"db": str(tmp_path / "c.sqlite"), "out_dir": str(tmp_path)}
    c["gate"].update(min_abs_score=0.05, min_agreeing_factors=2)   # make sure random data produces picks
    return c


# ---------------------------------------------------------------- original behaviour
def test_factors_are_causal(prices, cfg):
    """Adding future bars must not change past factor values (no look-ahead), for ALL factors."""
    full = ind.technical_factors(ind.build_panels(prices, cfg["benchmark"]))
    cut_ts = next(iter(prices.values())).index[-50]
    past_prices = {t: df.loc[:cut_ts] for t, df in prices.items()}
    past = ind.technical_factors(ind.build_panels(past_prices, cfg["benchmark"]))
    assert set(full) >= {"trend_ext", "momentum_ext", "breakout", "flow", "reversion"}
    for k in full:
        pd.testing.assert_series_equal(full[k].loc[cut_ts], past[k].loc[cut_ts], check_names=False)


def test_all_factors_bounded(prices, cfg):
    for k, panel in ind.technical_factors(ind.build_panels(prices, cfg["benchmark"])).items():
        assert panel.abs().max().max() <= 1.0 + 1e-9, k
        assert not panel.isna().any().any(), k


def test_no_signal_gives_neutral(cfg):
    tickers = ["AAA", "BBB", "CCC"]
    zero = pd.Series(0.0, index=tickers)
    factors = {k: zero for k in STOCK_FACTORS}
    d = decide(factors, pd.Series(50.0, index=tickers), pd.Series(1e9, index=tickers), "neutral", cfg)
    assert d["longs"].empty and d["shorts"].empty


def test_strong_agreeing_signal_is_picked(cfg):
    tickers = ["AAA", "BBB"]
    factors = {k: pd.Series([0.8, 0.0], index=tickers) for k in STOCK_FACTORS}
    d = decide(factors, pd.Series(55.0, index=tickers), pd.Series(1e9, index=tickers), "neutral", cfg)
    assert list(d["longs"].index) == ["AAA"]


def test_unweighted_factors_never_change_a_decision(cfg):
    """v1.0 must behave identically now that extra indicator factors exist."""
    tickers = ["AAA", "BBB"]
    base = {k: pd.Series([0.5, 0.0], index=tickers) for k in STOCK_FACTORS}
    extra = dict(base, trend_ext=pd.Series([-1.0, 1.0], index=tickers), flow=pd.Series([-1.0, 1.0], index=tickers))
    args = (pd.Series(55.0, index=tickers), pd.Series(1e9, index=tickers), "neutral", cfg)
    a, b = decide(base, *args), decide(extra, *args)
    pd.testing.assert_frame_equal(a["table"][["score", "agree", "side"]], b["table"][["score", "agree", "side"]])


def test_gate_blocks_overbought_illiquid_and_events(cfg):
    tickers = ["AAA"]
    factors = {k: pd.Series([0.8], index=tickers) for k in STOCK_FACTORS}
    liquid = pd.Series(1e9, index=tickers)
    assert decide(factors, pd.Series(85.0, index=tickers), liquid, "neutral", cfg)["longs"].empty
    assert decide(factors, pd.Series(55.0, index=tickers), pd.Series(1e3, index=tickers), "neutral", cfg)["longs"].empty
    assert decide(factors, pd.Series(55.0, index=tickers), liquid, "neutral", cfg, event_blackout=True)["longs"].empty


def test_llm_json_parsing_tolerates_fences():
    assert _extract_json('```json\n{"AAPL": {"score": 0.4}}\n```') == {"AAPL": {"score": 0.4}}


def test_forward_returns_and_costs():
    idx = pd.date_range("2026-01-05 09:30", periods=4, freq="h", tz="America/New_York")
    close = pd.DataFrame({"A": [100, 110, 121, 130], "B": [100, 100, 100, 100]}, index=idx, dtype=float)
    picks = pd.DataFrame([{"bar_ts": idx[0], "ticker": "A", "side": "LONG", "conviction": 50},
                          {"bar_ts": idx[0], "ticker": "B", "side": "SHORT", "conviction": 50}])
    out = ev.attach_returns(picks, close, h=1, cost_bps=10)
    assert out.loc[0, "net"] == pytest.approx(0.10 - 0.001)
    assert out.loc[0, "excess"] == pytest.approx(0.10 - 0.05 - 0.001)   # universe avg = 5%
    assert out.loc[1, "net"] == pytest.approx(-0.001)


def test_schedule_skips_weekend(cfg):
    fri_evening = datetime(2026, 9, 25, 17, 0, tzinfo=ZoneInfo("America/New_York"))
    nxt = next_run_time(cfg, fri_evening)
    assert nxt.weekday() == 0 and (nxt.hour, nxt.minute) == (10, 35)


# ---------------------------------------------------------------- crypto
def test_crypto_schedule_is_hourly_and_includes_weekends(crypto_cfg):
    sat = datetime(2026, 9, 26, 14, 7, tzinfo=ZoneInfo("UTC"))
    assert next_run_time(crypto_cfg, sat) == datetime(2026, 9, 26, 15, 5, tzinfo=ZoneInfo("UTC"))
    assert bars_per_day(crypto_cfg) == 24


def test_crypto_factors_causal_and_btc_is_scored(crypto_cfg):
    tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
    pr = _fetch_synthetic_crypto(tickers, 30)
    panels = ind.build_panels(pr, "BTCUSDT", 24, universe=tickers)
    assert "BTCUSDT" in panels["close"].columns                 # benchmark coin is also a candidate
    full = ind.technical_factors(panels)
    cut = next(iter(pr.values())).index[-100]
    past = ind.technical_factors(ind.build_panels({t: d.loc[:cut] for t, d in pr.items()}, "BTCUSDT", 24, tickers))
    for k in full:
        pd.testing.assert_series_equal(full[k].loc[cut], past[k].loc[cut], check_names=False)


def test_drop_incomplete_bar_crypto_is_24_7():
    idx = pd.date_range("2026-09-26 22:00", periods=4, freq="h", tz=TZ)          # a Saturday night
    df = pd.DataFrame({"close": 1.0}, index=idx)
    kept = data_prices.drop_incomplete_bar(df, idx[2] + pd.Timedelta(minutes=10), crypto=True)
    assert list(kept.index) == list(idx[:2])                                     # bar 2 is still forming


def test_binance_kline_parsing(monkeypatch):
    t0 = 1_780_000_000_000
    rows = [[t0 + i * 3_600_000, "1", "2", "0.5", "1.5", "100", 0, 0, 0, 0, 0, 0] for i in range(3)]
    monkeypatch.setattr(data_prices, "_binance_get", lambda path, params=None, retries=4: rows)
    df = data_prices._klines("BTCUSDT", 1)
    assert list(df.columns) == data_prices.FIELDS and len(df) == 3 and df["close"].iloc[0] == 1.5
    assert str(df.index.tz) == TZ


def test_top_crypto_filters_stablecoins_and_untradable(monkeypatch):
    coins = [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
             {"id": "tether", "symbol": "usdt", "name": "Tether"},
             {"id": "wrapped-bitcoin", "symbol": "wbtc", "name": "Wrapped Bitcoin"},
             {"id": "obscure", "symbol": "obs", "name": "Obscure"},
             {"id": "ethereum", "symbol": "eth", "name": "Ethereum"}]
    monkeypatch.setattr(universe, "_coingecko_top", lambda n: coins)
    monkeypatch.setattr(universe, "_binance_usdt_symbols", lambda: {"BTCUSDT", "ETHUSDT", "USDCUSDT", "WBTCUSDT"})
    assert universe.top_crypto(5) == ["BTCUSDT", "ETHUSDT"]


def test_config_hash_ignores_dynamic_universe(crypto_cfg):
    h = config_hash(crypto_cfg)
    crypto_cfg["universe"] = ["BTCUSDT"]
    assert config_hash(crypto_cfg) == h


# ---------------------------------------------------------------- outcome check + post-mortem
def test_diagnose_reasons():
    o = dict(direction_ok=True, net=-0.0005, mfe=0.0, excess=-0.001, ret=0.0015)
    assert outcomes.diagnose("LONG", o, {"atr_pct": 0.01}, {}, "neutral", 0.0, 0.002, 6)[0] == "cost_drag"
    o = dict(direction_ok=False, net=-0.02, mfe=0.015, excess=-0.02, ret=-0.02)
    assert outcomes.diagnose("LONG", o, {"atr_pct": 0.001}, {}, "neutral", 0.0, 0.002, 6)[0] == "reversal_after_favorable_move"
    o = dict(direction_ok=False, net=-0.02, mfe=0.0, excess=0.01, ret=0.02)      # SHORT, coin fell less than market
    assert outcomes.diagnose("SHORT", o, {"atr_pct": 0.001}, {}, "neutral", 0.03, 0.002, 6)[0] == "market_headwind"
    o = dict(direction_ok=False, net=-0.05, mfe=0.0, excess=-0.05, ret=0.05)
    reason, detail = outcomes.diagnose("SHORT", o, {"atr_pct": 0.001, "adx": 40, "rsi": 50}, {"trend": -0.9, "flow": 0.4},
                                       "neutral", 0.0, 0.002, 6)
    assert reason == "signal_wrong" and detail["factors_that_agreed"] == {"trend": -0.9}


def test_verify_and_postmortem_end_to_end(crypto_cfg, capsys):
    end = pd.Timestamp("2026-09-25 12:00", tz=TZ)
    payload = run_once(crypto_cfg, now=end - pd.Timedelta(hours=40))
    assert payload and (payload["longs"] or payload["shorts"])
    con = store.connect(crypto_cfg["paths"]["db"])
    assert con.execute("SELECT COUNT(*) FROM pick_features").fetchone()[0] > 0       # indicator snapshot stored

    # too early: nothing is due
    assert outcomes.verify_due(crypto_cfg, now=end - pd.Timedelta(hours=39)) == []
    # 6+ bars later: every pick gets an outcome, failures get a diagnosis
    done = outcomes.verify_due(crypto_cfg, now=end + pd.Timedelta(hours=2))
    assert len(done) == len(payload["longs"]) + len(payload["shorts"])
    for o in done:
        assert o["success"] or o["reason"] in outcomes.REASONS
    # running again does not re-check the same picks
    assert outcomes.verify_due(crypto_cfg, now=end + pd.Timedelta(hours=3)) == []
    assert con.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == len(done)
    # too few outcomes for conclusions: the post-mortem must refuse rather than guess
    assert outcomes.improve(crypto_cfg, str(ROOT / "config.synthetic_crypto.yaml")).startswith("Only")


def test_improve_suggests_only_with_enough_evidence(crypto_cfg, tmp_path):
    """Plant 200 outcomes where 'flow' agreeing always loses; only that weight should be cut."""
    rng = np.random.default_rng(0)
    con = store.connect(crypto_cfg["paths"]["db"])
    for i in range(200):
        flow = float(rng.choice([-0.5, 0.5]))
        win = flow < 0 if rng.random() < 0.9 else flow > 0          # flow agreeing (LONG, flow>0) loses 90%
        net = 0.01 if win else -0.01
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (f"r{i}", "", f"2026-09-01 00:{i % 60:02d}:00+00:00", crypto_cfg["strategy_version"], "h",
                     "neutral", None, 0, 0, "ok", "{}"))
        con.execute("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"r{i}", "AAAUSDT", 0.5, 4, 8, 50, "LONG", "LONG", "", "", 1.0,
                     json.dumps({"trend": float(rng.choice([-0.5, 0.5])), "flow": flow})))
        store.save_outcome(con, dict(run_id=f"r{i}", ticker="AAAUSDT", horizon=6, side="LONG", bar_ts="x",
                                     strategy_version=crypto_cfg["strategy_version"], entry=1, exit=1, ret=net,
                                     net=net, excess=net, mfe=0.0, mae=0.0, direction_ok=net > 0, success=net > 0,
                                     reason="success" if net > 0 else "signal_wrong",
                                     detail={} if net > 0 else {"flags": ["signal_wrong"]}, checked_ts=""))
    cfg_copy = tmp_path / "cfg.yaml"
    cfg_copy.write_text((ROOT / "config.synthetic_crypto.yaml").read_text())
    text = outcomes.improve(crypto_cfg, str(cfg_copy))
    assert "flow: 0.11 -> 0.077" in text and "trend:" not in text.split("## Suggested change")[1]
    assert (tmp_path / "config.suggested.yaml").exists()
