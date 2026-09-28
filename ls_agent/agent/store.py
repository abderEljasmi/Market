"""SQLite log of every run, every ticker score, indicator snapshots and 6h outcomes.

Logging ALL scores (not only picks) matters: it lets you re-test other
thresholds later and gives an honest baseline to compare picks against.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, run_ts TEXT, bar_ts TEXT, strategy_version TEXT,
    config_hash TEXT, regime TEXT, vix REAL, llm_used INTEGER, n_news INTEGER,
    status TEXT, thresholds TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT, ticker TEXT, score REAL, agree INTEGER, n_factors INTEGER,
    conviction INTEGER, side TEXT, candidate_side TEXT, blocked_by TEXT,
    why TEXT, close REAL, factors TEXT,
    PRIMARY KEY (run_id, ticker)
);
CREATE TABLE IF NOT EXISTS pick_features (
    run_id TEXT, ticker TEXT, features TEXT,
    PRIMARY KEY (run_id, ticker)
);
CREATE TABLE IF NOT EXISTS outcomes (
    run_id TEXT, ticker TEXT, horizon INTEGER, side TEXT, bar_ts TEXT, strategy_version TEXT,
    entry REAL, exit REAL, ret REAL, net REAL, excess REAL, mfe REAL, mae REAL,
    direction_ok INTEGER, success INTEGER, reason TEXT, detail TEXT, checked_ts TEXT,
    PRIMARY KEY (run_id, ticker, horizon)
);
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def already_logged(con, bar_ts: str, version: str) -> bool:
    row = con.execute("SELECT 1 FROM runs WHERE bar_ts=? AND strategy_version=?", (bar_ts, version)).fetchone()
    return row is not None


def log_run(con, run: dict, table: pd.DataFrame, closes: pd.Series, factor_rows: dict) -> None:
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
        run["run_id"], run["run_ts"], run["bar_ts"], run["strategy_version"], run["config_hash"],
        run["regime"], run["vix"], int(run["llm_used"]), run["n_news"], run["status"],
        json.dumps(run["thresholds"])))
    for t, r in table.iterrows():
        con.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            run["run_id"], t, r["score"], int(r["agree"]), int(r["n_factors"]), int(r["conviction"]),
            r["side"], r["candidate_side"], r["blocked_by"], r["why"],
            float(closes.get(t, float("nan"))), json.dumps(factor_rows.get(t, {}))))
    con.commit()


def log_features(con, run_id: str, feats: dict[str, dict]) -> None:
    for t, f in feats.items():
        con.execute("INSERT OR REPLACE INTO pick_features VALUES (?,?,?)", (run_id, t, json.dumps(f)))
    con.commit()


def load_picks(con, version: str | None = None) -> pd.DataFrame:
    q = """SELECT r.bar_ts, r.strategy_version, r.regime, s.* FROM scores s
           JOIN runs r USING (run_id) WHERE s.side IS NOT NULL"""
    params = ()
    if version:
        q += " AND r.strategy_version = ?"
        params = (version,)
    return pd.read_sql_query(q, con, params=params)


def load_runs(con, version: str | None = None) -> pd.DataFrame:
    q, params = "SELECT * FROM runs", ()
    if version:
        q, params = q + " WHERE strategy_version = ?", (version,)
    return pd.read_sql_query(q, con, params=params)


def load_unverified(con, horizon: int) -> pd.DataFrame:
    """Picks that have no outcome yet for this horizon, with their logged factors + indicators."""
    q = """SELECT r.bar_ts, r.strategy_version, r.regime, s.run_id, s.ticker, s.side, s.score, s.conviction,
                  s.close, s.factors, s.why, f.features
           FROM scores s JOIN runs r USING (run_id)
           LEFT JOIN pick_features f ON f.run_id = s.run_id AND f.ticker = s.ticker
           LEFT JOIN outcomes o ON o.run_id = s.run_id AND o.ticker = s.ticker AND o.horizon = ?
           WHERE s.side IS NOT NULL AND o.run_id IS NULL"""
    return pd.read_sql_query(q, con, params=(horizon,))


def save_outcome(con, o: dict) -> None:
    con.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        o["run_id"], o["ticker"], o["horizon"], o["side"], o["bar_ts"], o["strategy_version"],
        o["entry"], o["exit"], o["ret"], o["net"], o["excess"], o["mfe"], o["mae"],
        int(o["direction_ok"]), int(o["success"]), o["reason"], json.dumps(o["detail"]), o["checked_ts"]))
    con.commit()


def load_outcomes(con, horizon: int, version: str | None = None) -> pd.DataFrame:
    q = """SELECT o.*, s.factors, s.conviction, s.score, r.regime, f.features
           FROM outcomes o JOIN scores s ON s.run_id = o.run_id AND s.ticker = o.ticker
           JOIN runs r ON r.run_id = o.run_id
           LEFT JOIN pick_features f ON f.run_id = o.run_id AND f.ticker = o.ticker
           WHERE o.horizon = ?"""
    params: tuple = (horizon,)
    if version:
        q += " AND o.strategy_version = ?"
        params += (version,)
    return pd.read_sql_query(q, con, params=params)
