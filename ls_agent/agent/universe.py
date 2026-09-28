"""Resolve a dynamic universe (top-N crypto coins) into a concrete ticker list.

Source of truth for "top 200" is CoinGecko market cap. We can only trade/score
coins that have hourly data, so each coin is mapped to a Binance USDT pair;
coins without one (about a third of the top 200) are skipped and logged.
Stablecoins and wrapped/staked tokens are excluded: they track another asset
and their "trend" is noise.
The result is cached per day so a 24/7 loop does not hit CoinGecko every hour.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import requests

from .data_prices import _binance_get

log = logging.getLogger(__name__)

STABLES = {"usdt", "usdc", "dai", "fdusd", "usde", "tusd", "usdd", "pyusd", "usds", "usdp", "gusd", "busd",
           "frax", "lusd", "susd", "eurc", "eurt", "usdy", "usd0", "usdb", "usd1", "ustc", "rlusd", "bfusd"}
WRAPPED_HINTS = ("wrapped", "staked", "bridged", "restaked", "liquid staking", "lido", "coinbase wrapped")


def _excluded(coin: dict) -> bool:
    sym = coin["symbol"].lower()
    name = f"{coin.get('id', '')} {coin.get('name', '')}".lower()
    return sym in STABLES or any(h in name for h in WRAPPED_HINTS) or sym in {"wbtc", "weth", "steth", "wsteth", "weeth", "cbbtc"}


def _binance_usdt_symbols() -> set[str]:
    info = _binance_get("/api/v3/exchangeInfo")
    return {s["symbol"] for s in info["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}


def _coingecko_top(n: int) -> list[dict]:
    coins: list[dict] = []
    page = 1
    while len(coins) < n:
        r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                         params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
                         timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        coins += batch
        page += 1
    return coins[:n]


def top_crypto(n: int, min_dollar_volume: float = 0) -> list[str]:
    """Top-N by market cap, mapped to Binance USDT symbols. Falls back to 24h volume ranking."""
    tradable = _binance_usdt_symbols()
    try:
        picked = []
        for c in _coingecko_top(n):
            sym = c["symbol"].upper() + "USDT"
            if _excluded(c):
                continue
            if sym in tradable:
                picked.append(sym)
        log.info("universe: %d of the top %d coins by market cap have a Binance USDT pair", len(picked), n)
        return picked
    except Exception as e:
        log.warning("CoinGecko unavailable (%s); ranking Binance USDT pairs by 24h volume instead", e)
        tick = _binance_get("/api/v3/ticker/24hr")
        rows = [(t["symbol"], float(t["quoteVolume"])) for t in tick if t["symbol"] in tradable]
        stable_pairs = {s.upper() + "USDT" for s in STABLES}
        rows = [r for r in rows if r[0] not in stable_pairs]
        rows.sort(key=lambda r: r[1], reverse=True)
        return [s for s, _ in rows[:n]]


def resolve_universe(cfg: dict) -> list[str]:
    """Fill cfg['universe'] from cfg['universe_source'] when present. Returns the list."""
    src = cfg.get("universe_source")
    if not src:
        return cfg["universe"]
    if src["type"] == "synthetic_crypto":
        cfg["universe"] = ["BTCUSDT", "ETHUSDT"] + [f"COIN{i:03d}USDT" for i in range(src["n"] - 2)]
        return cfg["universe"]
    cache = Path(cfg["paths"]["db"]).parent / f"universe_{date.today().isoformat()}.json"
    if cache.exists():
        cfg["universe"] = json.loads(cache.read_text())
    else:
        cfg["universe"] = top_crypto(src["n"])
        cache.write_text(json.dumps(cfg["universe"]))
    return cfg["universe"]
