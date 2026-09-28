"""News and social data layer.

Returns {ticker: [ {"source", "time", "text"} ]} for the last N hours.
- news: yfinance headlines (free, delayed, sometimes sparse)
- X   : X API v2 recent search (paid tier, needs X_BEARER_TOKEN)
Add more sources by writing a function with the same return shape.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def yahoo_news(tickers: list[str], max_age_h: int, max_items: int) -> dict[str, list[dict]]:
    import yfinance as yf

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_h)
    out: dict[str, list[dict]] = {}
    for t in tickers:
        items = []
        try:
            for n in yf.Ticker(t).news or []:
                # yfinance changed its news format; handle old and new shapes
                c = n.get("content", n)
                title = c.get("title")
                when = _parse_time(c.get("pubDate") or c.get("providerPublishTime"))
                if title and when and when >= cutoff:
                    summary = (c.get("summary") or "")[:240]
                    items.append({"source": "news", "time": when.isoformat(), "text": f"{title}. {summary}".strip()})
        except Exception as e:  # network / format issues must not kill the run
            log.warning("news fetch failed for %s: %s", t, e)
        out[t] = sorted(items, key=lambda x: x["time"], reverse=True)[:max_items]
    return out


def x_posts(tickers: list[str], max_age_h: int, max_posts: int) -> dict[str, list[dict]]:
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        log.warning("X enabled but X_BEARER_TOKEN is not set; skipping social data")
        return {}
    start = (datetime.now(timezone.utc) - timedelta(hours=min(max_age_h, 24 * 7))).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {}
    for t in tickers:
        params = {
            # cashtag, English, no retweets/replies to cut spam
            "query": f"${t} lang:en -is:retweet -is:reply",
            "max_results": max(10, min(max_posts, 100)),
            "start_time": start,
            "tweet.fields": "created_at,public_metrics",
        }
        try:
            r = requests.get("https://api.x.com/2/tweets/search/recent", params=params,
                             headers={"Authorization": f"Bearer {token}"}, timeout=15)
            if r.status_code == 429:
                log.warning("X rate limit hit; stopping social fetch for this run")
                break
            r.raise_for_status()
            posts = r.json().get("data", [])
            # most-engaged posts first: engagement is a weak filter against bots
            posts.sort(key=lambda p: sum(p.get("public_metrics", {}).values()), reverse=True)
            out[t] = [{"source": "x", "time": p.get("created_at"), "text": p["text"][:280]} for p in posts[:max_posts]]
        except requests.RequestException as e:
            log.warning("X fetch failed for %s: %s", t, e)
    return out


def gather(cfg: dict, tickers: list[str]) -> dict[str, list[dict]]:
    # no free point-in-time crypto news source is wired in; crypto runs on technical factors only
    if cfg["data"]["provider"] in ("synthetic", "synthetic_crypto", "binance"):
        return {}
    llm = cfg["llm"]
    items = yahoo_news(tickers, llm["news_max_age_hours"], llm["max_items_per_ticker"])
    if cfg["social"]["x_enabled"]:
        for t, posts in x_posts(tickers, llm["news_max_age_hours"], cfg["social"]["x_max_posts_per_ticker"]).items():
            items.setdefault(t, []).extend(posts)
    return {t: v for t, v in items.items() if v}
