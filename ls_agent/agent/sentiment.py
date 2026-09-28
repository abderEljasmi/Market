"""LLM sentiment factor.

One Claude call per run scores every ticker that has fresh items.
The model only reads text; all numbers and decisions stay in code.
Factor value = score * confidence, in [-1, 1].
"""
from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger(__name__)

SYSTEM = """You are a financial news classifier for a short-term equity model.
For each ticker, judge how the provided news and social posts are likely to move
the stock over the next 1-3 trading days, relative to the overall market.

Rules:
- score: -1 (strongly bearish) to 1 (strongly bullish); 0 if nothing material.
- confidence: 0 to 1. Lower it for rumours, promotional or spammy posts, stale or
  already-priced-in news, and for items only loosely about the company.
- Social posts are weaker evidence than reported news; never let hype alone drive a high score.
- catalyst: at most 12 words naming the main driver, or "none".
Return ONLY a JSON object: {"TICKER": {"score": x, "confidence": y, "catalyst": "..."}}.
No prose, no markdown fences."""


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start != -1 else {}


def score_sentiment(cfg: dict, items: dict[str, list[dict]]) -> dict[str, dict]:
    """Returns {ticker: {"value", "score", "confidence", "catalyst"}}; empty dict on failure."""
    if not cfg["llm"]["enabled"] or not items:
        return {}
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("LLM enabled but ANTHROPIC_API_KEY is not set; running without sentiment")
        return {}

    import anthropic

    lines = []
    for t, its in items.items():
        lines.append(f"## {t}")
        lines += [f"- [{i['source']} {i['time'][:16]}] {i['text']}" for i in its]
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=cfg["llm"]["model"],
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        parsed = _extract_json(text)
    except Exception as e:  # the agent must keep running if the LLM call fails
        log.warning("sentiment call failed: %s", e)
        return {}

    out = {}
    for t, v in parsed.items():
        if t not in items or not isinstance(v, dict):
            continue  # ignore tickers the model invented
        try:
            s = max(-1.0, min(1.0, float(v.get("score", 0))))
            c = max(0.0, min(1.0, float(v.get("confidence", 0))))
        except (TypeError, ValueError):
            continue
        out[t] = {"value": s * c, "score": s, "confidence": c, "catalyst": str(v.get("catalyst", ""))[:120]}
    return out
