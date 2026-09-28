"""Load config.yaml into a plain dict and expose a short hash for run logging."""
import hashlib
import json
from pathlib import Path

import yaml

CRYPTO_PROVIDERS = ("binance", "synthetic_crypto")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = path
    cfg.setdefault("verify", {})
    for k, v in {"horizon_bars": 6, "min_outcomes_report": 20, "min_outcomes_tuning": 100,
                 "t_threshold": 2.0, "daily_postmortem": True}.items():
        cfg["verify"].setdefault(k, v)
    Path(cfg["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def is_crypto(cfg: dict) -> bool:
    return cfg["data"]["provider"] in CRYPTO_PROVIDERS


def bars_per_day(cfg: dict) -> int:
    """Hourly bars per day: 7 for US stocks (09:30-16:00), 24 for crypto."""
    return int(cfg["data"].get("bars_per_day", 24 if is_crypto(cfg) else 7))


def config_hash(cfg: dict) -> str:
    """Hash of everything that changes decisions (weights, gate, universe)."""
    relevant = {k: cfg[k] for k in ("weights", "gate", "strategy_version")}
    # a dynamic universe (top-N crypto) changes every day; hash the rule, not the list
    relevant["universe"] = cfg.get("universe_source") or cfg["universe"]
    blob = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
