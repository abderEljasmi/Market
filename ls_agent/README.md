# Hourly long/short agent - US stocks + top-200 crypto

Every hour the agent scores a liquid universe on ~25 technical indicators (grouped into 9 factors)
plus, for stocks, LLM-read news sentiment, runs a strict conviction gate, and outputs up to 3 longs and
3 shorts, or NEUTRAL for a side with no conviction. **6 bars (~6 hours) after each pick it re-checks whether
the call worked, diagnoses the failures, and writes a post-mortem with a suggested config.**

This is a research and paper-trading tool, not financial advice.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 18 tests: no look-ahead, gate, costs, schedule, outcome check
python main.py --config config.synthetic_crypto.yaml run    # offline dry run, 200 fake coins
python main.py --config config.synthetic.yaml run           # offline dry run, fake stocks
```

Optional environment variables (never commit them):

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | stock news sentiment (agent runs without it) |
| `X_BEARER_TOKEN` | only if `social.x_enabled: true` (paid X API tier) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | push picks, 6h outcomes and the daily post-mortem to your phone (see `agent/notify.py`) |

## Configs

| File | Market | Notes |
|---|---|---|
| `config.yaml` | US stocks | **v1.0 baseline**, original 4 technical factors + sentiment. Keep as the control. |
| `config.v2.yaml` | US stocks | v2.0: all indicator factors. Weights are a guess, not tuned. Separate DB. |
| `config.crypto.yaml` | top-200 crypto | c1.0: 24/7, hourly, Binance data, CoinGecko universe. Separate DB. |
| `config.synthetic*.yaml` | offline | fake prices for testing without network |

Every config writes to its own database and output folder, so versions never mix.

## Commands (add `--config <file>` before the command)

| Command | What it does |
|---|---|
| `run` | One run now; prints the report, writes `out*/latest.json` and `latest.txt` |
| `loop` | Runs on schedule (stocks: market hours, New York time; crypto: every hour at :05, 24/7). After every run it also re-checks due picks and once a day writes the post-mortem |
| `verify` | Re-check picks whose 6-bar horizon has completed; prints OK/MISS and *why* for every miss |
| `improve` | Post-mortem over all checked picks: failure reasons, per-factor check, suggested config |
| `backtest [--reveal-holdout] [--step N]` | Walk-forward backtest of the technical core |
| `evaluate [--version v1.0]` | Scorecard of logged paper picks and the reliability verdict |

## Indicators

| Factor | Built from |
|---|---|
| trend | EMA20 vs EMA50 and EMA50 slope, in ATR units |
| momentum | 1-day return + MACD histogram in ATR units, ranked across the universe |
| rel_strength | 5-day return minus benchmark (SPY / BTC), ranked |
| volume | relative volume signed by bar direction |
| trend_ext | ADX with +DI/-DI direction, Aroon oscillator, Ichimoku cloud + tenkan/kijun, distance to EMA200 |
| momentum_ext | 6-bar and 1-day rate of change (vol-scaled), TRIX |
| breakout | Donchian channel position, Bollinger %B (continuation reading) |
| flow | OBV slope, Chaikin money flow, MFI, rolling VWAP deviation |
| reversion | contrarian brake at extremes: Bollinger, CCI, Williams %R, StochRSI |
| sentiment | stocks only: Claude reads last-24h news |

Also computed and logged with every pick (for the post-mortem): RSI, ATR%, ADX, Stochastic %K, Bollinger bandwidth.
Only factors listed under `weights:` in a config affect decisions; the rest are ignored, so adding indicators
never silently changes an older version.

## The 6-hour check and the post-mortem

For each reported pick, `verify` compares the entry close with the close 6 bars later (side-adjusted, after
`cost_bps_round_trip`) and records the best/worst excursion inside the window.
A pick **worked** if its net return is positive. Each miss gets a primary reason and a list of contributing flags:

| Reason | Meaning |
|---|---|
| cost_drag | direction right, move smaller than costs |
| reversal_after_favorable_move | was in profit, then reversed: horizon too long or no exit rule |
| market_headwind | the pick beat the benchmark but the market moved against the side |
| counter_regime | traded against the SPY/BTC regime |
| stretched_entry | chased an already stretched move (RSI/Bollinger) |
| choppy_market | ADX < 20, trend signals unreliable |
| noise | move stayed inside a normal 1-ATR wiggle |
| signal_wrong | price moved decisively against the call |

`improve` (and the daily loop) aggregate this into `out*/postmortem.md`. If there is enough evidence
(>= 100 checked picks and a factor with |t| >= 2 versus the rest) it writes `config.suggested.yaml` with a bumped
version. **It never edits your live config.** Re-tuning automatically on a few dozen trades mostly fits noise, and
changing weights in place destroys the comparison that tells you whether the change helped. Apply one change,
run it as a new version in parallel, and keep it only if `evaluate` shows it beating the old version.

## The reliability loop

1. **Backtest (train split)**: `python main.py backtest`. If the technical core shows no edge net of costs, fix that first.
2. **Paper trade**: run `loop` for at least 8 weeks. Every pick is logged with the entry bar and price.
3. **Scorecard**: `python main.py evaluate`: hit rate, edge vs universe after costs, per-side edge, calibration, drawdown.
4. **Verdict**: the `reliability` block sets the targets: 300+ picks, 8+ weeks, hit rate >= 53%, edge >= 5 bps, drawdown <= 15%, calibrated.
   - **Fail**: change ONE thing, bump `strategy_version`, go back to step 1.
   - **Pass**: check `backtest --reveal-holdout` once. If that also holds, go live with a very small size.

Rules that keep it honest: one change per iteration, never tune on the holdout, set a maximum number of iterations.
"No edge" is a valid and common result.

## Running it unattended

- **VPS (recommended):** see `deploy/`. Two systemd units, one for stocks, one for crypto.
- **Windows PC:** Task Scheduler -> "Run whether user is logged on or not", trigger "At startup", action
  `C:\path\venv\Scripts\python.exe main.py --config config.crypto.yaml loop` with "Start in" set to the project folder.
  Disable sleep while plugged in. The PC has to stay on and online.

## Known limitations

- **yfinance** is unofficial and can be delayed, rate-limited or change format. For real money, use a paid feed.
- **Binance public API** may be geo-restricted on some hosts; `data-api.binance.vision` is tried first, then `api.binance.com`.
- **Crypto universe:** about a third of the top 200 by market cap have no Binance USDT pair and are skipped (logged at startup).
  Stablecoins and wrapped/staked tokens are excluded on purpose.
- **Crypto sentiment:** no crypto news source is wired in; crypto runs on technical factors only.
- **Shorting:** spot crypto cannot be shorted (needs perps/margin), stocks need borrow. Raise `cost_bps_round_trip` if you short for real.
- **Market holidays** are not in the stock schedule. A duplicate bar is detected and skipped.
- **Survivorship bias:** the backtest uses today's universe.
- **Fills:** returns are computed from bar closes, not real fills. The 6-bar check for stocks counts trading-hour bars, so it can span the overnight gap.
- **Multiple testing:** the more indicators you add, the easier it is to find a "pattern" that is only luck. That is why nothing is applied automatically.

## Files

| Path | Role |
|---|---|
| `agent/data_prices.py` | Hourly bars: yfinance, Binance, or synthetic; drops the still-forming bar |
| `agent/universe.py` | Top-N crypto universe (CoinGecko -> Binance USDT pairs), cached daily |
| `agent/indicators.py` | Causal indicators, 9 factor panels, regime |
| `agent/news.py`, `agent/sentiment.py` | Stock news + Claude sentiment |
| `agent/scoring.py` | Composite, agreement, conviction gate |
| `agent/pipeline.py` | One full run and report |
| `agent/outcomes.py` | 6-bar outcome check, failure diagnosis, post-mortem, suggested config |
| `agent/backtest.py`, `agent/evaluate.py` | Backtest, scorecard, reliability verdict |
| `agent/store.py` | SQLite log: runs, scores, indicator snapshots, outcomes |
| `agent/notify.py` | Optional Telegram push |
| `main.py` | CLI and scheduler |
