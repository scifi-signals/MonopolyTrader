# MonopolyTrader v4 — Claude IS the Trading Brain

## Overview

Autonomous AI paper trading agent. Manages $1,000 virtual portfolio trading TSLA. Claude makes every decision — no coded strategies, no signal aggregation, no stop losses.

**Dashboard:** https://scifi-signals.github.io/MonopolyTrader/dashboard/
**Server:** DigitalOcean (`science-intel`), systemd service `monopoly-trader` at `/root/monopoly-trader`

## Design Philosophy

- Paper money = learning budget, not capital to preserve
- Every scenario is an opportunity — trade aggressively to learn
- No HOLD bias — a trade that loses $30 but teaches something beats sitting in cash
- Claude sees raw data and decides everything: direction, sizing, entry/exit
- No stop losses — Claude evaluates underwater positions with context every cycle

## Three Rules (only 3)

1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK

## Architecture

Every 15 minutes during market hours:
1. **Gather data** — TSLA price/indicators + world snapshot + news + web search + events calendar
2. **Build brief** — Single text prompt with all raw data + portfolio + trade journal
3. **Claude decides** — One Sonnet call → JSON response (BUY/SELL/HOLD + shares + reasoning)
4. **Execute trade** — If BUY or SELL, simulate with slippage
5. **Record in journal** — On trade close, Haiku generates 50-word lesson
6. **Update dashboard** — Refresh dashboard data

## File Structure

```
src/
├── main.py          — 6-step cycle orchestrator
├── agent.py         — SYSTEM_PROMPT + build_market_brief() + make_decision()
├── tags.py          — Mechanical trade tagging (8 objective market condition tags)
├── thesis_builder.py — Nightly aggregation of tags into statistical playbook
├── market_data.py   — yfinance prices, indicators, regime, get_world_snapshot()
├── portfolio.py     — 2-rule validation, execute_trade(), cooldown
├── journal.py       — Trade journal with tags, Haiku lesson generation on close
├── events.py        — FOMC/CPI/NFP calendar + TSLA earnings from yfinance
├── news_feed.py     — yfinance + RSS news
├── web_search.py    — Web search (disabled, was Brave Search)
├── reporter.py      — Dashboard data generation (includes playbook + learning metrics)
├── observability.py — Anomaly detection, decision tracing, health checks
└── utils.py         — Config, logging, AI call helpers, time utils

tests/
└── test_thesis.py   — Tests for tags, aggregation, validation, playbook formatting

config.json          — ~40 lines. Ticker, risk params, model names, world tickers
data/
├── portfolio.json    — Current portfolio state
├── transactions.json — Full trade history
├── trade_journal.json — Journal entries with tags and lessons
├── thesis_ledger.json — Aggregated playbook stats (rebuilt nightly)
├── snapshots/         — Daily portfolio snapshots
└── latest_cycle.json  — Last decision for dashboard
```

## Learning System: Thesis Ledger

Every trade is mechanically tagged with 8 market conditions at execution:
- **rsi_zone**: oversold / neutral / overbought
- **trend**: above_sma50 / below_sma50
- **volatility**: low_vix / normal_vix / high_vix
- **regime**: trending / range_bound
- **macd**: bullish_cross / bearish_cross / neutral
- **market_context**: spy_up / spy_down / spy_flat
- **position_state**: opening_new / adding_to_winner / adding_to_loser / taking_profit / cutting_loss
- **event_proximity**: pre_event_24h / pre_event_72h / no_event

Nightly, `thesis_builder.py` aggregates all closed trades into single-tag stats (win rate, avg P&L). Claude sees its "playbook" every cycle — what setups work, what to avoid, whether its confidence is calibrated. Time-decay weights recent trades more heavily (30/60/90 day windows).

## What Claude Sees Each Cycle (the "brief")

- **TSLA indicators**: Price, RSI, SMA20/50, MACD, Bollinger, ATR, ADX, EMA, OBV, volume
- **Intraday 5min indicators**: RSI, SMA, MACD crossover, Bollinger
- **Last 5 days**: OHLCV
- **World**: Macro tickers (10Y yield, oil, BTC, NASDAQ, USD, SPY, VIX) + EV peers (RIVN, LCID, NIO, LI, XPEV)
- **Regime**: Trend direction, directional strength, volatility level
- **News**: yfinance + RSS headlines
- **Upcoming events**: FOMC, CPI, NFP within 72h + TSLA earnings date
- **Playbook**: Statistical performance by market condition (best/worst setups, confidence calibration)
- **Portfolio**: Cash, total value, P&L, position details, max BUY/SELL limits
- **Trade journal**: Last 5 trades with lessons learned

## Key Commands

```bash
python -m src.main           # Start scheduler
python -m src.main --once    # Single cycle
python -m src.main --report  # Generate dashboard
```

## Server

```bash
systemctl status monopoly-trader
systemctl restart monopoly-trader
tail -f /root/monopoly-trader/logs/agent.log
```

## Dependencies

yfinance, ta, anthropic, schedule, pandas, numpy, httpx (for Brave Search)

## Cost

- ~$0.27/day Sonnet (26 cycles × ~2000 token brief)
- ~$0.02/day Haiku (lessons on closed trades)
- ~$6/month total
