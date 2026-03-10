# MonopolyTrader v5 — Informed Trading with Persistent Intelligence

## Overview

Autonomous AI paper trading agent. Manages $1,000 virtual portfolio trading TSLA. Claude makes every decision — no coded strategies, no stop losses. v5 adds persistent market intelligence, nightly research, and richer data sources.

**Dashboard:** https://scifi-signals.github.io/MonopolyTrader/dashboard/
**Server:** DigitalOcean (`science-intel`), systemd service `monopoly-trader` at `/root/monopoly-trader`

## Design Philosophy

- Paper money = learning budget, not capital to preserve
- Every scenario is an opportunity — trade aggressively to learn
- Claude sees raw data AND accumulated intelligence — no amnesia between cycles
- No stop losses — Claude evaluates underwater positions with context every cycle
- Learning compounds: every trade feeds back into the intelligence system

## Three Rules (only 3)

1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK

## v5 Architecture

### Daily Cycle

**9:00 AM — Pre-Market Research** (Haiku)
- Reads Market Intelligence Document + pre-market price + overnight news
- Produces daily briefing with scenarios, thesis status, recommended posture

**Every 15 minutes — Trading Cycle** (Sonnet)
1. Gather data — TSLA price/indicators + options + analyst consensus + institutional data + world + news + events
2. Build brief — Market data + daily briefing + portfolio + playbook + journal
3. Claude decides — System prompt includes Market Intelligence Document
4. Execute trade — Simulate with slippage
5. Record in journal — Tags computed, entry logged
6. On trade close — Haiku lesson + immediate playbook rebuild
7. Monitor portfolio health
8. Update dashboard

**4:15 PM — Nightly Analysis** (Haiku)
- Rebuilds playbook from all closed trades
- Synthesizes today's trades, lessons, playbook stats, news into updated Market Intelligence Document
- Evaluates thesis: confirm, adjust confidence, or flip direction

### Market Intelligence Document (data/market_intelligence.json)

Persistent JSON in the SYSTEM PROMPT. Updated nightly by Haiku. Contains:
- **Thesis**: direction (bull/bear/neutral), confidence (0.3-0.9), evidence, invalidation criteria
- **Key levels**: support/resistance from recent price action
- **Active catalysts**: what's driving TSLA right now
- **Sector context**: EV sector and macro positioning
- **Lessons synthesis**: patterns from recent trade lessons
- **What's working/not working**: from playbook stats

Rules for thesis updates:
- 3+ consecutive losses OR major catalyst to flip direction
- P&L-based confidence decay (negative P&L over 7 days → lower confidence)
- Always includes contradicting evidence

## File Structure

```
src/
├── main.py          — Scheduler: 15min cycles + 9AM pre-market + 4:15PM nightly
├── analyst.py       — NEW: Nightly MID update + pre-market briefing (Haiku)
├── agent.py         — SYSTEM_PROMPT (with MID) + build_market_brief() + make_decision()
├── tags.py          — Mechanical trade tagging (8 market condition tags)
├── thesis_builder.py — Playbook aggregation (rebuilds on every trade close)
├── market_data.py   — Prices, indicators, regime, world, options, analyst, institutional
├── portfolio.py     — 2-rule validation, execute_trade(), cooldown
├── journal.py       — Trade journal with tags, Haiku lessons, get_entries_since()
├── events.py        — FOMC/CPI/NFP calendar + TSLA earnings
├── news_feed.py     — yfinance + RSS news
├── web_search.py    — Web search (disabled)
├── reporter.py      — Dashboard data generation
├── observability.py — Health checks
└── utils.py         — Config, logging, AI call helpers (supports model override)

data/
├── market_intelligence.json — Persistent MID (updated nightly)
├── daily_briefing.json      — Today's pre-market analysis
├── portfolio.json           — Current portfolio state
├── transactions.json        — Full trade history
├── trade_journal.json       — Journal entries with tags and lessons
├── thesis_ledger.json       — Playbook stats (rebuilt on every trade close)
├── snapshots/               — Daily portfolio snapshots
└── latest_cycle.json        — Last decision for dashboard
```

## What Claude Sees Each Cycle

**System Prompt (persistent context):**
- Trading rules (3 rules)
- Market Intelligence Document (thesis, levels, catalysts, lessons, what works)

**User Prompt (per cycle):**
- Today's daily briefing (scenarios, thesis status, posture)
- TSLA price + 18 daily indicators + 5 intraday indicators
- Options flow (put/call ratio, max pain, unusual volume)
- Analyst consensus (buy/hold/sell counts, price targets)
- Institutional data (ownership %, short interest)
- World macro (7 tickers) + EV peers (5 tickers)
- Regime classification
- News (yfinance + RSS, classified by catalyst type)
- Upcoming events (FOMC, CPI, NFP, TSLA earnings)
- Playbook (statistical performance by market condition)
- Portfolio state + position limits
- Last 5 trades with lessons

## Key Commands

```bash
python -m src.main             # Start scheduler
python -m src.main --once      # Single cycle
python -m src.main --report    # Generate dashboard
python -m src.main --analyst   # Run nightly analyst
python -m src.main --premarket # Run pre-market briefing
```

## Server

```bash
systemctl status monopoly-trader
systemctl restart monopoly-trader
tail -f /root/monopoly-trader/logs/agent.log
```

## Success Benchmark

Beat buy-and-hold TSLA over rolling 30-day windows, measured by:
- Total return vs buy-and-hold
- Max drawdown
- Win rate on closed trades
- Sharpe ratio

## Cost

- ~$0.30/day Sonnet (26 cycles × ~2500 token brief)
- ~$0.07/day Haiku (nightly analyst + pre-market + trade lessons)
- ~$8.50/month total
