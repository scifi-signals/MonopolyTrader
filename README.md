# MonopolyTrader

An autonomous AI paper trading agent that manages a $1,000 virtual portfolio trading TSLA. It treats every trade as a falsifiable hypothesis, learns from mistakes through a built-in skeptic layer, and must prove it can beat random traders before graduating to real money.

**[Live Dashboard](https://scifi-signals.github.io/MonopolyTrader/dashboard/)** | Running autonomously on a server since Feb 18, 2026

## How It Works

Every 15 minutes during market hours, the agent:

1. Fetches TSLA price data and calculates technical indicators
2. Runs 5 parallel strategies (momentum, mean reversion, sentiment, technical signals, DCA)
3. Queries its knowledge base for relevant lessons from past trades
4. Asks Claude (Sonnet 4) to make a BUY/SELL/HOLD decision with an explicit hypothesis
5. Executes the trade (paper), records the prediction, and waits to see if it was right
6. After the outcome is known, a **Skeptic layer** challenges the lesson before it enters the knowledge base

The agent also tracks every HOLD decision with counterfactual analysis — "what would have happened if I had acted?" — so it learns from inaction too.

### The Learning Loop

```
Trade/HOLD → Record hypothesis → Wait for outcome → Post-trade review
                                                          |
                    ┌─────────────────┬──────────────────┤
                    v                 v                   v
             Extract lesson    Adjust strategy     Discover patterns
             (Skeptic challenges)  weights          (recurring setups)
                    |                 |                   |
                    └─────────────────┴──────────────────┘
                                      v
                              Update knowledge base
                              (next cycle is smarter)
```

### Risk Management

- Dynamic ATR-based stop losses (not fixed %, which causes whipsaw on TSLA)
- Position sizing inversely proportional to volatility
- Macro gate: requires higher conviction when SPY drops >2% or VIX >30
- Max 65% in one stock, max 20% per trade, min 15% cash reserve
- 8% daily loss limit, 15% max drawdown kill switch
- 48-hour earnings blackout

### Graduation Path

The agent must meet 12 criteria sustained over 30 days before it gets real money:

- 90+ trading days, 50+ trades
- Above 75th percentile vs 100 simulated random traders
- Sharpe ratio > 0.5, max drawdown < 15%
- Prediction accuracy > 55%
- Beat buy-and-hold TSLA, SPY, and DCA baselines

Currently: **3/12 criteria met** (very early — 6 days in).

## Quick Start

```bash
git clone https://github.com/scifi-signals/MonopolyTrader.git
cd MonopolyTrader
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here

# Run a single decision cycle
python -m src.main --once

# Run continuously (15-min intervals during market hours)
python -m src.main

# Generate dashboard report
python -m src.main --report
```

Then open `dashboard/index.html` in a browser.

## Tech Stack

- **Python 3.11+** — core agent
- **Claude Sonnet 4** (Anthropic API) — decision engine + learning reviews
- **yfinance** — real-time and historical market data
- **ta** — technical indicator calculations
- **HTML5 + Chart.js** — interactive dashboard

## Project Structure

```
monopoly-trader/
├── config.json               # All settings (risk params, strategies, learning)
├── src/
│   ├── main.py               # Scheduler + orchestration
│   ├── agent.py              # Claude-powered decision engine
│   ├── learner.py            # Post-trade review + Skeptic layer
│   ├── strategies.py         # 5 strategy signal generators
│   ├── market_data.py        # Price data + indicators + regime detection
│   ├── portfolio.py          # Trade execution + risk enforcement
│   ├── knowledge_base.py     # Read/write the agent's growing brain
│   ├── researcher.py         # Deep TSLA research (daily)
│   ├── reporter.py           # Dashboard data generation
│   ├── benchmarks.py         # Agent vs buy-and-hold, SPY, DCA, random
│   ├── observability.py      # Decision tracing + anomaly detection
│   └── utils.py              # Logging, config, API helpers
├── dashboard/                # Live web dashboard
├── knowledge/                # Lessons, patterns, research, journal
├── data/                     # Portfolio state, transactions, snapshots
└── logs/                     # Agent logs + daily API cost reports
```

## Philosophy

Most "AI trading" projects are prompt-and-pray — they ask an LLM what to buy and hope for the best. MonopolyTrader is different:

- **Every trade is a hypothesis** that gets tested against reality
- **The Skeptic layer** challenges every lesson before it's saved — "was this really your signal working, or did the whole market move?"
- **Strategy weights are earned**, not assigned — strategies that perform well gain influence, ones that don't lose it
- **Benchmarks keep it honest** — if the agent can't beat a random trader, it's not ready for real money
- **Losses are data** — the agent is expected to lose early. What matters is that it gets less wrong over time.

## Disclaimer

This is a paper trading research project. No real money is involved. Past performance of a simulated portfolio does not indicate future results. This is not financial advice.
