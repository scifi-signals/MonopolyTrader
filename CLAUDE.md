# MonopolyTrader — An AI Stock Trading Agent That Learns

## Project Overview

MonopolyTrader is an autonomous AI-powered paper trading agent that manages a virtual portfolio of $1,000 ("Monopoly dollars") by analyzing real market data for **TSLA** and executing simulated trades. Unlike typical trading bots that follow fixed rules, MonopolyTrader is designed to **learn from its mistakes, build institutional knowledge over time, and evolve its strategies** based on accumulated experience.

**Goal**: Grow the $1,000 starting balance — not by being right on day one, but by developing increasingly accurate predictive ability through research, reflection, and adaptation. The agent will be wrong often at first. That's expected. What matters is that it gets *less wrong* over time.

## Core Philosophy: Learn, Don't Just Trade

The agent operates on three principles:

1. **Every trade is a hypothesis.** When the agent buys, it's saying "I believe X will happen because of Y." When the outcome is known, it compares prediction vs. reality and logs what it learned.

2. **Memory compounds into wisdom.** The agent maintains a growing "knowledge base" about TSLA — how it reacts to earnings, Elon's tweets, macro events, sector rotations, and seasonal patterns. This knowledge is built through both active research and post-trade reflection.

3. **Strategies earn trust.** Each strategy starts with equal weight. Over time, the agent adjusts confidence in each strategy based on real performance data. Strategies that consistently fail get deprioritized. New hybrid approaches can emerge from patterns the agent notices.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  DASHBOARD (HTML/JS)             │
│  Portfolio · Trade log · Strategy evolution ·    │
│  Learning journal · Prediction accuracy          │
├─────────────────────────────────────────────────┤
│                  REPORTER (Python)                │
│  Generates HTML dashboard + JSON API for live UI │
├─────────────────────────────────────────────────┤
│              ┌──────────────────────┐            │
│              │   LEARNING ENGINE    │            │
│              │  Post-trade reviews  │            │
│              │  Strategy evolution  │            │
│              │  Pattern discovery   │            │
│              └──────────┬───────────┘            │
│                         │                        │
│              ┌──────────▼───────────┐            │
│              │    KNOWLEDGE BASE    │            │
│              │  TSLA behavior log   │            │
│              │  Lessons learned     │            │
│              │  Strategy scores     │            │
│              │  Research findings   │            │
│              └──────────┬───────────┘            │
│                         │                        │
├─────────────────────────┼───────────────────────┤
│               AGENT BRAIN (Claude API)           │
│  Reads knowledge base · Makes predictions ·     │
│  States hypothesis before every trade            │
├─────────────────────────────────────────────────┤
│               EXECUTION ENGINE (Python)          │
│  Simulates trades · Enforces rules · Logs txns   │
├─────────────────────────────────────────────────┤
│         ┌───────────────┬───────────────┐       │
│    MARKET DATA     RESEARCH ENGINE   NEWS/SENT  │
│    Prices/OHLCV    Historical study  Headlines   │
│    Indicators      Event analysis    Sentiment   │
│    Volume          Correlation       Social      │
│         └───────────────┴───────────────┘       │
├─────────────────────────────────────────────────┤
│              PORTFOLIO STATE (JSON)               │
│  Cash · Holdings · History · Snapshots           │
└─────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Backend**: Python 3.11+
- **Market Data**: `yfinance` for price data, `ta` (Technical Analysis library) for indicators
- **AI Agent**: Anthropic Claude API (`claude-sonnet-4-20250514`) for decision-making
- **Dashboard**: HTML5 + Vanilla JS + Chart.js for interactive charts
- **Data Storage**: JSON files (portfolio state, transaction log, snapshots)
- **Scheduler**: Python `schedule` library for real-time polling (every 5 minutes during market hours)

---

## File Structure

```
monopoly-trader/
├── CLAUDE.md                 # This file — project spec and instructions
├── config.json               # Configuration (ticker, API keys, risk params)
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── portfolio.json        # Current portfolio state
│   ├── transactions.json     # Full trade history
│   ├── strategy_scores.json  # Evolving strategy confidence weights
│   └── snapshots/            # Daily portfolio value snapshots
│       └── YYYY-MM-DD.json
│
├── knowledge/                # THE AGENT'S GROWING BRAIN
│   ├── lessons.json          # Post-trade reflections and lessons learned
│   ├── tsla_profile.json     # Accumulated knowledge about TSLA behavior
│   ├── predictions.json      # Every prediction + outcome for accuracy tracking
│   ├── patterns.json         # Discovered patterns (e.g., "TSLA drops after...")
│   ├── research/             # Deep research reports the agent generates
│   │   ├── earnings_history.json
│   │   ├── catalyst_events.json
│   │   ├── correlation_notes.json
│   │   └── sector_context.json
│   └── journal.md            # Free-form agent journal — reflections in prose
│
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point — scheduler + main loop
│   ├── agent.py              # Claude-powered decision engine
│   ├── market_data.py        # Price fetching + technical indicators
│   ├── portfolio.py          # Portfolio management + trade execution
│   ├── strategies.py         # Strategy definitions + signal generators
│   ├── researcher.py         # Deep research engine — studies TSLA history
│   ├── learner.py            # Post-trade review + knowledge extraction
│   ├── knowledge_base.py     # Read/write to the knowledge/ directory
│   ├── reporter.py           # Dashboard HTML generation
│   └── utils.py              # Shared helpers (logging, time, formatting)
│
├── dashboard/
│   ├── index.html            # Main dashboard page
│   ├── style.css             # Dashboard styling
│   ├── app.js                # Dashboard interactivity + chart rendering
│   └── data.json             # Latest portfolio data for dashboard
│
└── logs/
    ├── agent.log             # Agent decision log
    └── trades.log            # Trade execution log
```

---

## Data Schemas

### config.json
```json
{
  "ticker": "TSLA",
  "starting_balance": 1000.00,
  "currency": "Monopoly Dollars",
  "poll_interval_minutes": 5,
  "market_hours": {
    "open": "09:30",
    "close": "16:00",
    "timezone": "America/New_York"
  },
  "risk_params": {
    "max_position_pct": 0.90,
    "max_single_trade_pct": 0.25,
    "stop_loss_pct": 0.05,
    "enable_fractional_shares": true
  },
  "anthropic_model": "claude-sonnet-4-20250514",
  "strategies_enabled": [
    "momentum",
    "mean_reversion",
    "sentiment",
    "technical_signals",
    "dca"
  ],
  "learning": {
    "research_on_startup": true,
    "review_after_every_trade": true,
    "deep_research_interval_hours": 24,
    "journal_entry_interval_hours": 12,
    "prediction_horizon_minutes": [30, 120, 1440]
  }
}
```

### portfolio.json
```json
{
  "cash": 1000.00,
  "holdings": {
    "TSLA": {
      "shares": 0,
      "avg_cost_basis": 0,
      "current_price": 0,
      "unrealized_pnl": 0
    }
  },
  "total_value": 1000.00,
  "total_pnl": 0,
  "total_pnl_pct": 0,
  "total_trades": 0,
  "winning_trades": 0,
  "losing_trades": 0,
  "created_at": "2026-02-18T00:00:00Z",
  "last_updated": "2026-02-18T00:00:00Z"
}
```

### Transaction Record (with hypothesis)
```json
{
  "id": "txn_001",
  "timestamp": "2026-02-18T10:35:00Z",
  "action": "BUY",
  "ticker": "TSLA",
  "shares": 1.5,
  "price": 350.25,
  "total_cost": 525.38,
  "cash_after": 474.62,
  "strategy": "momentum",
  "confidence": 0.78,
  "hypothesis": "TSLA will rise 2-4% over the next 2 hours based on strong upward momentum, positive delivery report sentiment, and RSI indicating room to run before overbought territory.",
  "reasoning": "RSI at 62 (bullish, not overbought), price above both SMA20 and SMA50 with widening gap, volume 1.3x average suggesting conviction. Recent delivery numbers beat expectations. Similar setups in my knowledge base have led to continued upside 65% of the time.",
  "signals": {
    "rsi": 62.3,
    "sma_20": 342.10,
    "sma_50": 335.80,
    "macd_signal": "bullish_crossover",
    "sentiment_score": 0.72,
    "volume_trend": "above_average"
  },
  "knowledge_applied": ["lesson_012", "pattern_005"],
  "review": null
}
```

### Prediction Record (predictions.json)
```json
{
  "id": "pred_001",
  "timestamp": "2026-02-18T10:35:00Z",
  "price_at_prediction": 350.25,
  "predictions": {
    "30min": {"direction": "up", "target": 353.50, "confidence": 0.72},
    "2hr": {"direction": "up", "target": 358.00, "confidence": 0.65},
    "1day": {"direction": "up", "target": 362.00, "confidence": 0.55}
  },
  "reasoning": "Momentum continuation after positive delivery report...",
  "outcomes": {
    "30min": {"actual": 351.80, "direction_correct": true, "error_pct": 0.48},
    "2hr": {"actual": 347.20, "direction_correct": false, "error_pct": 3.10},
    "1day": null
  },
  "linked_trade": "txn_001"
}
```

### Lesson Record (lessons.json)
```json
{
  "id": "lesson_012",
  "timestamp": "2026-02-18T14:45:00Z",
  "linked_trade": "txn_001",
  "what_i_predicted": "2-4% rise over 2 hours",
  "what_actually_happened": "Rose 0.4% in 30min then reversed, ending 2hr window down 0.9%",
  "why_i_was_wrong": "Overweighted delivery report sentiment. The positive news was already priced in by market open. Volume spike was actually profit-taking, not new buying conviction.",
  "lesson": "For TSLA, positive earnings/delivery beats that are reported pre-market are usually priced in by 10:30 AM. Momentum signals after this point may be misleading. Check if the news is already reflected in the opening gap.",
  "category": "sentiment_timing",
  "confidence_adjustment": {
    "sentiment": -0.08,
    "momentum": -0.05
  },
  "times_validated": 0,
  "times_contradicted": 0
}
```

### Pattern Record (patterns.json)
```json
{
  "id": "pattern_005",
  "discovered": "2026-02-25T16:10:00Z",
  "name": "TSLA Post-Earnings Drift",
  "description": "After TSLA earnings beats, the stock tends to drift upward for 3-5 trading days as analysts revise targets, then consolidate. After earnings misses, the drop is usually front-loaded (day 1) with partial recovery by day 3.",
  "evidence": ["txn_003", "txn_008", "research_earnings_q4_2025"],
  "reliability": 0.62,
  "sample_size": 8,
  "last_tested": "2026-02-25T00:00:00Z",
  "tags": ["earnings", "drift", "multi-day"]
}
```

### Strategy Scores (strategy_scores.json)
```json
{
  "last_updated": "2026-02-25T16:15:00Z",
  "strategies": {
    "momentum": {
      "weight": 0.22,
      "initial_weight": 0.20,
      "total_trades": 12,
      "winning_trades": 5,
      "win_rate": 0.42,
      "avg_return_pct": -0.3,
      "total_pnl": -14.50,
      "trend": "declining",
      "notes": "Performing poorly in current choppy market. Consider deprioritizing until clear trend emerges."
    },
    "mean_reversion": {
      "weight": 0.28,
      "initial_weight": 0.20,
      "total_trades": 8,
      "winning_trades": 6,
      "win_rate": 0.75,
      "avg_return_pct": 1.2,
      "total_pnl": 45.20,
      "trend": "improving",
      "notes": "Strong performer. TSLA's high volatility creates frequent reversion opportunities."
    }
  },
  "rebalance_history": [
    {
      "timestamp": "2026-02-22T16:10:00Z",
      "changes": {"momentum": -0.03, "mean_reversion": +0.05},
      "reason": "Weekly review: momentum underperforming in range-bound market"
    }
  ]
}
```

---

## Module Specifications

### 1. market_data.py — Market Data Service

**Responsibilities:**
- Fetch real-time and historical OHLCV data via `yfinance`
- Calculate technical indicators using `ta` library
- Provide a clean data bundle for the agent

**Key Functions:**
```python
get_current_price(ticker: str) -> dict
# Returns: {price, change, change_pct, volume, timestamp}

get_price_history(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame
# Returns OHLCV DataFrame

get_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame
# Returns intraday candles

calculate_indicators(df: pd.DataFrame) -> dict
# Returns: {rsi_14, sma_20, sma_50, ema_12, ema_26, macd, macd_signal,
#           bollinger_upper, bollinger_lower, atr, volume_sma_20, obv}

get_market_summary(ticker: str) -> dict
# Bundles current price + indicators + recent history into one payload
```

### 2. strategies.py — Strategy Engine

**Responsibilities:**
- Define each trading strategy as a signal generator
- Each strategy returns a signal: BUY / SELL / HOLD with confidence (0-1)
- **Confidence is weighted by the strategy's historical performance score**
- Combine signals from multiple strategies using evolved weights

**Strategies:**

| Strategy | Signal Logic |
|----------|-------------|
| **Momentum** | Buy when price > SMA20 > SMA50, RSI 50-70; Sell when price < SMA20, RSI > 75 |
| **Mean Reversion** | Buy when price < lower Bollinger, RSI < 30; Sell when price > upper Bollinger, RSI > 70 |
| **Sentiment** | Analyze recent news headlines via Claude; score -1 to +1 |
| **Technical Signals** | MACD crossovers, volume spikes, support/resistance levels |
| **DCA (Dollar Cost Averaging)** | Small periodic buys regardless of price, with tactical adjustments |

```python
class StrategySignal:
    action: str        # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0 (raw signal strength)
    weight: float      # from strategy_scores.json (earned trust)
    strategy: str      # strategy name
    reasoning: str     # human-readable explanation

def evaluate_all_strategies(market_data: dict, portfolio: dict, scores: dict) -> list[StrategySignal]
# Returns weighted signals from all enabled strategies

def aggregate_signals(signals: list[StrategySignal]) -> StrategySignal
# Weighted combination using evolved trust scores
```

### 3. researcher.py — Deep Research Engine (NEW)

**Responsibilities:**
- Study TSLA's historical behavior to build predictive knowledge
- Research runs on a schedule (daily) and on-demand before major events
- Uses Claude API + web search to gather and synthesize findings
- Writes findings to the knowledge base

**Research Tasks (run periodically):**

```python
async def research_earnings_history(ticker: str) -> dict:
    """Study how TSLA has reacted to past earnings reports.
    Analyze: beat/miss, pre-report drift, post-report move, 
    recovery timeline, options activity patterns."""

async def research_catalyst_events(ticker: str) -> dict:
    """Study impact of Elon tweets, product announcements, 
    delivery reports, regulatory news, analyst upgrades/downgrades.
    Build a catalog of event types → typical market reactions."""

async def research_correlations(ticker: str) -> dict:
    """Study TSLA's correlation with: NASDAQ, interest rates, 
    oil prices, BTC, other EV stocks. Identify leading indicators."""

async def research_seasonal_patterns(ticker: str) -> dict:
    """Study day-of-week effects, monthly patterns, 
    options expiration effects, quarter-end behavior."""

async def research_current_context(ticker: str) -> dict:
    """What's happening RIGHT NOW that matters? 
    Upcoming earnings, pending regulations, macro environment,
    competitor news, supply chain developments."""

async def research_on_demand(ticker: str, topic: str) -> dict:
    """Agent can request targeted research on a specific question
    it encounters during trading, e.g., 'How does TSLA react 
    when 10-year yield crosses 4.5%?'"""
```

**Output**: Each research task produces a structured report that gets stored in `knowledge/research/` and summarized into `knowledge/tsla_profile.json`.

### 4. learner.py — Learning Engine (NEW)

**Responsibilities:**
- Conduct post-trade reviews comparing predictions to outcomes
- Extract lessons and update the knowledge base
- Adjust strategy confidence weights based on performance
- Discover new patterns from trade history
- Write journal entries reflecting on overall progress

**The Learning Loop:**

```
TRADE HAPPENS → Wait for outcome → REVIEW
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
           Extract Lesson      Adjust Strategy     Discover Pattern
           (what went wrong    Scores (reward/     (do I see this
            or right, why)      punish strategy)    pattern repeating?)
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                              UPDATE KNOWLEDGE BASE
                              (lessons, patterns, scores, journal)
```

**Key Functions:**

```python
async def review_trade(trade: dict, outcome: dict) -> dict:
    """After a trade closes (or after prediction horizon passes):
    - Compare hypothesis vs actual outcome
    - Call Claude to analyze what went right/wrong
    - Generate a lesson record
    - Return confidence adjustments for strategies used"""

async def review_predictions(predictions: list) -> dict:
    """Check all pending predictions whose time horizon has passed.
    Score accuracy. Update running accuracy metrics."""

async def discover_patterns(trades: list, lessons: list) -> list:
    """Periodically analyze trade history + lessons looking for
    recurring patterns. E.g., 'I keep losing on momentum trades
    during the first hour — maybe TSLA's opening volatility 
    makes momentum unreliable early in the session.'"""

async def evolve_strategy_weights(scores: dict, recent_trades: list) -> dict:
    """Recalculate strategy weights based on recent performance.
    Uses exponential decay — recent trades matter more than old ones.
    Ensures no strategy drops below 0.05 weight (always give it a chance)."""

async def write_journal_entry(portfolio: dict, knowledge: dict) -> str:
    """Periodic reflection: How am I doing? What's working? 
    What should I try differently? Written in natural language 
    and appended to knowledge/journal.md"""
```

### 5. knowledge_base.py — Knowledge Manager (NEW)

**Responsibilities:**
- CRUD operations for all knowledge files
- Query knowledge relevant to current trading decisions
- Summarize knowledge for inclusion in agent prompts
- Manage knowledge size (prune old, low-value entries)

**Key Functions:**

```python
def get_relevant_knowledge(market_context: dict) -> dict:
    """Given current market conditions, retrieve the most relevant:
    - Lessons (matching current signals/conditions)
    - Patterns (matching current setup)
    - Research findings (relevant to current catalysts)
    Returns a condensed knowledge bundle for the agent prompt."""

def add_lesson(lesson: dict) -> None
def add_pattern(pattern: dict) -> None  
def add_research(topic: str, findings: dict) -> None
def update_strategy_scores(scores: dict) -> None

def get_prediction_accuracy() -> dict:
    """Calculate rolling accuracy stats:
    - Overall direction accuracy (30min, 2hr, 1day)
    - Accuracy by strategy
    - Accuracy trend over time (improving?)
    - Biggest blind spots"""

def get_knowledge_summary() -> str:
    """One-paragraph summary of what the agent has learned so far.
    Used in dashboard and journal entries."""
```

### 6. agent.py — AI Decision Engine

**Responsibilities:**
- Receive market data + strategy signals + portfolio state + **relevant knowledge**
- Make predictions with explicit hypotheses
- Use Claude API to make final BUY/SELL/HOLD decision
- Every decision references what it's learned

**Agent Prompt Structure:**
```
SYSTEM: You are MonopolyTrader, an AI trading agent managing a virtual 
portfolio. You are LEARNING — you started with $1,000 and your goal is to 
grow it by getting better at predicting TSLA's movements over time.

You have a knowledge base of lessons from past trades, patterns you've 
discovered, and research you've conducted. USE THIS KNOWLEDGE. Reference 
specific lessons and patterns when making decisions. If you're uncertain, 
say so — it's better to HOLD than to make a low-conviction trade.

Every trade is a hypothesis. State clearly what you predict will happen 
and why. You will be reviewed on this prediction later.

USER: 
<market_data>{current price, indicators, recent candles}</market_data>
<strategy_signals>{weighted signals from all strategies}</strategy_signals>
<portfolio>{current holdings, cash, P&L, recent trades}</portfolio>
<relevant_knowledge>
  <lessons>{lessons matching current conditions}</lessons>
  <patterns>{patterns that may apply}</patterns>
  <research>{relevant research findings}</research>
  <prediction_accuracy>{your recent accuracy stats}</prediction_accuracy>
</relevant_knowledge>

Analyze the data and decide: BUY, SELL, or HOLD.
Respond in JSON: {
  action, shares, confidence, strategy,
  hypothesis: "I predict TSLA will [direction] by [amount] over [timeframe] because [reasoning]",
  reasoning,
  knowledge_applied: [list of lesson/pattern IDs used],
  risk_note,
  research_request: "optional — if you want to research something before next decision"
}
```

### 7. portfolio.py — Portfolio Manager

**Responsibilities:**
- Execute simulated trades (update cash, holdings)
- Enforce risk rules (position limits, stop losses)
- Calculate P&L (realized + unrealized)
- Save/load portfolio state
- Create daily snapshots

**Key Functions:**
```python
def execute_trade(action: str, shares: float, price: float, decision: dict) -> dict
# Validates trade against risk rules, executes, logs, returns result

def check_stop_losses(current_price: float) -> Optional[dict]
# Returns forced sell signal if stop loss triggered

def get_portfolio_summary() -> dict
# Current state with all P&L calculations

def save_snapshot() -> None
# Save current state to snapshots/YYYY-MM-DD.json

def get_performance_vs_benchmark() -> dict
# Compare portfolio returns vs buy-and-hold TSLA + SPY
```

### 8. reporter.py — Dashboard Generator

**Responsibilities:**
- Generate the HTML dashboard with embedded data
- Create JSON data files for the JS frontend

**Dashboard Sections:**
- 📈 **Portfolio Value Chart** — Line chart of total value over time vs TSLA buy-and-hold vs SPY
- 💰 **Current Position** — Holdings, cash, total value, P&L
- 🧠 **Learning Progress** — Prediction accuracy over time (is the agent improving?)
- 📊 **Strategy Evolution** — How strategy weights have shifted over time (area chart)
- 🎯 **Prediction Scoreboard** — Recent predictions with outcomes (correct/incorrect)
- 📋 **Trade Log** — Sortable table with hypothesis, outcome, and lesson for each trade
- 📓 **Agent Journal** — The agent's own reflections on its progress
- 🔬 **Knowledge Base Browser** — Lessons, patterns, research findings
- ⚡ **Live Status** — Current price, last decision, next poll time

### 9. main.py — Orchestrator

**The Decision Cycle (every 5 minutes during market hours):**
```python
def run_cycle():
    1. Fetch market data + calculate indicators
    2. Run strategy signals (weighted by evolved scores)
    3. Check stop losses (auto-sell if triggered)
    4. Query knowledge base for relevant context
    5. Call agent for decision (with knowledge context)
    6. If action != HOLD: execute trade, record hypothesis
    7. Check pending predictions — score any whose horizon has passed
    8. If a recent trade closed: run post-trade review → extract lesson
    9. Update dashboard data
    10. Log everything
```

**The Research Cycle (daily, after market close):**
```python
def run_daily_research():
    1. Run all research tasks (earnings, catalysts, correlations, context)
    2. Discover new patterns from recent trade history
    3. Evolve strategy weights based on recent performance
    4. Write journal entry
    5. Generate full dashboard report
    6. Save daily snapshot
```

**The Bootstrap (first run):**
```python
def bootstrap():
    """On very first run, before any trading:
    1. Initialize portfolio with $1,000
    2. Run ALL research tasks to build initial knowledge base
    3. Study last 2 years of TSLA price action
    4. Identify current market regime (trending, range-bound, volatile)
    5. Write initial journal entry: 'Day 1 — here's what I know so far'
    6. Start trading with informed (but humble) initial positions"""
```

---

## Risk Management Rules

1. **Max Position Size**: Never invest more than 90% of portfolio in a single stock
2. **Max Single Trade**: No single trade larger than 25% of portfolio value
3. **Stop Loss**: Auto-sell if position drops 5% below cost basis
4. **Fractional Shares**: Enabled — allows meaningful positions even with small portfolio
5. **Cash Reserve**: Always maintain at least 10% cash for opportunities
6. **Cool-down**: Minimum 15 minutes between trades on same ticker
7. **Daily Loss Limit**: If portfolio drops 8% in a day, stop trading until next day

---

## Dependencies (requirements.txt)

```
yfinance>=0.2.31
ta>=0.11.0
anthropic>=0.40.0
schedule>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0
```

---

## Build Order

Build and test in this order:

### Phase 1 — Foundation
1. `config.json` — Set up configuration
2. `market_data.py` — Get price data flowing and indicators calculating
3. `portfolio.py` — Portfolio state management and trade simulation
4. `knowledge_base.py` — Read/write knowledge files (start with empty scaffolding)
5. **Test**: Verify you can fetch TSLA data, calculate indicators, and execute a mock trade

### Phase 2 — Intelligence
6. `strategies.py` — Implement all 5 strategy signal generators with dynamic weighting
7. `agent.py` — Wire up Claude API for decision-making with knowledge context
8. **Test**: Feed real data through strategies → agent → mock trade. Verify hypothesis is recorded.

### Phase 3 — Learning (the heart of the project)
9. `learner.py` — Post-trade review, lesson extraction, pattern discovery, strategy evolution
10. `researcher.py` — Deep research tasks that build TSLA knowledge
11. **Test**: Execute a trade, wait for outcome, run review, verify lesson is created and strategy scores update

### Phase 4 — Automation
12. `main.py` — Build the scheduler with both decision cycle (5min) and research cycle (daily)
13. Implement bootstrap sequence for first run (research before trading)
14. `utils.py` — Logging, time helpers, formatting
15. **Test**: Run for a full market day. Verify predictions are scored, lessons extracted, journal written.

### Phase 5 — Dashboard
16. `reporter.py` — Generate data JSON for dashboard
17. `dashboard/index.html` + `style.css` + `app.js` — Build interactive dashboard
18. Include: portfolio chart, prediction accuracy trend, strategy evolution chart, trade log with lessons, journal viewer
19. **Test**: Run for several days, verify dashboard tells the story of the agent's learning journey

### Phase 6 — Polish & Maturation
20. Add benchmark comparison (buy-and-hold TSLA, SPY)
21. Tune the learning loop — are lessons actually improving decisions?
22. Add "what I'd do differently" retrospective analysis
23. Stress test error handling (market closed, API down, etc.)
24. Add ability for the agent to request custom research topics

---

## Running the Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your-key-here

# Initialize portfolio and start trading
python src/main.py

# Or run a single decision cycle (for testing)
python src/main.py --once

# Generate dashboard report
python src/main.py --report

# View dashboard
open dashboard/index.html
```

---

## Success Metrics

The agent's success is measured not just by P&L, but by **learning trajectory**:

1. **Prediction Accuracy Trend** — Is the agent getting better at predicting direction over time? Track 7-day rolling accuracy for 30min, 2hr, and 1-day predictions. Week 1 might be 50% (coin flip). If it's 60%+ by week 4, that's real learning.

2. **Strategy Evolution** — Are strategy weights shifting in response to performance? Are the winning strategies getting more weight?

3. **Lesson Quality** — Are lessons specific and actionable, or vague? Good: "TSLA drops after Elon tweets about non-Tesla ventures during market hours." Bad: "Sometimes the stock goes down."

4. **Journal Insight** — Read the agent's journal entries. Is it developing a coherent mental model of TSLA's behavior?

5. **P&L vs Benchmark** — Is the portfolio outperforming simple buy-and-hold? (Don't expect this early — the learning has to accumulate first.)

---

## Notes

- The agent uses **Monopoly dollars** — this is paper trading only, no real money involved
- All trades execute at real market prices fetched from Yahoo Finance
- A small slippage model (0.1%) is applied to simulate real execution
- The agent's reasoning is fully logged for post-analysis
- Dashboard auto-refreshes every 60 seconds when open
- **The agent is expected to lose money early.** The design prioritizes learning rate over initial returns. A smart agent that loses $50 in week 1 but develops accurate TSLA intuition is better than one that gets lucky on a few trades.
- The knowledge base is the agent's most valuable asset — protect it, back it up, and watch it grow
