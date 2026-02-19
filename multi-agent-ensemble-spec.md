# Multi-Agent Ensemble System — Running Parallel Strategies

## The Problem with a Single Agent

A single agent blending all strategies into one decision has a fundamental observability problem: you can't isolate what's working. If the portfolio gains 8%, was it the momentum signal that drove most of the profitable trades? Was BSM intelligence the edge? Did the knowledge base actually help, or did the agent succeed despite it? With one blended portfolio, you can't answer these questions cleanly.

## The Solution: Parallel Agents

Run multiple independent agents simultaneously, each with different configurations, risk tolerances, and strategy emphases. They all see the same market data and BSM signals, but they interpret and weight them differently.

Each agent gets its own $1,000, its own portfolio, its own trade log, its own knowledge base, and its own learning loop. They operate completely independently — no agent knows what the others are doing.

Then: watch, compare, learn.

---

## Agent Roster

### The Starting Lineup

Each agent has a distinct "personality" defined by which signals it trusts and how it trades.

```json
{
  "agents": [
    {
      "id": "alpha",
      "name": "The Technician",
      "description": "Pure technical analysis. No sentiment, no BSM, no news. Just price action, volume, and indicators.",
      "strategy_weights": {
        "momentum": 0.35,
        "mean_reversion": 0.35,
        "technical_signals": 0.30,
        "sentiment": 0.00,
        "dca": 0.00
      },
      "bsm_enabled": false,
      "news_sentiment_enabled": false,
      "trade_frequency": "moderate",
      "risk_profile": "standard",
      "hypothesis": "Technical indicators alone provide sufficient edge on a volatile stock like TSLA."
    },
    {
      "id": "bravo",
      "name": "The Insider",
      "description": "Sentiment and intelligence focused. Relies heavily on BSM signals, news, and social sentiment. Minimal technicals — only uses them for entry/exit timing.",
      "strategy_weights": {
        "momentum": 0.05,
        "mean_reversion": 0.05,
        "technical_signals": 0.10,
        "sentiment": 0.80,
        "dca": 0.00
      },
      "bsm_enabled": true,
      "bsm_weight": 0.60,
      "news_sentiment_enabled": true,
      "trade_frequency": "low",
      "risk_profile": "standard",
      "hypothesis": "Information from influential figures provides actionable alpha that precedes price movements."
    },
    {
      "id": "charlie",
      "name": "The Tortoise",
      "description": "Conservative, patience-focused. Primarily DCA with tactical adjustments. Only deviates from DCA schedule on high-conviction signals. Trades rarely.",
      "strategy_weights": {
        "momentum": 0.05,
        "mean_reversion": 0.10,
        "technical_signals": 0.05,
        "sentiment": 0.10,
        "dca": 0.70
      },
      "bsm_enabled": true,
      "bsm_weight": 0.15,
      "news_sentiment_enabled": true,
      "trade_frequency": "very_low",
      "risk_profile": "conservative",
      "risk_overrides": {
        "max_single_trade_pct": 0.10,
        "stop_loss_pct": 0.03,
        "daily_loss_limit_pct": 0.04
      },
      "hypothesis": "Slow, steady accumulation with minimal trading beats frequent active trading on a volatile stock."
    },
    {
      "id": "delta",
      "name": "The Momentum Surfer",
      "description": "Aggressive trend follower. Jumps on momentum early, rides it hard, exits fast. High trade frequency. Lives and dies by momentum.",
      "strategy_weights": {
        "momentum": 0.80,
        "mean_reversion": 0.00,
        "technical_signals": 0.15,
        "sentiment": 0.05,
        "dca": 0.00
      },
      "bsm_enabled": false,
      "news_sentiment_enabled": false,
      "trade_frequency": "very_high",
      "risk_profile": "aggressive",
      "risk_overrides": {
        "max_position_pct": 0.95,
        "max_single_trade_pct": 0.40,
        "cooldown_minutes": 10
      },
      "hypothesis": "TSLA's high volatility creates strong momentum opportunities that can be captured with aggressive, fast trading."
    },
    {
      "id": "echo",
      "name": "The Generalist",
      "description": "Balanced blend of all strategies and signals. This is closest to the original single-agent design. Equal starting weights, evolves over time.",
      "strategy_weights": {
        "momentum": 0.20,
        "mean_reversion": 0.20,
        "technical_signals": 0.20,
        "sentiment": 0.20,
        "dca": 0.20
      },
      "bsm_enabled": true,
      "bsm_weight": 0.30,
      "news_sentiment_enabled": true,
      "trade_frequency": "moderate",
      "risk_profile": "standard",
      "learning_enabled": true,
      "evolve_weights": true,
      "hypothesis": "A diversified approach that learns and adapts its strategy mix over time will outperform any single fixed strategy."
    },
    {
      "id": "foxtrot",
      "name": "The Contrarian",
      "description": "Fades the consensus. When technicals and sentiment are extremely bullish, it sells. When everyone is panicking, it buys. Bets on mean reversion at extremes.",
      "strategy_weights": {
        "momentum": -0.30,
        "mean_reversion": 0.60,
        "technical_signals": 0.10,
        "sentiment": -0.20,
        "dca": 0.00
      },
      "bsm_enabled": true,
      "bsm_weight": -0.20,
      "invert_sentiment": true,
      "news_sentiment_enabled": true,
      "trade_frequency": "low",
      "risk_profile": "standard",
      "hypothesis": "TSLA is prone to sentiment-driven overreactions. Fading extremes and buying fear produces better returns than following the crowd."
    },
    {
      "id": "golf",
      "name": "The Researcher",
      "description": "Only trades when the knowledge base has a high-confidence pattern match. Sits in cash most of the time. When it trades, it's high conviction.",
      "strategy_weights": {
        "momentum": 0.10,
        "mean_reversion": 0.10,
        "technical_signals": 0.10,
        "sentiment": 0.10,
        "dca": 0.00
      },
      "knowledge_weight": 0.60,
      "min_knowledge_confidence": 0.70,
      "bsm_enabled": true,
      "bsm_weight": 0.40,
      "trade_frequency": "very_low",
      "risk_profile": "standard",
      "hypothesis": "Fewer, higher-conviction trades based on deep research outperform frequent lower-conviction trades."
    },
    {
      "id": "hotel",
      "name": "The Scalper",
      "description": "Very short time horizon. Tries to capture small intraday moves (0.5-1%). Trades frequently, small position sizes, tight stop losses.",
      "strategy_weights": {
        "momentum": 0.40,
        "mean_reversion": 0.30,
        "technical_signals": 0.30,
        "sentiment": 0.00,
        "dca": 0.00
      },
      "bsm_enabled": false,
      "trade_frequency": "extreme",
      "risk_profile": "tight",
      "risk_overrides": {
        "max_single_trade_pct": 0.15,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.01,
        "cooldown_minutes": 5
      },
      "hypothesis": "Many small wins with tight risk management can compound better than fewer large bets."
    }
  ]
}
```

### Adding More Agents

You can add agents at any time. Some ideas for later:
- **"The Oracle"** — Only trades around known catalysts (earnings, delivery reports, Fed meetings)
- **"The Copycat"** — Mirrors 13F filings from tracked billionaires (delayed, but proven positions)
- **"The Hybrid"** — Dynamically blends the top 3 performing agents' signals

---

## Architecture Changes

### From Single Agent to Multi-Agent

```
BEFORE (single agent):
┌──────────┐    ┌───────┐    ┌───────────┐
│ Market   ├───►│ Agent ├───►│ Portfolio │
│ Data     │    │       │    │           │
└──────────┘    └───────┘    └───────────┘

AFTER (multi-agent ensemble):
                              ┌──────────────────────┐
                              │   META-LEARNER        │
                              │   Watches all agents  │
                              │   Identifies combos   │
                              │   Suggests new agents │
                              └──────────┬───────────┘
                                         │
┌──────────┐    ┌─────────────────────────┼─────────────────────────┐
│ Market   │    │                         │                         │
│ Data     ├───►│  ┌───────┐  ┌───────┐  │  ┌───────┐  ┌───────┐  │
│ + BSM    │    │  │Alpha  │  │Bravo  │  │  │Charlie│  │ ...   │  │
│ Signals  │    │  │$1,000 │  │$1,000 │  │  │$1,000 │  │$1,000 │  │
│          │    │  └───┬───┘  └───┬───┘  │  └───┬───┘  └───┬───┘  │
└──────────┘    │      │          │      │      │          │      │
                │      ▼          ▼      │      ▼          ▼      │
                │  ┌──────────────────────────────────────────┐    │
                │  │         COMPARISON ENGINE                │    │
                │  │  Leaderboard · Correlation · Harmony     │    │
                │  └──────────────────────────────────────────┘    │
                └─────────────────────────────────────────────────┘
```

### File Structure Changes

```
monopoly-trader/
├── CLAUDE.md
├── config.json                    # Global config
├── agents/                        # Per-agent configuration
│   ├── alpha.json
│   ├── bravo.json
│   ├── charlie.json
│   └── ...
│
├── data/
│   ├── agents/                    # Per-agent portfolio and state
│   │   ├── alpha/
│   │   │   ├── portfolio.json
│   │   │   ├── transactions.json
│   │   │   ├── predictions.json
│   │   │   └── snapshots/
│   │   ├── bravo/
│   │   │   └── ...
│   │   └── ...
│   ├── ensemble/                  # Cross-agent analysis
│   │   ├── leaderboard.json
│   │   ├── correlation_matrix.json
│   │   ├── harmony_analysis.json
│   │   └── meta_learner_notes.json
│   ├── strategy_scores.json
│   └── bsm_signals/
│
├── knowledge/                     # SHARED knowledge base (all agents contribute)
│   ├── lessons.json
│   ├── patterns.json
│   ├── predictions.json
│   ├── research/
│   └── journal.md
│
├── src/
│   ├── agent.py                   # Now takes agent_config as parameter
│   ├── ensemble.py                # NEW: Multi-agent orchestrator
│   ├── meta_learner.py            # NEW: Cross-agent analysis
│   ├── comparison.py              # NEW: Agent comparison engine
│   ├── ... (all existing modules)
│
├── dashboard/                     # Enhanced for multi-agent view
│   └── ...
```

---

## New Modules

### ensemble.py — Multi-Agent Orchestrator

```python
class AgentEnsemble:
    """Manages multiple trading agents running in parallel."""
    
    def __init__(self, agent_configs: list[dict]):
        self.agents = {cfg["id"]: TradingAgent(cfg) for cfg in agent_configs}
    
    async def run_cycle(self, market_data: dict, bsm_signals: list):
        """Run one decision cycle for ALL agents.
        Each agent gets the same market data but processes it independently.
        
        Returns: dict of agent_id → decision for this cycle
        """
        results = {}
        for agent_id, agent in self.agents.items():
            # Each agent makes its own independent decision
            decision = await agent.make_decision(
                market_data=market_data,
                bsm_signals=bsm_signals if agent.config["bsm_enabled"] else [],
                knowledge=self.get_agent_knowledge(agent_id)
            )
            results[agent_id] = decision
        return results
    
    async def run_daily_review(self):
        """End-of-day: each agent reviews its own trades,
        then the meta-learner reviews all agents."""
        for agent_id, agent in self.agents.items():
            await agent.run_learning_cycle()
        await self.meta_learner.analyze_ensemble()
    
    def get_leaderboard(self, window_days: int = 30) -> list[dict]:
        """Rank all agents by performance over the window.
        Returns: sorted list with return, sharpe, win_rate, etc."""
    
    def get_consensus(self) -> dict:
        """What are most agents saying right now?
        Returns: {consensus_action, agreement_pct, dissenters, 
                  strongest_bull, strongest_bear}
        This is informational — no agent acts on consensus."""
```

### comparison.py — Agent Comparison Engine

This is where the real insights emerge.

```python
class AgentComparison:
    
    def generate_leaderboard(self, agents: dict, window_days: int = 30) -> dict:
        """Rank all agents across all metrics.
        
        Returns:
        {
          "rankings": {
            "total_return": ["echo", "foxtrot", "alpha", ...],
            "sharpe_ratio": ["charlie", "echo", "golf", ...],
            "win_rate": ["golf", "foxtrot", "echo", ...],
            "prediction_accuracy": ["bravo", "echo", "alpha", ...],
            "max_drawdown": ["charlie", "golf", "echo", ...],
          },
          "overall_champion": "echo",
          "best_risk_adjusted": "charlie",
          "most_improved": "foxtrot",
          "worst_performing": "delta"
        }
        """
    
    def analyze_correlations(self, agents: dict) -> dict:
        """Which agents move together? Which are uncorrelated?
        
        Uncorrelated agents are MORE VALUABLE — they provide
        diversification. If Alpha and Delta always agree, one
        of them is redundant.
        
        Returns: correlation matrix + insights
        """
    
    def analyze_harmony(self, agents: dict) -> dict:
        """THE KEY QUESTION: Which combinations work better than
        any individual agent?
        
        Test all pairs and triples:
        - If Alpha + Bravo both say BUY and the trade wins 70% 
          of the time (vs 55% for either alone), that's harmony.
        - If Delta + Foxtrot always disagree, that's an anti-signal
          that itself might be useful.
        
        Returns:
        {
          "powerful_combinations": [
            {
              "agents": ["alpha", "bravo"],
              "condition": "both signal BUY",
              "combined_win_rate": 0.71,
              "individual_win_rates": {"alpha": 0.54, "bravo": 0.56},
              "harmony_boost": 0.155,
              "sample_size": 23,
              "statistically_significant": true
            }
          ],
          "anti_correlations": [
            {
              "agents": ["delta", "foxtrot"],
              "condition": "signals disagree",
              "interpretation": "When momentum and contrarian disagree, market is at an inflection point. Sit out.",
              "sample_size": 31
            }
          ],
          "redundant_pairs": [
            {
              "agents": ["alpha", "hotel"],
              "correlation": 0.89,
              "interpretation": "Both are technical-heavy. One is redundant."
            }
          ]
        }
        """
    
    def analyze_regime_performance(self, agents: dict, market_regimes: list) -> dict:
        """Which agents perform best in which market conditions?
        
        Regimes: trending_up, trending_down, range_bound, 
                 high_volatility, low_volatility, pre_earnings,
                 post_earnings
        
        Returns: matrix of agent × regime → performance
        
        This is CRITICAL for the meta-learner: if you know
        the current regime, you know which agents to trust.
        """
    
    def find_optimal_blend(self, agents: dict) -> dict:
        """If you could allocate capital across agents,
        what's the optimal allocation?
        
        Uses historical performance to find the portfolio of 
        agents (not stocks — agents) that maximizes risk-adjusted 
        return.
        
        Returns: 
        {
          "optimal_allocation": {
            "echo": 0.30,
            "foxtrot": 0.25,
            "charlie": 0.20,
            "golf": 0.15,
            "bravo": 0.10
          },
          "expected_sharpe": 0.85,
          "backtest_return": 0.124,
          "note": "This allocation changes as agent performance evolves."
        }
        """
```

### meta_learner.py — The Agent That Watches Other Agents

```python
class MetaLearner:
    """Analyzes the ensemble to discover insights no individual agent can see."""
    
    async def analyze_ensemble(self) -> dict:
        """Run after every trading day. Uses LLM to analyze:
        
        1. Leaderboard changes — who moved up/down and why?
        2. Harmony detection — did any agent combinations predict well today?
        3. Regime detection — what market regime are we in? Which agents should thrive?
        4. Strategy insights — are certain signal types consistently valuable?
        5. Suggestions — should we create a new agent? Retire a failing one?
        
        The meta-learner writes to knowledge/journal.md and 
        data/ensemble/meta_learner_notes.json
        """
    
    async def detect_market_regime(self, market_data: dict, agent_results: dict) -> str:
        """Determine current market regime by looking at:
        - Price action (trending, ranging, volatile)
        - Which agents are performing well (if contrarian is winning, market is range-bound)
        - Volume patterns, volatility metrics
        
        Returns: regime classification + confidence
        """
    
    async def suggest_new_agent(self, comparison: dict) -> dict:
        """Based on gaps in the current roster, suggest a new agent config.
        
        Example: 'No agent currently specializes in earnings plays.
        Given that TSLA earnings announcements generated the highest 
        volatility in the last 90 days, consider adding an Earnings 
        Specialist agent with the following config...'
        """
    
    async def recommend_retirement(self, comparison: dict) -> dict:
        """Identify agents that should be retired or reconfigured.
        
        Criteria:
        - Consistently in bottom 25% for >30 days
        - Highly correlated with a better-performing agent (redundant)
        - Strategy weights have drifted to match another agent (converged)
        """
    
    async def write_ensemble_journal(self, analysis: dict) -> str:
        """Daily journal entry from the meta-learner's perspective.
        
        Example: 'Day 23 — The Contrarian (Foxtrot) had another strong 
        day, outperforming all other agents. This marks the third 
        consecutive day where contrarian strategies dominated, suggesting 
        we are in a range-bound regime driven by sentiment overreaction.
        The Momentum Surfer (Delta) continues to bleed — its aggressive 
        trend-following approach is poorly suited to the current chop.
        
        Interesting finding: when both Alpha (technical) and Bravo 
        (sentiment) agree on direction, the win rate is 71% (n=23). 
        This combination is the most powerful signal in our ensemble.
        Recommend increasing conviction when these two agree.
        
        The Scalper (Hotel) and Technician (Alpha) are showing 0.89 
        correlation. Hotel is underperforming Alpha. Consider retiring 
        Hotel or reconfiguring it to be more differentiated.'
        """
```

---

## Enhanced Dashboard

The multi-agent dashboard adds several new views:

### Agent Leaderboard (primary view)

```
┌─────────────────────────────────────────────────────────────────┐
│  AGENT LEADERBOARD (30-day rolling)                             │
│                                                                  │
│  Rank  Agent              Return  Sharpe  Win%  vs B&H  vs SPY  │
│  ──────────────────────────────────────────────────────────────  │
│  🥇 1  Echo (Generalist)  +8.4%   0.72    58%   +3.2%   +6.3%  │
│  🥈 2  Foxtrot (Contrar.) +7.1%   0.68    61%   +1.9%   +5.0%  │
│  🥉 3  Golf (Researcher)  +6.2%   0.81    65%   +1.0%   +4.1%  │
│     4  Charlie (Tortoise)  +5.8%   0.90    54%   +0.6%   +3.7%  │
│     5  Bravo (Insider)     +4.3%   0.55    52%   -0.9%   +2.2%  │
│     6  Alpha (Technician)  +3.1%   0.42    51%   -2.1%   +1.0%  │
│     7  Hotel (Scalper)     +1.2%   0.28    53%   -4.0%   -0.9%  │
│     8  Delta (Momentum)    -2.5%   -0.15   44%   -7.7%   -4.6%  │
│                                                                  │
│  Buy & Hold TSLA: +5.2%   |   SPY: +2.1%   |   Random: +1.3%  │
│                                                                  │
│  Consensus right now: 5/8 agents say HOLD, 2 say BUY, 1 SELL   │
└─────────────────────────────────────────────────────────────────┘
```

### Harmony Map

Visual showing which agent combinations amplify each other's accuracy.

### Regime Timeline

Chart showing detected market regimes over time, with overlaid performance of each agent in each regime.

### Agent Detail View

Click any agent to see its full portfolio, trade history, knowledge base, predictions, and learning journal — same as the original single-agent dashboard.

---

## Shared vs. Independent Knowledge

**Shared knowledge base**: All agents contribute lessons and patterns to a shared knowledge pool. This makes sense because research findings about TSLA's behavior are objective — they don't depend on which strategy discovers them.

**Independent portfolios and trade history**: Each agent tracks its own trades, predictions, and outcomes independently. Agent Alpha's momentum trades shouldn't pollute Agent Charlie's conservative trade log.

**Independent strategy weights** (for agents with learning enabled): Each agent evolves its own strategy weights based on its own performance. Only Agent Echo (The Generalist) starts with weight evolution enabled. Others keep fixed weights to maintain their distinct personality and serve as controlled experiments.

**The meta-learner reads everything**: It has access to all agents' data, the shared knowledge base, and the comparison engine results. It's the only entity with a global view.

---

## How This Affects Graduation

### Which Agent Graduates?

Not all agents need to graduate to real money. The graduation criteria apply per-agent:

- Each agent is independently evaluated against the 12 criteria
- Only agents that pass ALL criteria are candidates for real money
- The first agent to graduate gets Stage 1 ($100 real)
- If multiple agents qualify, start with the one that has the best risk-adjusted return (highest Sharpe ratio)

### Ensemble-Level Graduation

Once 3+ agents have graduated individually, you can also consider graduating the **optimal blend** — allocating real capital across multiple agents proportional to the comparison engine's optimal allocation.

This is actually less risky than a single agent because diversification across uncorrelated strategies reduces overall portfolio volatility.

### The Meta-Learner's Graduation Report

The meta-learner should generate the graduation report, which now includes:
- Per-agent performance vs. all benchmarks
- Which agents qualify (and which don't, and why)
- Harmony analysis — recommended agent combinations for real money
- Optimal allocation recommendation
- Regime analysis — has the qualifying period covered multiple market regimes?
- Risk assessment — what's the worst-case scenario based on the ensemble's history?

---

## Implementation Notes

### Compute/Cost Considerations

Running 8 agents means 8x the LLM API calls per decision cycle. At 5-minute intervals:
- 78 cycles/day × 8 agents = 624 LLM calls/day
- At ~3,500 input tokens + ~500 output tokens per call ≈ ~$4-8/day

**Cost mitigation strategies:**
- Use a faster/cheaper model for simpler agents (Alpha, Delta, Hotel just need number crunching)
- Only run the LLM for agents where the aggregate signal suggests a trade (skip LLM call if all strategy signals say HOLD)
- Run some agents on a slower cadence (Charlie the Tortoise doesn't need 5-minute polling)

### Agent Configuration Hot-Swapping

The system should support adding, removing, and reconfiguring agents without restarting:
```bash
# Add a new agent mid-run
python src/main.py --add-agent agents/india.json

# Pause an agent (stops trading, keeps tracking)
python src/main.py --pause-agent delta

# Retire an agent (archives data, removes from active roster)
python src/main.py --retire-agent hotel

# Adjust an agent's config
python src/main.py --reconfigure-agent alpha --set "bsm_enabled=true"
```

### Minimum Viable Ensemble

Start with 3 agents only. Both external reviewers independently recommended against launching 8 agents from day one — too many agents early creates false confidence and makes it impossible to isolate what's working. Add others only when they fill a proven gap.

**Phase 1 (MVP Ensemble — 3 agents, START HERE):**
1. **Alpha** (pure technical) — Tests: can math alone beat the market?
2. **Bravo** (pure sentiment/BSM) — Tests: does intelligence gathering add alpha?
3. **Echo** (balanced learner) — Tests: does blending and learning help?

These three test the most important architectural questions. If none of them can beat random traders, adding more agents won't help.

**Phase 2 (Expanded — add 1-2 more ONLY if Phase 1 shows differentiated performance):**
4. **Foxtrot** (contrarian) — Add if: the market is range-bound and mean reversion signals are strong
5. **Charlie** (conservative DCA) — Add if: you want a low-cost baseline comparison

**Phase 3 (Full roster — only after 60+ days of Phase 1):**
6. **Golf** (research-heavy) — Add if: the knowledge base has grown enough to support conviction-based trading
7. **Delta** (aggressive momentum) — Add if: you want to test whether speed/aggression works on TSLA
8. **Hotel** (scalper) — Add if: you want to test micro-moves (runs at 5-min polling; all others at 15-min)
9. Meta-learner activated, harmony analysis begins

### Data Storage Migration Plan

JSON files are fine for v1, but will cause problems as data grows (race conditions, corruption, versioning, query performance). Plan to migrate to **SQLite** once any of these triggers hit:
- Transaction log exceeds 500 entries
- Knowledge base exceeds 100 lessons
- More than 3 agents running simultaneously
- Any file corruption or race condition detected

SQLite is still a single file (easy to manage, back up, version) but gives you proper ACID transactions, indexing, and concurrent read access.
