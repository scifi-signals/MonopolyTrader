# Benchmarking & Graduation Criteria — When Does the Agent Earn Real Money?

## The Fundamental Question

If the agent can't consistently beat someone who simply bought TSLA on day one and held it, then all the strategies, learning, and intelligence gathering are just expensive noise. Every trade has a cost (slippage, opportunity cost, API fees), so the agent must generate enough alpha to overcome those costs AND beat passive holding.

This spec defines:
1. **Benchmarks** — What we're measuring against
2. **Performance metrics** — How we measure the agent's skill vs. luck
3. **Graduation criteria** — Specific, data-driven thresholds that must be met before real money is involved
4. **Graduation stages** — A gradual ramp from Monopoly dollars to real capital

---

## 1. Benchmarks — The Opponents the Agent Must Beat

The agent is tracked against four benchmarks simultaneously. All benchmarks start with the same $1,000 on the same day.

### Benchmark A: Buy-and-Hold TSLA
The simplest test. On day one, take $1,000 and buy as much TSLA as possible. Never sell. This is the agent's primary opponent.

```python
def calculate_buy_and_hold(start_date: str, start_balance: float, ticker: str) -> dict:
    """Buy on start_date at market open. Track value daily.
    Returns: current_value, total_return_pct, max_drawdown, sharpe_ratio"""
```

**Why this matters**: If the agent can't beat this, all its trading activity is negative value. Many professional hedge funds fail this test.

### Benchmark B: S&P 500 (SPY)
Buy $1,000 of SPY and hold. This represents "the market" — the return anyone could get with zero effort.

**Why this matters**: Even if the agent beats TSLA buy-and-hold, if it can't beat SPY, you'd be better off in an index fund. This benchmark answers "is TSLA-specific intelligence valuable?"

### Benchmark C: Dollar Cost Average TSLA
Invest $50/week into TSLA regardless of price. This is what a disciplined retail investor without any intelligence would do.

**Why this matters**: DCA smooths out volatility. If the agent's timing doesn't beat mechanical DCA, its market-timing ability is unproven.

### Benchmark D: Random Trader
A simulated trader that makes random BUY/SELL/HOLD decisions with the same risk rules as the agent (same position limits, stop losses, trade sizes). Run 100 random simulations and track the median outcome.

**Why this matters**: This is the most important benchmark. It separates skill from luck. If the agent can't consistently outperform random decisions with the same constraints, it has no real edge. Any profit is just luck.

```python
def simulate_random_trader(
    start_date: str, 
    start_balance: float, 
    ticker: str, 
    num_simulations: int = 100,
    trade_probability: float = 0.3,
    risk_rules: dict = None
) -> dict:
    """Simulate N random traders with same rules as the agent.
    Returns: median_return, percentile_25, percentile_75, 
    agent_percentile (where does our agent rank among random traders?)"""
```

---

## 2. Performance Metrics — Measuring Skill vs. Luck

Raw return isn't enough. A strategy that returns 20% but could have lost 50% isn't better than one that returns 12% with max 5% drawdown.

### Core Metrics (calculated daily, displayed on dashboard)

| Metric | What It Measures | Target for Graduation |
|--------|------------------|----------------------|
| **Total Return %** | Raw profit/loss | Must beat Buy-and-Hold TSLA |
| **Alpha** | Excess return vs. benchmark (SPY) | Must be positive over 30-day rolling window |
| **Sharpe Ratio** | Risk-adjusted return (return / volatility) | Must be > 0.5 (annualized) |
| **Max Drawdown** | Worst peak-to-trough decline | Must be < TSLA buy-and-hold drawdown |
| **Win Rate** | % of trades that were profitable | Must be > 52% (enough to overcome costs) |
| **Profit Factor** | Gross profit / gross loss | Must be > 1.2 |
| **Expectancy** | Average $ gained per trade | Must be positive |
| **Agent vs. Random Percentile** | Where does the agent rank vs. 100 random traders? | Must be > 75th percentile |
| **Prediction Accuracy (direction)** | % of directional predictions correct | Must be > 55% on 2-hour predictions |
| **Prediction Accuracy Trend** | Is accuracy improving over time? | Must show statistically significant upward trend |

### Rolling Windows

All metrics are tracked across multiple time windows:
- **7-day rolling** — Recent performance, noisy but shows momentum
- **30-day rolling** — Primary evaluation window
- **90-day rolling** — Long-term trend, most statistically significant
- **All-time** — Overall track record

### Statistical Significance

Short-term results can be pure luck. We need statistical confidence that the agent has genuine skill.

```python
def calculate_statistical_significance(
    agent_returns: list[float], 
    benchmark_returns: list[float],
    confidence_level: float = 0.95
) -> dict:
    """Run multiple statistical tests. Financial data violates standard 
    t-test assumptions (non-iid, fat tails, autocorrelation), so we 
    use multiple methods:
    
    1. Paired t-test (standard, but squishy on financial data)
    2. Bootstrap confidence intervals (more robust, fewer assumptions)
    3. Deflated Sharpe Ratio (corrects for multiple testing and non-normality)
    4. White's Reality Check (critical when testing multiple agents — 
       corrects for data snooping across the ensemble)
    
    Returns: {t_statistic, p_value, bootstrap_ci, deflated_sharpe,
              whites_p_value, is_significant, min_trades_needed}"""
```

**Rule of thumb**: With typical stock volatility, you need roughly 60-100 trades to separate skill from luck at 95% confidence. At 15-minute intervals with selective trading, this means at least 4-6 weeks of active trading before any conclusions are meaningful. The 90-day minimum for Stage 0 ensures we have sufficient data.

---

## 3. The Benchmark Dashboard

This should be the FIRST thing you see when you open the dashboard — not buried in a tab.

### Dashboard Section: "Am I Beating the Market?"

```
┌─────────────────────────────────────────────────────────────┐
│  PERFORMANCE vs. BENCHMARKS (30-day rolling)                │
│                                                              │
│  MonopolyTrader:     +8.4%  ████████████████████ ← YOU      │
│  Buy & Hold TSLA:    +5.2%  █████████████                    │
│  DCA TSLA:           +4.8%  ████████████                     │
│  S&P 500 (SPY):      +2.1%  █████                            │
│  Random (median):    +1.3%  ███                               │
│                                                              │
│  Alpha vs SPY: +6.3%    Sharpe: 0.72    Win Rate: 58%       │
│  Agent vs Random: 82nd percentile (beating 82/100 randoms)  │
│  Statistical significance: YES (p=0.03)                      │
│                                                              │
│  Verdict: ✅ OUTPERFORMING — Agent shows genuine skill       │
└─────────────────────────────────────────────────────────────┘
```

### The "Verdict" Logic

```python
def calculate_verdict(metrics: dict) -> dict:
    """Generate a plain-English verdict on agent performance.
    
    Returns:
    - verdict: one of ['no_data', 'too_early', 'underperforming', 
               'inconclusive', 'promising', 'outperforming', 'graduating']
    - explanation: why this verdict
    - recommendation: what to do next
    """
    
    if total_trades < 30:
        return "too_early", "Need more trades for meaningful comparison"
    
    if agent_return < buy_hold_return and agent_return < spy_return:
        return "underperforming", "Agent is losing to passive strategies"
    
    if agent_vs_random_percentile < 60:
        return "inconclusive", "Agent is not clearly beating random trading"
    
    if agent_return > buy_hold_return and sharpe > 0.3 and not statistically_significant:
        return "promising", "Positive signs but need more data for confidence"
    
    if all graduation criteria met:
        return "graduating", "Agent has met all criteria for real money"
    
    ...
```

---

## 4. Graduation Criteria — The Checklist

The agent does NOT get real money until ALL of the following are met. No exceptions, no "it feels ready."

### Stage Gate: Paper Trading → Real Money

**Minimum Duration**: 90 trading days (approximately 4.5 months) — 60 days is too short; one market regime can make a bad agent look good.

**Performance Criteria (ALL must be met simultaneously over a 30-day rolling window):**

| # | Criterion | Threshold | Why |
|---|-----------|-----------|-----|
| 1 | Total return beats Buy & Hold TSLA | Agent > TSLA B&H over 30 days | Core value proposition |
| 2 | Total return beats SPY | Agent > SPY over 30 days | Proves stock-picking adds value |
| 3 | Agent vs. Random percentile | > 75th percentile | Proves skill, not luck |
| 4 | Sharpe Ratio (annualized) | > 0.5 | Acceptable risk-adjusted return |
| 5 | Max drawdown | < 15% | Proves risk management works |
| 6 | Win rate | > 52% | Enough edge to overcome costs |
| 7 | Profit factor | > 1.2 | Winning more than losing |
| 8 | Prediction accuracy (2hr, direction) | > 55% | Agent can actually predict |
| 9 | Prediction accuracy trend | Positive slope over 30 days | Agent is still improving |
| 10 | Statistical significance vs. benchmark | p < 0.05 | 95% confident this isn't luck (p < 0.10 is too noisy for financial data) |
| 11 | No CRITICAL alerts in last 14 days | 0 critical anomalies | System is stable |
| 12 | Consecutive positive weeks | ≥ 3 of last 4 weeks profitable | Consistency, not one lucky streak |

**All 12 criteria must be green simultaneously.** If any single criterion fails, the agent stays on Monopoly dollars.

### Graduation Report

When all criteria are met, the system generates a formal graduation report:

```python
def generate_graduation_report(metrics: dict, history: dict) -> dict:
    """Generate comprehensive report making the case for real money.
    
    Includes:
    - All 12 criteria with current values and pass/fail status
    - Full performance comparison vs. all 4 benchmarks
    - Statistical confidence analysis
    - Risk analysis (worst case scenarios based on historical drawdowns)
    - Recommended starting real capital amount
    - Recommended risk parameters for real trading (more conservative than paper)
    - Strategy breakdown (which strategies earned the most alpha)
    - Known weaknesses and blind spots
    - BSM signal accuracy contribution
    """
```

---

## 5. Graduation Stages — Scaling Into Real Money

Even after graduating from paper trading, you don't dump your life savings in. Scale gradually with increasing trust.

### Stage 0: Paper Trading (Monopoly Dollars)
- **Capital**: $1,000 virtual
- **Duration**: Minimum 90 trading days (not 60 — one regime doesn't prove anything)
- **Exit criteria**: All 12 graduation criteria met
- **Risk rules**: Standard (as defined in config.json)

### Stage 1: Micro-Real ("Prove It With Skin in the Game")
- **Capital**: $100 real money
- **Duration**: Minimum 60 trading days (extended from 30 — real execution is a different beast)
- **Purpose**: Verify that real execution (actual slippage, real-time fills) matches paper trading assumptions
- **Exit criteria**: Same 12 criteria, PLUS real P&L within 2% of what paper trading would have produced (execution quality check)
- **Risk rules**: More conservative — max position 50% (not 90%), max single trade 10% (not 25%), stop loss 3% (not 5%)
- **Kill switch**: If real portfolio drops below $85 (-15%), immediately halt and revert to paper trading for investigation

### Stage 2: Small-Real ("Building Confidence")
- **Capital**: $500 real money
- **Duration**: Minimum 30 trading days
- **Exit criteria**: All 12 criteria sustained, Stage 1 validated execution quality
- **Risk rules**: Moderate — max position 70%, max single trade 15%, stop loss 4%
- **Kill switch**: Portfolio drops below $400 (-20%)

### Stage 3: Medium-Real ("Trusted Agent")
- **Capital**: $2,000-5,000 real money
- **Duration**: Ongoing
- **Entry criteria**: 6+ months total track record (paper + Stages 1-2), sustained outperformance
- **Risk rules**: Standard (matching paper trading rules, now validated)
- **Kill switch**: Portfolio drops below 75% of starting capital

### Stage 4: Significant-Real ("Full Autonomy")
- **Capital**: $10,000+ (user-determined)
- **Entry criteria**: 12+ months track record, proven through multiple market conditions (trending, range-bound, volatile, bearish)
- **Risk rules**: Operator-defined, informed by full performance history
- **Note**: At this stage, you may also want to diversify beyond TSLA to reduce single-stock risk

### Stage Regression

**Important**: If the agent fails criteria at any stage, it drops back one stage. If it fails at Stage 1, it goes back to paper trading. If it fails at Stage 3, it drops to Stage 2 capital levels. The agent must re-earn trust.

**However**: Negative milestones trigger a structured decision point, not an automatic kill. The operator chooses between retool, pause, or kill based on a Decision Point Report that includes available model upgrades, cost analysis, and research value assessment. See the MonopolyTrader CLAUDE.md "Milestone Alert & Decision System" section for the full framework.

**Model Upgrade Consideration**: LLM capabilities improve rapidly. If the agent is underperforming, always check if a newer model is available before killing the project. After a model upgrade, allow 14 days of recalibration before judging performance.

```python
def check_stage_regression(current_stage: int, metrics: dict) -> dict:
    """Check if the agent should be demoted to a lower stage.
    
    Triggers:
    - Kill switch hit (drawdown exceeded)
    - 2+ graduation criteria failed for 7 consecutive days
    - CRITICAL anomaly detected
    - Execution quality diverges >5% from paper trading
    
    Returns: {should_regress, new_stage, reason, recommended_action}
    """
```

---

## 6. Benchmark Implementation

### Where It Lives

MonopolyTrader should have a `benchmarks.py` module:

```python
class BenchmarkTracker:
    """Tracks all four benchmarks alongside the agent's performance."""
    
    def __init__(self, start_date: str, start_balance: float, ticker: str):
        self.start_date = start_date
        self.start_balance = start_balance
        self.ticker = ticker
        self.benchmarks = {
            "buy_hold_tsla": self._init_buy_hold(ticker),
            "buy_hold_spy": self._init_buy_hold("SPY"),
            "dca_tsla": self._init_dca(ticker),
            "random_traders": self._init_random_simulations(ticker, n=100)
        }
    
    def update_daily(self, current_prices: dict) -> None:
        """Update all benchmark values with today's prices."""
    
    def get_comparison(self, agent_portfolio: dict, window_days: int = 30) -> dict:
        """Full comparison of agent vs. all benchmarks.
        Returns: per-benchmark return comparison, alpha, agent rank,
        statistical tests, verdict."""
    
    def get_agent_percentile_vs_random(self, agent_return: float) -> float:
        """Where does the agent rank among 100 random traders?"""
    
    def run_statistical_test(self, agent_returns: list, benchmark_returns: list) -> dict:
        """Paired t-test for statistical significance."""
    
    def check_graduation_criteria(self, agent_metrics: dict) -> dict:
        """Check all 12 criteria. Return pass/fail for each."""
    
    def generate_graduation_report(self, agent_metrics: dict) -> str:
        """If all criteria pass, generate the formal graduation report."""


class RandomTraderSimulator:
    """Simulates random trading decisions with same rules as agent."""
    
    def __init__(self, price_history: pd.DataFrame, start_balance: float, risk_rules: dict):
        self.price_history = price_history
        self.start_balance = start_balance
        self.risk_rules = risk_rules
    
    def run_simulation(self) -> dict:
        """Run one random trader simulation over the full price history."""
    
    def run_monte_carlo(self, n: int = 100) -> dict:
        """Run N simulations. Return distribution of outcomes."""
```

### Integration with Dashboard

The benchmark comparison should be:
- **Always visible** on the main dashboard (not a sub-tab)
- **Updated daily** after market close
- **Includes a chart** showing all 5 lines (agent + 4 benchmarks) over time
- **Shows the verdict** prominently with color coding (red/yellow/green)
- **Shows graduation progress** — which of the 12 criteria are met, which aren't

### Integration with Build Order

Add to MonopolyTrader's build order, early — this should be Phase 2 or 3, not Phase 6:

```
Phase 2 (after basic trading works):
- benchmarks.py — Implement all 4 benchmarks
- RandomTraderSimulator — Run 100 random simulations
- Add benchmark comparison to every daily snapshot
- Add benchmark chart to dashboard as the PRIMARY view
```

---

## 7. The Honest Reality Check

Some hard truths to keep in mind:

**Most active traders underperform buy-and-hold.** Across all professional hedge funds, roughly 85-90% fail to beat the S&P 500 over a 10-year period. An AI agent, while tireless and unemotional, faces the same fundamental challenge: markets are highly efficient, and consistently finding edge is extremely difficult.

**60 days is the minimum, not the target.** 60 trading days gives you enough data to start making statistical claims. But true confidence comes from seeing the agent perform across different market regimes — bull runs, corrections, sideways chop, high volatility events (earnings, Fed meetings), black swans. A full year covering multiple regimes is ideal before meaningful real capital.

**The learning component is the real edge.** Most trading bots run fixed strategies. MonopolyTrader's ability to learn, adapt, and accumulate knowledge is what might actually give it a genuine edge over time. But that learning takes time to compound. Don't rush graduation.

**Paper trading to real money execution gap.** Paper trading assumes you can buy/sell at the exact price you see. Real markets have slippage, liquidity issues, and delayed fills. Stage 1 (micro-real, $100) exists specifically to measure this gap before scaling up.

**The Billionaire Signal Monitor could be the differentiator.** Technical analysis is commoditized — every trading bot does it. What might actually give MonopolyTrader a unique edge is intelligence from BSM that other market participants aren't synthesizing as systematically. If BSM signals correlate with subsequent price movements at >60% accuracy, that's real alpha.

---

## 8. Config Additions

Add to MonopolyTrader's `config.json`:

```json
{
  "benchmarks": {
    "enabled": true,
    "benchmarks_tracked": ["buy_hold_tsla", "buy_hold_spy", "dca_tsla", "random_100"],
    "dca_weekly_amount": 50.00,
    "random_simulations": 100,
    "statistical_confidence_level": 0.95
  },
  "graduation": {
    "current_stage": 0,
    "stage_0_min_days": 90,
    "stage_1_real_capital": 100,
    "stage_1_min_days": 60,
    "stage_2_real_capital": 500,
    "stage_2_min_days": 30,
    "stage_3_real_capital": 2000,
    "criteria": {
      "beat_buy_hold_tsla": true,
      "beat_spy": true,
      "random_percentile_min": 75,
      "sharpe_min": 0.5,
      "max_drawdown_max": 0.15,
      "win_rate_min": 0.52,
      "profit_factor_min": 1.2,
      "prediction_accuracy_min": 0.55,
      "prediction_trend_positive": true,
      "statistical_p_value_max": 0.05,
      "no_critical_alerts_days": 14,
      "consecutive_profitable_weeks_min": 3
    },
    "regression_triggers": {
      "drawdown_kill_switch_pct": 0.15,
      "criteria_failed_consecutive_days": 7,
      "execution_divergence_max_pct": 0.05
    }
  }
}
```
