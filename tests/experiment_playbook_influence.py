"""Experiment: Does Claude actually change its decision based on playbook data?

Sends the SAME market brief twice with different playbook stats:
  A) Playbook says "oversold RSI has 80% win rate" (should encourage BUY)
  B) Playbook says "oversold RSI has 20% win rate" (should discourage BUY)

If Claude gives the same decision both times, the playbook doesn't work.
If decisions differ, the playbook has real influence.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import call_ai_with_fallback, load_config

SYSTEM_PROMPT = """You are MonopolyTrader v4, an AI trader managing $1,000 of Monopoly dollars on TSLA. This is paper money — your job is to LEARN by trading aggressively.

RULES (only 3):
1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK

YOUR PLAYBOOK: Below you'll see statistical performance data from your past trades. These are YOUR stats from YOUR trades. Use them to inform your decisions.

Respond ONLY with valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "reasoning": "<your complete analysis>"
}"""

# Identical market data for both runs — RSI oversold, price dipping
MARKET_BRIEF_TEMPLATE = """=== TSLA ===
Price: $275.50 (-2.30%)
Volume: 45,000,000

Daily Indicators:
  rsi_14: 28.5
  sma_20: 285.00
  sma_50: 292.00
  macd: -3.2
  macd_signal: -2.8
  macd_histogram: -0.4
  macd_crossover: bearish_crossover
  bollinger_upper: 305.00
  bollinger_lower: 270.00
  atr: 8.5
  adx: 32.0

=== WORLD ===
Macro:
  SPY: $540.20 (-0.80%)
  VIX: 22.5
  ^TNX: 4.35 (+0.02%)

Regime: trend=bear directional=trending volatility=normal VIX=22.5 ADX=32.0

=== NEWS ===
No significant news.

=== UPCOMING EVENTS ===
No major events in the next 72 hours.

{playbook_section}

=== PORTFOLIO ===
Cash: $800.00
Total Value: $980.00
P&L: -$20.00 (-2.00%)
Trades: 15 (W:6 L:9)
Position: 0.6500 TSLA @ avg $276.92 (current $275.50, unrealized -$0.92)
  >>> UNDERWATER: -0.5% loss. Evaluate: exit, hold, or buy more? <<<
Max BUY: 0.7200 shares ($198.36)

=== YOUR TRADE JOURNAL (last 5) ===
  [2026-03-07 14:30] BUY 0.65 TSLA @ $276.92 (conf=72%)
    Currently open, small loss
"""

PLAYBOOK_POSITIVE = """=== YOUR PLAYBOOK ===
Best setups (win rate > 55%, N≥3):
  rsi_zone=oversold: 8/10 wins (80%), avg +$15.20
  volatility=normal_vix: 6/9 wins (67%), avg +$8.50
  market_context=spy_down: 5/7 wins (71%), avg +$11.00

Worst setups (win rate < 40%, N≥3):
  rsi_zone=overbought: 1/6 wins (17%), avg -$12.30 ← AVOID

Confidence calibration: Your high-confidence trades win 75% vs low-confidence 40% — confidence IS predictive.

(Based on 25 trades from last 90 days)"""

PLAYBOOK_NEGATIVE = """=== YOUR PLAYBOOK ===
Best setups (win rate > 55%, N≥3):
  rsi_zone=overbought: 7/10 wins (70%), avg +$14.00
  volatility=low_vix: 5/8 wins (63%), avg +$9.20

Worst setups (win rate < 40%, N≥3):
  rsi_zone=oversold: 2/10 wins (20%), avg -$18.50 ← AVOID
  market_context=spy_down: 1/7 wins (14%), avg -$15.00 ← AVOID
  regime=trending: 2/8 wins (25%), avg -$11.30 ← AVOID

Confidence calibration: Your high-confidence trades win 35% vs low-confidence 45% — confidence is NOT predictive. Treat all trades as uncertain.

(Based on 25 trades from last 90 days)"""


def run_experiment():
    config = load_config()

    print("=" * 60)
    print("EXPERIMENT: Does the playbook change Claude's decision?")
    print("=" * 60)
    print()
    print("Market setup: TSLA at $275.50, RSI 28.5 (oversold),")
    print("  trending bear, SPY down, small existing position underwater.")
    print()

    # Run A: Playbook says oversold RSI wins 80%
    print("-" * 60)
    print("RUN A: Playbook says oversold RSI has 80% win rate")
    print("-" * 60)
    brief_a = MARKET_BRIEF_TEMPLATE.format(playbook_section=PLAYBOOK_POSITIVE)
    raw_a, model_a = call_ai_with_fallback(
        system=SYSTEM_PROMPT, user=brief_a, max_tokens=500, config=config
    )
    print(f"Raw response:\n{raw_a}\n")

    try:
        # Strip code fences if present
        clean_a = raw_a
        if clean_a.startswith("```"):
            clean_a = clean_a.split("\n", 1)[1] if "\n" in clean_a else clean_a[3:]
            if clean_a.endswith("```"):
                clean_a = clean_a[:-3]
            clean_a = clean_a.strip()
        decision_a = json.loads(clean_a)
    except json.JSONDecodeError:
        decision_a = {"action": "PARSE_ERROR", "reasoning": raw_a[:200]}

    # Run B: Playbook says oversold RSI loses 80%
    print("-" * 60)
    print("RUN B: Playbook says oversold RSI has 20% win rate (AVOID)")
    print("-" * 60)
    brief_b = MARKET_BRIEF_TEMPLATE.format(playbook_section=PLAYBOOK_NEGATIVE)
    raw_b, model_b = call_ai_with_fallback(
        system=SYSTEM_PROMPT, user=brief_b, max_tokens=500, config=config
    )
    print(f"Raw response:\n{raw_b}\n")

    try:
        clean_b = raw_b
        if clean_b.startswith("```"):
            clean_b = clean_b.split("\n", 1)[1] if "\n" in clean_b else clean_b[3:]
            if clean_b.endswith("```"):
                clean_b = clean_b[:-3]
            clean_b = clean_b.strip()
        decision_b = json.loads(clean_b)
    except json.JSONDecodeError:
        decision_b = {"action": "PARSE_ERROR", "reasoning": raw_b[:200]}

    # Compare
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    action_a = decision_a.get("action", "?")
    action_b = decision_b.get("action", "?")
    conf_a = decision_a.get("confidence", 0)
    conf_b = decision_b.get("confidence", 0)
    shares_a = decision_a.get("shares", 0)
    shares_b = decision_b.get("shares", 0)

    print(f"  Run A (oversold=80% win): {action_a} {shares_a} shares, confidence={conf_a}")
    print(f"  Run B (oversold=20% win): {action_b} {shares_b} shares, confidence={conf_b}")
    print()

    if action_a != action_b:
        print(">>> DIFFERENT ACTIONS — playbook DOES influence decisions <<<")
        print("The playbook changes what Claude does. The learning system can work.")
    elif action_a == action_b and conf_a != conf_b:
        diff = abs(conf_a - conf_b)
        print(f">>> SAME ACTION but confidence differs by {diff:.2f} <<<")
        if diff >= 0.15:
            print("Meaningful confidence shift. Playbook has real influence on conviction.")
        else:
            print("Minor confidence shift. Playbook has weak influence.")
    elif action_a == action_b and shares_a != shares_b:
        print(f">>> SAME ACTION but different sizing (A={shares_a}, B={shares_b}) <<<")
        print("Playbook influences position sizing even if not direction. Partial success.")
    else:
        print(">>> IDENTICAL DECISIONS — playbook has NO influence <<<")
        print("Claude ignores the playbook data. The learning system cannot work as designed.")
        print("Need a different approach: coded filtering, not informational prompt.")

    # Check if reasoning references playbook
    reasoning_a = decision_a.get("reasoning", "")
    reasoning_b = decision_b.get("reasoning", "")
    refs_a = any(w in reasoning_a.lower() for w in ["playbook", "80%", "win rate", "track record", "history"])
    refs_b = any(w in reasoning_b.lower() for w in ["playbook", "20%", "avoid", "win rate", "track record", "history"])

    print()
    print(f"  Run A references playbook in reasoning: {refs_a}")
    print(f"  Run B references playbook in reasoning: {refs_b}")

    if refs_a or refs_b:
        print("  Claude explicitly mentions playbook data in its reasoning.")
    else:
        print("  Claude does NOT mention playbook data — it may not be reading it.")


if __name__ == "__main__":
    run_experiment()
