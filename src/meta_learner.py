"""Meta-learner — cross-agent analysis, regime detection, agent suggestions."""

import json
import os
from anthropic import Anthropic
from .utils import load_config, load_json, save_json, iso_now, setup_logging, DATA_DIR
from .ensemble import list_agents, load_agent_portfolio, load_agent_transactions, load_agent_config
from .comparison import calculate_agent_metrics, generate_leaderboard

logger = setup_logging("meta_learner")

ENSEMBLE_DIR = DATA_DIR / "ensemble"


def _get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        for path in ["anthropic_api_key.txt", "../anthropic_api_key.txt"]:
            try:
                with open(path) as f:
                    api_key = f.read().strip()
                    break
            except FileNotFoundError:
                continue
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")
    return Anthropic(api_key=api_key)


async def daily_ensemble_analysis() -> dict:
    """Run daily cross-agent analysis after market close.

    Compares all agents, detects patterns, makes suggestions.
    """
    agent_names = list_agents()
    if not agent_names:
        return {"note": "No agents configured"}

    # Gather metrics
    leaderboard = generate_leaderboard()
    metrics_summary = []
    for m in leaderboard:
        metrics_summary.append(
            f"  {m['display_name']}: return={m['total_pnl_pct']:+.2f}%, "
            f"trades={m['total_trades']}, win_rate={m['win_rate']:.0f}%, "
            f"sharpe={m['sharpe_ratio']:.3f}, max_dd={m['max_drawdown_pct']:.1f}%"
        )

    # Look for patterns across agents
    analysis = {
        "timestamp": iso_now(),
        "agent_count": len(agent_names),
        "leaderboard_summary": metrics_summary,
        "regime_performance": _analyze_regime_performance(agent_names),
        "suggestions": _generate_suggestions(leaderboard),
        "retirement_candidates": _check_retirement(leaderboard),
    }

    save_json(ENSEMBLE_DIR / "daily_analysis.json", analysis)
    logger.info(f"Daily ensemble analysis complete: {len(agent_names)} agents analyzed")
    return analysis


def _analyze_regime_performance(agent_names: list[str]) -> dict:
    """Check which agents perform best in which market regimes."""
    # Load the latest cycle data which has regime info
    latest = load_json(ENSEMBLE_DIR / "latest_cycle.json", default={})
    current_regime = latest.get("regime", {})

    regime_data = {
        "current_regime": current_regime,
        "agents": {},
    }

    for name in agent_names:
        txns = load_agent_transactions(name)
        # Group trades by the regime at time of trade (if available)
        regime_trades = {}
        for t in txns:
            # We don't have per-trade regime tags yet, but we can
            # note the agent's performance trajectory
            pass

        portfolio = load_agent_portfolio(name)
        regime_data["agents"][name] = {
            "current_value": portfolio.get("total_value", 1000),
            "trend_in_current_regime": "insufficient_data",
        }

    return regime_data


def _generate_suggestions(leaderboard: list[dict]) -> list[str]:
    """Generate actionable suggestions based on agent performance."""
    suggestions = []

    if not leaderboard:
        return ["No agents running yet. Start with Alpha, Bravo, Echo."]

    # Check if any agent is significantly outperforming
    returns = [m["total_pnl_pct"] for m in leaderboard]
    if len(returns) >= 2:
        best = leaderboard[0]
        worst = leaderboard[-1]

        if best["total_pnl_pct"] - worst["total_pnl_pct"] > 5:
            suggestions.append(
                f"{best['display_name']} is outperforming {worst['display_name']} "
                f"by {best['total_pnl_pct'] - worst['total_pnl_pct']:.1f}%. "
                f"Consider increasing allocation to winning strategy."
            )

    # Check for stagnant agents (many trades but no edge)
    for m in leaderboard:
        if m["total_trades"] > 20 and abs(m["total_pnl_pct"]) < 1:
            suggestions.append(
                f"{m['display_name']} has {m['total_trades']} trades but near-zero return. "
                f"May need parameter tuning."
            )

    # Check if learning agent (Echo) is outperforming non-learning (Alpha)
    learning = [m for m in leaderboard if m.get("learning_enabled")]
    static = [m for m in leaderboard if not m.get("learning_enabled")]
    if learning and static:
        best_learning = max(learning, key=lambda m: m["total_pnl_pct"])
        best_static = max(static, key=lambda m: m["total_pnl_pct"])
        if best_static["total_pnl_pct"] > best_learning["total_pnl_pct"] + 2:
            suggestions.append(
                f"STATIC_BEATS_LEARNER: {best_static['display_name']} is outperforming "
                f"learning agents. The learning loop may be adding noise."
            )
        elif best_learning["total_pnl_pct"] > best_static["total_pnl_pct"] + 2:
            suggestions.append(
                f"Learning agent {best_learning['display_name']} is outperforming "
                f"static agents — the learning loop is producing edge."
            )

    if not suggestions:
        suggestions.append("All agents performing similarly. Need more trading days for differentiation.")

    return suggestions


def _check_retirement(leaderboard: list[dict]) -> list[dict]:
    """Identify agents that should be considered for retirement."""
    candidates = []

    for m in leaderboard:
        reasons = []

        # Consistently losing money after many trades
        if m["total_trades"] > 30 and m["total_pnl_pct"] < -10:
            reasons.append(f"Down {m['total_pnl_pct']:.1f}% after {m['total_trades']} trades")

        # Very high drawdown
        if m["max_drawdown_pct"] > 12:
            reasons.append(f"Max drawdown {m['max_drawdown_pct']:.1f}% exceeds comfort level")

        # Negative Sharpe after many trades
        if m["total_trades"] > 20 and m["sharpe_ratio"] < -0.5:
            reasons.append(f"Negative risk-adjusted return (Sharpe: {m['sharpe_ratio']:.3f})")

        if reasons:
            candidates.append({
                "agent": m["agent"],
                "display_name": m["display_name"],
                "reasons": reasons,
                "recommendation": "review" if len(reasons) == 1 else "consider_retirement",
            })

    return candidates


async def write_ensemble_journal(leaderboard: list[dict] = None) -> str:
    """Write a cross-agent journal entry using Claude.

    Analyzes the ensemble as a whole and reflects on what's working.
    """
    if leaderboard is None:
        leaderboard = generate_leaderboard()

    if not leaderboard:
        return "No agents running yet."

    config = load_config()
    metrics_text = "\n".join([
        f"  #{m['rank']} {m['display_name']}: "
        f"return={m['total_pnl_pct']:+.2f}%, trades={m['total_trades']}, "
        f"win_rate={m['win_rate']:.0f}%, sharpe={m['sharpe_ratio']:.3f}"
        for m in leaderboard
    ])

    prompt = f"""You are the Meta-Learner observing a multi-agent trading ensemble.

Current Leaderboard:
{metrics_text}

Write a brief journal entry (3-5 paragraphs) analyzing:
1. Which agents are performing best and why
2. Are the learning agents showing improvement over static ones?
3. What market conditions favor which agents?
4. Suggestions for the ensemble going forward

Be data-driven. Reference specific metrics. Keep it concise."""

    try:
        client = _get_client()
        response = client.messages.create(
            model=config["anthropic_model"],
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        entry = response.content[0].text.strip()

        # Append to ensemble journal
        journal_path = ENSEMBLE_DIR / "journal.md"
        existing = ""
        if journal_path.exists():
            existing = journal_path.read_text()

        with open(journal_path, "w") as f:
            f.write(f"# Ensemble Journal\n\n---\n\n## {iso_now()}\n\n{entry}\n\n{existing}")

        return entry
    except Exception as e:
        logger.error(f"Ensemble journal failed: {e}")
        return f"Journal generation failed: {e}"
