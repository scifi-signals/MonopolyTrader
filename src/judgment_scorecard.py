"""Judgment Scorecard — measures whether AI judgment adds value.

Tracks outcomes by decision source (AI-confirmed, AI-overridden, code-skipped,
contrarian, stops) and compares them. Answers the question: "Is the AI helping?"

Rebuilt nightly. No AI involved — pure statistics on observed outcomes.
"""

from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR
from .outcome_tracker import get_resolved_outcomes

logger = setup_logging("judgment_scorecard")

SCORECARD_PATH = DATA_DIR / "judgment_scorecard.json"

# Minimum outcomes per source to report meaningful stats
MIN_SAMPLE = 3


def build_scorecard(horizon: str = "1h", lookback_days: int = 90) -> dict:
    """Compute judgment scorecard from resolved outcomes with decision metadata.

    Groups outcomes by decision source and computes:
    - Count, avg price change, win rate, avg win, avg loss
    - Comparison metrics between key decision sources

    Returns scorecard dict (also saved to disk).
    """
    outcomes = get_resolved_outcomes(min_horizon=horizon, lookback_days=lookback_days)

    # Filter to outcomes that have decision metadata
    with_meta = [o for o in outcomes if o.get("decision_meta")]
    without_meta = len(outcomes) - len(with_meta)

    if not with_meta:
        scorecard = {
            "last_updated": iso_now(),
            "horizon": horizon,
            "total_outcomes": len(outcomes),
            "outcomes_with_meta": 0,
            "outcomes_without_meta": without_meta,
            "sources": {},
            "comparisons": {},
            "verdict": "Insufficient data — no outcomes with decision metadata yet.",
        }
        save_json(SCORECARD_PATH, scorecard)
        return scorecard

    # Group by decision source
    by_source = {}
    for o in with_meta:
        meta = o["decision_meta"]
        source = meta.get("source", "unknown")
        change = o.get("changes", {}).get(horizon)
        if change is None:
            continue

        if source not in by_source:
            by_source[source] = []
        by_source[source].append({
            "change": change,
            "signal_score": meta.get("signal_score", 0),
            "ai_action": meta.get("ai_action"),
            "ai_confidence": meta.get("ai_confidence"),
            "final_action": meta.get("final_action"),
            "signal_agreed": meta.get("signal_agreed"),
        })

    # Compute stats per source
    sources = {}
    for source, entries in by_source.items():
        sources[source] = _compute_source_stats(entries)

    # Compute comparison metrics
    comparisons = _compute_comparisons(sources, by_source)

    # Generate human-readable verdict
    verdict = _generate_verdict(sources, comparisons)

    scorecard = {
        "last_updated": iso_now(),
        "horizon": horizon,
        "total_outcomes": len(outcomes),
        "outcomes_with_meta": len(with_meta),
        "outcomes_without_meta": without_meta,
        "sources": sources,
        "comparisons": comparisons,
        "verdict": verdict,
    }

    save_json(SCORECARD_PATH, scorecard)
    logger.info(
        f"Judgment scorecard built: {len(with_meta)} outcomes, "
        f"{len(sources)} sources"
    )
    return scorecard


def _compute_source_stats(entries: list[dict]) -> dict:
    """Compute statistics for a single decision source."""
    n = len(entries)
    if n == 0:
        return {"n": 0}

    changes = [e["change"] for e in entries]

    # A "win" depends on what action was taken:
    # BUY → price went up is a win
    # SELL → price went down is a win (we exited before further drop)
    # HOLD → we measure opportunity cost (did price go up = missed opportunity)
    wins = 0
    losses = 0
    for e in entries:
        action = e.get("final_action", "HOLD")
        change = e["change"]
        if action == "BUY":
            if change > 0.001:
                wins += 1
            elif change < -0.001:
                losses += 1
        elif action == "SELL":
            if change < -0.001:
                wins += 1  # price dropped after we sold — good exit
            elif change > 0.001:
                losses += 1  # price rose after we sold — bad exit
        else:  # HOLD
            # For holds, we track what happened — neither win nor loss
            pass

    up = [c for c in changes if c > 0.001]
    down = [c for c in changes if c < -0.001]

    return {
        "n": n,
        "avg_change": round(sum(changes) / n, 6),
        "median_change": round(sorted(changes)[n // 2], 6),
        "up_count": len(up),
        "down_count": len(down),
        "flat_count": n - len(up) - len(down),
        "up_rate": round(len(up) / n, 3),
        "avg_up": round(sum(up) / len(up), 6) if up else 0,
        "avg_down": round(sum(down) / len(down), 6) if down else 0,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / (wins + losses), 3) if (wins + losses) > 0 else None,
        "avg_signal_score": round(
            sum(e.get("signal_score", 0) for e in entries) / n, 6
        ),
    }


def _compute_comparisons(sources: dict, by_source: dict) -> dict:
    """Compare key decision source pairs to measure AI value-add."""
    comparisons = {}

    # 1. AI confirm vs code skip — does calling AI for bullish signals help?
    ai_confirm = sources.get("ai_confirm", {})
    code_skip = sources.get("code_skip", {})
    if ai_confirm.get("n", 0) >= MIN_SAMPLE and code_skip.get("n", 0) >= MIN_SAMPLE:
        comparisons["ai_confirm_vs_code_skip"] = {
            "description": "Does AI confirmation of bullish signals improve outcomes vs skipping?",
            "ai_confirm_avg": ai_confirm["avg_change"],
            "code_skip_avg": code_skip["avg_change"],
            "delta": round(ai_confirm["avg_change"] - code_skip["avg_change"], 6),
            "ai_helps": ai_confirm["avg_change"] > code_skip["avg_change"],
        }

    # 2. AI-agreed vs AI-disagreed (within ai_confirm source)
    ai_confirm_entries = by_source.get("ai_confirm", [])
    agreed = [e for e in ai_confirm_entries if e.get("signal_agreed") is True]
    disagreed = [e for e in ai_confirm_entries if e.get("signal_agreed") is False]
    if len(agreed) >= MIN_SAMPLE and len(disagreed) >= MIN_SAMPLE:
        agreed_avg = sum(e["change"] for e in agreed) / len(agreed)
        disagreed_avg = sum(e["change"] for e in disagreed) / len(disagreed)
        comparisons["ai_agreed_vs_disagreed"] = {
            "description": "When AI agrees with bullish signal vs disagrees — which is better?",
            "agreed_n": len(agreed),
            "agreed_avg": round(agreed_avg, 6),
            "disagreed_n": len(disagreed),
            "disagreed_avg": round(disagreed_avg, 6),
            "delta": round(agreed_avg - disagreed_avg, 6),
            "overrides_help": disagreed_avg > agreed_avg,
        }

    # 3. AI exit vs mechanical stops — who times exits better?
    ai_exit = sources.get("ai_exit", {})
    stop_trailing = sources.get("stop_trailing", {})
    stop_time = sources.get("stop_time", {})
    # Combine mechanical stops
    stop_n = stop_trailing.get("n", 0) + stop_time.get("n", 0)
    if ai_exit.get("n", 0) >= MIN_SAMPLE and stop_n >= MIN_SAMPLE:
        # For exits: negative avg_change after selling = good (price kept falling)
        stop_entries = by_source.get("stop_trailing", []) + by_source.get("stop_time", [])
        stop_avg = sum(e["change"] for e in stop_entries) / len(stop_entries)
        comparisons["ai_exit_vs_stops"] = {
            "description": "AI-timed exits vs mechanical stops — which preserves more value?",
            "ai_exit_avg": ai_exit["avg_change"],
            "ai_exit_n": ai_exit["n"],
            "stop_avg": round(stop_avg, 6),
            "stop_n": stop_n,
            "delta": round(ai_exit["avg_change"] - stop_avg, 6),
            "ai_better": ai_exit["avg_change"] < stop_avg,  # lower = better for exits
        }

    # 4. Contrarian entries — do they work?
    contrarian = sources.get("code_contrarian", {})
    if contrarian.get("n", 0) >= MIN_SAMPLE:
        comparisons["contrarian_performance"] = {
            "description": "Do code-driven contrarian entries generate positive returns?",
            "n": contrarian["n"],
            "avg_change": contrarian["avg_change"],
            "win_rate": contrarian.get("win_rate"),
            "profitable": contrarian["avg_change"] > 0,
        }

    return comparisons


def _generate_verdict(sources: dict, comparisons: dict) -> str:
    """Generate a human-readable summary of whether AI is helping."""
    parts = []

    total_ai = sum(
        s.get("n", 0) for name, s in sources.items()
        if name in ("ai_confirm", "ai_exit")
    )
    total_code = sum(
        s.get("n", 0) for name, s in sources.items()
        if name in ("code_skip", "code_contrarian", "stop_trailing", "stop_time")
    )

    parts.append(f"Data: {total_ai} AI-decided cycles, {total_code} code-decided cycles.")

    # Report each comparison
    comp = comparisons.get("ai_confirm_vs_code_skip", {})
    if comp:
        if comp.get("ai_helps"):
            parts.append(
                f"AI confirmation adds value: {comp['delta']:+.4f} avg edge over code-skip."
            )
        else:
            parts.append(
                f"AI confirmation not helping: {comp['delta']:+.4f} vs code-skip."
            )

    comp = comparisons.get("ai_agreed_vs_disagreed", {})
    if comp:
        if comp.get("overrides_help"):
            parts.append(
                f"AI overrides are valuable: disagreements avg {comp['disagreed_avg']:+.4f} "
                f"vs agreements {comp['agreed_avg']:+.4f}."
            )
        else:
            parts.append(
                f"AI overrides not helping: disagreements avg {comp['disagreed_avg']:+.4f} "
                f"vs agreements {comp['agreed_avg']:+.4f}."
            )

    comp = comparisons.get("ai_exit_vs_stops", {})
    if comp:
        if comp.get("ai_better"):
            parts.append(f"AI times exits better than stops ({comp['delta']:+.4f}).")
        else:
            parts.append(f"Mechanical stops time exits better than AI ({comp['delta']:+.4f}).")

    comp = comparisons.get("contrarian_performance", {})
    if comp:
        if comp.get("profitable"):
            parts.append(
                f"Contrarian entries profitable: {comp['avg_change']:+.4f} avg "
                f"(win rate: {comp['win_rate']})."
            )
        else:
            parts.append(
                f"Contrarian entries losing: {comp['avg_change']:+.4f} avg "
                f"(win rate: {comp['win_rate']})."
            )

    if not comparisons:
        parts.append("Not enough data for comparisons yet.")

    return " ".join(parts)


def load_scorecard() -> dict:
    """Load the most recent scorecard from disk."""
    return load_json(SCORECARD_PATH, default={"sources": {}, "comparisons": {}})


def get_scorecard_for_brief() -> str:
    """Format scorecard as compact text for the agent brief.

    Shows AI its own track record so it can calibrate.
    """
    sc = load_scorecard()
    if not sc.get("sources"):
        return ""

    parts = ["=== JUDGMENT SCORECARD ==="]
    parts.append(sc.get("verdict", "No data."))

    # Show per-source stats if meaningful
    for source, stats in sc.get("sources", {}).items():
        if stats.get("n", 0) >= MIN_SAMPLE:
            wr = f", win_rate={stats['win_rate']}" if stats.get("win_rate") is not None else ""
            parts.append(
                f"  {source}: n={stats['n']}, avg={stats['avg_change']:+.4f}{wr}"
            )

    return "\n".join(parts)
