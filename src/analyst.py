"""v5 Analyst — nightly intelligence update + pre-market briefing.

Separates research from trading. Sonnet builds persistent market intelligence
that the trading Sonnet uses to make informed decisions every cycle.
"""

import json
from datetime import datetime, timezone, timedelta

from .utils import (
    load_config, load_json, save_json, iso_now, setup_logging,
    call_ai_with_fallback, DATA_DIR,
)
from .journal import get_entries_since, load_journal
from .thesis_builder import format_playbook_for_brief, LEDGER_PATH
from .market_data import get_current_price
from .news_feed import fetch_news_feed, format_news_for_prompt

logger = setup_logging("analyst")

MID_PATH = DATA_DIR / "market_intelligence.json"
BRIEFING_PATH = DATA_DIR / "daily_briefing.json"


NIGHTLY_PROMPT = """You are the MonopolyTrader Analyst. Your job is to update the Market Intelligence Document for tomorrow's trading.

You will receive:
- The CURRENT Market Intelligence Document (what you wrote yesterday)
- Today's CLOSED TRADES with outcomes (P&L, tags, lessons)
- Current PLAYBOOK STATS (win rates by market condition)
- OVERNIGHT NEWS headlines
- Current TSLA PRICE and KEY INDICATORS

OUTPUT: Return ONLY a valid JSON object matching this exact schema. Every field must be populated.

{
  "last_updated": "<current ISO timestamp>",
  "update_number": <previous + 1>,
  "thesis": {
    "direction": "bullish" | "bearish" | "neutral",
    "confidence": <float 0.3 to 0.9>,
    "reasoning": "<2-3 sentences explaining why>",
    "established": "<date thesis was first set to this direction>",
    "status": "active",
    "supporting_evidence": ["<specific evidence, max 5 items>"],
    "contradicting_evidence": ["<specific evidence, max 5 items, ALWAYS at least 1>"],
    "invalidation_criteria": "<specific price or condition that proves this thesis wrong>"
  },
  "key_levels": {
    "strong_support": <price>,
    "support": <price>,
    "resistance": <price>,
    "strong_resistance": <price>,
    "notes": "<brief explanation of levels>"
  },
  "active_catalysts": [
    {"catalyst": "<what>", "impact": "bullish" | "bearish" | "uncertain", "age_days": <int>, "still_relevant": true | false}
  ],
  "sector_context": "<1-2 sentences on EV sector and macro>",
  "lessons_synthesis": "<from recent trade lessons, identify the single most profitable pattern and single most unprofitable pattern>",
  "what_working": "<from playbook stats with N>=3, cite specific tag:value win rates>",
  "what_not_working": "<from playbook stats with N>=3, cite specific tag:value win rates marked AVOID>",
  "pre_market_override": null
}

RULES FOR UPDATING:

THESIS DIRECTION:
- Start from yesterday's thesis. Do NOT flip direction without strong evidence.
- To CHANGE direction requires: (a) 3+ consecutive losses on current thesis OR (b) major catalyst that invalidates the thesis OR (c) price breaks through invalidation_criteria
- To LOWER confidence: 1-2 losses on the thesis, contradicting evidence grows, OR net P&L on trades in last 7 days is negative (reduce by at least 0.1)
- To RAISE confidence: 2+ wins confirming thesis, or supporting evidence grows
- Never go above 0.9 or below 0.3.
- If flipping direction, reset "established" to today's date.

EVIDENCE:
- Be specific: "$2.3M insider buying on March 5" not "insider buying"
- Remove evidence older than 14 days unless it's a major structural factor
- ALWAYS include at least 1 contradicting evidence item — there is ALWAYS a bear case

KEY LEVELS:
- Use recent highs/lows, SMA levels, round numbers
- Update if price has moved significantly from previous levels

ACTIVE CATALYSTS:
- Remove catalysts older than 14 days unless still actively impacting price
- Mark "still_relevant: false" for catalysts whose impact has been fully absorbed
- Add new catalysts from overnight news

LESSONS SYNTHESIS:
- From provided trade lessons, identify the MOST FREQUENT profitable pattern and MOST FREQUENT unprofitable pattern
- Be specific and actionable, not generic advice

WHAT'S WORKING / NOT WORKING:
- Pull directly from playbook stats with N >= 3
- Cite exact numbers: "rsi_zone=oversold: 4/5 wins (80%)"

Do NOT:
- Invent data or stats you weren't given
- Write more than 800 tokens total
- Include any text outside the JSON object
- Set confidence above 0.9 or below 0.3"""


PRE_MARKET_PROMPT = """You are the MonopolyTrader Pre-Market Analyst. Produce a daily trading briefing.

You will receive:
- The current Market Intelligence Document (thesis, levels, catalysts)
- Pre-market / latest TSLA price
- Overnight news

OUTPUT: Return ONLY a valid JSON object:

{
  "date": "<today's date YYYY-MM-DD>",
  "pre_market_price": <float or null>,
  "overnight_change_pct": <float or null>,
  "thesis_status": "confirmed" | "challenged" | "invalidated",
  "scenarios": [
    {"condition": "<if price does X>", "action": "<what to consider>"}
  ],
  "key_news_overnight": ["<headline 1>", "<headline 2>"],
  "recommended_posture": "aggressive" | "cautiously_bullish" | "cautiously_bearish" | "defensive" | "neutral",
  "posture_reasoning": "<1-2 sentences>"
}

THESIS STATUS RULES:
- "confirmed": overnight data aligns with the current thesis direction
- "challenged": some contradicting evidence but not enough to flip (e.g., modest adverse price move, mixed news)
- "invalidated": clear evidence against thesis (price broke invalidation level, major negative catalyst)

SCENARIOS: Provide 2-3 scenarios based on likely price ranges today. Be specific about price levels.

Do NOT include any text outside the JSON object."""


def load_mid() -> dict:
    """Load the Market Intelligence Document."""
    return load_json(MID_PATH, default={})


def run_nightly_update():
    """After market close: update the Market Intelligence Document.

    Reads today's trades, playbook, and news. Haiku synthesizes into
    an updated MID that Claude Sonnet sees every cycle tomorrow.
    """
    config = load_config()
    analyst_model = config.get("analyst_model", config.get("anthropic_model", "claude-sonnet-4-20250514"))

    # Load current MID
    current_mid = load_mid()

    # Get today's closed trades (last 7 days for lesson synthesis)
    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_entries = get_entries_since(seven_days_ago)
    closed_today = [
        e for e in recent_entries
        if e.get("realized_pnl") is not None
    ]

    # Format trades for the analyst
    trades_text = "No closed trades in last 7 days."
    if closed_today:
        lines = []
        for e in closed_today:
            pnl = e.get("realized_pnl", 0)
            lesson = e.get("lesson", "")
            tags = e.get("tags", {})
            lines.append(
                f"  {e.get('action')} {e.get('shares', 0):.2f} @ ${e.get('price', 0):.2f} "
                f"→ closed @ ${e.get('close_price', 0):.2f}, P&L: ${pnl:+.2f}"
            )
            if lesson:
                lines.append(f"    Lesson: {lesson}")
            if tags:
                tag_str = ", ".join(f"{k}={v}" for k, v in tags.items())
                lines.append(f"    Tags: {tag_str}")
        trades_text = f"Closed trades (last 7 days, {len(closed_today)} trades):\n" + "\n".join(lines)

    # Get playbook stats
    ledger = load_json(LEDGER_PATH, default={})
    playbook_text = format_playbook_for_brief(ledger)

    # Get latest price
    try:
        current = get_current_price(config["ticker"])
        price_text = f"TSLA: ${current['price']} ({current['change_pct']:+.2f}%)"
    except Exception:
        price_text = "TSLA price unavailable"

    # Get news
    try:
        news_feed = fetch_news_feed(config["ticker"])
        news_text = format_news_for_prompt(news_feed, max_items=8) if news_feed.items else "No news."
    except Exception:
        news_text = "News unavailable."

    # Build the user prompt
    user_prompt = f"""=== CURRENT MARKET INTELLIGENCE DOCUMENT ===
{json.dumps(current_mid, indent=2)}

=== TODAY'S TRADES AND LESSONS ===
{trades_text}

=== PLAYBOOK STATS ===
{playbook_text}

=== LATEST PRICE ===
{price_text}

=== NEWS ===
{news_text}

Update the Market Intelligence Document based on all of the above. Return ONLY valid JSON."""

    try:
        raw, model_used = call_ai_with_fallback(
            system=NIGHTLY_PROMPT,
            user=user_prompt,
            max_tokens=1500,
            config=config,
            model=analyst_model,
        )

        # Parse JSON response — strip code fences and any preamble
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

        # Find JSON object if there's text before it
        json_start = clean.find("{")
        if json_start > 0:
            clean = clean[json_start:]

        updated_mid = json.loads(clean)

        # Validate essential fields exist
        if "thesis" not in updated_mid or "direction" not in updated_mid.get("thesis", {}):
            raise ValueError("MID missing required thesis fields")

        # Ensure confidence bounds
        conf = updated_mid.get("thesis", {}).get("confidence", 0.5)
        updated_mid["thesis"]["confidence"] = max(0.3, min(0.9, conf))

        # Clear any stale pre_market_override
        updated_mid["pre_market_override"] = None

        save_json(MID_PATH, updated_mid)
        logger.info(
            f"MID updated: thesis={updated_mid['thesis']['direction']} "
            f"confidence={updated_mid['thesis']['confidence']:.1f} "
            f"(model={model_used})"
        )
        return updated_mid

    except json.JSONDecodeError as e:
        logger.error(f"Nightly analyst failed to parse JSON: {e}\nRaw: {raw[:500]}")
        logger.info("Keeping existing MID unchanged")
        return current_mid
    except Exception as e:
        logger.error(f"Nightly analyst failed: {e}")
        logger.info("Keeping existing MID unchanged")
        return current_mid


def run_pre_market():
    """Before market open: produce daily briefing with scenarios.

    Reads MID + pre-market price + overnight news. Produces a daily
    briefing with thesis status and trading scenarios.
    """
    config = load_config()
    analyst_model = config.get("analyst_model", config.get("anthropic_model", "claude-sonnet-4-20250514"))
    ticker = config["ticker"]

    # Load current MID
    mid = load_mid()

    # Get pre-market / latest price
    pre_market_price = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        pre_market_price = info.get("preMarketPrice") or info.get("regularMarketPrice")
    except Exception:
        pass

    if pre_market_price is None:
        try:
            current = get_current_price(ticker)
            pre_market_price = current["price"]
        except Exception:
            pre_market_price = None

    price_text = f"Pre-market TSLA: ${pre_market_price:.2f}" if pre_market_price else "Pre-market price unavailable"

    # Get overnight news
    try:
        news_feed = fetch_news_feed(ticker)
        news_text = format_news_for_prompt(news_feed, max_items=5) if news_feed.items else "No overnight news."
    except Exception:
        news_text = "News unavailable."

    user_prompt = f"""=== MARKET INTELLIGENCE DOCUMENT ===
{json.dumps(mid, indent=2)}

=== PRE-MARKET PRICE ===
{price_text}

=== OVERNIGHT NEWS ===
{news_text}

Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

Produce the daily briefing. Return ONLY valid JSON."""

    try:
        raw, model_used = call_ai_with_fallback(
            system=PRE_MARKET_PROMPT,
            user=user_prompt,
            max_tokens=600,
            config=config,
            model=analyst_model,
        )

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

        briefing = json.loads(clean)

        # If thesis is invalidated, update MID with override
        if briefing.get("thesis_status") == "invalidated":
            mid["pre_market_override"] = {
                "status": "invalidated",
                "reason": briefing.get("posture_reasoning", "Pre-market analysis invalidated thesis"),
                "timestamp": iso_now(),
            }
            save_json(MID_PATH, mid)
            logger.warning(f"Pre-market INVALIDATED thesis: {briefing.get('posture_reasoning')}")

        save_json(BRIEFING_PATH, briefing)
        logger.info(
            f"Daily briefing: thesis_status={briefing.get('thesis_status')}, "
            f"posture={briefing.get('recommended_posture')} (model={model_used})"
        )
        return briefing

    except json.JSONDecodeError as e:
        logger.error(f"Pre-market briefing failed to parse: {e}\nRaw: {raw[:500]}")
        # Write minimal briefing
        fallback = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "thesis_status": "unknown",
            "scenarios": [],
            "recommended_posture": "neutral",
            "posture_reasoning": "Pre-market analysis failed — proceed with caution",
        }
        save_json(BRIEFING_PATH, fallback)
        return fallback
    except Exception as e:
        logger.error(f"Pre-market briefing failed: {e}")
        fallback = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "thesis_status": "unknown",
            "scenarios": [],
            "recommended_posture": "neutral",
            "posture_reasoning": f"Pre-market analysis error: {e}",
        }
        save_json(BRIEFING_PATH, fallback)
        return fallback


def format_mid_for_system_prompt(mid: dict) -> str:
    """Format the MID for inclusion in Claude's system prompt."""
    if not mid or not mid.get("thesis"):
        return ""

    thesis = mid["thesis"]
    parts = [
        "=== YOUR MARKET INTELLIGENCE (updated nightly) ===",
        f"Thesis: {thesis.get('direction', '?').upper()} (confidence: {thesis.get('confidence', 0):.1f})",
        f"  {thesis.get('reasoning', 'No reasoning')}",
    ]

    # Pre-market override warning
    override = mid.get("pre_market_override")
    if override:
        parts.append(f"  >>> WARNING: Thesis {override.get('status', 'challenged')} by pre-market analysis <<<")
        parts.append(f"  Reason: {override.get('reason', 'Unknown')}")

    # Evidence
    supporting = thesis.get("supporting_evidence", [])
    if supporting:
        parts.append(f"  Supporting: {'; '.join(supporting[:3])}")
    contradicting = thesis.get("contradicting_evidence", [])
    if contradicting:
        parts.append(f"  Contradicting: {'; '.join(contradicting[:3])}")
    if thesis.get("invalidation_criteria"):
        parts.append(f"  Invalidation: {thesis['invalidation_criteria']}")

    # Key levels
    levels = mid.get("key_levels", {})
    if levels.get("support") or levels.get("resistance"):
        level_parts = []
        for name in ["strong_support", "support", "resistance", "strong_resistance"]:
            val = levels.get(name)
            if val:
                level_parts.append(f"{name.replace('_', ' ')}=${val}")
        if level_parts:
            parts.append(f"  Key levels: {', '.join(level_parts)}")
        if levels.get("notes"):
            parts.append(f"  ({levels['notes']})")

    # Catalysts
    catalysts = [c for c in mid.get("active_catalysts", []) if c.get("still_relevant", True)]
    if catalysts:
        parts.append("  Active catalysts:")
        for c in catalysts[:4]:
            parts.append(f"    [{c.get('impact', '?')}] {c.get('catalyst', '?')} ({c.get('age_days', '?')}d ago)")

    # Sector
    if mid.get("sector_context"):
        parts.append(f"  Sector: {mid['sector_context']}")

    # Lessons
    if mid.get("lessons_synthesis"):
        parts.append(f"  Recent lessons: {mid['lessons_synthesis']}")

    # What's working
    if mid.get("what_working"):
        parts.append(f"  Working: {mid['what_working']}")
    if mid.get("what_not_working"):
        parts.append(f"  Not working: {mid['what_not_working']}")

    return "\n".join(parts)


def format_briefing_for_prompt(briefing: dict) -> str:
    """Format the daily briefing for inclusion in the user prompt."""
    if not briefing:
        return "No daily briefing available."

    parts = [
        f"Thesis status: {briefing.get('thesis_status', 'unknown').upper()}",
    ]

    if briefing.get("recommended_posture"):
        parts.append(f"Recommended posture: {briefing['recommended_posture']}")
    if briefing.get("posture_reasoning"):
        parts.append(f"  {briefing['posture_reasoning']}")

    scenarios = briefing.get("scenarios", [])
    if scenarios:
        parts.append("Scenarios:")
        for s in scenarios:
            parts.append(f"  IF {s.get('condition', '?')}: {s.get('action', '?')}")

    news = briefing.get("key_news_overnight", [])
    if news:
        parts.append("Overnight news:")
        for headline in news[:3]:
            parts.append(f"  - {headline}")

    return "\n".join(parts)
