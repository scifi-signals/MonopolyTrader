"""Deep research engine — studies TSLA history to build predictive knowledge."""

import json
import os
from anthropic import Anthropic
from .utils import load_config, iso_now, setup_logging
from .knowledge_base import add_research, get_research, update_tsla_profile, get_tsla_profile
from .market_data import get_price_history

logger = setup_logging("researcher")


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


def _call_claude(system: str, user: str, max_tokens: int = 2000) -> tuple[str, str]:
    """Call Claude and return (response_text, model_version)."""
    config = load_config()
    client = _get_client()
    model = config["anthropic_model"]
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip(), model


def _parse_json(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def _price_summary(ticker: str, period: str = "6mo") -> str:
    """Generate a text summary of recent price action."""
    try:
        df = get_price_history(ticker, period=period, interval="1d")
        recent = df.tail(30)

        lines = [f"Last 30 trading days for {ticker}:"]
        for idx, row in recent.iterrows():
            lines.append(
                f"  {idx.strftime('%Y-%m-%d')}: O=${row['Open']:.2f} H=${row['High']:.2f} "
                f"L=${row['Low']:.2f} C=${row['Close']:.2f} V={int(row['Volume']):,}"
            )

        # Key stats
        close = recent["Close"]
        lines.append(f"\n30-day range: ${close.min():.2f} - ${close.max():.2f}")
        lines.append(f"30-day avg: ${close.mean():.2f}")
        lines.append(f"Current: ${close.iloc[-1]:.2f}")
        lines.append(f"30-day return: {((close.iloc[-1] / close.iloc[0]) - 1) * 100:+.1f}%")

        # Longer context
        full_close = df["Close"]
        lines.append(f"\n{period} range: ${full_close.min():.2f} - ${full_close.max():.2f}")
        lines.append(f"{period} return: {((full_close.iloc[-1] / full_close.iloc[0]) - 1) * 100:+.1f}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Price data unavailable: {e}"


# --- Research Tasks ---

RESEARCH_SYSTEM = """You are a financial research analyst studying TSLA (Tesla) stock behavior for an AI trading agent. Your research will be stored in a knowledge base and used to make better trading decisions.

Be specific and data-driven. Cite price levels, percentages, and timeframes. Focus on actionable insights — patterns that could help predict future price movements.

Return your findings as JSON:
{
  "summary": "<1-2 sentence summary>",
  "key_findings": ["<specific, actionable finding>", ...],
  "actionable_rules": ["<if X then Y rule for trading>", ...],
  "confidence": <0-1 how confident you are in these findings>,
  "caveats": ["<limitations or things to watch>"]
}"""


async def research_earnings_history(ticker: str = "TSLA") -> dict:
    """Study how TSLA has reacted to past earnings reports."""
    price_data = _price_summary(ticker)

    user_prompt = f"""Research TSLA's earnings reaction patterns.

Based on your knowledge of TSLA's earnings history (you have training data through early 2025):
1. How does TSLA typically move in the 5 days BEFORE earnings?
2. What's the average post-earnings move (beat vs miss)?
3. How long does the post-earnings drift last?
4. Are there patterns in which quarter tends to surprise?
5. How does options implied volatility typically behave?

Recent price action for context:
{price_data}

Provide specific, actionable trading rules based on earnings patterns."""

    try:
        raw, model_ver = _call_claude(RESEARCH_SYSTEM, user_prompt)
        findings = _parse_json(raw)
        findings["model_version"] = model_ver
        add_research("earnings_history", findings)
        logger.info(f"Earnings research complete: {len(findings.get('key_findings', []))} findings")

        # Fetch actual upcoming earnings dates via yfinance
        _fetch_upcoming_earnings(ticker)

        return findings
    except Exception as e:
        logger.error(f"Earnings research failed: {e}")
        return {}


def _fetch_upcoming_earnings(ticker: str = "TSLA"):
    """Fetch upcoming earnings dates via yfinance and save to earnings_history.json."""
    try:
        import yfinance as yf
        from datetime import datetime, timezone

        stock = yf.Ticker(ticker)
        dates = stock.get_earnings_dates(limit=4)
        if dates is not None and not dates.empty:
            upcoming = []
            now = datetime.now(timezone.utc)
            for dt in dates.index:
                # Convert to timezone-aware UTC datetime
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                else:
                    dt = dt.tz_convert("UTC")
                # Only keep future dates
                if dt >= now:
                    upcoming.append(dt.isoformat())
            if upcoming:
                # Merge into existing earnings_history.json
                research = get_research("earnings_history")
                research["upcoming_earnings"] = sorted(upcoming)
                research["upcoming_earnings_updated"] = iso_now()
                from .utils import save_json, KNOWLEDGE_DIR
                save_json(KNOWLEDGE_DIR / "research" / "earnings_history.json", research)
                logger.info(f"Saved {len(upcoming)} upcoming earnings dates for {ticker}")
        else:
            logger.info(f"No earnings dates returned by yfinance for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to fetch upcoming earnings dates: {e}")


async def research_catalyst_events(ticker: str = "TSLA") -> dict:
    """Study impact of major event types on TSLA."""
    price_data = _price_summary(ticker)

    user_prompt = f"""Research TSLA's sensitivity to different catalyst events.

Based on your knowledge of TSLA's history:
1. How does TSLA react to Elon Musk's tweets/posts (especially controversial ones)?
2. What happens after product announcements (new models, FSD updates, Optimus)?
3. How do delivery report beats/misses affect the stock?
4. What about regulatory news (NHTSA investigations, EU regulations)?
5. Analyst upgrade/downgrade impact?
6. Macro events (rate decisions, CPI, China news)?

Recent price action:
{price_data}

For each catalyst type, provide: typical magnitude, duration, and a trading rule."""

    try:
        raw, model_ver = _call_claude(RESEARCH_SYSTEM, user_prompt)
        findings = _parse_json(raw)
        findings["model_version"] = model_ver
        add_research("catalyst_events", findings)
        logger.info(f"Catalyst research complete: {len(findings.get('key_findings', []))} findings")
        return findings
    except Exception as e:
        logger.error(f"Catalyst research failed: {e}")
        return {}


async def research_correlations(ticker: str = "TSLA") -> dict:
    """Study TSLA's correlations with other assets."""
    price_data = _price_summary(ticker)

    user_prompt = f"""Research TSLA's correlations with other assets and indicators.

Based on your knowledge:
1. How correlated is TSLA with NASDAQ/QQQ? When do they decouple?
2. Does TSLA track Bitcoin/crypto sentiment?
3. How does TSLA relate to interest rates / 10-year yield?
4. Oil prices — does TSLA benefit from high oil (EV narrative) or suffer (risk-off)?
5. Other EV stocks (RIVN, LCID, NIO) — do they lead or lag TSLA?
6. VIX — how does TSLA behave in high vs low volatility regimes?

Recent price action:
{price_data}

Identify the strongest correlations and any leading indicators."""

    try:
        raw, model_ver = _call_claude(RESEARCH_SYSTEM, user_prompt)
        findings = _parse_json(raw)
        findings["model_version"] = model_ver
        add_research("correlation_notes", findings)
        logger.info(f"Correlation research complete: {len(findings.get('key_findings', []))} findings")
        return findings
    except Exception as e:
        logger.error(f"Correlation research failed: {e}")
        return {}


async def research_seasonal_patterns(ticker: str = "TSLA") -> dict:
    """Study seasonal and timing patterns."""
    price_data = _price_summary(ticker)

    user_prompt = f"""Research TSLA's seasonal and timing patterns.

Based on your knowledge:
1. Day-of-week effects — does TSLA tend to perform differently on certain days?
2. Time-of-day patterns — opening volatility, lunch lull, power hour?
3. Monthly patterns — are certain months historically better?
4. Options expiration effects — monthly and quarterly OPEX impact?
5. Quarter-end and year-end behavior?
6. January effect or tax-loss selling patterns?

Recent price action:
{price_data}

Focus on statistically meaningful patterns with enough history to be reliable."""

    try:
        raw, model_ver = _call_claude(RESEARCH_SYSTEM, user_prompt)
        findings = _parse_json(raw)
        findings["model_version"] = model_ver
        add_research("seasonal_patterns", findings)
        logger.info(f"Seasonal research complete: {len(findings.get('key_findings', []))} findings")
        return findings
    except Exception as e:
        logger.error(f"Seasonal research failed: {e}")
        return {}


async def research_current_context(ticker: str = "TSLA") -> dict:
    """Research what's happening RIGHT NOW that matters for TSLA."""
    price_data = _price_summary(ticker)

    user_prompt = f"""Analyze the current market context for TSLA.

Recent price action:
{price_data}

Based on the price action and your knowledge up to early 2025:
1. What's the current market regime? (trending, range-bound, volatile)
2. Where are the key support and resistance levels?
3. What's the likely impact of current macro conditions?
4. What upcoming events could be catalysts?
5. What's the overall risk/reward setup right now?

Provide a current assessment with specific price levels and scenarios."""

    try:
        raw, model_ver = _call_claude(RESEARCH_SYSTEM, user_prompt)
        findings = _parse_json(raw)
        findings["model_version"] = model_ver
        add_research("sector_context", findings)

        # Update TSLA profile with current context
        rules = findings.get("actionable_rules", [])
        profile_updates = {
            "market_regime": _extract_regime(findings),
            "key_levels": _extract_levels(findings),
            "behavioral_notes": findings.get("key_findings", [])[:5],
        }
        update_tsla_profile(profile_updates)

        logger.info(f"Current context research complete")
        return findings
    except Exception as e:
        logger.error(f"Current context research failed: {e}")
        return {}


async def research_on_demand(ticker: str, topic: str) -> dict:
    """Agent-requested targeted research on a specific question."""
    price_data = _price_summary(ticker)

    user_prompt = f"""The trading agent has requested research on this specific topic:

"{topic}"

Context — recent {ticker} price action:
{price_data}

Research this topic thoroughly. Provide specific, actionable findings."""

    try:
        raw, model_ver = _call_claude(RESEARCH_SYSTEM, user_prompt)
        findings = _parse_json(raw)
        findings["model_version"] = model_ver
        # Store with sanitized topic name
        safe_topic = topic[:50].replace(" ", "_").replace("/", "_").lower()
        add_research(f"ondemand_{safe_topic}", findings)
        logger.info(f"On-demand research complete: {topic[:50]}")
        return findings
    except Exception as e:
        logger.error(f"On-demand research failed: {e}")
        return {}


def _extract_regime(findings: dict) -> str:
    """Extract market regime from research findings."""
    summary = findings.get("summary", "").lower()
    for regime in ["trending up", "bullish", "uptrend"]:
        if regime in summary:
            return "trending_up"
    for regime in ["trending down", "bearish", "downtrend"]:
        if regime in summary:
            return "trending_down"
    for regime in ["range", "sideways", "consolidat"]:
        if regime in summary:
            return "range_bound"
    for regime in ["volatile", "choppy", "uncertain"]:
        if regime in summary:
            return "volatile"
    return "unknown"


def _extract_levels(findings: dict) -> dict:
    """Try to extract support/resistance levels from findings."""
    levels = {"support": [], "resistance": []}
    for finding in findings.get("key_findings", []):
        finding_lower = finding.lower()
        # Simple extraction — look for price-like numbers near support/resistance keywords
        if "support" in finding_lower:
            levels["support"].append(finding[:100])
        if "resistance" in finding_lower:
            levels["resistance"].append(finding[:100])
    return levels


# --- Run All Research ---

async def run_full_research(ticker: str = "TSLA"):
    """Run all research tasks. Called on bootstrap and daily."""
    logger.info(f"Starting full research for {ticker}...")

    results = {}
    for name, func in [
        ("earnings_history", research_earnings_history),
        ("catalyst_events", research_catalyst_events),
        ("correlations", research_correlations),
        ("seasonal_patterns", research_seasonal_patterns),
        ("current_context", research_current_context),
    ]:
        logger.info(f"  Researching: {name}...")
        try:
            results[name] = await func(ticker)
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            results[name] = {}

    logger.info("Full research complete")
    return results
