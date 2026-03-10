I now have a complete understanding of the entire v3 codebase. Here is the full v4 implementation plan.

---

# MonopolyTrader v4 Implementation Plan

## Design Philosophy Summary

v3 had Claude as a rubber-stamper for coded strategy signals. The strategies.py computed signals, aggregated them with weighted votes, applied statistical constraints, ATR-sized the position, and then asked Claude "here's what the math says -- agree?" Claude's job was to parse a pre-digested brief and output JSON. v4 flips this: Claude IS the brain. Raw data goes in, decisions come out. No intermediary strategy logic telling Claude what to think.

---

## File Disposition

### DELETE (9 files)

1. **`src/strategies.py`** -- DELETE. The five coded strategies (momentum, mean_reversion, technical_signals, range_trader, thesis_alignment) and all their signal aggregation logic are the core of what v4 removes. Claude sees the raw indicators and decides for itself.

2. **`src/thesis.py`** -- DELETE. The thesis system (Analyst/Trader two-phase architecture) is replaced by Claude reasoning directly in every cycle. There is no persistent thesis object.

3. **`src/trade_stats.py`** -- DELETE. The per-dimension hit rate engine, statistical constraints, and conviction calibration system are all part of the constraint machinery v4 eliminates.

4. **`src/learner.py`** -- DELETE. The full learning engine (Skeptic layer, pattern discovery, strategy weight evolution, journal entries, trade reviews, hold reviews) is replaced by the trade journal. One Haiku call per closed trade produces one 50-word lesson.

5. **`src/knowledge_base.py`** -- DELETE. The entire knowledge directory structure (lessons.json, patterns.json, predictions.json, tsla_profile.json, strategy_scores.json, journal.md, research/) is replaced by a single `data/trade_journal.json` file.

6. **`src/researcher.py`** -- DELETE. The deep research engine that produced canned Claude summaries about TSLA behavior is removed. Claude gets live context (news, web search, macro data) every cycle instead.

7. **`src/ensemble.py`** -- DELETE. Multi-agent orchestration is not part of v4.

8. **`src/comparison.py`** -- DELETE. Agent comparison/leaderboard is not part of v4.

9. **`src/meta_learner.py`** -- DELETE. Cross-agent analysis is not part of v4.

### KEEP AS-IS (3 files)

1. **`src/__init__.py`** -- KEEP AS-IS.

2. **`src/observability.py`** -- KEEP AS-IS. Health checks still useful. (If it imports deleted modules, those imports will need guarding, but the module itself stays.)

3. **`src/benchmarks.py`** -- KEEP AS-IS. Benchmark tracking is orthogonal to the trading brain redesign.

### MODIFY (5 files)

1. **`src/utils.py`** -- MODIFY
2. **`src/market_data.py`** -- MODIFY
3. **`src/portfolio.py`** -- MODIFY
4. **`src/reporter.py`** -- MODIFY
5. **`dashboard/index.html`** -- MODIFY
6. **`dashboard/app.js`** -- MODIFY
7. **`config.json`** -- MODIFY

### REPLACE (2 files)

1. **`src/agent.py`** -- REPLACE (completely new Claude-as-brain architecture)
2. **`src/main.py`** -- REPLACE (simplified 15-minute cycle)

### NEW (2 files)

1. **`src/journal.py`** -- NEW (trade journal system)
2. **`src/web_search.py`** -- NEW (Brave Search API integration)

---

## Detailed Specifications

### 1. `config.json` -- MODIFY

Replace the entire file with:

```json
{
  "ticker": "TSLA",
  "starting_balance": 1000.00,
  "currency": "Monopoly Dollars",
  "poll_interval_minutes": 15,
  "market_hours": {
    "open": "09:30",
    "close": "16:00",
    "timezone": "America/New_York"
  },
  "risk_params": {
    "max_position_pct": 0.50,
    "min_cash_reserve": 100.00,
    "slippage_per_side_pct": 0.0005,
    "slippage_volatile_per_side_pct": 0.0015,
    "enable_fractional_shares": true,
    "cooldown_minutes": 15
  },
  "anthropic_model": "claude-sonnet-4-20250514",
  "haiku_model": "claude-haiku-4-5-20251001",
  "brave_search": {
    "enabled": true,
    "max_results": 5,
    "query_templates": [
      "TSLA Tesla stock news today",
      "Tesla {catalyst} latest"
    ]
  },
  "world_tickers": {
    "macro": ["^TNX", "CL=F", "BTC-USD", "^IXIC", "DX-Y.NYB", "SPY", "^VIX"],
    "ev_peers": ["RIVN", "LCID", "NIO", "LI", "XPEV"]
  },
  "journal": {
    "max_entries_in_brief": 10,
    "lesson_max_words": 50
  },
  "news_feed": {
    "yfinance": true,
    "rss": true,
    "brave_search": true
  }
}
```

Key changes:
- `risk_params` simplified: removed stop_loss_method, ATR multipliers, position_sizing_method, max_risk_per_trade_pct, daily_loss_limit_pct, earnings_blackout_hours, gap_risk_size_reduction_pct, bsm_conviction_cap_pct, macro_gate, kill_switches. Now just: max_position_pct=0.50, min_cash_reserve=$100, slippage, fractional shares, cooldown.
- Removed `strategies_enabled`, `multi_step_reasoning`, `learning`, `event_triggers`, `benchmarks`, `graduation`, `bsm`.
- Added `brave_search`, `world_tickers`, `journal`, `haiku_model`.

### 2. `src/utils.py` -- MODIFY

Changes:
- Remove `KNOWLEDGE_DIR` constant (no more knowledge/ directory).
- Remove `get_cost_summary` (keep `_log_api_usage` for cost tracking -- it is useful).
- Keep everything else: `load_config`, `load_json`, `save_json`, `now_utc`, `now_et`, `iso_now`, `is_market_open`, `setup_logging`, `format_currency`, `format_pct`, `call_ai_with_fallback`, `generate_id`, `ROOT_DIR`, `DATA_DIR`, `LOGS_DIR`, `COSTS_DIR`, `DASHBOARD_DIR`, `CONFIG_PATH`, `MODEL_PRICING`.

Specifically, change line 15 from:
```python
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
```
to remove it entirely (or keep but don't create it). The journal lives in `DATA_DIR / "trade_journal.json"`.

Actually, keep `KNOWLEDGE_DIR` defined but don't use it. The old knowledge/ directory will still exist on disk; we just stop writing to it. Cleaner to remove the constant and let any imports fail loudly.

Final decision: **Remove the `KNOWLEDGE_DIR` line. Remove `get_cost_summary` function.** Everything else stays.

### 3. `src/web_search.py` -- NEW

Brave Search API integration.

```python
"""Web search via Brave Search API — free tier, 1 query/second, 2000/month."""

import os
import httpx
from .utils import setup_logging

logger = setup_logging("web_search")

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def brave_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Brave and return simplified results.
    
    Args:
        query: Search query string.
        max_results: Number of results to return (max 20).
    
    Returns:
        List of dicts: [{"title": str, "url": str, "snippet": str, "age": str}, ...]
        Returns empty list on any error (never crashes the cycle).
    """
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        for path in ["brave_api_key.txt", "../brave_api_key.txt"]:
            try:
                with open(path) as f:
                    api_key = f.read().strip()
                    break
            except FileNotFoundError:
                continue
    
    if not api_key:
        logger.info("No BRAVE_API_KEY found — web search disabled")
        return []
    
    try:
        resp = httpx.get(
            BRAVE_ENDPOINT,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={
                "q": query,
                "count": min(max_results, 20),
                "freshness": "pd",  # past day
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", "")[:300],
                "age": item.get("age", ""),
            })
        
        logger.info(f"Brave search '{query[:40]}': {len(results)} results")
        return results
    
    except Exception as e:
        logger.warning(f"Brave search failed: {e}")
        return []


def search_tsla_news() -> list[dict]:
    """Run the standard TSLA news search query."""
    return brave_search("TSLA Tesla stock news today", max_results=5)


def search_catalyst(catalyst: str) -> list[dict]:
    """Search for a specific TSLA catalyst."""
    return brave_search(f"Tesla TSLA {catalyst} latest", max_results=3)


def format_search_results(results: list[dict]) -> str:
    """Format search results as text for the Claude brief."""
    if not results:
        return "No web search results."
    
    lines = []
    for r in results:
        age = f" ({r['age']})" if r.get("age") else ""
        lines.append(f"- {r['title']}{age}")
        if r.get("snippet"):
            lines.append(f"  {r['snippet'][:150]}")
    
    return "\n".join(lines)
```

Brave Search API details:
- Endpoint: `https://api.search.brave.com/res/v1/web/search`
- Auth: `X-Subscription-Token` header with API key
- Free tier: 2,000 queries/month, 1 query/second
- We use `freshness=pd` (past day) for current news
- Key stored in env var `BRAVE_API_KEY` or file `brave_api_key.txt`

### 4. `src/market_data.py` -- MODIFY

Changes to the existing module:

**Add:** `get_world_snapshot()` function that fetches all macro + EV peer data in one call.

**Keep:** `get_current_price`, `get_price_history`, `get_intraday`, `calculate_indicators`, `_last`. These all work fine.

**Remove:** `check_macro_gate` (Claude evaluates macro conditions itself), `get_bsm_signals` (BSM integration removed), `check_volume_spike` (not needed as separate trigger), `check_vix_change` (not needed as separate trigger), `_load_previous_regime`, `_save_current_regime`.

**Simplify:** `classify_regime` -- keep but simplify. Remove hysteresis file saving. Just return the regime dict.

**Simplify:** `get_macro_data` -- keep but rename internal references.

**Simplify:** `get_market_summary` -- keep but it no longer needs to be as heavy. The new `get_world_snapshot` supplements it.

**New function:**

```python
def get_world_snapshot(config: dict = None) -> dict:
    """Fetch macro instruments + EV peers for world context.
    
    Returns:
        {
            "macro": {
                "^TNX": {"price": float, "change_pct": float, "name": "10Y Treasury Yield"},
                "CL=F": {"price": float, "change_pct": float, "name": "Crude Oil"},
                ...
            },
            "ev_peers": {
                "RIVN": {"price": float, "change_pct": float},
                ...
            },
            "fetched_at": str
        }
    """
    if config is None:
        config = load_config()
    
    world_tickers = config.get("world_tickers", {})
    macro_tickers = world_tickers.get("macro", [])
    peer_tickers = world_tickers.get("ev_peers", [])
    
    NAMES = {
        "^TNX": "10Y Treasury Yield",
        "CL=F": "Crude Oil WTI",
        "BTC-USD": "Bitcoin",
        "^IXIC": "NASDAQ Composite",
        "DX-Y.NYB": "US Dollar Index",
        "SPY": "S&P 500 ETF",
        "^VIX": "VIX Volatility",
    }
    
    result = {"macro": {}, "ev_peers": {}, "fetched_at": str(datetime.utcnow())}
    
    for ticker in macro_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d", interval="1d")
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((curr - prev) / prev * 100, 2)
                result["macro"][ticker] = {
                    "price": round(curr, 2),
                    "change_pct": change_pct,
                    "name": NAMES.get(ticker, ticker),
                }
        except Exception as e:
            logger.warning(f"World snapshot {ticker}: {e}")
    
    for ticker in peer_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d", interval="1d")
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((curr - prev) / prev * 100, 2)
                result["ev_peers"][ticker] = {
                    "price": round(curr, 2),
                    "change_pct": change_pct,
                }
        except Exception as e:
            logger.warning(f"World snapshot {ticker}: {e}")
    
    return result
```

### 5. `src/journal.py` -- NEW

The trade journal replaces the entire learning stack.

```python
"""Trade journal — one entry per trade, one lesson per close.

The journal is the only persistent learning mechanism in v4.
Every trade gets an entry when opened. When closed, Haiku writes
a 50-word lesson. The last 10 entries appear in every trading brief.
"""

import os
from anthropic import Anthropic
from .utils import load_json, save_json, iso_now, setup_logging, load_config, DATA_DIR

logger = setup_logging("journal")

JOURNAL_PATH = DATA_DIR / "trade_journal.json"

# Haiku for cheap lessons
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def load_journal() -> list[dict]:
    """Load all journal entries.
    
    Returns:
        List of journal entry dicts, ordered oldest-first.
    """
    return load_json(JOURNAL_PATH, default=[])


def save_journal(entries: list[dict]) -> None:
    """Save journal entries to disk."""
    save_json(JOURNAL_PATH, entries)


def add_entry(
    trade_id: str,
    action: str,
    ticker: str,
    shares: float,
    price: float,
    reasoning: str,
    confidence: float,
    portfolio_value: float,
    market_snapshot: str,
) -> dict:
    """Record a new trade in the journal.
    
    Called immediately after a trade executes.
    
    Args:
        trade_id: Transaction ID (e.g. "txn_042").
        action: "BUY" or "SELL".
        ticker: Stock ticker.
        shares: Number of shares traded.
        price: Execution price.
        reasoning: Claude's reasoning for this trade.
        confidence: Claude's stated confidence (0-1).
        portfolio_value: Portfolio value after trade.
        market_snapshot: One-line market context (price, VIX, etc.).
    
    Returns:
        The journal entry dict.
    """
    entries = load_journal()
    
    entry = {
        "trade_id": trade_id,
        "timestamp": iso_now(),
        "action": action,
        "ticker": ticker,
        "shares": round(shares, 4),
        "price": round(price, 2),
        "total_value": round(shares * price, 2),
        "reasoning": reasoning[:500],
        "confidence": round(confidence, 2),
        "portfolio_value": round(portfolio_value, 2),
        "market_snapshot": market_snapshot[:200],
        "lesson": None,          # filled on close
        "close_trade_id": None,  # filled when position closes
        "close_price": None,     # filled when position closes
        "realized_pnl": None,    # filled when position closes
        "closed_at": None,       # filled when position closes
    }
    
    entries.append(entry)
    save_journal(entries)
    logger.info(f"Journal: recorded {action} {shares:.4f} {ticker} @ ${price:.2f}")
    return entry


def close_entry(
    open_trade_id: str,
    close_trade_id: str,
    close_price: float,
    realized_pnl: float,
) -> dict | None:
    """Mark a journal entry as closed and generate a lesson via Haiku.
    
    Called when a SELL closes a position that was opened by a previous BUY.
    
    Args:
        open_trade_id: The trade_id of the original BUY entry.
        close_trade_id: The trade_id of the closing SELL.
        close_price: Price at which position was closed.
        realized_pnl: Realized P&L from this round trip.
    
    Returns:
        The updated journal entry with lesson, or None if entry not found.
    """
    entries = load_journal()
    
    target = None
    for entry in entries:
        if entry["trade_id"] == open_trade_id and entry["lesson"] is None:
            target = entry
            break
    
    if target is None:
        logger.warning(f"Journal: no open entry found for {open_trade_id}")
        return None
    
    target["close_trade_id"] = close_trade_id
    target["close_price"] = round(close_price, 2)
    target["realized_pnl"] = round(realized_pnl, 2)
    target["closed_at"] = iso_now()
    
    # Generate lesson via Haiku
    lesson = _generate_lesson(target)
    target["lesson"] = lesson
    
    save_journal(entries)
    logger.info(
        f"Journal: closed {open_trade_id} → P&L ${realized_pnl:.2f} | "
        f"Lesson: {lesson[:80]}"
    )
    return target


def _generate_lesson(entry: dict) -> str:
    """Call Haiku to generate a 50-word lesson from a closed trade.
    
    Args:
        entry: Journal entry dict with trade details and outcome.
    
    Returns:
        A lesson string, max 50 words. Returns a fallback on error.
    """
    config = load_config()
    model = config.get("haiku_model", HAIKU_MODEL)
    
    pnl = entry.get("realized_pnl", 0)
    outcome = "profit" if pnl >= 0 else "loss"
    pnl_pct = (pnl / entry["total_value"] * 100) if entry["total_value"] > 0 else 0
    
    prompt = f"""Write a 50-word trading lesson from this trade:

Action: {entry['action']} {entry['shares']} shares of {entry['ticker']}
Entry: ${entry['price']:.2f} | Exit: ${entry.get('close_price', 0):.2f}
Result: ${pnl:.2f} ({pnl_pct:+.1f}%) — {outcome}
Reasoning at entry: {entry['reasoning'][:300]}
Market context at entry: {entry['market_snapshot']}

Write exactly one lesson in 50 words or fewer. Be specific and actionable. 
Do not start with "Lesson:" — just state the insight directly."""

    try:
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
            return f"{'Win' if pnl >= 0 else 'Loss'}: ${pnl:.2f} on {entry['ticker']}. No API key for lesson generation."
        
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        lesson = response.content[0].text.strip()
        
        # Truncate to ~50 words if Haiku was verbose
        words = lesson.split()
        if len(words) > 60:
            lesson = " ".join(words[:50]) + "..."
        
        return lesson
    
    except Exception as e:
        logger.warning(f"Lesson generation failed: {e}")
        return f"{'Win' if pnl >= 0 else 'Loss'}: ${pnl:.2f}. {entry['reasoning'][:100]}"


def get_recent_entries(n: int = 10) -> list[dict]:
    """Get the last N journal entries for inclusion in the trading brief.
    
    Args:
        n: Number of entries to return.
    
    Returns:
        Last N entries, newest-first.
    """
    entries = load_journal()
    return list(reversed(entries[-n:]))


def format_journal_for_brief(entries: list[dict]) -> str:
    """Format journal entries as text for the Claude trading brief.
    
    Args:
        entries: List of journal entry dicts (newest-first).
    
    Returns:
        Formatted text block for inclusion in the system prompt.
    """
    if not entries:
        return "No previous trades recorded."
    
    lines = []
    for e in entries:
        pnl_str = ""
        lesson_str = ""
        if e.get("realized_pnl") is not None:
            pnl_str = f" → ${e['realized_pnl']:+.2f}"
            if e.get("lesson"):
                lesson_str = f"\n    LESSON: {e['lesson']}"
        
        lines.append(
            f"  [{e['timestamp'][:16]}] {e['action']} {e['shares']:.2f} {e['ticker']} "
            f"@ ${e['price']:.2f} (conf={e['confidence']:.0%}){pnl_str}"
            f"{lesson_str}"
        )
    
    return "\n".join(lines)


def get_journal_stats() -> dict:
    """Compute summary statistics from the journal.
    
    Returns:
        {
            "total_trades": int,
            "closed_trades": int,
            "open_trades": int,
            "total_pnl": float,
            "wins": int,
            "losses": int,
            "win_rate": float,
            "avg_win": float,
            "avg_loss": float,
            "biggest_win": float,
            "biggest_loss": float,
            "lessons": list[str],  # all lessons
        }
    """
    entries = load_journal()
    
    closed = [e for e in entries if e.get("realized_pnl") is not None]
    wins = [e for e in closed if e["realized_pnl"] >= 0]
    losses = [e for e in closed if e["realized_pnl"] < 0]
    
    return {
        "total_trades": len(entries),
        "closed_trades": len(closed),
        "open_trades": len(entries) - len(closed),
        "total_pnl": round(sum(e["realized_pnl"] for e in closed), 2) if closed else 0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "avg_win": round(sum(e["realized_pnl"] for e in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(e["realized_pnl"] for e in losses) / len(losses), 2) if losses else 0,
        "biggest_win": max((e["realized_pnl"] for e in wins), default=0),
        "biggest_loss": min((e["realized_pnl"] for e in losses), default=0),
        "lessons": [e["lesson"] for e in closed if e.get("lesson")],
    }
```

**Trade journal entry format** (as stored in `data/trade_journal.json`):
```json
[
  {
    "trade_id": "txn_042",
    "timestamp": "2026-03-10T14:30:00+00:00",
    "action": "BUY",
    "ticker": "TSLA",
    "shares": 0.85,
    "price": 285.50,
    "total_value": 242.68,
    "reasoning": "TSLA oversold after broad market selloff, RSI at 28, VIX elevated but declining...",
    "confidence": 0.72,
    "portfolio_value": 957.32,
    "market_snapshot": "TSLA $285.50 (-3.2%), VIX 24.5, SPY -1.1%, 10Y 4.32%",
    "lesson": null,
    "close_trade_id": null,
    "close_price": null,
    "realized_pnl": null,
    "closed_at": null
  },
  {
    "trade_id": "txn_043",
    "timestamp": "2026-03-11T10:45:00+00:00",
    "action": "SELL",
    "ticker": "TSLA",
    "shares": 0.85,
    "price": 294.20,
    "total_value": 250.07,
    "reasoning": "Taking profit on overnight gap up, RSI now 61, approaching resistance at 295...",
    "confidence": 0.65,
    "portfolio_value": 964.73,
    "market_snapshot": "TSLA $294.20 (+2.8%), VIX 21.3, SPY +0.8%, 10Y 4.28%",
    "lesson": "Buying oversold TSLA during broad selloffs works when VIX is declining — the fear is subsiding. Key was RSI under 30 with falling VIX, not just the price drop. Wait for VIX direction, not just level.",
    "close_trade_id": "txn_043",
    "close_price": 294.20,
    "realized_pnl": 7.40,
    "closed_at": "2026-03-11T10:45:00+00:00"
  }
]
```

### 6. `src/portfolio.py` -- MODIFY

Changes:
- **Remove:** `check_stop_losses` (Claude decides whether to exit underwater positions), `calculate_position_size` (Claude decides sizing), `apply_gap_risk_reduction` (removed), `check_daily_loss_limit` (removed), `check_end_of_day_close` (removed -- Claude decides if it wants to close before EOD).
- **Simplify:** `validate_trade` -- reduce to two rules: (1) max 50% portfolio in position, (2) keep $100 cash minimum. Remove max_single_trade_pct, min_cash_reserve_pct (now a flat $100).
- **Keep:** `load_portfolio`, `save_portfolio`, `load_transactions`, `save_transactions`, `update_market_price`, `execute_trade`, `get_portfolio_summary`, `save_snapshot`, `_default_portfolio`, `check_cooldown` (simplified -- flat 15min, no ATR scaling).

**Modified `validate_trade`:**
```python
def validate_trade(action: str, shares: float, price: float, portfolio: dict, config: dict = None) -> tuple[bool, str]:
    """Check if a trade passes v4 risk rules. Returns (ok, reason).
    
    v4 rules (simple):
    - Max 50% of portfolio value in any position
    - Keep $100 cash minimum
    - Shares must be positive
    """
    if config is None:
        config = load_config()
    risk = config["risk_params"]
    ticker = config["ticker"]

    if shares <= 0:
        return False, "Shares must be positive"

    if not risk.get("enable_fractional_shares", True) and shares != int(shares):
        return False, "Fractional shares disabled"

    total_cost = shares * price

    if action == "BUY":
        if total_cost > portfolio["cash"]:
            return False, f"Insufficient cash: need ${total_cost:.2f}, have ${portfolio['cash']:.2f}"

        # Max position size: 50% of portfolio
        max_position_pct = risk.get("max_position_pct", 0.50)
        current_holding_value = portfolio["holdings"].get(ticker, {}).get("shares", 0) * price
        new_position_value = current_holding_value + total_cost
        max_position = portfolio["total_value"] * max_position_pct
        if new_position_value > max_position:
            return False, f"Position would be ${new_position_value:.2f}, exceeds max ${max_position:.2f} ({max_position_pct*100:.0f}%)"

        # Cash reserve: keep $100 minimum
        min_cash = risk.get("min_cash_reserve", 100.00)
        cash_after = portfolio["cash"] - total_cost
        if cash_after < min_cash:
            return False, f"Would leave ${cash_after:.2f} cash, below minimum ${min_cash:.2f}"

    elif action == "SELL":
        held = portfolio["holdings"].get(ticker, {}).get("shares", 0)
        if shares > held:
            return False, f"Can't sell {shares:.4f} shares, only hold {held:.4f}"
    else:
        return False, f"Unknown action: {action}"

    return True, "OK"
```

**Simplified `check_cooldown`:**
```python
def check_cooldown(ticker: str) -> bool:
    """Returns True if cooldown period has passed since last trade.
    
    v4: flat 15-minute cooldown, no ATR scaling.
    """
    config = load_config()
    cooldown = config["risk_params"].get("cooldown_minutes", 15)

    transactions = load_transactions()
    ticker_trades = [t for t in transactions if t["ticker"] == ticker]
    if not ticker_trades:
        return True

    last_trade_time = datetime.fromisoformat(ticker_trades[-1]["timestamp"])
    now = datetime.now(timezone.utc)
    minutes_since = (now - last_trade_time).total_seconds() / 60

    return minutes_since >= cooldown
```

**Modified `execute_trade`:** Simplify the `decision` dict expectations. Remove `trade_context` parameter. Remove `constraints_applied`. Keep slippage (it is realistic). Remove slippage VIX scaling (use flat rate).

```python
def execute_trade(action: str, shares: float, price: float, decision: dict = None) -> dict:
    """Execute a simulated trade. Returns the transaction record.
    
    v4: simplified. No trade_context, no constraints. Claude decided, we execute.
    """
    config = load_config()
    ticker = config["ticker"]
    portfolio = load_portfolio()
    transactions = load_transactions()

    # Apply flat slippage
    risk = config["risk_params"]
    slippage = risk.get("slippage_per_side_pct", 0.0005)
    if action == "BUY":
        exec_price = round(price * (1 + slippage), 2)
    else:
        exec_price = round(price * (1 - slippage), 2)

    # Validate
    ok, reason = validate_trade(action, shares, exec_price, portfolio)
    if not ok:
        logger.warning(f"Trade rejected: {reason}")
        return {"status": "rejected", "reason": reason}

    total_cost = round(shares * exec_price, 2)
    h = portfolio["holdings"].setdefault(ticker, {
        "shares": 0.0, "avg_cost_basis": 0.0,
        "current_price": 0.0, "unrealized_pnl": 0.0,
    })

    realized_pnl = 0.0

    if action == "BUY":
        old_total = h["shares"] * h["avg_cost_basis"]
        new_total = old_total + total_cost
        h["shares"] = round(h["shares"] + shares, 6)
        h["avg_cost_basis"] = round(new_total / h["shares"], 2) if h["shares"] > 0 else 0.0
        portfolio["cash"] = round(portfolio["cash"] - total_cost, 2)
    elif action == "SELL":
        realized_pnl = round((exec_price - h["avg_cost_basis"]) * shares, 2)
        h["shares"] = round(h["shares"] - shares, 6)
        if h["shares"] < 0.0001:
            h["shares"] = 0.0
            h["avg_cost_basis"] = 0.0
        portfolio["cash"] = round(portfolio["cash"] + total_cost, 2)

    portfolio = update_market_price(portfolio, ticker, exec_price)
    portfolio["total_trades"] += 1
    if action == "SELL":
        if realized_pnl >= 0:
            portfolio["winning_trades"] += 1
        else:
            portfolio["losing_trades"] += 1

    # Build transaction record (simplified for v4)
    txn_id = generate_id("txn", [t["id"] for t in transactions])
    txn = {
        "id": txn_id,
        "timestamp": iso_now(),
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "price": exec_price,
        "total_cost": total_cost,
        "realized_pnl": realized_pnl if action == "SELL" else None,
        "cash_after": portfolio["cash"],
        "portfolio_value_after": portfolio["total_value"],
        "confidence": decision.get("confidence", 0) if decision else 0,
        "reasoning": decision.get("reasoning", "") if decision else "",
    }

    transactions.append(txn)
    save_transactions(transactions)
    save_portfolio(portfolio)

    logger.info(
        f"{action} {shares:.4f} {ticker} @ ${exec_price:.2f} "
        f"(total: ${total_cost:.2f}, cash: ${portfolio['cash']:.2f}, "
        f"value: ${portfolio['total_value']:.2f})"
    )

    return {"status": "executed", "transaction": txn, "portfolio": portfolio}
```

### 7. `src/agent.py` -- REPLACE

The entire file is replaced. This is the heart of v4.

```python
"""AI decision engine — Claude IS the trading brain.

v4: No coded strategies. No signal aggregation. No thesis system.
Raw market data + world context + trade journal → Claude reasons → Claude decides.
"""

import json
import os
from anthropic import Anthropic
from .utils import (
    load_config, format_currency, iso_now, setup_logging,
    call_ai_with_fallback, _log_api_usage,
)
from .journal import get_recent_entries, format_journal_for_brief
from .news_feed import NewsFeed, format_news_for_prompt
from .web_search import search_tsla_news, format_search_results

logger = setup_logging("agent")


SYSTEM_PROMPT = """You are MonopolyTrader v4, an AI trader managing $1,000 of Monopoly dollars on TSLA. This is paper money — your job is to LEARN by trading aggressively. Every scenario is an opportunity. No fear.

You receive a market brief every 15 minutes during market hours. You see raw indicators, world context, news, web search results, and your own trade journal with lessons from past trades.

YOU decide what matters. YOU interpret the indicators. YOU size the position. There are no coded strategies telling you what to think.

RULES (only 3):
1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK

NO STOP LOSSES. If your position is underwater, you evaluate it with context every cycle:
- Why is it falling? Broad market or TSLA-specific?
- Is the thesis that led to the buy still valid?
- Should you buy MORE at this lower price, or exit?
You are never forced out. You decide.

EVERY SCENARIO IS AN OPPORTUNITY:
- Stock dropping? Maybe it's oversold — buy the dip
- Stock flat? Maybe a breakout is coming — position early
- Stock ripping? Maybe momentum continues — ride it
- Holding cash? Fine, but only if you genuinely see no setup
- Underwater position? Evaluate with fresh eyes — average down or cut

Paper money means you can be WRONG and learn from it. A trade that loses $30 but teaches you something is worth more than sitting in cash for a week learning nothing.

TRADE JOURNAL: Your last 10 trades and their lessons appear below. These are YOUR lessons from YOUR experience. Use them. If you lost money buying into high RSI, don't do it again. If you made money buying VIX spikes, look for that setup.

Respond ONLY with valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "reasoning": "<your complete analysis: what you see in the data, why you're acting or not acting, what you expect to happen>",
  "risk_note": "<what could go wrong with this trade>"
}"""


def build_market_brief(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    web_results: list[dict],
    journal_entries: list[dict],
    config: dict,
) -> str:
    """Build the complete market brief for Claude.
    
    This is the single prompt that contains everything Claude needs
    to make a trading decision.
    
    Args:
        market_data: From get_market_summary() — price, indicators, recent days.
        world: From get_world_snapshot() — macro + EV peers.
        portfolio: Current portfolio state.
        news_feed: NewsFeed object from fetch_news_feed().
        web_results: From brave_search().
        journal_entries: Recent journal entries (newest-first).
        config: Config dict.
    
    Returns:
        Formatted text brief for the user message to Claude.
    """
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    intraday = market_data.get("intraday_indicators") or {}
    recent = market_data.get("recent_days", [])
    regime = market_data.get("regime", {})
    ticker = config["ticker"]
    
    parts = []
    
    # === TSLA Price & Indicators ===
    parts.append(f"=== {ticker} ===")
    parts.append(f"Price: ${current.get('price', 'N/A')} ({current.get('change_pct', 0):+.2f}%)")
    parts.append(f"Volume: {current.get('volume', 0):,}")
    parts.append("")
    
    parts.append("Daily Indicators:")
    indicator_keys = [
        "rsi_14", "sma_20", "sma_50",
        "macd", "macd_signal", "macd_histogram", "macd_crossover",
        "bollinger_upper", "bollinger_lower", "bollinger_mid",
        "atr", "adx", "adx_pos", "adx_neg",
        "ema_12", "ema_26", "volume_sma_20", "obv",
    ]
    for key in indicator_keys:
        val = daily.get(key)
        if val is not None:
            parts.append(f"  {key}: {val}")
    
    if intraday and intraday.get("current_price"):
        parts.append("")
        parts.append("Intraday (5min) Indicators:")
        for key in ["rsi_14", "sma_20", "macd_crossover", "bollinger_upper", "bollinger_lower"]:
            val = intraday.get(key)
            if val is not None:
                parts.append(f"  {key}: {val}")
    
    if recent:
        parts.append("")
        parts.append("Last 5 Days:")
        for day in recent:
            parts.append(
                f"  {day['date']}: O={day['open']} H={day['high']} "
                f"L={day['low']} C={day['close']} V={day['volume']:,}"
            )
    
    # === World Context ===
    parts.append("")
    parts.append("=== WORLD ===")
    
    macro = world.get("macro", {})
    if macro:
        parts.append("Macro:")
        for sym, data in macro.items():
            name = data.get("name", sym)
            parts.append(f"  {name} ({sym}): {data['price']} ({data['change_pct']:+.2f}%)")
    
    peers = world.get("ev_peers", {})
    if peers:
        parts.append("EV Peers:")
        for sym, data in peers.items():
            parts.append(f"  {sym}: ${data['price']} ({data['change_pct']:+.2f}%)")
    
    # Regime (from market_data)
    if regime:
        parts.append("")
        parts.append(f"Regime: trend={regime.get('trend','?')} directional={regime.get('directional','?')} "
                     f"volatility={regime.get('volatility','?')} VIX={regime.get('vix',0):.1f} "
                     f"ADX={regime.get('adx',0):.1f}")
    
    # === News ===
    parts.append("")
    parts.append("=== NEWS ===")
    if news_feed and news_feed.items:
        parts.append(format_news_for_prompt(news_feed, max_items=10))
    else:
        parts.append("No news available.")
    
    # === Web Search ===
    if web_results:
        parts.append("")
        parts.append("=== WEB SEARCH (last 24h) ===")
        parts.append(format_search_results(web_results))
    
    # === Portfolio ===
    parts.append("")
    parts.append("=== PORTFOLIO ===")
    parts.append(f"Cash: {format_currency(portfolio.get('cash', 0))}")
    parts.append(f"Total Value: {format_currency(portfolio.get('total_value', 0))}")
    parts.append(f"P&L: {format_currency(portfolio.get('total_pnl', 0))} ({portfolio.get('total_pnl_pct', 0):+.2f}%)")
    parts.append(f"Trades: {portfolio.get('total_trades', 0)} (W:{portfolio.get('winning_trades', 0)} L:{portfolio.get('losing_trades', 0)})")
    
    holdings = portfolio.get("holdings", {})
    h = holdings.get(ticker, {})
    if h.get("shares", 0) > 0:
        parts.append(
            f"Position: {h['shares']:.4f} {ticker} @ avg ${h['avg_cost_basis']:.2f} "
            f"(current ${h['current_price']:.2f}, unrealized {format_currency(h['unrealized_pnl'])})"
        )
        # Explicitly tell Claude what the position looks like
        pnl_pct = ((h['current_price'] - h['avg_cost_basis']) / h['avg_cost_basis'] * 100) if h['avg_cost_basis'] > 0 else 0
        if pnl_pct < 0:
            parts.append(f"  >>> UNDERWATER: {pnl_pct:.1f}% loss. Evaluate: exit, hold, or buy more? <<<")
    else:
        parts.append("Position: FLAT (all cash)")
    
    # Position limits for Claude
    max_position = portfolio.get("total_value", 1000) * config["risk_params"].get("max_position_pct", 0.50)
    max_buy_value = min(max_position, portfolio.get("cash", 0) - config["risk_params"].get("min_cash_reserve", 100))
    if max_buy_value > 0 and current.get("price", 0) > 0:
        max_shares = max_buy_value / current["price"]
        parts.append(f"Max BUY: {max_shares:.4f} shares (${max_buy_value:.2f})")
    elif h.get("shares", 0) > 0:
        parts.append(f"Can SELL up to {h['shares']:.4f} shares")
    
    # === Trade Journal ===
    parts.append("")
    parts.append("=== YOUR TRADE JOURNAL (last 10) ===")
    parts.append(format_journal_for_brief(journal_entries))
    
    return "\n".join(parts)


def make_decision(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    web_results: list[dict],
    journal_entries: list[dict],
    config: dict,
) -> dict:
    """Call Claude to make a trading decision.
    
    This is the single Claude call per cycle. No multi-step, no analyst/trader
    split. One brief, one decision.
    
    Args:
        market_data: From get_market_summary().
        world: From get_world_snapshot().
        portfolio: Current portfolio state.
        news_feed: NewsFeed object.
        web_results: Brave search results.
        journal_entries: Recent journal entries.
        config: Config dict.
    
    Returns:
        Parsed decision dict: {action, shares, confidence, reasoning, risk_note}
    """
    brief = build_market_brief(
        market_data, world, portfolio, news_feed,
        web_results, journal_entries, config,
    )
    
    logger.info(f"Calling Claude for decision (brief ~{len(brief)} chars)")
    
    try:
        raw, model_used = call_ai_with_fallback(
            system=SYSTEM_PROMPT,
            user=brief,
            max_tokens=800,
            config=config,
        )
        
        # Strip code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        
        decision = json.loads(raw)
        
        # Validate required fields
        action = decision.get("action", "HOLD")
        if action not in ("BUY", "SELL", "HOLD"):
            logger.warning(f"Invalid action '{action}', defaulting to HOLD")
            decision["action"] = "HOLD"
            decision["shares"] = 0
        
        if action == "HOLD":
            decision["shares"] = 0
        
        decision["_model_version"] = model_used
        
        logger.info(
            f"Decision: {decision.get('action')} "
            f"{decision.get('shares', 0):.4f} shares, "
            f"confidence={decision.get('confidence', 0):.2f}"
        )
        
        return decision
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response: {e}\nRaw: {raw[:500]}")
        return {
            "action": "HOLD", "shares": 0, "confidence": 0,
            "reasoning": f"JSON parse error: {e}",
            "risk_note": "Defaulting to HOLD due to parse error",
        }
    except Exception as e:
        logger.error(f"Agent decision failed: {e}")
        return {
            "action": "HOLD", "shares": 0, "confidence": 0,
            "reasoning": f"Error: {e}",
            "risk_note": f"Agent error: {e}",
        }
```

### 8. `src/main.py` -- REPLACE

```python
"""MonopolyTrader v4 — main entry point.

Every 15 minutes during market hours:
  1. Gather data (TSLA + world + news + web search)
  2. Build brief (raw data + portfolio + journal)
  3. Claude decides (BUY / SELL / HOLD)
  4. Execute trade if any
  5. Record in journal
  6. Update dashboard
"""

import argparse
import signal
import sys
import time
import schedule

from .utils import (
    load_config, is_market_open, now_et, iso_now,
    format_currency, setup_logging, DATA_DIR, save_json,
)
from .market_data import get_market_summary, get_world_snapshot
from .portfolio import (
    load_portfolio, execute_trade, save_portfolio, save_snapshot,
    update_market_price, get_portfolio_summary, check_cooldown,
)
from .agent import make_decision
from .news_feed import fetch_news_feed
from .web_search import search_tsla_news
from .journal import (
    add_entry as journal_add_entry,
    close_entry as journal_close_entry,
    get_recent_entries,
    load_journal,
)

logger = setup_logging("main")

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_cycle():
    """The v4 trading cycle. Runs every 15 minutes during market hours."""
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    try:
        logger.info("--- v4 Decision Cycle ---")

        # 1. Gather data
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        regime = market_data.get("regime", {})
        vix = regime.get("vix", 0)

        logger.info(
            f"{ticker}: ${current_price} "
            f"({market_data['current'].get('change_pct', 0):+.2f}%) "
            f"| VIX: {vix:.1f}"
        )

        # World snapshot (macro + EV peers)
        world = {}
        try:
            world = get_world_snapshot(config)
        except Exception as e:
            logger.warning(f"World snapshot failed: {e}")

        # News feed
        news_feed = None
        try:
            news_feed = fetch_news_feed(ticker)
        except Exception as e:
            logger.warning(f"News feed failed: {e}")

        # Web search
        web_results = []
        try:
            if config.get("brave_search", {}).get("enabled", False):
                web_results = search_tsla_news()
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        # 2. Update portfolio with current price
        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # 3. Get journal entries
        journal_entries = get_recent_entries(
            config.get("journal", {}).get("max_entries_in_brief", 10)
        )

        # 4. Check cooldown
        if not check_cooldown(ticker):
            logger.info("Cooldown active — skipping Claude call")
            _update_dashboard(market_data, portfolio)
            return

        # 5. Claude decides
        decision = make_decision(
            market_data=market_data,
            world=world,
            portfolio=portfolio,
            news_feed=news_feed,
            web_results=web_results,
            journal_entries=journal_entries,
            config=config,
        )

        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        # 6. Execute
        if action in ("BUY", "SELL") and shares > 0:
            result = execute_trade(action, shares, current_price, decision)

            if result["status"] == "executed":
                txn = result["transaction"]
                logger.info(
                    f"EXECUTED: {action} {shares:.4f} @ ${txn['price']:.2f} "
                    f"| Cash: {format_currency(result['portfolio']['cash'])} "
                    f"| Value: {format_currency(result['portfolio']['total_value'])}"
                )

                # Build market snapshot for journal
                market_snapshot = (
                    f"{ticker} ${current_price} "
                    f"({market_data['current'].get('change_pct', 0):+.2f}%), "
                    f"VIX {vix:.1f}"
                )
                spy_data = world.get("macro", {}).get("SPY", {})
                if spy_data:
                    market_snapshot += f", SPY {spy_data.get('change_pct', 0):+.2f}%"

                # Record in journal
                journal_add_entry(
                    trade_id=txn["id"],
                    action=action,
                    ticker=ticker,
                    shares=txn["shares"],
                    price=txn["price"],
                    reasoning=decision.get("reasoning", ""),
                    confidence=decision.get("confidence", 0),
                    portfolio_value=result["portfolio"]["total_value"],
                    market_snapshot=market_snapshot,
                )

                # If this was a SELL, close the corresponding BUY journal entry
                if action == "SELL" and txn.get("realized_pnl") is not None:
                    _close_journal_for_sell(txn)

            elif result["status"] == "rejected":
                logger.warning(f"Trade rejected: {result['reason']}")
        else:
            logger.info(f"HOLD — {decision.get('reasoning', 'N/A')[:120]}")

        # 7. Save latest cycle for dashboard
        _save_latest_cycle(decision, market_data, regime, world)

        # 8. Update dashboard
        _update_dashboard(market_data, portfolio)

    except Exception as e:
        logger.error(f"v4 decision cycle error: {e}", exc_info=True)


def _close_journal_for_sell(sell_txn: dict):
    """Find the most recent open BUY journal entry and close it."""
    journal = load_journal()
    # Find the most recent unclosed BUY entry for this ticker
    for entry in reversed(journal):
        if (entry["action"] == "BUY" 
            and entry["ticker"] == sell_txn["ticker"]
            and entry.get("lesson") is None):
            journal_close_entry(
                open_trade_id=entry["trade_id"],
                close_trade_id=sell_txn["id"],
                close_price=sell_txn["price"],
                realized_pnl=sell_txn["realized_pnl"],
            )
            return
    logger.warning(f"No open BUY journal entry found to close for {sell_txn['id']}")


def _save_latest_cycle(decision: dict, market_data: dict, regime: dict, world: dict):
    """Save the latest cycle data for the dashboard Agent's Mind card."""
    cycle_data = {
        "timestamp": iso_now(),
        "action": decision.get("action", "HOLD"),
        "shares": decision.get("shares", 0),
        "confidence": decision.get("confidence", 0),
        "reasoning": decision.get("reasoning", ""),
        "risk_note": decision.get("risk_note", ""),
        "price": market_data.get("current", {}).get("price", 0),
        "regime": regime,
        "vix": regime.get("vix", 0),
    }
    save_json(DATA_DIR / "latest_cycle.json", cycle_data)


def _update_dashboard(market_data: dict, portfolio: dict):
    """Refresh dashboard data."""
    try:
        from .reporter import generate_dashboard_data
        generate_dashboard_data()
    except Exception as e:
        logger.warning(f"Dashboard update failed: {e}")


def run_daily_tasks():
    """Run after market close: snapshot."""
    try:
        save_snapshot()
    except Exception as e:
        logger.warning(f"Daily tasks failed: {e}")

    try:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="MonopolyTrader v4")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--report", action="store_true", help="Generate dashboard and exit")
    args = parser.parse_args()

    config = load_config()
    logger.info("MonopolyTrader v4 starting")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "snapshots").mkdir(parents=True, exist_ok=True)

    if args.report:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
        logger.info("Dashboard generated")
        return

    if args.once:
        run_cycle()
        return

    # Schedule
    interval = config.get("poll_interval_minutes", 15)
    schedule.every(interval).minutes.do(run_cycle)
    schedule.every().day.at("16:15").do(run_daily_tasks)

    logger.info(f"Scheduler started: every {interval}min during market hours")
    run_cycle()  # Run immediately

    while _running:
        schedule.run_pending()
        time.sleep(10)

    logger.info("MonopolyTrader v4 stopped")


if __name__ == "__main__":
    main()
```

### 9. `src/news_feed.py` -- KEEP AS-IS

No changes needed. It already does yfinance + RSS news fetching, classification, scoring, and dedup. Works perfectly for v4.

### 10. `src/reporter.py` -- MODIFY

Major simplification. Remove all references to:
- `knowledge_base` (lessons, patterns, predictions, strategy scores, journal.md, tsla_profile)
- `strategies` (weight history, rebalance history)
- `thesis` module
- `trade_stats` (constraints, stats)
- `benchmarks` comparison (keep if module exists, but make it optional)
- `ensemble` data
- `observability` health data (keep if module exists)
- Prediction accuracy tracking
- Hold log / counterfactual stats

**Add:** Trade journal display in dashboard data.

The `generate_dashboard_data` function should produce:
```python
data = {
    "generated_at": iso_now(),
    "ticker": ticker,
    "current_price": current,
    "latest_cycle": latest_cycle,  # from data/latest_cycle.json
    "portfolio": summary,
    "transactions": transactions,
    "snapshots": snapshots,
    "trade_journal": journal_entries,  # from journal.py
    "journal_stats": journal_stats,   # from journal.get_journal_stats()
    "market_open": is_market_open(),
    "time_et": now_et().strftime("%Y-%m-%d %H:%M ET"),
}
```

Import from `journal` module instead of `knowledge_base`.

### 11. `dashboard/index.html` -- MODIFY

Changes:
- Update title to "MonopolyTrader v4 Dashboard"
- Remove: benchmark comparison card, graduation progress, strategy weight section, weight evolution chart, prediction scoreboard, knowledge base tabs (lessons/patterns/journal), hold log, ensemble leaderboard, accuracy trend chart, random dist chart, drawdown chart, Sharpe chart.
- Add: Trade Journal section showing recent journal entries with lessons.
- Keep: Portfolio value chart (without benchmark lines), KPI strip, trade log, Agent's Mind card, status banner.

The simplified dashboard has:
1. Header with v4 tag
2. Agent's Mind card (latest decision)
3. KPI strip (value, P&L, cash, trades, win rate)
4. Portfolio value chart (simple line, no benchmarks)
5. Trade Journal (newest first, showing entry/exit/lesson)
6. Trade Log table

### 12. `dashboard/app.js` -- MODIFY

Simplify to render only the v4 data structure. Remove all benchmark rendering, graduation checklist, strategy bars, weight evolution chart, prediction scoreboard, knowledge base browser, hold log, ensemble leaderboard, performance analytics charts.

Add `renderTradeJournal()` function:
```javascript
function renderTradeJournal() {
    const entries = DATA.trade_journal || [];
    const container = document.getElementById('journal-entries');
    if (!entries.length) {
        container.innerHTML = '<p class="empty">No trades recorded yet.</p>';
        return;
    }
    
    let html = '';
    for (const e of entries) {
        const pnlClass = e.realized_pnl > 0 ? 'positive' : e.realized_pnl < 0 ? 'negative' : '';
        const pnlStr = e.realized_pnl !== null 
            ? `<span class="${pnlClass}">${fmt(e.realized_pnl)}</span>` 
            : '<span class="pending">open</span>';
        
        html += `<div class="journal-entry ${e.action.toLowerCase()}">
            <div class="journal-header">
                <span class="journal-action ${e.action.toLowerCase()}">${e.action}</span>
                <span>${e.shares.toFixed(2)} ${e.ticker} @ ${fmt(e.price)}</span>
                <span class="journal-time">${timeAgo(e.timestamp)}</span>
                <span class="journal-pnl">${pnlStr}</span>
            </div>
            <div class="journal-reasoning">${e.reasoning.substring(0, 200)}${e.reasoning.length > 200 ? '...' : ''}</div>
            ${e.lesson ? `<div class="journal-lesson">LESSON: ${e.lesson}</div>` : ''}
        </div>`;
    }
    container.innerHTML = html;
}
```

---

## v3-to-v4 Data Transition

### What to keep:
- `data/portfolio.json` -- Keep as-is. The portfolio state carries over. Cash balance, holdings, total trades, winning/losing counts.
- `data/transactions.json` -- Keep as-is. Historical trades remain viewable in the trade log.
- `data/snapshots/` -- Keep as-is. Portfolio value chart continues from existing data.

### What to create:
- `data/trade_journal.json` -- Starts empty (`[]`). New trades from v4 onward get journal entries.

### What becomes unused (leave on disk, don't delete):
- `knowledge/` directory and all its contents
- `data/strategy_scores.json`
- `data/hold_log.json`
- `data/latest_cycle.json` (will be overwritten by v4 format)
- `data/benchmarks.json`
- `data/seen_news.json`
- `agents/` directory

### Server transition:
The server currently runs as `monopoly-trader` systemd service. After deploying v4 code:
1. Stop service
2. Deploy new code (git pull)
3. `pip install httpx` (for Brave Search)
4. Set `BRAVE_API_KEY` environment variable in systemd unit file
5. Start service

No data migration script needed. The portfolio carries over seamlessly.

---

## Dependency Changes

**Add to `requirements.txt`:**
```
httpx>=0.27.0
```

**Remove dependency on (but these stay installed, just unused):**
- `feedparser` (still used by news_feed.py RSS -- keep)
- All others remain: `yfinance`, `ta`, `anthropic`, `schedule`, `pandas`, `numpy`

---

## Cost Estimate

Per 15-minute cycle:
- 1 Claude Sonnet call (~2000 token input brief, ~300 token output) = ~$0.0105
- Occasional Haiku lesson call on trade close (~500 input, ~80 output) = ~$0.0007
- ~26 cycles per trading day = ~$0.27/day for Sonnet
- Brave Search: ~26 calls/day = free tier (2000/month, ~520/month at 26/day)

Monthly estimate: ~$5.70 Sonnet + ~$0.50 Haiku = ~$6.20/month. Well within budget.

---

## Summary: What Gets Simpler

| v3 Component | Lines of code | v4 Replacement | Lines |
|---|---|---|---|
| strategies.py | 634 | DELETED | 0 |
| thesis.py | 185 | DELETED | 0 |
| trade_stats.py | 522 | DELETED | 0 |
| learner.py | ~1000 | journal.py | ~250 |
| knowledge_base.py | 620 | DELETED | 0 |
| researcher.py | 380 | DELETED | 0 |
| agent.py (v3) | ~1050 | agent.py (v4) | ~250 |
| main.py (v3) | ~800 | main.py (v4) | ~250 |
| NEW: web_search.py | - | web_search.py | ~100 |
| **Total removed** | **~5200** | **Total new** | **~850** |

The codebase shrinks by roughly 4,000 lines. Claude gets raw data and makes decisions. The trade journal is the only learning mechanism. Three rules instead of twenty.

### Critical Files for Implementation
- `C:\Users\chris\Downloads\monopoly-trader\src\agent.py` - Complete replacement: the new Claude-as-brain architecture with SYSTEM_PROMPT and build_market_brief
- `C:\Users\chris\Downloads\monopoly-trader\src\main.py` - Complete replacement: simplified 15-min cycle with no triggers/gates/constraints
- `C:\Users\chris\Downloads\monopoly-trader\src\journal.py` - New file: trade journal with Haiku lesson generation, replaces entire learning stack
- `C:\Users\chris\Downloads\monopoly-trader\src\portfolio.py` - Modify: simplify validate_trade to 2 rules, remove stop losses/ATR sizing/EOD close
- `C:\Users\chris\Downloads\monopoly-trader\src\market_data.py` - Modify: add get_world_snapshot(), remove macro gates and trigger checks