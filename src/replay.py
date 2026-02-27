"""Historical Replay Mode — practice trading on past data to accelerate learning."""

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .backtest import WalkForwardBacktest
from .knowledge_base import get_relevant_knowledge, get_strategy_scores
from .learner import review_trade
from .portfolio import validate_trade
from .strategies import evaluate_all_strategies, aggregate_signals
from .agent import make_decision
from .utils import (
    load_config, save_json, load_json, iso_now, generate_id,
    setup_logging, DATA_DIR, call_ai_with_fallback,
)

logger = setup_logging("replay")

# ---------------------------------------------------------------------------
# Targeted practice scenarios
# ---------------------------------------------------------------------------
SCENARIOS = {
    "crash_2022":       {"start": "2022-01-01", "end": "2022-06-30", "label": "2022 Bear Market"},
    "rally_2023":       {"start": "2023-01-01", "end": "2023-07-31", "label": "2023 Recovery Rally"},
    "volatile_2022_q4": {"start": "2022-10-01", "end": "2022-12-31", "label": "2022 Q4 Volatility"},
    "sideways_2023_h2": {"start": "2023-07-01", "end": "2023-12-31", "label": "2023 H2 Sideways"},
    "breakout_2024":    {"start": "2024-01-01", "end": "2024-06-30", "label": "2024 Breakout"},
}

# How many bars after a trade before we review it
REVIEW_DELAY_BARS = 5
# How often to save a checkpoint (bars)
CHECKPOINT_INTERVAL = 25


# ---------------------------------------------------------------------------
# ReplayPortfolio — isolated portfolio that mirrors live execution logic
# ---------------------------------------------------------------------------
class ReplayPortfolio:
    """Isolated portfolio for replay sessions. Writes to data/replay/{session_id}/."""

    def __init__(self, session_id: str, starting_balance: float = 1000.0):
        self.session_id = session_id
        self.dir = DATA_DIR / "replay" / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.portfolio = {
            "cash": starting_balance,
            "holdings": {
                "TSLA": {
                    "shares": 0.0,
                    "avg_cost_basis": 0.0,
                    "current_price": 0.0,
                    "unrealized_pnl": 0.0,
                }
            },
            "total_value": starting_balance,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "created_at": iso_now(),
            "last_updated": iso_now(),
        }
        self.transactions: list[dict] = []
        self.starting_balance = starting_balance

    def update_price(self, price: float):
        """Update holdings with current bar price."""
        h = self.portfolio["holdings"]["TSLA"]
        h["current_price"] = round(price, 2)
        if h["shares"] > 0:
            h["unrealized_pnl"] = round((price - h["avg_cost_basis"]) * h["shares"], 2)
        else:
            h["unrealized_pnl"] = 0.0

        holdings_value = sum(
            hld["shares"] * hld["current_price"]
            for hld in self.portfolio["holdings"].values()
        )
        self.portfolio["total_value"] = round(self.portfolio["cash"] + holdings_value, 2)
        self.portfolio["total_pnl"] = round(
            self.portfolio["total_value"] - self.starting_balance, 2
        )
        pct = (self.portfolio["total_pnl"] / self.starting_balance) * 100 if self.starting_balance else 0
        self.portfolio["total_pnl_pct"] = round(pct, 2)
        self.portfolio["last_updated"] = iso_now()

    def execute_trade(self, action: str, shares: float, price: float,
                      decision: dict, bar_index: int) -> dict | None:
        """Execute a replay trade using the same validation as live."""
        config = load_config()

        # Apply slippage
        risk = config["risk_params"]
        vix = (decision or {}).get("_vix", 0)
        base_slippage = risk.get("slippage_per_side_pct", 0.0005)
        volatile_slippage = risk.get("slippage_volatile_per_side_pct", base_slippage)
        slippage = volatile_slippage if vix > 25 else base_slippage

        if action == "BUY":
            exec_price = round(price * (1 + slippage), 2)
        else:
            exec_price = round(price * (1 - slippage), 2)

        # Validate using the shared function
        ok, reason = validate_trade(action, shares, exec_price, self.portfolio, config)
        if not ok:
            logger.info(f"Replay trade rejected: {reason}")
            return None

        total_cost = round(shares * exec_price, 2)
        h = self.portfolio["holdings"]["TSLA"]
        realized_pnl = 0.0

        if action == "BUY":
            old_total = h["shares"] * h["avg_cost_basis"]
            new_total = old_total + total_cost
            h["shares"] = round(h["shares"] + shares, 6)
            h["avg_cost_basis"] = round(new_total / h["shares"], 2) if h["shares"] > 0 else 0.0
            self.portfolio["cash"] = round(self.portfolio["cash"] - total_cost, 2)
        elif action == "SELL":
            realized_pnl = round((exec_price - h["avg_cost_basis"]) * shares, 2)
            h["shares"] = round(h["shares"] - shares, 6)
            if h["shares"] < 0.0001:
                h["shares"] = 0.0
                h["avg_cost_basis"] = 0.0
            self.portfolio["cash"] = round(self.portfolio["cash"] + total_cost, 2)

        self.update_price(exec_price)
        self.portfolio["total_trades"] += 1
        if action == "SELL":
            if realized_pnl >= 0:
                self.portfolio["winning_trades"] += 1
            else:
                self.portfolio["losing_trades"] += 1

        txn_id = generate_id("rtxn", [t["id"] for t in self.transactions])
        txn = {
            "id": txn_id,
            "timestamp": iso_now(),
            "action": action,
            "ticker": "TSLA",
            "shares": shares,
            "price": exec_price,
            "total_cost": total_cost,
            "realized_pnl": realized_pnl if action == "SELL" else None,
            "cash_after": self.portfolio["cash"],
            "portfolio_value_after": self.portfolio["total_value"],
            "strategy": (decision or {}).get("strategy", "replay"),
            "confidence": (decision or {}).get("confidence", 0),
            "hypothesis": (decision or {}).get("hypothesis", ""),
            "reasoning": (decision or {}).get("reasoning", ""),
            "signals": (decision or {}).get("signals", {}),
            "knowledge_applied": (decision or {}).get("knowledge_applied", []),
            "regime": (decision or {}).get("_regime", {}),
            "review": None,
            "_replay_source": self.session_id,
            "_bar_index": bar_index,
        }
        self.transactions.append(txn)
        logger.info(
            f"Replay {action} {shares:.4f} @ ${exec_price:.2f} "
            f"(bar {bar_index}, value ${self.portfolio['total_value']:.2f})"
        )
        return txn

    def save_checkpoint(self, bar_index: int, api_cost: float):
        """Save current state for resumability."""
        checkpoint = {
            "bar_index": bar_index,
            "portfolio": self.portfolio,
            "transactions": self.transactions,
            "api_cost": api_cost,
            "saved_at": iso_now(),
        }
        save_json(self.dir / "checkpoint.json", checkpoint)

    def load_checkpoint(self) -> dict | None:
        """Load a previous checkpoint if it exists."""
        path = self.dir / "checkpoint.json"
        if path.exists():
            return load_json(path)
        return None

    def save_final(self):
        """Save final results."""
        save_json(self.dir / "portfolio.json", self.portfolio)
        save_json(self.dir / "transactions.json", self.transactions)


# ---------------------------------------------------------------------------
# DataObfuscator — anti-hindsight bias
# ---------------------------------------------------------------------------
class DataObfuscator:
    """Mask real prices/tickers to prevent the agent from recognizing historical data."""

    def __init__(self):
        self.scale_factor = random.uniform(0.3, 3.0)
        logger.info(f"Obfuscator scale factor: {self.scale_factor:.4f}")

    def obfuscate_bar(self, bar: dict, bar_index: int) -> dict:
        """Return an obfuscated copy of a bar dict."""
        ob = dict(bar)

        # Scale all price-based fields
        price_fields = [
            "close", "open", "high", "low",
            "sma_20", "sma_50", "ema_12", "ema_26",
            "bollinger_upper", "bollinger_lower", "bollinger_mid",
            "atr", "current_price",
        ]
        for f in price_fields:
            if f in ob and ob[f] is not None:
                ob[f] = round(ob[f] * self.scale_factor, 2)

        # MACD values are price-derived, scale them too
        for f in ["macd", "macd_signal", "macd_histogram"]:
            if f in ob and ob[f] is not None:
                ob[f] = round(ob[f] * self.scale_factor, 4)

        # Replace ticker references
        # (The bar dict doesn't have a ticker field, but we handle it in market_data building)

        # Replace date with sequential bar number
        ob["date"] = f"bar_{bar_index:04d}"

        # Percentage-based indicators left as-is: rsi_14, macd_crossover
        # VIX and volume left as-is (market-wide / ratio data)

        return ob

    def obfuscate_market_data(self, market_data: dict, bar_index: int) -> dict:
        """Obfuscate a full market_data dict (get_market_summary shape)."""
        md = json.loads(json.dumps(market_data))  # deep copy

        md["ticker"] = "ASSET_A"

        # Scale current price fields
        current = md.get("current", {})
        for f in ["price"]:
            if f in current and current[f] is not None:
                current[f] = round(current[f] * self.scale_factor, 2)
        # change is price-derived
        if "change" in current and current["change"] is not None:
            current["change"] = round(current["change"] * self.scale_factor, 2)
        # change_pct is percentage — leave as-is

        # Scale daily indicators
        daily = md.get("daily_indicators", {})
        price_fields = [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "bollinger_upper", "bollinger_lower", "bollinger_mid",
            "atr", "current_price",
        ]
        for f in price_fields:
            if f in daily and daily[f] is not None:
                daily[f] = round(daily[f] * self.scale_factor, 2)
        for f in ["macd", "macd_signal", "macd_histogram"]:
            if f in daily and daily[f] is not None:
                daily[f] = round(daily[f] * self.scale_factor, 4)

        # Scale recent_days
        for day in md.get("recent_days", []):
            for f in ["open", "high", "low", "close"]:
                if f in day and day[f] is not None:
                    day[f] = round(day[f] * self.scale_factor, 2)
            day["date"] = f"bar_{bar_index:04d}"

        md["fetched_at"] = f"bar_{bar_index:04d}"
        return md

    def scale_price(self, price: float) -> float:
        """Scale a single price value (for portfolio display)."""
        return round(price * self.scale_factor, 2)

    def unscale_price(self, obfuscated_price: float) -> float:
        """Reverse the scaling."""
        return round(obfuscated_price / self.scale_factor, 2) if self.scale_factor else 0


# ---------------------------------------------------------------------------
# ReplayProgress — console reporting
# ---------------------------------------------------------------------------
class ReplayProgress:
    """Print progress updates during replay."""

    def __init__(self, total_bars: int):
        self.total_bars = total_bars
        self.start_time = time.time()

    def report(self, bar_index: int, portfolio_value: float, starting_balance: float,
               trade_count: int, lesson_count: int, api_cost: float):
        """Print progress every N bars."""
        elapsed = time.time() - self.start_time
        pct_done = bar_index / self.total_bars if self.total_bars else 0
        remaining = (elapsed / pct_done - elapsed) if pct_done > 0.01 else 0

        pnl_pct = ((portfolio_value - starting_balance) / starting_balance * 100
                    if starting_balance else 0)

        print(
            f"  [{bar_index}/{self.total_bars}] "
            f"${portfolio_value:,.2f} ({pnl_pct:+.1f}%) | "
            f"{trade_count} trades, {lesson_count} lessons | "
            f"~${api_cost:.2f} spent | "
            f"~{remaining / 60:.0f}min left"
        )

    def summary(self, portfolio_value: float, starting_balance: float,
                trade_count: int, lessons: list, api_cost: float):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        pnl_pct = ((portfolio_value - starting_balance) / starting_balance * 100
                    if starting_balance else 0)

        # Lesson quality distribution
        quality = {"validated": 0, "unvalidated": 0, "rejected": 0}
        for l in lessons:
            skeptic = l.get("skeptic_review", {})
            if skeptic.get("validated"):
                quality["validated"] += 1
            elif skeptic.get("validated") is False:
                quality["rejected"] += 1
            else:
                quality["unvalidated"] += 1

        print("\n" + "=" * 60)
        print("REPLAY COMPLETE")
        print("=" * 60)
        print(f"  P&L:      ${portfolio_value - starting_balance:+.2f} ({pnl_pct:+.1f}%)")
        print(f"  Trades:   {trade_count}")
        print(f"  Lessons:  {len(lessons)} "
              f"(validated: {quality['validated']}, "
              f"unvalidated: {quality['unvalidated']}, "
              f"rejected: {quality['rejected']})")
        print(f"  API Cost: ${api_cost:.2f}")
        print(f"  Duration: {elapsed / 60:.1f} minutes")
        print("=" * 60)


# ---------------------------------------------------------------------------
# ReplayEngine — main orchestrator
# ---------------------------------------------------------------------------
class ReplayEngine:
    """Walk through historical bars, calling the real agent + learning loop."""

    def __init__(
        self,
        year: int | None = None,
        scenario: str | None = None,
        obfuscate: bool = True,
        resume: bool = True,
    ):
        # Determine date range
        if scenario and scenario in SCENARIOS:
            sc = SCENARIOS[scenario]
            self.start = sc["start"]
            self.end = sc["end"]
            self.label = sc["label"]
        elif year:
            self.start = f"{year}-01-01"
            self.end = f"{year}-12-31"
            self.label = f"Full Year {year}"
        else:
            raise ValueError("Must provide either year or scenario")

        self.obfuscate = obfuscate
        self.resume = resume

        # Session ID for isolation
        short_id = uuid.uuid4().hex[:8]
        tag = scenario or str(year)
        self.session_id = f"replay_{tag}_{short_id}"

        self.obfuscator = DataObfuscator() if obfuscate else None
        self.portfolio = ReplayPortfolio(self.session_id)
        self.config = load_config()

        # Tracking
        self.api_cost = 0.0
        self.lessons_created: list[dict] = []
        self.pending_reviews: list[dict] = []  # (txn, bar_index_at_trade)
        self.flagged_bars: set[int] = set()

    def _load_historical_data(self) -> list[dict]:
        """Use WalkForwardBacktest to download and prepare bars."""
        bt = WalkForwardBacktest(
            ticker="TSLA",
            start=self.start,
            end=self.end,
        )
        bt.load_data()
        bars = bt._prepare_indicators()
        logger.info(f"Loaded {len(bars)} bars for {self.label} ({self.start} to {self.end})")
        return bars

    def _flag_bad_bars(self, bars: list[dict]):
        """Flag bars with data quality issues (Proposals 5, 8)."""
        for i, bar in enumerate(bars):
            # >50% single-day move = likely split artifact
            if i > 0:
                prev_close = bars[i - 1]["close"]
                if prev_close > 0:
                    day_change = abs(bar["close"] - prev_close) / prev_close
                    if day_change > 0.50:
                        self.flagged_bars.add(i)
                        logger.warning(
                            f"Flagged bar {i} ({bar['date']}): "
                            f"{day_change*100:.1f}% move — possible split artifact"
                        )

            # Zero volume = missing data
            if bar.get("volume", 0) == 0:
                self.flagged_bars.add(i)
                logger.warning(f"Flagged bar {i} ({bar['date']}): zero volume")

        if self.flagged_bars:
            logger.info(f"Flagged {len(self.flagged_bars)} bars for data quality issues")

    def _build_market_data(self, bar: dict, bar_index: int, bars: list[dict]) -> dict:
        """Construct a dict matching get_market_summary() output shape from a bar."""
        # Recent days context (last 5 bars)
        recent_start = max(0, bar_index - 5)
        recent_days = []
        for j in range(recent_start, bar_index):
            b = bars[j]
            recent_days.append({
                "date": b["date"],
                "open": round(b["open"], 2),
                "high": round(b["high"], 2),
                "low": round(b["low"], 2),
                "close": round(b["close"], 2),
                "volume": b.get("volume", 0),
            })

        # Pre-calculate max trade value so the agent knows its budget
        risk = self.config["risk_params"]
        max_trade_value = round(self.portfolio.portfolio["total_value"] * risk["max_single_trade_pct"], 2)
        max_shares = round(max_trade_value / bar["close"], 4) if bar["close"] > 0 else 0

        market_data = {
            "ticker": "TSLA",
            "current": {
                "price": round(bar["close"], 2),
                "change": round(bar["close"] - bar["open"], 2),
                "change_pct": round((bar["close"] - bar["open"]) / bar["open"] * 100, 2) if bar["open"] else 0,
                "volume": bar.get("volume", 0),
            },
            "position_limits": {
                "max_trade_value": max_trade_value,
                "max_shares_at_current_price": max_shares,
            },
            "daily_indicators": {
                "rsi_14": bar.get("rsi_14"),
                "sma_20": bar.get("sma_20"),
                "sma_50": bar.get("sma_50"),
                "ema_12": bar.get("ema_12"),
                "ema_26": bar.get("ema_26"),
                "macd": bar.get("macd"),
                "macd_signal": bar.get("macd_signal"),
                "macd_histogram": bar.get("macd_histogram"),
                "macd_crossover": bar.get("macd_crossover", "none"),
                "bollinger_upper": bar.get("bollinger_upper"),
                "bollinger_lower": bar.get("bollinger_lower"),
                "bollinger_mid": bar.get("bollinger_mid"),
                "atr": bar.get("atr"),
                "volume_sma_20": bar.get("volume_sma_20"),
                "obv": bar.get("obv"),
                "current_price": round(bar["close"], 2),
            },
            "intraday_indicators": None,  # Not available in daily replay
            "recent_days": recent_days,
            "macro": {
                "spy_change_pct": bar.get("spy_daily_change", 0),
                "vix_level": bar.get("vix", 20),
                "fetched": True,
            },
            "regime": {
                "trend": bar.get("spy_trend", "unknown"),
                "volatility": bar.get("vol_regime", "normal"),
                "vix": bar.get("vix", 20),
            },
            "fetched_at": bar["date"],
        }

        if self.obfuscate and self.obfuscator:
            market_data = self.obfuscator.obfuscate_market_data(market_data, bar_index)

        return market_data

    def _build_macro_gate(self, bar: dict) -> dict:
        """Construct macro gate from bar data."""
        vix = bar.get("vix", 20)
        spy_change = bar.get("spy_daily_change", 0)
        config = self.config
        mg = config["risk_params"].get("macro_gate", {})

        gate_active = False
        reason = ""
        if spy_change < mg.get("spy_daily_drop_threshold", -0.02):
            gate_active = True
            reason = f"SPY down {spy_change*100:.1f}%"
        if vix > mg.get("vix_threshold", 30):
            gate_active = True
            reason = (reason + " + " if reason else "") + f"VIX at {vix:.1f}"

        return {
            "gate_active": gate_active,
            "reason": reason,
            "spy_change_pct": spy_change,
            "vix": vix,
            "confidence_threshold_override": mg.get("elevated_confidence_required", 0.80) if gate_active else None,
        }

    async def _call_agent_with_retry(self, market_data, signals, aggregate,
                                     portfolio, knowledge, macro_gate, regime) -> dict:
        """Call make_decision with exponential backoff. Falls back to HOLD."""
        delays = [5, 15, 45]
        last_error = None

        for attempt, delay in enumerate(delays):
            try:
                decision = make_decision(
                    market_data=market_data,
                    signals=signals,
                    aggregate=aggregate,
                    portfolio=portfolio,
                    knowledge=knowledge,
                    macro_gate=macro_gate,
                    regime=regime,
                )
                # Estimate API cost (~2K input + 500 output tokens)
                self.api_cost += 0.01  # rough estimate per call
                return decision
            except Exception as e:
                last_error = e
                logger.warning(f"Agent call attempt {attempt+1} failed: {e}")
                if attempt < len(delays) - 1:
                    await asyncio.sleep(delay)

        logger.error(f"All agent retries failed: {last_error}")
        return {
            "action": "HOLD",
            "shares": 0,
            "confidence": 0,
            "strategy": "retry_exhausted",
            "hypothesis": "Agent unavailable — defaulting to HOLD",
            "reasoning": str(last_error),
            "knowledge_applied": [],
            "risk_note": "Retry exhausted",
        }

    async def _review_matured_trades(self, bar: dict, bar_index: int, bars: list[dict]):
        """Review trades that are 5+ bars old."""
        current_price = bar["close"]
        still_pending = []

        for txn, trade_bar in self.pending_reviews:
            if bar_index - trade_bar >= REVIEW_DELAY_BARS:
                try:
                    # Attach historical SPY change so the skeptic has real data
                    spy_change = bar.get("spy_daily_change", 0)
                    txn["_replay_spy_change"] = spy_change
                    lesson = await review_trade(txn, current_price)
                    if lesson:
                        self.lessons_created.append(lesson)
                        logger.info(
                            f"Replay lesson {lesson['id']}: "
                            f"{lesson.get('category', '?')} — {lesson.get('lesson', '')[:80]}"
                        )
                except Exception as e:
                    logger.error(f"Replay trade review failed: {e}")
            else:
                still_pending.append((txn, trade_bar))

        self.pending_reviews = still_pending

    async def run(self):
        """Main replay loop."""
        print(f"\n{'='*60}")
        print(f"REPLAY MODE: {self.label}")
        print(f"Session: {self.session_id}")
        print(f"Obfuscation: {'ON' if self.obfuscate else 'OFF'}")
        print(f"{'='*60}\n")

        # Load historical data
        bars = self._load_historical_data()
        if not bars:
            print("No bars loaded. Check date range.")
            return

        # Flag bad data
        self._flag_bad_bars(bars)

        progress = ReplayProgress(len(bars))
        start_bar = 0

        # Resume from checkpoint if available
        if self.resume:
            checkpoint = self.portfolio.load_checkpoint()
            if checkpoint:
                start_bar = checkpoint["bar_index"] + 1
                self.portfolio.portfolio = checkpoint["portfolio"]
                self.portfolio.transactions = checkpoint["transactions"]
                self.api_cost = checkpoint.get("api_cost", 0)
                print(f"Resuming from bar {start_bar} (checkpoint found)")

        # Main loop
        for i in range(start_bar, len(bars)):
            bar = bars[i]

            # Skip flagged bars
            if i in self.flagged_bars:
                logger.info(f"Skipping flagged bar {i} ({bar['date']})")
                continue

            # 1. Update replay portfolio with bar's close price
            self.portfolio.update_price(bar["close"])

            # 2. Build market_data dict (obfuscated if enabled)
            market_data = self._build_market_data(bar, i, bars)

            # 3. Run strategies
            try:
                scores = get_strategy_scores()
                signals = evaluate_all_strategies(
                    market_data=market_data,
                    portfolio=self.portfolio.portfolio,
                    scores=scores.get("strategies"),
                    regime=market_data.get("regime"),
                )
                aggregate = aggregate_signals(signals)
            except Exception as e:
                logger.warning(f"Strategy evaluation failed at bar {i}: {e}")
                continue

            # 4. Get knowledge
            knowledge = get_relevant_knowledge(market_data)

            # 5. Build macro gate
            macro_gate = self._build_macro_gate(bar)
            regime = market_data.get("regime", {})

            # 6. Call agent
            decision = await self._call_agent_with_retry(
                market_data, signals, aggregate,
                self.portfolio.portfolio, knowledge, macro_gate, regime,
            )

            # Attach regime for trade record
            decision["_regime"] = regime
            decision["_vix"] = bar.get("vix", 0)

            action = decision.get("action", "HOLD")

            # 7. Execute if BUY/SELL
            if action in ("BUY", "SELL"):
                shares = decision.get("shares", 0)
                if shares > 0:
                    price = bar["close"]
                    if self.obfuscate and self.obfuscator:
                        # Agent sees obfuscated prices, but we execute at real prices
                        pass
                    txn = self.portfolio.execute_trade(action, shares, price, decision, i)
                    if txn:
                        self.pending_reviews.append((txn, i))

            # 8. Review matured trades
            await self._review_matured_trades(bar, i, bars)

            # 9. Checkpoint
            if i > 0 and i % CHECKPOINT_INTERVAL == 0:
                self.portfolio.save_checkpoint(i, self.api_cost)

            # 10. Progress
            if i > 0 and i % 10 == 0:
                progress.report(
                    bar_index=i,
                    portfolio_value=self.portfolio.portfolio["total_value"],
                    starting_balance=self.portfolio.starting_balance,
                    trade_count=self.portfolio.portfolio["total_trades"],
                    lesson_count=len(self.lessons_created),
                    api_cost=self.api_cost,
                )

        # Review any remaining pending trades
        if bars:
            last_bar = bars[-1]
            for txn, _ in self.pending_reviews:
                try:
                    txn["_replay_spy_change"] = last_bar.get("spy_daily_change", 0)
                    lesson = await review_trade(txn, last_bar["close"])
                    if lesson:
                        self.lessons_created.append(lesson)
                except Exception as e:
                    logger.error(f"Final review failed: {e}")

        # Save results
        self.portfolio.save_final()
        save_json(
            self.portfolio.dir / "lessons.json",
            self.lessons_created,
        )
        save_json(
            self.portfolio.dir / "summary.json",
            {
                "session_id": self.session_id,
                "label": self.label,
                "start": self.start,
                "end": self.end,
                "obfuscated": self.obfuscate,
                "total_bars": len(bars),
                "flagged_bars": len(self.flagged_bars),
                "trades": self.portfolio.portfolio["total_trades"],
                "lessons": len(self.lessons_created),
                "final_value": self.portfolio.portfolio["total_value"],
                "pnl": self.portfolio.portfolio["total_pnl"],
                "pnl_pct": self.portfolio.portfolio["total_pnl_pct"],
                "api_cost": self.api_cost,
                "completed_at": iso_now(),
            },
        )

        # Print summary
        progress.summary(
            portfolio_value=self.portfolio.portfolio["total_value"],
            starting_balance=self.portfolio.starting_balance,
            trade_count=self.portfolio.portfolio["total_trades"],
            lessons=self.lessons_created,
            api_cost=self.api_cost,
        )

        print(f"\nResults saved to: {self.portfolio.dir}")
        print(f"Lessons added to shared knowledge base: {len(self.lessons_created)}")
