"""Tests for the Thesis Ledger learning system.

Validates tag computation, thesis aggregation, playbook formatting,
and data integrity with mock trades.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# We test the functions directly, not through the module system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_compute_tags_basic():
    """Tags are computed correctly from market data."""
    from src.tags import compute_tags

    market_data = {
        "current": {"price": 300.0},
        "daily_indicators": {
            "rsi_14": 25.0,  # oversold
            "sma_50": 310.0,  # price below sma50
            "macd_crossover": "bullish_crossover",
        },
        "regime": {
            "vix": 28.0,  # high
            "directional": "trending",
        },
    }
    world = {"macro": {"SPY": {"change_pct": -1.5}}}  # spy_down
    portfolio = {
        "holdings": {"TSLA": {"shares": 0, "avg_cost_basis": 0}},
    }
    config = {"ticker": "TSLA"}
    events = {"macro_events": [], "tsla_earnings": None}

    tags = compute_tags(market_data, world, portfolio, config, events, action="BUY")

    assert tags["rsi_zone"] == "oversold", f"Expected oversold, got {tags['rsi_zone']}"
    assert tags["trend"] == "below_sma50", f"Expected below_sma50, got {tags['trend']}"
    assert tags["volatility"] == "high_vix", f"Expected high_vix, got {tags['volatility']}"
    assert tags["regime"] == "trending", f"Expected trending, got {tags['regime']}"
    assert tags["macd"] == "bullish_cross", f"Expected bullish_cross, got {tags['macd']}"
    assert tags["market_context"] == "spy_down", f"Expected spy_down, got {tags['market_context']}"
    assert tags["position_state"] == "opening_new", f"Expected opening_new, got {tags['position_state']}"
    assert tags["event_proximity"] == "no_event", f"Expected no_event, got {tags['event_proximity']}"
    print("  PASS: test_compute_tags_basic")


def test_compute_tags_position_states():
    """Position state tag correctly distinguishes winner/loser adds."""
    from src.tags import compute_tags

    base_market = {
        "current": {"price": 300.0},
        "daily_indicators": {"rsi_14": 50, "sma_50": 290, "macd_crossover": "none"},
        "regime": {"vix": 20, "directional": "range_bound"},
    }
    world = {"macro": {"SPY": {"change_pct": 0.1}}}
    config = {"ticker": "TSLA"}
    events = {"macro_events": [], "tsla_earnings": None}

    # Buying when no position = opening_new
    portfolio = {"holdings": {"TSLA": {"shares": 0, "avg_cost_basis": 0}}}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["position_state"] == "opening_new"

    # Buying when price > avg_cost = adding_to_winner
    portfolio = {"holdings": {"TSLA": {"shares": 1.0, "avg_cost_basis": 280.0}}}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["position_state"] == "adding_to_winner"

    # Buying when price < avg_cost = adding_to_loser
    portfolio = {"holdings": {"TSLA": {"shares": 1.0, "avg_cost_basis": 320.0}}}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["position_state"] == "adding_to_loser"

    # Selling when price > avg_cost = taking_profit
    portfolio = {"holdings": {"TSLA": {"shares": 1.0, "avg_cost_basis": 280.0}}}
    tags = compute_tags(base_market, world, portfolio, config, events, "SELL")
    assert tags["position_state"] == "taking_profit"

    # Selling when price < avg_cost = cutting_loss
    portfolio = {"holdings": {"TSLA": {"shares": 1.0, "avg_cost_basis": 320.0}}}
    tags = compute_tags(base_market, world, portfolio, config, events, "SELL")
    assert tags["position_state"] == "cutting_loss"

    print("  PASS: test_compute_tags_position_states")


def test_compute_tags_events():
    """Event proximity tag detects upcoming events correctly."""
    from src.tags import compute_tags

    base_market = {
        "current": {"price": 300.0},
        "daily_indicators": {"rsi_14": 50, "sma_50": 290, "macd_crossover": "none"},
        "regime": {"vix": 20, "directional": "range_bound"},
    }
    world = {"macro": {}}
    portfolio = {"holdings": {"TSLA": {"shares": 0, "avg_cost_basis": 0}}}
    config = {"ticker": "TSLA"}

    # No events
    events = {"macro_events": [], "tsla_earnings": None}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["event_proximity"] == "no_event"

    # CPI tomorrow
    events = {"macro_events": [{"event": "CPI", "hours_until": 18}], "tsla_earnings": None}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["event_proximity"] == "pre_event_24h"

    # FOMC in 48h
    events = {"macro_events": [{"event": "FOMC", "hours_until": 48}], "tsla_earnings": None}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["event_proximity"] == "pre_event_72h"

    # TSLA earnings in 2 days
    events = {"macro_events": [], "tsla_earnings": {"days_until": 2}}
    tags = compute_tags(base_market, world, portfolio, config, events, "BUY")
    assert tags["event_proximity"] == "pre_event_24h"

    print("  PASS: test_compute_tags_events")


def test_thesis_builder_aggregation():
    """Thesis builder correctly aggregates trades into single-tag stats."""
    from src.thesis_builder import build_ledger, LEDGER_PATH, JOURNAL_PATH
    from src.utils import save_json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    # Create mock journal with closed trades
    mock_journal = [
        {
            "trade_id": "txn_001", "action": "BUY", "ticker": "TSLA",
            "shares": 1.0, "price": 280.0, "total_value": 280.0,
            "reasoning": "test", "confidence": 0.8, "portfolio_value": 1000,
            "market_snapshot": "test", "timestamp": now,
            "tags": {"rsi_zone": "oversold", "trend": "below_sma50", "volatility": "high_vix",
                     "regime": "trending", "macd": "bullish_cross", "market_context": "spy_down",
                     "position_state": "opening_new", "event_proximity": "no_event"},
            "lesson": "test lesson", "close_trade_id": "txn_002",
            "close_price": 295.0, "realized_pnl": 15.0, "closed_at": now,
        },
        {
            "trade_id": "txn_003", "action": "BUY", "ticker": "TSLA",
            "shares": 0.5, "price": 310.0, "total_value": 155.0,
            "reasoning": "test", "confidence": 0.6, "portfolio_value": 1000,
            "market_snapshot": "test", "timestamp": now,
            "tags": {"rsi_zone": "overbought", "trend": "above_sma50", "volatility": "normal_vix",
                     "regime": "trending", "macd": "neutral", "market_context": "spy_up",
                     "position_state": "opening_new", "event_proximity": "no_event"},
            "lesson": "test lesson", "close_trade_id": "txn_004",
            "close_price": 300.0, "realized_pnl": -5.0, "closed_at": now,
        },
        {
            "trade_id": "txn_005", "action": "BUY", "ticker": "TSLA",
            "shares": 1.0, "price": 270.0, "total_value": 270.0,
            "reasoning": "test", "confidence": 0.75, "portfolio_value": 1000,
            "market_snapshot": "test", "timestamp": now,
            "tags": {"rsi_zone": "oversold", "trend": "below_sma50", "volatility": "high_vix",
                     "regime": "range_bound", "macd": "bullish_cross", "market_context": "spy_down",
                     "position_state": "opening_new", "event_proximity": "pre_event_72h"},
            "lesson": "test lesson", "close_trade_id": "txn_006",
            "close_price": 285.0, "realized_pnl": 15.0, "closed_at": now,
        },
    ]

    # Write mock data and build ledger
    save_json(JOURNAL_PATH, mock_journal)
    ledger = build_ledger()

    # Verify structure
    assert ledger["total_trades"] == 3
    assert "theses" in ledger
    assert "calibration" in ledger

    # Check rsi_zone:oversold thesis (2 trades, both wins)
    oversold = ledger["theses"].get("rsi_zone:oversold")
    assert oversold is not None, "rsi_zone:oversold thesis missing"
    assert oversold["trades"] == 2, f"Expected 2 trades, got {oversold['trades']}"
    assert oversold["wins"] == 2, f"Expected 2 wins, got {oversold['wins']}"
    assert oversold["avg_pnl"] == 15.0, f"Expected avg_pnl 15.0, got {oversold['avg_pnl']}"

    # Check rsi_zone:overbought thesis (1 trade, loss)
    overbought = ledger["theses"].get("rsi_zone:overbought")
    assert overbought is not None, "rsi_zone:overbought thesis missing"
    assert overbought["trades"] == 1
    assert overbought["wins"] == 0

    # Check regime:trending thesis (2 trades: 1 win, 1 loss)
    trending = ledger["theses"].get("regime:trending")
    assert trending is not None, "regime:trending thesis missing"
    assert trending["trades"] == 2

    # Verify each trade appears in exactly 8 tag theses
    total_thesis_trades = sum(t["trades"] for t in ledger["theses"].values())
    assert total_thesis_trades == 3 * 8, f"Expected {3*8} thesis-trade pairs, got {total_thesis_trades}"

    print("  PASS: test_thesis_builder_aggregation")


def test_thesis_builder_validation():
    """Validation catches suspicious patterns."""
    from src.thesis_builder import validate_ledger

    theses = {
        "rsi_zone:oversold": {"trades": 5, "wins": 5, "win_rate": 1.0},
        "rsi_zone:neutral": {"trades": 3, "wins": 1, "win_rate": 0.333},
        "bad_data": {"trades": 2, "wins": 3, "win_rate": 1.5},  # impossible
    }

    warnings = validate_ledger(theses, total_trades=10)

    # Should flag 100% win rate
    assert any("100% win rate" in w for w in warnings), "Should flag 100% win rate"
    # Should flag wins > trades
    assert any("DATA ERROR" in w for w in warnings), "Should flag wins > trades"

    print("  PASS: test_thesis_builder_validation")


def test_playbook_formatting():
    """Playbook formats correctly for the brief."""
    from src.thesis_builder import format_playbook_for_brief

    # Empty ledger
    empty = {"total_trades": 0, "theses": {}, "calibration": {}}
    text = format_playbook_for_brief(empty)
    assert "No closed trades" in text

    # Ledger with significant theses
    ledger = {
        "total_trades": 20,
        "active_trades": 18,
        "theses": {
            "rsi_zone:oversold": {
                "tag": "rsi_zone", "value": "oversold", "trades": 5,
                "wins": 4, "win_rate": 0.8, "avg_pnl": 12.5,
            },
            "rsi_zone:overbought": {
                "tag": "rsi_zone", "value": "overbought", "trades": 4,
                "wins": 1, "win_rate": 0.25, "avg_pnl": -8.3,
            },
            "trend:above_sma50": {
                "tag": "trend", "value": "above_sma50", "trades": 3,
                "wins": 1, "win_rate": 0.50, "avg_pnl": 0.5,
            },
        },
        "calibration": {
            "calibrated": False,
            "high_confidence_win_rate": 0.45,
            "low_confidence_win_rate": 0.50,
        },
    }

    text = format_playbook_for_brief(ledger)

    assert "Best setups" in text, "Should show best setups"
    assert "oversold" in text, "Should mention oversold"
    assert "AVOID" in text, "Should mark worst setups"
    assert "NOT predictive" in text, "Should show calibration warning"

    print("  PASS: test_playbook_formatting")


def test_playbook_bootstrap_phase():
    """During bootstrap, shows preliminary data without claiming stats."""
    from src.thesis_builder import format_playbook_for_brief

    # All theses have N<3
    ledger = {
        "total_trades": 4,
        "active_trades": 4,
        "theses": {
            "rsi_zone:neutral": {
                "tag": "rsi_zone", "value": "neutral", "trades": 2,
                "wins": 1, "win_rate": 0.5, "avg_pnl": 2.0,
            },
            "trend:above_sma50": {
                "tag": "trend", "value": "above_sma50", "trades": 2,
                "wins": 0, "win_rate": 0.0, "avg_pnl": -3.0,
            },
        },
        "calibration": {"status": "insufficient_data"},
    }

    text = format_playbook_for_brief(ledger)
    assert "Building playbook" in text, "Should indicate bootstrap phase"
    assert "preliminary" in text, "Should mark data as preliminary"

    print("  PASS: test_playbook_bootstrap_phase")


if __name__ == "__main__":
    print("\n=== Thesis Ledger Tests ===\n")
    test_compute_tags_basic()
    test_compute_tags_position_states()
    test_compute_tags_events()
    test_thesis_builder_aggregation()
    test_thesis_builder_validation()
    test_playbook_formatting()
    test_playbook_bootstrap_phase()
    print("\n=== All tests passed ===\n")
