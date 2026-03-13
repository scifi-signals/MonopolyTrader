"""Thesis system — persistent, versioned market thesis for TSLA."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from .utils import load_json, save_json, iso_now, setup_logging, KNOWLEDGE_DIR

logger = setup_logging("thesis")

THESIS_PATH = KNOWLEDGE_DIR / "thesis.json"
MAX_VERSIONS = 5


@dataclass
class Thesis:
    """The agent's running thesis about TSLA."""
    narrative: str = ""
    direction: str = "neutral"  # bearish | bullish | neutral
    conviction: float = 0.5  # 0.0 to 1.0
    key_levels: dict = field(default_factory=lambda: {"support": [], "resistance": []})
    bull_case: str = ""
    bear_case: str = ""
    invalidation: str = ""
    key_catalysts: list = field(default_factory=list)
    version: int = 1
    updated_at: str = ""
    updated_by: str = ""  # "analyst" or "manual"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Thesis":
        """Create Thesis from a dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def is_stale(self, hours: float = 4.0) -> bool:
        """Check if thesis hasn't been updated in `hours` hours."""
        if not self.updated_at:
            return True
        try:
            updated = datetime.fromisoformat(self.updated_at)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            elapsed = (now - updated).total_seconds() / 3600
            return elapsed > hours
        except (ValueError, TypeError):
            return True

    def price_breaks_level(self, current_price: float) -> dict | None:
        """Check if current price has broken a thesis support or resistance level.

        Returns dict with break info, or None if no break.
        """
        supports = self.key_levels.get("support", [])
        resistances = self.key_levels.get("resistance", [])

        # Check support break (price below support)
        for level in supports:
            if current_price < level * 0.995:  # 0.5% buffer
                return {
                    "type": "support_break",
                    "level": level,
                    "price": current_price,
                    "distance_pct": round((level - current_price) / level * 100, 2),
                }

        # Check resistance break (price above resistance)
        for level in resistances:
            if current_price > level * 1.005:  # 0.5% buffer
                return {
                    "type": "resistance_break",
                    "level": level,
                    "price": current_price,
                    "distance_pct": round((current_price - level) / level * 100, 2),
                }

        return None

    def format_for_prompt(self) -> str:
        """Format thesis for inclusion in agent prompts."""
        if not self.narrative:
            return "No thesis established yet. Build one from current news and market context."

        parts = [
            f"CURRENT THESIS (v{self.version}, updated {self.updated_at}):",
            f"  Direction: {self.direction.upper()} (conviction: {self.conviction:.2f})",
            f"  Narrative: {self.narrative}",
        ]

        supports = self.key_levels.get("support", [])
        resistances = self.key_levels.get("resistance", [])
        if supports or resistances:
            parts.append(f"  Key Levels: Support {supports}, Resistance {resistances}")

        if self.bull_case:
            parts.append(f"  Bull case: {self.bull_case}")
        if self.bear_case:
            parts.append(f"  Bear case: {self.bear_case}")
        if self.invalidation:
            parts.append(f"  Invalidation: {self.invalidation}")
        if self.key_catalysts:
            parts.append(f"  Key catalysts: {', '.join(self.key_catalysts)}")

        return "\n".join(parts)


def load_thesis() -> Thesis:
    """Load the current thesis from disk."""
    data = load_json(THESIS_PATH, default={})
    current = data.get("current")
    if current:
        return Thesis.from_dict(current)
    return Thesis()


def save_thesis(thesis: Thesis, reason: str = ""):
    """Save thesis to disk, maintaining version history (last 5)."""
    data = load_json(THESIS_PATH, default={"current": None, "history": []})

    # Archive current version before overwriting
    old_current = data.get("current")
    if old_current:
        history = data.get("history", [])
        history.append(old_current)
        # Keep only last MAX_VERSIONS
        data["history"] = history[-MAX_VERSIONS:]

    thesis.version = (old_current.get("version", 0) if old_current else 0) + 1
    thesis.updated_at = iso_now()

    data["current"] = thesis.to_dict()
    save_json(THESIS_PATH, data)

    logger.info(
        f"Thesis updated (v{thesis.version}): {thesis.direction} "
        f"conviction={thesis.conviction:.2f} — {reason or 'no reason given'}"
    )


def get_thesis_history() -> list[dict]:
    """Get the version history of thesis changes."""
    data = load_json(THESIS_PATH, default={})
    history = data.get("history", [])
    current = data.get("current")
    if current:
        history = history + [current]
    return history


def thesis_changed_meaningfully(old: Thesis, new: Thesis) -> bool:
    """Check if the thesis changed enough to warrant a trader call.

    Direction change or conviction shift > 0.2 counts as meaningful.
    """
    if old.direction != new.direction:
        return True
    if abs(old.conviction - new.conviction) > 0.2:
        return True
    # Key level change
    if old.key_levels != new.key_levels:
        return True
    return False


def create_initial_thesis() -> Thesis:
    """Create a default initial thesis for bootstrapping."""
    thesis = Thesis(
        narrative="Initial thesis — no established view yet. Gathering news and market data.",
        direction="neutral",
        conviction=0.3,
        key_levels={"support": [], "resistance": []},
        bull_case="Unknown — needs research",
        bear_case="Unknown — needs research",
        invalidation="Will be set after first analyst run",
        key_catalysts=[],
        updated_by="bootstrap",
    )
    save_thesis(thesis, reason="Initial bootstrap")
    return thesis
