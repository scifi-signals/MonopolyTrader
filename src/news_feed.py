"""News feed — multi-source news aggregation with classification and scoring."""

import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from .utils import setup_logging

logger = setup_logging("news_feed")

# Catalyst type keywords — lowercase matching against headline + summary
CATALYST_KEYWORDS = {
    "earnings": [
        "earnings", "revenue", "profit", "eps", "quarter", "q1", "q2", "q3", "q4",
        "guidance", "beat", "miss", "forecast", "outlook", "annual report",
    ],
    "regulatory": [
        "fsd", "autopilot", "nhtsa", "faa", "sec ", "regulation", "regulatory",
        "investigation", "probe", "recall", "compliance", "approval", "permit",
        "robotaxi", "self-driving", "autonomous", "fda",
    ],
    "product": [
        "model ", "cybertruck", "semi", "roadster", "megapack", "powerwall",
        "launch", "production", "factory", "gigafactory", "delivery", "deliveries",
        "supercharger", "charging", "battery", "4680", "optimus", "robot",
    ],
    "macro": [
        "fed ", "interest rate", "inflation", "cpi", "gdp", "recession",
        "tariff", "trade war", "oil price", "treasury", "yield", "unemployment",
        "nasdaq", "s&p 500", "sp500", "dow jones", "market crash", "bear market",
        "bull market", "rally",
    ],
    "musk_personal": [
        "elon musk", "musk tweet", "musk said", "musk personal", "musk political",
        "doge ", "x.com", "twitter", "spacex", "neuralink", "boring company",
        "musk controversy", "musk ceo",
    ],
    "analyst": [
        "upgrade", "downgrade", "price target", "analyst", "rating",
        "overweight", "underweight", "buy rating", "sell rating", "hold rating",
        "outperform", "underperform", "neutral rating",
    ],
    "delivery": [
        "delivery", "deliveries", "production numbers", "units delivered",
        "vehicle deliveries", "quarterly deliveries",
    ],
    "competition": [
        "rivian", "lucid", "nio", "byd", "ford ev", "gm ev", "volkswagen ev",
        "hyundai ev", "competition", "market share", "ev market",
    ],
}

# Tesla-specific relevance keywords
TSLA_KEYWORDS = [
    "tesla", "tsla", "elon musk", "musk", "gigafactory", "cybertruck",
    "model 3", "model y", "model s", "model x", "fsd", "autopilot",
    "supercharger", "megapack", "optimus", "robotaxi",
]


@dataclass
class NewsItem:
    """A single news item with classification and relevance scoring."""
    title: str
    source: str
    published: str  # ISO timestamp or approximate
    url: str = ""
    summary: str = ""
    catalyst_type: str = "unknown"
    relevance: float = 0.0  # 0.0 to 1.0
    _hash: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "published": self.published,
            "url": self.url,
            "summary": self.summary,
            "catalyst_type": self.catalyst_type,
            "relevance": self.relevance,
        }


@dataclass
class NewsFeed:
    """Aggregated, classified, scored news feed."""
    items: list[NewsItem] = field(default_factory=list)
    fetched_at: str = ""
    source_counts: dict = field(default_factory=dict)

    @property
    def high_impact(self) -> list[NewsItem]:
        """Items with relevance > 0.7, sorted by relevance descending."""
        return sorted(
            [i for i in self.items if i.relevance > 0.7],
            key=lambda x: x.relevance,
            reverse=True,
        )

    def to_dict(self) -> dict:
        return {
            "fetched_at": self.fetched_at,
            "total_items": len(self.items),
            "high_impact_count": len(self.high_impact),
            "source_counts": self.source_counts,
            "items": [i.to_dict() for i in self.items],
            "high_impact": [i.to_dict() for i in self.high_impact],
        }


def _headline_hash(title: str) -> str:
    """Generate a hash for deduplication based on normalized headline."""
    normalized = re.sub(r'[^a-z0-9\s]', '', title.lower().strip())
    # Remove common filler words for better dedup
    for word in ["the", "a", "an", "is", "are", "was", "were", "has", "have"]:
        normalized = normalized.replace(f" {word} ", " ")
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def _classify_catalyst(text: str) -> str:
    """Classify a headline/summary into a catalyst type using keyword matching."""
    text_lower = text.lower()
    scores = {}
    for catalyst, keywords in CATALYST_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[catalyst] = score

    if not scores:
        return "unknown"
    return max(scores, key=scores.get)


def _score_relevance(title: str, summary: str = "") -> float:
    """Score relevance of a news item to TSLA trading decisions.

    Tesla in title = 0.8+, general market = 0.3, Tesla in summary = 0.5.
    """
    text = f"{title} {summary}".lower()
    title_lower = title.lower()

    # Direct Tesla mention in title = high relevance
    if any(kw in title_lower for kw in ["tesla", "tsla"]):
        base = 0.85
    elif any(kw in title_lower for kw in ["elon musk", "musk"]):
        base = 0.80
    elif any(kw in title_lower for kw in TSLA_KEYWORDS):
        base = 0.75
    # Tesla in summary but not title
    elif any(kw in text for kw in ["tesla", "tsla"]):
        base = 0.50
    # EV market / competition news
    elif any(kw in text for kw in ["ev market", "electric vehicle", "ev sales"]):
        base = 0.40
    # General macro/market news
    elif any(kw in text for kw in ["fed ", "interest rate", "nasdaq", "s&p"]):
        base = 0.30
    else:
        base = 0.15

    # Boost for high-impact catalyst keywords in title
    for catalyst_type in ["earnings", "delivery", "regulatory"]:
        if any(kw in title_lower for kw in CATALYST_KEYWORDS.get(catalyst_type, [])):
            base = min(base + 0.10, 1.0)
            break

    return round(base, 2)


def _is_similar(title1: str, title2: str, threshold: float = 0.6) -> bool:
    """Check headline similarity for deduplication.

    Uses word overlap ratio — if >60% of words overlap, consider duplicate.
    """
    words1 = set(re.sub(r'[^a-z0-9\s]', '', title1.lower()).split())
    words2 = set(re.sub(r'[^a-z0-9\s]', '', title2.lower()).split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    smaller = min(len(words1), len(words2))
    return (overlap / smaller) >= threshold if smaller > 0 else False


def _fetch_yfinance_news(ticker: str) -> list[NewsItem]:
    """Fetch expanded news from yfinance (up to ~50 items with metadata)."""
    items = []
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return items

        for item in news:  # Use full list, not truncated to 8
            content = item.get("content", {})
            title = content.get("title", item.get("title", ""))
            if not title:
                continue

            provider = content.get("provider", {}).get("displayName", "yfinance")
            pub_date = content.get("pubDate", item.get("providerPublishTime", ""))
            url = content.get("canonicalUrl", {}).get("url", "")
            summary = content.get("summary", "")

            # Convert unix timestamp if needed
            if isinstance(pub_date, (int, float)):
                pub_date = datetime.fromtimestamp(pub_date, tz=timezone.utc).isoformat()

            news_item = NewsItem(
                title=title,
                source=provider,
                published=str(pub_date),
                url=url,
                summary=summary[:300] if summary else "",
            )
            items.append(news_item)

    except Exception as e:
        logger.warning(f"yfinance news fetch failed: {e}")

    return items


def _fetch_rss_feeds() -> list[NewsItem]:
    """Fetch news from RSS feeds (Electrek, Reuters, CNBC). Graceful fallback."""
    items = []

    feeds = {
        "Electrek": "https://electrek.co/feed/",
        "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    }

    try:
        import feedparser
    except ImportError:
        logger.info("feedparser not installed — RSS feeds disabled")
        return items

    for source_name, feed_url in feeds.items():
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo and not feed.entries:
                logger.debug(f"RSS feed {source_name} returned no entries")
                continue

            for entry in feed.entries[:15]:  # Cap per source
                title = entry.get("title", "")
                if not title:
                    continue

                # Parse published date
                pub_date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                    except Exception:
                        pub_date = entry.get("published", "")
                elif hasattr(entry, "published"):
                    pub_date = entry.published

                summary = entry.get("summary", "")
                # Strip HTML tags from summary
                summary = re.sub(r'<[^>]+>', '', summary)[:300]

                news_item = NewsItem(
                    title=title,
                    source=source_name,
                    published=pub_date,
                    url=entry.get("link", ""),
                    summary=summary,
                )
                items.append(news_item)

        except Exception as e:
            logger.warning(f"RSS feed {source_name} failed: {e}")
            continue

    return items


def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    """Remove duplicate headlines using word overlap similarity."""
    if not items:
        return items

    unique = []
    seen_hashes = set()

    for item in items:
        h = _headline_hash(item.title)
        if h in seen_hashes:
            continue

        # Check against existing unique items for fuzzy similarity
        is_dup = False
        for existing in unique:
            if _is_similar(item.title, existing.title):
                # Keep the one with higher relevance
                if item.relevance > existing.relevance:
                    unique.remove(existing)
                    unique.append(item)
                is_dup = True
                break

        if not is_dup:
            unique.append(item)
            seen_hashes.add(h)

    return unique


def fetch_news_feed(ticker: str = "TSLA") -> NewsFeed:
    """Fetch, classify, score, and deduplicate news from all sources.

    Returns a NewsFeed with all items classified and high_impact filtered.
    """
    all_items = []
    source_counts = {}

    # Source 1: yfinance expanded
    yf_items = _fetch_yfinance_news(ticker)
    source_counts["yfinance"] = len(yf_items)
    all_items.extend(yf_items)

    # Source 2: RSS feeds
    rss_items = _fetch_rss_feeds()
    source_counts["rss"] = len(rss_items)
    all_items.extend(rss_items)

    # Classify and score each item
    for item in all_items:
        text = f"{item.title} {item.summary}"
        item.catalyst_type = _classify_catalyst(text)
        item.relevance = _score_relevance(item.title, item.summary)
        item._hash = _headline_hash(item.title)

    # Deduplicate
    unique_items = _deduplicate(all_items)

    # Sort by relevance descending
    unique_items.sort(key=lambda x: x.relevance, reverse=True)

    feed = NewsFeed(
        items=unique_items,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        source_counts=source_counts,
    )

    high_count = len(feed.high_impact)
    logger.info(
        f"News feed: {len(unique_items)} items ({len(all_items)} raw, "
        f"{len(all_items) - len(unique_items)} deduped), "
        f"{high_count} high-impact"
    )

    return feed


def format_news_for_prompt(feed: NewsFeed, max_items: int = 15) -> str:
    """Format news feed for inclusion in agent prompts.

    Shows high-impact items first, then top remaining items.
    """
    if not feed.items:
        return "No recent news available."

    parts = []

    # High-impact items with full detail
    high = feed.high_impact
    if high:
        parts.append(f"HIGH-IMPACT NEWS ({len(high)} items):")
        for item in high[:8]:
            parts.append(
                f"  [{item.catalyst_type.upper()}] {item.title} "
                f"(relevance={item.relevance:.2f}, source={item.source})"
            )
            if item.summary:
                parts.append(f"    {item.summary[:150]}")
    else:
        parts.append("No high-impact news detected.")

    # Remaining notable items
    remaining = [i for i in feed.items if i.relevance <= 0.7 and i.relevance >= 0.3]
    if remaining:
        parts.append(f"\nOTHER NOTABLE ({len(remaining)} items):")
        for item in remaining[:max_items - len(high)]:
            parts.append(
                f"  [{item.catalyst_type}] {item.title} "
                f"(relevance={item.relevance:.2f})"
            )

    return "\n".join(parts)
