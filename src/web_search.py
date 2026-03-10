"""Web search via Brave Search API — free tier, 1 query/second, 2000/month."""

import os
import httpx
from .utils import setup_logging

logger = setup_logging("web_search")

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def brave_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Brave and return simplified results.

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
