import feedparser
import logging
from typing import List, Dict
from src.vector_store import store_news, retrieve_news

logger = logging.getLogger(__name__)

# Trusted health news RSS feeds
RSS_FEEDS = [
    {"name": "WHO News", "url": "https://www.who.int/rss-feeds/news-english.xml"},
    {"name": "Harvard Health", "url": "https://www.health.harvard.edu/blog/feed"},
    {"name": "Medical News Today", "url": "https://www.medicalnewstoday.com/newsfeeds/rss/nutrition.xml"},
    {"name": "NHS Health News", "url": "https://www.nhs.uk/rss/live-well"},
]


def fetch_health_news(max_per_feed: int = 3) -> List[Dict[str, str]]:
    """
    Fetch latest health/nutrition news from trusted RSS feeds.
    Returns raw article entries (title, summary, link, source).
    """
    articles = []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:max_per_feed]:
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", "")),
                    "link": entry.get("link", ""),
                    "source": feed_info["name"],
                })
        except Exception as e:
            logger.warning(f"Failed to fetch from {feed_info['name']}: {e}")
    return articles


def process_and_store_news(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Store fetched news articles into ChromaDB for future retrieval.
    Returns the list of stored articles with metadata.
    """
    stored = []
    for article in articles:
        # Use the summary as the health tip if short enough, otherwise truncate
        health_tip = article["summary"][:300] if article["summary"] else article["title"]
        store_news(
            headline=article["title"],
            summary=article["summary"][:500],
            health_tip=health_tip,
            source=article["source"],
        )
        stored.append(article)
    return stored


def get_relevant_health_tips(query: str, n_results: int = 3) -> List[Dict]:
    """Retrieve health tips relevant to the given ingredients/context."""
    return retrieve_news(query=query, n_results=n_results)
