"""Collector tools: Google News RSS, Hacker News (Algolia API) and Reddit search.

Each fetcher returns ``Article`` objects and never raises on a bad source: a
failed source is reported in ``CollectResult.errors`` so one outage doesn't
kill the run. Parsing is separated from HTTP so it can be unit-tested offline.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from .config import Settings
from .schemas import Article

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
USER_AGENT = "NewsLens/1.0 (research briefing tool)"
_TAG = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def _id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:10]


def _iso(rfc822: str | None) -> str:
    try:
        return parsedate_to_datetime(rfc822).isoformat() if rfc822 else ""
    except (TypeError, ValueError):
        return ""


# ------------------------------------------------------------------ parsers (pure)
def parse_google_news_rss(xml_text: str, limit: int) -> list[Article]:
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        url = (item.findtext("link") or "").strip()
        title = _clean(item.findtext("title"))
        publisher = _clean(item.findtext("source")) or "Google News"
        # Google News titles end with " - Publisher"; drop the suffix.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        if not url or not title:
            continue
        out.append(Article(id=_id(url), title=title, summary=_clean(item.findtext("description")), url=url,
                           publisher=publisher, origin="google_news", published=_iso(item.findtext("pubDate"))))
        if len(out) >= limit:
            break
    return out


def parse_hackernews(payload: dict, limit: int) -> list[Article]:
    out = []
    for hit in payload.get("hits", []):
        title = _clean(hit.get("title") or hit.get("story_title"))
        oid = hit.get("objectID")
        url = hit.get("url") or (f"https://news.ycombinator.com/item?id={oid}" if oid else "")
        if not title or not url:
            continue
        out.append(Article(id=_id(url), title=title, summary=_clean(hit.get("story_text"))[:600], url=url,
                           publisher="Hacker News", origin="hackernews", published=hit.get("created_at", "")))
        if len(out) >= limit:
            break
    return out


def parse_reddit(payload: dict, limit: int) -> list[Article]:
    out = []
    for child in payload.get("data", {}).get("children", []):
        d = child.get("data", {})
        title = _clean(d.get("title"))
        permalink = d.get("permalink")
        if not title or not permalink:
            continue
        url = f"https://www.reddit.com{permalink}"
        out.append(Article(id=_id(url), title=title, summary=_clean(d.get("selftext"))[:600], url=url,
                           publisher=f"r/{d.get('subreddit', 'reddit')}", origin="reddit",
                           published=str(d.get("created_utc", ""))))
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ HTTP
def google_news_url(query: str, region: str) -> str:
    return (f"https://news.google.com/rss/search?q={quote_plus(query)}"
            f"&hl=en-{region}&gl={region}&ceid={region}:en")


@dataclass
class CollectResult:
    articles: list[Article] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def dedupe(articles: list[Article]) -> list[Article]:
    """Drop repeated URLs and near-identical titles from the same publisher."""
    seen_url, seen_title, out = set(), set(), []
    for a in articles:
        key = (a.publisher.lower(), re.sub(r"\W+", " ", a.title.lower()).strip())
        if a.url in seen_url or key in seen_title:
            continue
        seen_url.add(a.url)
        seen_title.add(key)
        out.append(a)
    return out


def collect_live(queries: list[str], settings: Settings, sources: tuple[str, ...] = ("google_news", "hackernews",
                                                                                      "reddit")) -> CollectResult:
    result = CollectResult()
    limit = settings.max_articles_per_query
    with httpx.Client(timeout=settings.request_timeout, headers={"User-Agent": USER_AGENT},
                      follow_redirects=True) as client:
        for q in queries:
            jobs = {
                "google_news": lambda q=q: parse_google_news_rss(
                    client.get(google_news_url(q, settings.region)).raise_for_status().text, limit),
                "hackernews": lambda q=q: parse_hackernews(client.get(
                    "https://hn.algolia.com/api/v1/search", params={"query": q, "tags": "story"}
                ).raise_for_status().json(), limit),
                "reddit": lambda q=q: parse_reddit(client.get(
                    "https://www.reddit.com/search.json", params={"q": q, "sort": "relevance", "t": "week",
                                                                  "limit": limit}
                ).raise_for_status().json(), limit),
            }
            for name in sources:
                try:
                    result.articles.extend(jobs[name]())
                except Exception as exc:  # one bad source must not stop the run
                    result.errors.append(f"{name} / '{q}': {type(exc).__name__}: {str(exc)[:120]}")
    result.articles = dedupe(result.articles)
    return result


def list_samples() -> list[str]:
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.json"))


def load_sample(name: str) -> tuple[str, list[Article]]:
    payload = json.loads((SAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return payload["topic"], [Article(**a) for a in payload["articles"]]
