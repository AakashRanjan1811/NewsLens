from pathlib import Path

from sklearn.metrics import adjusted_rand_score

from newslens.clustering import build_stories, cluster_labels
from newslens.schemas import Article
from newslens.sources import dedupe, google_news_url, parse_google_news_rss, parse_hackernews, parse_reddit

FIXTURES = Path(__file__).parent / "fixtures"


def test_google_news_rss_parsing():
    arts = parse_google_news_rss((FIXTURES / "google_news.xml").read_text(), limit=10)
    assert [a.title for a in arts] == ["Voltara launches V2 scooter", "Subsidy cut announced"]  # suffix stripped
    assert arts[0].publisher == "Daily Ledger" and arts[0].origin == "google_news"
    assert arts[0].published.startswith("2026-11-10T09:15")
    assert "<" not in arts[0].summary and "&nbsp;" not in arts[0].summary  # HTML cleaned
    assert arts[1].published == ""  # bad date tolerated
    assert len(parse_google_news_rss((FIXTURES / "google_news.xml").read_text(), limit=1)) == 1


def test_hackernews_and_reddit_parsing():
    hn = parse_hackernews({"hits": [
        {"title": "ChargeGrid raises $40M", "url": "https://x.example/1", "created_at": "2026-11-07T10:00:00Z"},
        {"title": "Ask HN: EV scooters?", "url": None, "objectID": "42"},
        {"title": None, "url": "https://x.example/skip"},
    ]}, limit=10)
    assert [a.url for a in hn] == ["https://x.example/1", "https://news.ycombinator.com/item?id=42"]
    rd = parse_reddit({"data": {"children": [
        {"data": {"title": "Zephyr recall", "permalink": "/r/IndianEVs/comments/abc/", "subreddit": "IndianEVs",
                  "selftext": "check your VIN", "created_utc": 1.7e9}},
        {"data": {"title": "no link"}},
    ]}}, limit=10)
    assert len(rd) == 1 and rd[0].publisher == "r/IndianEVs" and rd[0].url.startswith("https://www.reddit.com/r/")


def test_google_news_url_encodes_query():
    assert "q=electric+two-wheelers" in google_news_url("electric two-wheelers", "IN")


def test_dedupe_by_url_and_same_publisher_title():
    a = Article(id="1", title="Same title!", url="u1", publisher="P")
    b = Article(id="2", title="same  title", url="u2", publisher="P")
    c = Article(id="3", title="Same title", url="u3", publisher="Q")  # other publisher: keep
    d = Article(id="4", title="Different", url="u1", publisher="R")  # repeated URL: drop
    assert [x.id for x in dedupe([a, b, c, d])] == ["1", "3"]


def test_clustering_matches_human_labels(articles, settings):
    pred = cluster_labels(articles, settings.cluster_distance_threshold)
    assert adjusted_rand_score([a.story_id for a in articles], pred) > 0.9


def test_stories_ranked_by_publisher_count(articles, settings):
    stories = build_stories(articles, settings.cluster_distance_threshold, max_stories=5)
    assert len(stories) == 5 and [s.story_id for s in stories] == [1, 2, 3, 4, 5]
    counts = [len(s.publishers) for s in stories]
    assert counts == sorted(counts, reverse=True)


def test_single_article_and_empty_input():
    one = Article(id="1", title="Only story", url="u", publisher="P")
    assert len(build_stories([one], 0.8, 5)) == 1
    assert build_stories([], 0.8, 5) == []
