"""Story clustering: group articles that report the same event.

TF-IDF (word uni+bi-grams, sublinear tf) + agglomerative clustering with
average linkage on cosine distance. The number of stories is not fixed in
advance: a distance threshold decides when two articles are "the same story".
Lightweight by design (no model download), so it runs anywhere, including CI.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import Article, Story


def cluster_labels(articles: list[Article], distance_threshold: float) -> np.ndarray:
    if len(articles) == 1:
        return np.zeros(1, dtype=int)
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                        token_pattern=r"(?u)\b[\w][\w,.]*\w\b|\b\w\b").fit_transform([a.text for a in articles])
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                    distance_threshold=distance_threshold)
    return model.fit_predict(X.toarray())


def build_stories(articles: list[Article], distance_threshold: float, max_stories: int) -> list[Story]:
    """Return stories ranked by number of distinct publishers, then recency."""
    if not articles:
        return []
    labels = cluster_labels(articles, distance_threshold)
    groups: dict[int, list[Article]] = {}
    for label, art in zip(labels, articles, strict=True):
        groups.setdefault(int(label), []).append(art)
    stories = []
    for members in groups.values():
        members.sort(key=lambda a: a.published)
        publishers = sorted({a.publisher for a in members})
        stories.append(Story(story_id=0, headline=members[0].title, article_ids=[a.id for a in members],
                             publishers=publishers, latest=members[-1].published))
    stories.sort(key=lambda s: (len(s.publishers), s.latest), reverse=True)
    stories = stories[:max_stories]
    for i, s in enumerate(stories, start=1):
        s.story_id = i
    return stories
