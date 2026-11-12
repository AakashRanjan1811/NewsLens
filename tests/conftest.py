from dataclasses import replace

import pytest

from newslens.config import Settings
from newslens.sources import load_sample


@pytest.fixture(scope="session")
def sample():
    return load_sample("ev_two_wheelers")


@pytest.fixture
def articles(sample):
    return sample[1]


@pytest.fixture
def by_id(articles):
    return {a.id: a for a in articles}


@pytest.fixture
def settings():
    return replace(Settings(), llm_provider="mock", groq_api_key="", cluster_distance_threshold=0.8,
                   support_overlap=0.6, max_unmatched_words=2, max_revisions=2)
