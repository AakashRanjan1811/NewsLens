"""Runtime configuration from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _i(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock").lower())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.1))

    # Collection
    max_articles_per_query: int = field(default_factory=lambda: _i("MAX_ARTICLES_PER_QUERY", 15))
    max_queries: int = 4
    request_timeout: float = 10.0
    region: str = field(default_factory=lambda: os.getenv("NEWS_REGION", "IN"))

    # Clustering: cosine distance below which two articles are the same story
    cluster_distance_threshold: float = field(default_factory=lambda: _f("CLUSTER_DISTANCE_THRESHOLD", 0.8))
    max_stories: int = 10

    # Verification: share of a claim's content words that must appear in the source
    support_overlap: float = field(default_factory=lambda: _f("SUPPORT_OVERLAP", 0.6))
    min_corroborating_publishers: int = 2
    max_revisions: int = field(default_factory=lambda: _i("MAX_REVISIONS", 2))

    def describe_llm(self) -> str:
        return {"groq": f"Groq · {self.groq_model}", "ollama": f"Ollama · {self.ollama_model}"}.get(
            self.llm_provider, "Mock analyst (offline)")
