"""Typed contracts between agents. LLM output is validated against these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Article(BaseModel):
    id: str
    title: str
    summary: str = ""
    url: str
    publisher: str
    origin: Literal["google_news", "hackernews", "reddit", "sample", "other"] = "other"
    published: str = ""  # ISO-8601
    story_id: str | None = None  # human label (sample corpus only)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


class SearchPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6, description="Short web-news search queries")
    focus: str = Field(max_length=300, description="One sentence on what the briefing should cover")


class Story(BaseModel):
    story_id: int
    headline: str
    article_ids: list[str]
    publishers: list[str]
    latest: str


class ExtractedClaim(BaseModel):
    text: str = Field(min_length=8, max_length=300, description="One factual, self-contained statement")
    article_ids: list[str] = Field(min_length=1, max_length=6, description="IDs of articles stating this claim")


class StoryClaims(BaseModel):
    story_id: int = Field(description="story_id copied from the input")
    claims: list[ExtractedClaim] = Field(max_length=4)


class AnalystOutput(BaseModel):
    stories: list[StoryClaims]


class VerifiedClaim(BaseModel):
    claim_id: str
    story_id: int
    text: str
    cited_article_ids: list[str]
    supporting_article_ids: list[str]
    supporting_publishers: list[str]
    status: Literal["corroborated", "single_source", "unsupported"]
    reason: str


class BriefingBullet(BaseModel):
    text: str = Field(min_length=8, max_length=400)
    claim_ids: list[str] = Field(min_length=1, max_length=4, description="IDs of the verified claims this bullet uses")


class Briefing(BaseModel):
    headline: str = Field(min_length=5, max_length=150)
    bullets: list[BriefingBullet] = Field(min_length=1, max_length=10)


class CriticIssue(BaseModel):
    bullet_index: int | None
    rule: str
    detail: str
