"""Verifier Agent (deterministic): is each claim actually in the sources?

An article *supports* a claim only if
1. every number in the claim appears in the article (catches altered figures),
2. every capitalised name in the claim appears in the article (catches swapped
   companies, places, people), and
3. most of the claim's content words appear in the article (catches invented
   statements).

Claims are then labelled:
- corroborated  : supported by >= 2 distinct publishers
- single_source : supported by exactly 1 publisher
- unsupported   : no article supports it -> never reaches the briefing
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..config import Settings
from ..schemas import AnalystOutput, Article, Story, VerifiedClaim

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z][a-z0-9'-]+")
_CAP = re.compile(r"\b[A-Z][A-Za-z0-9-]{2,}\b")
_SUFFIXES = ("ing", "ed", "es", "s")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in ENGLISH_STOP_WORDS}


def names(text: str) -> tuple[set[str], set[str]]:
    """Capitalised tokens, split into (mid-sentence names, sentence-initial words).

    A sentence-initial capital may just be an ordinary word ("Charging startup ..."),
    so those are checked more leniently (stem as substring) than true mid-sentence names.
    """
    mid, initial = set(), set()
    for m in _CAP.finditer(text):
        word = m.group().lower()
        if word in ENGLISH_STOP_WORDS:
            continue
        before = text[: m.start()].rstrip()
        (initial if not before or before[-1] in ".!?:" else mid).add(word)
    return mid, initial


def support_check(claim: str, article_text: str, min_overlap: float) -> tuple[bool, str]:
    missing_nums = numbers(claim) - numbers(article_text)
    if missing_nums:
        return False, f"numbers not in source: {sorted(missing_nums)}"
    lowered = article_text.lower()
    mid, initial = names(claim)
    missing_names = {n for n in mid if n not in lowered} | {n for n in initial if _stem(n) not in lowered}
    if missing_names:
        return False, f"names not in source: {sorted(missing_names)}"
    words = content_words(claim)
    if not words:
        return False, "claim has no content words"
    overlap = len(words & content_words(article_text)) / len(words)
    if overlap < min_overlap:
        return False, f"only {overlap:.0%} of claim words in source"
    return True, f"{overlap:.0%} word overlap"


def verify_claims(output: AnalystOutput, stories: list[Story], articles: dict[str, Article],
                  settings: Settings) -> list[VerifiedClaim]:
    by_story = {s.story_id: s for s in stories}
    verified: list[VerifiedClaim] = []
    for story_claims in output.stories:
        story = by_story.get(story_claims.story_id)
        if story is None:
            continue  # invented story id: drop silently, nothing to verify against
        for claim in story_claims.claims:
            cited = [a for a in claim.article_ids if a in articles]
            # Check cited articles first, then the rest of the story (LLMs often under-cite).
            candidates = list(dict.fromkeys(cited + story.article_ids))
            supporting, reasons = [], []
            for a_id in candidates:
                ok, why = support_check(claim.text, articles[a_id].text, settings.support_overlap)
                if ok:
                    supporting.append(a_id)
                elif a_id in cited:
                    reasons.append(f"{a_id}: {why}")
            publishers = sorted({articles[a].publisher for a in supporting})
            if not supporting:
                status, reason = "unsupported", "; ".join(reasons) or "no cited article found"
            elif len(publishers) >= settings.min_corroborating_publishers:
                status, reason = "corroborated", f"{len(publishers)} publishers"
            else:
                status, reason = "single_source", f"only {publishers[0]}"
            verified.append(VerifiedClaim(
                claim_id=f"C{len(verified) + 1}", story_id=story.story_id, text=claim.text,
                cited_article_ids=claim.article_ids, supporting_article_ids=supporting,
                supporting_publishers=publishers, status=status, reason=reason))
    return verified
