"""Critic Agent (deterministic): checks the Writer's briefing against verified claims."""

from __future__ import annotations

import re

from ..schemas import Briefing, CriticIssue, VerifiedClaim
from .verifier import numbers

HEDGES = ("reportedly", "single source", "according to", "unconfirmed", "claims", "rumour", "rumor", "alleged")


def review(briefing: Briefing, claims: list[VerifiedClaim]) -> list[CriticIssue]:
    by_id = {c.claim_id: c for c in claims}
    issues: list[CriticIssue] = []
    for i, bullet in enumerate(briefing.bullets):
        used = [by_id.get(cid) for cid in bullet.claim_ids]
        unknown = [cid for cid, c in zip(bullet.claim_ids, used, strict=True) if c is None]
        if unknown:
            issues.append(CriticIssue(bullet_index=i, rule="unknown_claim", detail=f"unknown claim ids {unknown}"))
        known = [c for c in used if c is not None]
        bad = [c.claim_id for c in known if c.status == "unsupported"]
        if bad:
            issues.append(CriticIssue(bullet_index=i, rule="unsupported_claim",
                                      detail=f"uses unsupported claims {bad}"))
        if not known:
            continue
        extra = numbers(bullet.text) - set().union(*(numbers(c.text) for c in known))
        if extra:
            issues.append(CriticIssue(bullet_index=i, rule="unsupported_number",
                                      detail=f"numbers {sorted(extra)} are not in the cited claims"))
        if all(c.status == "single_source" for c in known) and not any(h in bullet.text.lower() for h in HEDGES):
            issues.append(CriticIssue(bullet_index=i, rule="unhedged_single_source",
                                      detail="single-source claim stated as fact; hedge it (e.g. 'Reportedly')"))
    texts = [re.sub(r"\W+", " ", b.text.lower()).strip() for b in briefing.bullets]
    for i, t in enumerate(texts):
        if t in texts[:i]:
            issues.append(CriticIssue(bullet_index=i, rule="duplicate_bullet", detail="repeats an earlier bullet"))
    return issues


def drop_failing_bullets(briefing: Briefing, issues: list[CriticIssue]) -> Briefing:
    """Fail-safe after max revisions: remove every bullet that still has an issue."""
    bad = {x.bullet_index for x in issues if x.bullet_index is not None}
    kept = [b for i, b in enumerate(briefing.bullets) if i not in bad]
    return Briefing.model_construct(headline=briefing.headline, bullets=kept)
