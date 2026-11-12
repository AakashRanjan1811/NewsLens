"""LangGraph pipeline.

    planner -> collector -> clusterer -> analyst -> verifier -> writer -> critic
                                                                  ^         |
                                                                  └─ revise ┘  (at most N times)
                                                                            |
                                                                       finalize -> END

LLM agents: planner, analyst, writer. Deterministic agents: collector,
clusterer, verifier, critic, finalize. Every node appends to an audit trail.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.critic import drop_failing_bullets, review
from .agents.llm_agents import run_analyst, run_planner, run_writer
from .agents.verifier import verify_claims
from .clustering import build_stories
from .config import Settings
from .llm import LLMClient, LLMError
from .schemas import Article, Briefing, CriticIssue, Story, VerifiedClaim
from .sources import collect_live


class LensState(TypedDict, total=False):
    topic: str
    queries: list[str]
    focus: str
    articles: list[dict]
    source_errors: list[str]
    stories: list[dict]
    claims: list[dict]
    briefing: dict | None
    issues: list[dict]
    revisions: int
    report_md: str
    error: str | None
    audit: Annotated[list[dict], operator.add]


@dataclass
class LensContext:
    settings: Settings
    llm: LLMClient
    sample_articles: list[Article] | None = None  # offline mode: skip live collection
    sources: tuple[str, ...] = ("google_news", "hackernews", "reddit")


def _audit(agent: str, event: str, detail: str = "") -> list[dict]:
    return [{"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agent": agent,
             "event": event, "detail": detail}]


def after_critic(state: LensState, max_revisions: int) -> str:
    if state.get("error"):
        return END
    if state.get("issues") and state.get("revisions", 0) < max_revisions:
        return "writer"
    return "finalize"


def _guard(fn):
    """Stop the graph cleanly if an earlier node failed or an LLM call errors."""
    def wrapper(state: LensState) -> dict:
        if state.get("error"):
            return {}
        try:
            return fn(state)
        except LLMError as exc:
            return {"error": str(exc), "audit": _audit(fn.__name__, "LLM error", str(exc))}
    wrapper.__name__ = fn.__name__
    return wrapper


def render_markdown(topic: str, briefing: Briefing, claims: list[VerifiedClaim], stories: list[Story],
                    articles: dict[str, Article]) -> str:
    by_id = {c.claim_id: c for c in claims}
    refs: dict[str, int] = {}
    lines = [f"# {briefing.headline}", ""]
    for b in briefing.bullets:
        cites = []
        for cid in b.claim_ids:
            for a_id in by_id[cid].supporting_article_ids:
                refs.setdefault(a_id, len(refs) + 1)
                cites.append(refs[a_id])
        badge = "✅" if any(by_id[c].status == "corroborated" for c in b.claim_ids) else "⚠️"
        lines.append(f"- {badge} {b.text} " + "".join(f"[{n}]" for n in sorted(set(cites))))
    lines += ["", "✅ confirmed by 2+ publishers · ⚠️ single source", "", "## Sources"]
    for a_id, n in refs.items():
        a = articles[a_id]
        lines.append(f"{n}. [{a.title}]({a.url}) — {a.publisher}")
    lines += ["", f"*{len(stories)} stories from {len(articles)} articles on '{topic}'.*"]
    return "\n".join(lines)


def build_graph(ctx: LensContext):
    s = ctx.settings

    @_guard
    def planner(state: LensState) -> dict:
        plan = run_planner(ctx.llm, state["topic"], s)
        return {"queries": plan.queries, "focus": plan.focus,
                "audit": _audit("planner", f"{len(plan.queries)} queries", " | ".join(plan.queries))}

    @_guard
    def collector(state: LensState) -> dict:
        if ctx.sample_articles is not None:
            arts, errors, mode = ctx.sample_articles, [], "sample corpus"
        else:
            res = collect_live(state["queries"], s, ctx.sources)
            arts, errors, mode = res.articles, res.errors, "live"
        if not arts:
            return {"error": "No articles found. " + "; ".join(errors[:3]), "source_errors": errors,
                    "audit": _audit("collector", "no articles", "; ".join(errors[:3]))}
        origins = sorted({a.origin for a in arts})
        return {"articles": [a.model_dump() for a in arts], "source_errors": errors,
                "audit": _audit("collector", f"{len(arts)} articles ({mode})",
                                f"origins: {', '.join(origins)}; failed sources: {len(errors)}")}

    @_guard
    def clusterer(state: LensState) -> dict:
        arts = [Article(**a) for a in state["articles"]]
        stories = build_stories(arts, s.cluster_distance_threshold, s.max_stories)
        detail = "; ".join(f"#{x.story_id} {x.headline[:60]} ({len(x.publishers)} pubs)" for x in stories)
        return {"stories": [x.model_dump() for x in stories],
                "audit": _audit("clusterer", f"{len(stories)} stories", detail)}

    @_guard
    def analyst(state: LensState) -> dict:
        arts = {a["id"]: Article(**a) for a in state["articles"]}
        stories = [Story(**x) for x in state["stories"]]
        out = run_analyst(ctx.llm, stories, arts)
        claims = verify_claims(out, stories, arts, s)
        counts = {k: sum(c.status == k for c in claims) for k in ("corroborated", "single_source", "unsupported")}
        return {"claims": [c.model_dump() for c in claims],
                "audit": _audit("analyst", f"{len(claims)} claims extracted", "") +
                _audit("verifier", "claims checked against sources",
                       ", ".join(f"{k}={v}" for k, v in counts.items()))}

    @_guard
    def writer(state: LensState) -> dict:
        claims = [VerifiedClaim(**c) for c in state["claims"]]
        revising = bool(state.get("issues"))
        b = run_writer(ctx.llm, state["topic"], state.get("focus", ""), claims,
                       state.get("issues") if revising else None)
        revisions = state.get("revisions", 0) + (1 if revising else 0)
        return {"briefing": b.model_dump(), "revisions": revisions,
                "audit": _audit("writer", f"revision {revisions}" if revising else "draft", f"{len(b.bullets)} bullets")}

    @_guard
    def critic(state: LensState) -> dict:
        issues = review(Briefing(**state["briefing"]), [VerifiedClaim(**c) for c in state["claims"]])
        detail = "; ".join(f"bullet {i.bullet_index}: {i.rule}" for i in issues) or "all checks passed"
        return {"issues": [i.model_dump() for i in issues],
                "audit": _audit("critic", "revise" if issues else "approved", detail)}

    @_guard
    def finalize(state: LensState) -> dict:
        claims = [VerifiedClaim(**c) for c in state["claims"]]
        briefing = Briefing(**state["briefing"])
        issues = [CriticIssue(**i) for i in state.get("issues", [])]
        dropped = 0
        if issues:  # out of revisions: fail safe
            before = len(briefing.bullets)
            briefing = drop_failing_bullets(briefing, issues)
            dropped = before - len(briefing.bullets)
        arts = {a["id"]: Article(**a) for a in state["articles"]}
        md = render_markdown(state["topic"], briefing, claims, [Story(**x) for x in state["stories"]], arts)
        return {"briefing": briefing.model_dump(), "report_md": md,
                "audit": _audit("finalize", "briefing published", f"dropped {dropped} failing bullets")}

    g = StateGraph(LensState)
    for fn in (planner, collector, clusterer, analyst, writer, critic, finalize):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "planner")
    for a, b in [("planner", "collector"), ("collector", "clusterer"), ("clusterer", "analyst"),
                 ("analyst", "writer"), ("writer", "critic"), ("finalize", END)]:
        g.add_edge(a, b)
    g.add_conditional_edges("critic", lambda st: after_critic(st, s.max_revisions), ["writer", "finalize", END])
    return g.compile()


def run(topic: str, ctx: LensContext) -> LensState:
    graph = build_graph(ctx)
    return graph.invoke({"topic": topic, "revisions": 0, "audit": []}, {"recursion_limit": 40})
