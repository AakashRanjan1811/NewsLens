from dataclasses import replace

import pytest
from langgraph.graph import END
from pydantic import BaseModel

import newslens.graph as graph_mod
from eval.evaluate import InjectingLLM
from newslens.graph import LensContext, after_critic, run
from newslens.llm import GroqLLM, LLMError, MockLLM, _LangChainLLM, make_llm
from newslens.schemas import Briefing
from newslens.sources import CollectResult


def test_routing_after_critic():
    assert after_critic({"issues": [{}], "revisions": 0}, 2) == "writer"
    assert after_critic({"issues": [{}], "revisions": 2}, 2) == "finalize"
    assert after_critic({"issues": [], "revisions": 0}, 2) == "finalize"
    assert after_critic({"error": "x"}, 2) == END


def test_full_run_on_sample(sample, settings):
    topic, articles = sample
    state = run(topic, LensContext(settings, MockLLM(), sample_articles=articles))
    assert not state.get("error")
    assert state["revisions"] == 1  # critic caught unhedged single-source bullets once
    assert "## Sources" in state["report_md"]
    claim_status = {c["claim_id"]: c["status"] for c in state["claims"]}
    for b in state["briefing"]["bullets"]:
        assert b["claim_ids"] and all(claim_status[c] != "unsupported" for c in b["claim_ids"])
    agents = {a["agent"] for a in state["audit"]}
    assert {"planner", "collector", "clusterer", "analyst", "verifier", "writer", "critic", "finalize"} <= agents


def test_injected_fabrications_never_reach_the_briefing(sample, settings):
    topic, articles = sample
    llm = InjectingLLM()
    state = run(topic, LensContext(settings, llm, sample_articles=articles))
    published = " ".join(b["text"] for b in state["briefing"]["bullets"])
    assert llm.injected and not [f for f in llm.injected if f.rstrip(".") in published]


def test_stubborn_writer_is_cut_back_after_max_revisions(sample, settings):
    class Stubborn(MockLLM):
        def structured(self, system, user, schema):  # never hedges
            return super().structured(system, user.replace('"critic_feedback": [{', '"ignored": [{'), schema)

    topic, articles = sample
    state = run(topic, LensContext(replace(settings, max_revisions=2), Stubborn(), sample_articles=articles))
    assert state["revisions"] == 2
    assert all("single source" not in b["text"].lower() for b in state["briefing"]["bullets"])
    assert any("dropped 2" in a["detail"] for a in state["audit"] if a["agent"] == "finalize")


def test_llm_error_stops_cleanly(sample, settings):
    class Broken:
        name = "broken"

        def structured(self, *a):
            raise LLMError("rate limited")

    state = run("x", LensContext(settings, Broken(), sample_articles=sample[1]))
    assert "rate limited" in state["error"] and "report_md" not in state


def test_live_mode_with_no_articles_reports_source_errors(monkeypatch, settings):
    monkeypatch.setattr(graph_mod, "collect_live", lambda q, s, src: CollectResult([], ["google_news: timeout"]))
    state = run("anything", LensContext(settings, MockLLM()))
    assert "No articles" in state["error"] and state["source_errors"] == ["google_news: timeout"]


def test_make_llm_validation(settings):
    assert isinstance(make_llm(settings), MockLLM)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqLLM(replace(settings, llm_provider="groq"))
    with pytest.raises(LLMError, match="Unknown"):
        make_llm(replace(settings, llm_provider="gpt-9"))


class Out(BaseModel):
    value: int


class Fake(_LangChainLLM):
    name = "fake"

    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    def _bind(self, schema):
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1][1])
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_retry_on_invalid_output_and_tool_use_failure():
    f = Fake([{"value": "nope"}, RuntimeError("400 tool_use_failed"), {"value": 7}])
    assert f.structured("s", "u", Out).value == 7 and "invalid" in f.prompts[1]
    with pytest.raises(LLMError, match="request failed"):
        Fake([RuntimeError("401 bad key")]).structured("s", "u", Out)


def test_briefing_schema_rejects_empty_bullets():
    with pytest.raises(ValueError):
        Briefing(headline="Hello world", bullets=[])
