"""LLM providers behind one interface: ``structured(system, user, schema)``.

* ``GroqLLM``   – hosted Llama models on Groq (free tier).
* ``OllamaLLM`` – local models through Ollama.
* ``MockLLM``   – deterministic offline stand-in for tests, CI and demos.

All providers return validated Pydantic objects.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
PAYLOAD_MARKER = "INPUT_JSON:"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...


class _LangChainLLM:
    name = "langchain"
    max_attempts = 3

    def _model(self):  # pragma: no cover
        raise NotImplementedError

    def _bind(self, schema):
        return self._model().with_structured_output(schema)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        runnable = self._bind(schema)
        prompt, last_error = user, None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = runnable.invoke([("system", system), ("human", prompt)])
                if isinstance(result, dict):
                    result = schema.model_validate(result)
                if result is None:
                    raise LLMError("model returned no structured output")
                return result
            except (ValidationError, LLMError, ValueError) as exc:
                last_error = exc
                log.warning("%s attempt %d invalid output: %s", self.name, attempt, exc)
                prompt = (f"{user}\n\nYour previous answer was invalid: {str(exc)[:600]}\n"
                          "Return output that matches the schema exactly.")
            except Exception as exc:
                if "tool_use_failed" in str(exc) and attempt < self.max_attempts:
                    last_error = exc
                    continue
                raise LLMError(f"{self.name} request failed: {exc}") from exc
        raise LLMError(f"{self.name} gave no valid output after {self.max_attempts} attempts: {last_error}")


class GroqLLM(_LangChainLLM):
    def __init__(self, settings: Settings):
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env or choose another provider.")
        self.settings, self.name = settings, f"groq:{settings.groq_model}"

    def _model(self):
        from langchain_groq import ChatGroq

        return ChatGroq(api_key=self.settings.groq_api_key, model=self.settings.groq_model,
                        temperature=self.settings.llm_temperature, max_retries=2)


class OllamaLLM(_LangChainLLM):
    def __init__(self, settings: Settings):
        self.settings, self.name = settings, f"ollama:{settings.ollama_model}"

    def _model(self):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=self.settings.ollama_model, base_url=self.settings.ollama_base_url,
                          temperature=self.settings.llm_temperature)

    def _bind(self, schema):
        return self._model().with_structured_output(schema, method="json_schema")


# ---------------------------------------------------------------------- mock
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class MockLLM:
    """Extractive, deterministic stand-in.

    Like real models it is imperfect on purpose: its first briefing draft
    states a single-source claim as plain fact (no hedge) so the critic ->
    revision loop is exercised in tests and demos.
    """

    name = "mock"

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        if PAYLOAD_MARKER not in user:
            raise LLMError("MockLLM needs an INPUT_JSON payload")
        payload = json.loads(user.split(PAYLOAD_MARKER, 1)[1])
        handler = {"SearchPlan": _plan, "AnalystOutput": _analyse, "Briefing": _write}.get(schema.__name__)
        if handler is None:
            raise LLMError(f"MockLLM has no handler for {schema.__name__}")
        return schema.model_validate(handler(payload))


def _plan(p: dict) -> dict:
    topic = p["topic"].strip()
    return {"queries": [topic, f"{topic} latest news", f"{topic} policy", f"{topic} market"][: p.get("max_queries", 4)],
            "focus": f"Most important recent developments about {topic}, with multi-source confirmation."}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9,.$%]*", text.lower()) if len(w) > 3}


def _analyse(p: dict) -> dict:
    stories = []
    for story in p["stories"]:
        arts = story["articles"]
        lead = arts[0]
        claims = []
        for sentence in _SENTENCE.split(lead["summary"])[:2]:
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue
            words = _words(sentence)
            cited = [a["id"] for a in arts if len(words & _words(a["title"] + " " + a["summary"])) >= 0.5 * len(words)]
            claims.append({"text": sentence, "article_ids": cited or [lead["id"]]})
        stories.append({"story_id": story["story_id"], "claims": claims[:3]})
    return {"stories": stories}


def _write(p: dict) -> dict:
    usable = [c for c in p["claims"] if c["status"] != "unsupported"]
    first_per_story: dict[int, dict] = {}
    for c in usable:
        first_per_story.setdefault(c["story_id"], c)  # one bullet per story
    claims = sorted(first_per_story.values(), key=lambda c: (c["status"] != "corroborated", c["story_id"]))
    revising = bool(p.get("critic_feedback"))
    bullets = []
    for c in claims[:10]:
        text = c["text"].rstrip(".")
        if c["status"] == "single_source" and revising:
            text = f"Reportedly (single source): {text[0].lower()}{text[1:]}"
        bullets.append({"text": text + ".", "claim_ids": [c["claim_id"]]})
    topic = p["topic"]
    return {"headline": f"{topic.title()}: {len(bullets)} key developments", "bullets": bullets}


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "groq":
        return GroqLLM(settings)
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings)
    if settings.llm_provider == "mock":
        return MockLLM()
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use mock, groq or ollama.")
