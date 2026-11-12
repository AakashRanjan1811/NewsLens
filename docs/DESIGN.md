# Design decisions & trade-offs

Each section answers a question an interviewer is likely to ask.

## 1. Why split work between LLM agents and deterministic agents?

LLMs are good at language (turning a topic into queries, pulling claims out of prose, writing clearly) and bad at guarantees. Anything that must be *true every time* — "is this number in the source?", "is this bullet hedged?" — is plain Python. The LLM proposes; rules verify. A misbehaving model can make the briefing shorter, never wrong in the ways we check.

## 2. Why a fixed pipeline with one loop instead of a free-form agent?

The task has a natural order (plan → collect → cluster → extract → verify → write). A fixed LangGraph pipeline is predictable, cheap (3 LLM calls per run + revisions) and testable. The one place iteration helps — fixing the writer's mistakes — is an explicit writer ⇄ critic loop with a revision cap. LangGraph gives shared typed state, conditional edges and an audit trail without hand-rolled control flow.

## 3. How does claim verification work?

An article supports a claim only if:
1. **Numbers:** every number in the claim appears in the article (after removing thousands separators). Catches "3,200" → "4,800".
2. **Names:** every capitalised mid-sentence token is in the article. A sentence-initial capital might be an ordinary word ("Charging startup…"), so it is checked leniently (its stem as a substring). Catches swapped companies and places.
3. **Content words:** ≥60% of the claim's content words (stop-words removed, light stemming) are in the article **and** at most 2 are missing. The second rule catches an invented clause tacked onto a true sentence ("…after its CEO resigned").

The verifier checks the cited articles first and then the rest of the story, because LLMs often under-cite. Corroboration counts **distinct publishers** that support the claim, not articles.

**Why lexical and not an NLI model?** It is fast, explainable ("numbers not in source: ['4800']"), needs no model download, and is strong on the most damaging errors (numbers and names). Its blind spots — heavy paraphrase and misleading-but-literal claims — are documented. The upgrade path is an entailment model (e.g. a DeBERTa NLI cross-encoder) as an additional check.

## 4. How are rumours handled?

Single-source claims are not deleted — sometimes one outlet has the scoop — but the critic forces the writer to hedge them ("Reportedly (single source): …") and the renderer marks them ⚠️. Corroborated claims get ✅.

## 5. Why TF-IDF clustering instead of embeddings?

Same-story news articles share rare words (company names, figures), which TF-IDF weights highly. It needs no model download, runs in milliseconds and works in CI. Agglomerative clustering with a distance threshold means the number of stories is not fixed in advance. Sentence embeddings would help with heavily paraphrased coverage and are a drop-in replacement for `cluster_labels`.

## 6. What if a source is down or returns garbage?

Each source/query is fetched independently; failures are collected in `source_errors` and shown in the UI. Parsers skip malformed items (missing title/link, bad dates). Only if *no* articles arrive does the run stop, with the errors shown.

## 7. How do you evaluate a system like this?

Three layers, all in `eval/evaluate.py` and run in CI:
- **Component:** clustering vs human story labels (ARI, homogeneity, completeness).
- **Adversarial:** hand-written paraphrased true claims vs programmatically generated fabrications (number, entity, invented fact) → precision/recall of the verifier.
- **End-to-end:** inject fabricated claims into the analyst's output and count how many reach the published briefing.

While building the eval I found a bug in my own fabrication generator (it turned "V2" into "V2" and produced a "fake" identical to the truth). Lesson: validate the evaluation harness itself; the fix is an assertion that every fabrication differs from its source claim.

## 8. Prompt injection?

Article text is untrusted and every prompt says so, but the real defence is structural: the writer can only cite claim IDs, every claim is re-verified against source text, and the critic re-checks every bullet. An article that says "ignore previous instructions" cannot get an unsupported sentence published.

## 9. What would change in production?

- Fetch full article text (with robots.txt respect and caching), not just RSS snippets.
- Add an NLI entailment check and source credibility weights (wire services vs forums).
- Scheduled runs with diffing ("what changed since yesterday").
- Persistent storage of articles and claims; LangSmith/OpenTelemetry tracing.

## 10. Key numbers

- 8 agents/steps, 3 LLM calls per run (+ revisions), 34 tests.
- Clustering ARI 1.0 on the labelled corpus (stable for thresholds 0.7–0.9).
- Verifier: 34/34 fabrications caught, 0/14 true claims rejected; 0/9 injected fabrications published.
