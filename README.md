# NewsLens

[![CI](https://github.com/AakashRanjan1811/NewsLens/actions/workflows/ci.yml/badge.svg)](https://github.com/AakashRanjan1811/NewsLens/actions/workflows/ci.yml)

A multi-agent tool that makes a short news briefing on a topic. It collects articles from Google News, Hacker News and Reddit, groups them into stories, pulls out the main claims, checks each claim against the source articles, and writes a summary where every point has a source. Claims that only one source reports are marked.

I made this because LLM news summaries often get numbers or names wrong, or repeat a single Reddit post as if it were confirmed. Here the LLM does the reading and writing, and plain Python code does the checking.

## How it works

```
topic
  -> planner     (LLM: makes 3-4 search queries)
  -> collector   (Google News RSS, Hacker News API, Reddit search)
  -> clusterer   (TF-IDF + agglomerative clustering, groups articles into stories)
  -> analyst     (LLM: up to 3 claims per story)
  -> verifier    (checks every claim against the article text)
  -> writer      (LLM: bullet-point briefing using only verified claims)
  -> critic      (checks the bullets; sends them back to the writer if needed, max 2 times)
  -> final briefing with sources
```

The verifier accepts a claim only if the numbers and names in it appear in the article, and almost all of its other words do too. Each claim is marked as:
- corroborated: found in 2 or more publishers
- single source: found in only one
- unsupported: not found, so it is not used

The critic rejects bullets that use unsupported claims, add numbers that are not in the claims, repeat each other, or state a single-source claim without saying so.

The LLM can be Groq, a local Ollama model, or an offline mock (default).

## Evaluation

`python -m eval.evaluate` runs on a small sample set of 22 articles about 9 stories. The sample data is fictional (made-up companies and news sites) so it works offline.

- Clustering: matched the hand-labelled stories exactly (ARI = 1.0) for distance thresholds from 0.7 to 0.9.
- Fake claim detection: I wrote 14 true claims (reworded) and generated 34 fake versions by changing a number, swapping a company or place name, or adding a made-up detail. The verifier caught all 34 and accepted all 14 true ones.
- End-to-end: when the analyst is made to add one fake claim per story, none of the 9 fake claims end up in the final briefing.

Limitations:
- The test set is small and I wrote it myself, so it is cleaner than real news. Expect worse results on live data. The clustering threshold and the "max 2 unmatched words" rule were tuned on this set.
- The verifier only compares words. It can reject a correct claim that is reworded a lot, and it cannot catch a claim that uses the article's exact words in a misleading way.
- In live mode it only uses article titles and RSS summaries, not the full article.

## Setup

```bash
git clone https://github.com/AakashRanjan1811/NewsLens.git
cd NewsLens
python -m venv .venv
.venv\Scripts\activate        # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # on Mac/Linux: cp .env.example .env
streamlit run app.py
```

The app starts with the offline sample. To use live news, choose "Live news" in the sidebar. For a real LLM, get a free Groq key from https://console.groq.com/keys and set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env`, or use Ollama with `LLM_PROVIDER=ollama`.

Command line:

```bash
python -m newslens --sample ev_two_wheelers
python -m newslens "electric two-wheelers India" --out briefing.md
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

34 tests, all offline: the RSS/HN/Reddit parsers, clustering, each verifier and critic rule, the full pipeline, fake claim injection, LLM errors, the CLI and the Streamlit app.

## Project structure

```
newslens/        sources, clustering, agents, LLM wrappers, LangGraph pipeline
app.py           Streamlit app
eval/            evaluation script
data/samples/    sample articles (fictional)
tests/           pytest tests
docs/DESIGN.md   notes on the design choices
```

## License

MIT
