"""NewsLens - Streamlit dashboard.   Run: streamlit run app.py"""

from __future__ import annotations

import os
from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from newslens.config import Settings
from newslens.graph import LensContext, run
from newslens.llm import LLMError, make_llm
from newslens.sources import list_samples, load_sample

st.set_page_config(page_title="NewsLens", page_icon="🔎", layout="wide")


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:
        return os.getenv(name, "")


defaults = Settings()
st.sidebar.title("⚙️ Configuration")
providers = ["mock", "groq", "ollama"]
provider = st.sidebar.selectbox("LLM", providers, index=providers.index(defaults.llm_provider)
                                if defaults.llm_provider in providers else 0,
                                help="mock = offline extractive analyst (no API key)")
groq_key = secret("GROQ_API_KEY")
if provider == "groq":
    groq_key = st.sidebar.text_input("Groq API key", value=groq_key, type="password")
mode = st.sidebar.radio("News source", ["Sample corpus (offline)", "Live news"])
sources = ("google_news", "hackernews", "reddit")
if mode == "Live news":
    sources = tuple(st.sidebar.multiselect("Sources", list(sources), default=list(sources)))
    sample = None
else:
    sample = st.sidebar.selectbox("Sample", list_samples())
threshold = st.sidebar.slider("Story clustering threshold (cosine distance)", 0.5, 0.95,
                              defaults.cluster_distance_threshold, 0.05)
settings = replace(defaults, llm_provider=provider, groq_api_key=groq_key, cluster_distance_threshold=threshold)

st.title("🔎 NewsLens")
st.caption("Multi-agent news analyst: plan → collect → cluster → extract claims → verify against sources → "
           "write → critique. Every bullet is cited; single-source claims are flagged.")

if mode == "Live news":
    topic = st.text_input("Topic", placeholder="e.g. electric two-wheelers India")
else:
    topic = load_sample(sample)[0]
    st.info("Sample corpus is **fictional** (invented companies and publishers) and works fully offline.")

if st.button("🚀 Build briefing", type="primary", disabled=not topic):
    try:
        llm = make_llm(settings)
        articles = load_sample(sample)[1] if sample else None
        with st.spinner("Agents researching..."):
            st.session_state["state"] = run(topic, LensContext(settings, llm, sample_articles=articles,
                                                               sources=sources))
    except LLMError as exc:
        st.error(str(exc))

state = st.session_state.get("state")
if not state:
    st.stop()
if state.get("error"):
    st.error(state["error"])
    st.stop()

claims = pd.DataFrame(state["claims"])
stories = pd.DataFrame(state["stories"])
k = st.columns(5)
k[0].metric("Articles", len(state["articles"]))
k[1].metric("Stories", len(stories))
k[2].metric("Corroborated claims", int((claims["status"] == "corroborated").sum()))
k[3].metric("Unsupported (blocked)", int((claims["status"] == "unsupported").sum()))
k[4].metric("Writer revisions", state.get("revisions", 0))

brief_tab, claims_tab, stories_tab, audit_tab = st.tabs(["📰 Briefing", "✅ Claims", "🧩 Stories", "Audit trail"])
with brief_tab:
    st.markdown(state["report_md"])
    st.download_button("Download briefing (.md)", state["report_md"], "briefing.md", "text/markdown")
    if state.get("source_errors"):
        with st.expander(f"{len(state['source_errors'])} source errors"):
            st.write("\n".join(f"- {e}" for e in state["source_errors"]))
with claims_tab:
    st.dataframe(claims[["claim_id", "story_id", "status", "text", "supporting_publishers", "reason"]],
                 hide_index=True, width="stretch")
    st.plotly_chart(px.histogram(claims, x="status", color="status", title="Verification outcome"),
                    width="stretch")
with stories_tab:
    stories["n_publishers"] = stories["publishers"].apply(len)
    st.dataframe(stories[["story_id", "headline", "n_publishers", "publishers", "latest"]], hide_index=True,
                 width="stretch")
with audit_tab:
    st.dataframe(pd.DataFrame(state["audit"]), hide_index=True, width="stretch")
