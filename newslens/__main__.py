"""Command line: python -m newslens "electric two-wheelers" [--sample ev_two_wheelers] [--out briefing.md]"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .graph import LensContext, run
from .llm import LLMError, make_llm
from .sources import list_samples, load_sample


def main() -> int:
    parser = argparse.ArgumentParser(prog="newslens", description="Multi-agent, source-verified news briefings.")
    parser.add_argument("topic", nargs="?", help="Topic to research (ignored with --sample)")
    parser.add_argument("--sample", choices=list_samples(), help="Use an offline sample corpus instead of live news")
    parser.add_argument("--out", help="Write the Markdown briefing to this file")
    args = parser.parse_args()
    settings = Settings()
    try:
        llm = make_llm(settings)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.sample:
        topic, articles = load_sample(args.sample)
        ctx = LensContext(settings, llm, sample_articles=articles)
    elif args.topic:
        topic, ctx = args.topic, LensContext(settings, llm)
    else:
        parser.error("give a topic or --sample")
    state = run(topic, ctx)
    if state.get("error"):
        print(f"error: {state['error']}", file=sys.stderr)
        return 1
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(state["report_md"])
        print(f"Briefing written to {args.out}")
    else:
        print(state["report_md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
