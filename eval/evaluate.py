"""NewsLens evaluation (offline, reproducible).

    python -m eval.evaluate

1. Story clustering vs human labels (Adjusted Rand Index, homogeneity, completeness).
2. Hallucination detection: hand-written *paraphrased* true claims plus
   fabricated variants (altered number, swapped entity, invented extra fact).
   Measures how well the verifier flags fabrications without rejecting true claims.
3. End-to-end injection test: an analyst that slips one fabricated claim into
   every story. Measures how many fabrications reach the published briefing.
"""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score

from newslens.agents.verifier import support_check
from newslens.clustering import build_stories, cluster_labels
from newslens.config import Settings
from newslens.graph import LensContext, run
from newslens.llm import MockLLM
from newslens.schemas import AnalystOutput
from newslens.sources import load_sample

RESULTS = Path(__file__).parent / "results"

# Paraphrased (not copied) true statements, keyed by the sample corpus story label.
TRUE_CLAIMS = [
    ("voltara_v2_launch", "Voltara Motors launched the V2 electric scooter priced at Rs 89,999."),
    ("voltara_v2_launch", "The Voltara V2 has a claimed range of 140 km."),
    ("voltara_v2_launch", "Deliveries of the Voltara V2 begin in January."),
    ("subsidy_cut", "The subsidy for electric two-wheelers was cut from Rs 15,000 to Rs 10,000 per vehicle."),
    ("subsidy_cut", "The revised electric two-wheeler subsidy takes effect from April 1."),
    ("zephyr_recall", "Zephyr Mobility recalled 3,200 electric scooters after battery fire reports."),
    ("zephyr_recall", "Zephyr Mobility owners get a free battery inspection."),
    ("ravipur_charging", "Ravipur Municipal Corporation plans 500 public charging points by March."),
    ("ravipur_charging", "Many Ravipur chargers will be placed at metro stations."),
    ("october_sales", "Electric two-wheeler sales rose 32% year on year in October."),
    ("october_sales", "October electric two-wheeler sales reached 1.1 lakh units."),
    ("chargegrid_funding", "ChargeGrid raised $40 million in a Series B round."),
    ("chargegrid_funding", "Northstar Ventures led the ChargeGrid Series B round."),
    ("lithium_prices", "Lithium carbonate prices fell 8% this quarter."),
]

ENTITY_SWAPS = {"Voltara Motors": "Zephyr Mobility", "Voltara": "Zephyr", "Zephyr Mobility": "Voltara Motors",
                "Ravipur": "Northbridge", "ChargeGrid": "VoltSwap", "Northstar Ventures": "Bluepeak Capital",
                "Lithium carbonate": "Cobalt"}
INVENTED_FACTS = ["after its chief executive resigned", "amid a government investigation",
                  "despite a nationwide strike", "following a Supreme Court order"]


def alter_number(text: str) -> str | None:
    # Skip digits glued to letters (model names like "V2").
    match = re.search(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])", text)
    if not match:
        return None
    raw = match.group()
    value = float(raw.replace(",", ""))
    if "." in raw:
        formatted = f"{value + 0.4:.1f}"
    elif value < 10:
        formatted = str(int(value) + 3)
    else:
        new = value * 1.5
        formatted = f"{new:,.0f}" if "," in raw else f"{new:.0f}"
    fake = text[: match.start()] + formatted + text[match.end():]
    return fake if fake != text else None


def swap_entity(text: str) -> str | None:
    for old, new in ENTITY_SWAPS.items():
        if old in text:
            return text.replace(old, new, 1)
    return None


def add_fact(text: str, i: int) -> str:
    return text.rstrip(".") + " " + INVENTED_FACTS[i % len(INVENTED_FACTS)] + "."


def fabricated_variants() -> list[tuple[str, str, str]]:
    out = []
    for i, (story, claim) in enumerate(TRUE_CLAIMS):
        for kind, fake in (("altered_number", alter_number(claim)), ("swapped_entity", swap_entity(claim)),
                           ("invented_fact", add_fact(claim, i))):
            if fake and fake != claim:
                out.append((story, kind, fake))
    return out


def is_supported(story: str, claim: str, articles, settings: Settings) -> bool:
    return any(support_check(claim, a.text, settings.support_overlap, settings.max_unmatched_words)[0]
               for a in articles if a.story_id == story)


class InjectingLLM(MockLLM):
    """Analyst that adds one fabricated claim (altered number) to every story."""

    def structured(self, system, user, schema):
        result = super().structured(system, user, schema)
        if schema is AnalystOutput:
            for story in result.stories:
                if story.claims:
                    real = story.claims[0]
                    fake = alter_number(real.text) or add_fact(real.text, story.story_id)
                    assert fake != real.text
                    story.claims.append(real.model_copy(update={"text": fake}))
            self.injected = [c.text for s in result.stories for c in s.claims[-1:]]
        return result


def main() -> None:
    settings = Settings()
    topic, articles = load_sample("ev_two_wheelers")
    gold = [a.story_id for a in articles]
    lines = [f"# NewsLens evaluation\n\nSample corpus: {len(articles)} articles, {len(set(gold))} labelled stories "
             "(fictional). Analyst/writer: offline mock.\n"]

    # 1. clustering
    lines.append("## 1. Story clustering\n\n| distance threshold | stories | ARI | homogeneity | completeness |")
    lines.append("|--:|--:|--:|--:|--:|")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = cluster_labels(articles, th)
        mark = " **(default)**" if th == settings.cluster_distance_threshold else ""
        lines.append(f"| {th}{mark} | {len(set(pred))} | {adjusted_rand_score(gold, pred):.3f} | "
                     f"{homogeneity_score(gold, pred):.3f} | {completeness_score(gold, pred):.3f} |")

    # 2. hallucination detection
    fakes = fabricated_variants()
    true_ok = [is_supported(s, c, articles, settings) for s, c in TRUE_CLAIMS]
    caught = {}
    for story, kind, fake in fakes:
        caught.setdefault(kind, []).append(not is_supported(story, fake, articles, settings))
    tp = sum(sum(v) for v in caught.values())
    fp = sum(not ok for ok in true_ok)
    fn = len(fakes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(fakes)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lines.append(f"\n## 2. Fabricated-claim detection\n\n{len(TRUE_CLAIMS)} paraphrased true claims, "
                 f"{len(fakes)} fabricated variants.\n")
    lines.append("| fabrication type | caught |\n|:--|--:|")
    for kind, flags in caught.items():
        lines.append(f"| {kind} | {sum(flags)}/{len(flags)} |")
    lines.append(f"| **all** | **{tp}/{len(fakes)}** |\n")
    lines.append(f"- True claims accepted: **{sum(true_ok)}/{len(true_ok)}** (false-positive rate "
                 f"{fp / len(true_ok):.0%})")
    lines.append(f"- Precision {precision:.2f} · Recall {recall:.2f} · F1 {f1:.2f} (missed fabrications: {fn})")
    rejected = [c for (s, c), ok in zip(TRUE_CLAIMS, true_ok, strict=True) if not ok]
    if rejected:
        lines.append("- Rejected true claims: " + "; ".join(f"'{c}'" for c in rejected))

    # 3. end-to-end injection
    llm = InjectingLLM()
    state = run(topic, LensContext(settings, llm, sample_articles=articles))
    published = " ".join(b["text"] for b in state["briefing"]["bullets"])
    leaked = [f for f in llm.injected if f.rstrip(".") in published]
    statuses = [c["status"] for c in state["claims"]]
    lines.append(f"\n## 3. End-to-end injection test\n\nAnalyst injected {len(llm.injected)} fabricated claims "
                 f"(one per story).\n")
    lines.append(f"- Marked unsupported by the verifier: **{statuses.count('unsupported')}**")
    lines.append(f"- Fabrications that reached the published briefing: **{len(leaked)}/{len(llm.injected)}**")
    lines.append(f"- Bullets published: {len(state['briefing']['bullets'])}, every one with ≥1 source citation; "
                 f"writer revisions: {state['revisions']}")
    stories = build_stories(articles, settings.cluster_distance_threshold, settings.max_stories)
    multi = sum(len(s.publishers) >= 2 for s in stories)
    lines.append(f"- Stories covered by 2+ publishers: {multi}/{len(stories)}")

    report = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
