import pytest

from newslens.agents.critic import drop_failing_bullets, review
from newslens.agents.verifier import numbers, support_check, verify_claims
from newslens.clustering import build_stories
from newslens.schemas import AnalystOutput, Briefing, BriefingBullet, ExtractedClaim, StoryClaims, VerifiedClaim

SRC = ("Zephyr Mobility recalls 3,200 scooters after battery fires. Zephyr Mobility is recalling 3,200 electric "
       "scooters after reports of battery fires. Owners will get a free battery inspection at service centres.")


def test_number_normalisation():
    assert numbers("Rs 89,999 and 1.1 lakh, 32%.") == {"89999", "1.1", "32"}


@pytest.mark.parametrize("claim,ok", [
    ("Zephyr Mobility recalled 3,200 electric scooters after battery fire reports.", True),
    ("Charging startup ChargeGrid raised money", False),  # nothing in source
    ("Zephyr Mobility recalls 4,800 scooters after battery fires.", False),  # altered number
    ("Voltara Motors recalls 3,200 scooters after battery fires.", False),  # swapped name (mid-sentence word)
    ("Voltara recalls 3,200 scooters after battery fires.", False),  # swapped sentence-initial name
    ("Zephyr Mobility recalls 3,200 scooters after battery fires despite a nationwide strike.", False),  # invented
])
def test_support_check(claim, ok):
    assert support_check(claim, SRC, 0.6, 2)[0] is ok


def test_sentence_initial_common_word_is_not_treated_as_a_name():
    src = "ChargeGrid, a battery swapping startup, raised $40 million in Series B funding."
    assert support_check("Charging startup ChargeGrid raised $40 million in Series B funding.", src, 0.6, 3)[0]


def _output(story_id, text, cited):
    return AnalystOutput(stories=[StoryClaims(story_id=story_id, claims=[ExtractedClaim(text=text,
                                                                                         article_ids=cited)])])


def test_statuses_and_under_citation(articles, by_id, settings):
    stories = build_stories(articles, settings.cluster_distance_threshold, settings.max_stories)
    recall = next(s for s in stories if "Zephyr" in s.headline)
    # Cites only one article, but the verifier finds the others in the story.
    out = _output(recall.story_id, "Zephyr Mobility recalled 3,200 electric scooters after battery fire reports.",
                  [recall.article_ids[0]])
    [c] = verify_claims(out, stories, by_id, settings)
    assert c.status == "corroborated" and len(c.supporting_publishers) == 3
    out = _output(recall.story_id, "Zephyr Mobility recalled 9,000 electric scooters.", recall.article_ids)
    assert verify_claims(out, stories, by_id, settings)[0].status == "unsupported"
    lithium = next(s for s in stories if "Lithium" in s.headline)
    out = _output(lithium.story_id, "Lithium carbonate prices fell 8% this quarter.", lithium.article_ids)
    assert verify_claims(out, stories, by_id, settings)[0].status == "single_source"


def test_invented_story_ids_and_article_ids_are_ignored(articles, by_id, settings):
    stories = build_stories(articles, settings.cluster_distance_threshold, settings.max_stories)
    assert verify_claims(_output(999, "Something happened today.", ["a01"]), stories, by_id, settings) == []
    out = _output(stories[0].story_id, "Voltara Motors launched the V2 electric scooter priced at Rs 89,999.",
                  ["does-not-exist"])
    assert verify_claims(out, stories, by_id, settings)[0].status == "corroborated"


def claim(cid, status, text="Zephyr Mobility recalled 3,200 scooters."):
    return VerifiedClaim(claim_id=cid, story_id=1, text=text, cited_article_ids=[], supporting_article_ids=["a"],
                         supporting_publishers=["P"], status=status, reason="")


def brief(*bullets):
    return Briefing(headline="Test briefing", bullets=[BriefingBullet(text=t, claim_ids=ids) for t, ids in bullets])


CLAIMS = [claim("C1", "corroborated"), claim("C2", "single_source"), claim("C3", "unsupported")]


@pytest.mark.parametrize("bullet,rule", [
    (("Zephyr recalled scooters.", ["C9"]), "unknown_claim"),
    (("Zephyr recalled scooters.", ["C3"]), "unsupported_claim"),
    (("Zephyr recalled 5,000 scooters.", ["C1"]), "unsupported_number"),
    (("Zephyr Mobility recalled 3,200 scooters.", ["C2"]), "unhedged_single_source"),
])
def test_each_critic_rule_fires(bullet, rule):
    assert rule in {i.rule for i in review(brief(bullet), CLAIMS)}


def test_clean_briefing_passes_and_duplicates_are_caught():
    ok = brief(("Zephyr Mobility recalled 3,200 scooters.", ["C1"]),
               ("Reportedly (single source): Zephyr recalled 3,200 scooters.", ["C2"]))
    assert review(ok, CLAIMS) == []
    dup = brief(("Zephyr Mobility recalled 3,200 scooters.", ["C1"]), ("Zephyr Mobility recalled 3,200 scooters!",
                                                                        ["C1"]))
    assert [i.rule for i in review(dup, CLAIMS)] == ["duplicate_bullet"]


def test_fail_safe_drops_only_failing_bullets():
    b = brief(("Zephyr Mobility recalled 3,200 scooters.", ["C1"]), ("Bad bullet here.", ["C3"]))
    kept = drop_failing_bullets(b, review(b, CLAIMS))
    assert [x.claim_ids for x in kept.bullets] == [["C1"]]
