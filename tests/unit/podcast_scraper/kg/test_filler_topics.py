"""Conversational boilerplate must not become a navigation destination.

A ``Topic`` node does not stay a node. It becomes a theme-cluster member, a co-occurrence pair, a
trending chip, and a followable interest — so filler that survives extraction is eventually
offered to a listener as a storyline. Found during local validation 2026-09-03: of 13 topics
extracted into ``tests/fixtures/viewer-validation-corpus/v3``, **six** were filler or sentence
fragments, and one of them (``welcome-back-to``) had become a theme cluster.

``_normalize_topic_label`` cannot catch these. It trims stopwords and caps tokens, and by those
rules "welcome back" is a perfectly ordinary two-token label. This is a different question — not
"is the label tidy" but "is this a subject at all" — so it is a separate, deliberately
conservative predicate.

The policy is the same one ``KNOWN_ORGS`` states: false negatives beat false positives. Letting a
junk topic through is cosmetic; dropping a real one silently removes evidence from every downstream
surface. Every rule fires only where no content word survives.
"""

from __future__ import annotations

import pytest

from podcast_scraper.kg.filters import is_filler_topic

pytestmark = pytest.mark.unit


# --- the labels that motivated this, verbatim from the corpus ---------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "welcome back to",
        "great to be back",
        "excited for this one",
        "without the",
    ],
)
def test_the_measured_filler_is_dropped(label: str) -> None:
    assert is_filler_topic(label), f"{label!r} would be offered as a storyline"


# --- and the ones that must survive, including deliberate near-misses -------------------------


@pytest.mark.parametrize(
    "label",
    [
        # From the same corpus — real subjects sitting beside the filler.
        "business markets",
        "environment",
        "fed vice chair",
        "gear",
        "health",
        "outdoor activities",
        "science research",
        "technology",
        # Real prod topics.
        "ai regulation",
        "open source ai models",
        "us-china ai competition",
        "federal reserve policy",
        # NEAR-MISSES. These start with a word on the conversational-lead list or look like
        # fragments, and are exactly what a sloppier rule would eat.
        "great firewall",
        "welcome to the machine",
        "thanks economy",
        "happy hour",
    ],
)
def test_real_topics_survive(label: str) -> None:
    assert not is_filler_topic(label), f"{label!r} is a real subject and was dropped"


# --- the rules, stated individually so a future edit can see which one it changed --------------


def test_a_two_word_fragment_is_an_ACCEPTED_MISS() -> None:
    """"diversify or" survives, deliberately.

    It is two words ending in a function word — structurally identical to "Down Under", "Coming
    Out", "Inside Out". Only world knowledge separates a truncation from a proper title at that
    length. This module's policy is that false negatives beat false positives, and the asymmetry
    is stark: keeping "diversify or" is cosmetic, while deleting "Down Under" removes a real
    topic from every surface silently. Asserted so the trade is visible rather than looking like
    an oversight.
    """
    assert not is_filler_topic("diversify or")
    assert not is_filler_topic("Down Under")
    assert not is_filler_topic("Coming Out")


def test_a_dangling_conjunction_is_a_fragment() -> None:
    """The rule has to run on the RAW label.

    ``_normalize_topic_label`` strips trailing stopwords, so by the time it is done "diversify or"
    is "diversify" and "without the" is "without" — both indistinguishable from ordinary one-word
    topics. The evidence that they are fragments is the very token normalization removes.
    """
    assert is_filler_topic("ai regulation and")
    assert is_filler_topic("supply chain resilience or")
    assert not is_filler_topic("crime and punishment"), "a conjunction mid-label is fine"


def test_function_words_alone_are_not_a_subject() -> None:
    assert is_filler_topic("the")
    assert is_filler_topic("this one here")
    # Two-word all-function-word labels are spared — see the accepted-miss test above.
    assert not is_filler_topic("this one")


def test_a_greeting_needs_no_content_word_to_be_dropped() -> None:
    """But a greeting word FOLLOWED by content is a real topic."""
    assert is_filler_topic("welcome back")
    assert is_filler_topic("thanks")
    assert not is_filler_topic("welcome wagon"), "a content word after the lead makes it a subject"


@pytest.mark.parametrize(
    "label",
    ["Today", "Coming Out", "Down Under", "Great Firewall", "Happy Hour", "Stay Interviews"],
)
def test_ordinary_english_words_are_not_boilerplate_markers(label: str) -> None:
    """The lead list was trimmed after measuring these.

    "today", "coming", "great", "happy", "wonderful" and "stay" were all treated as conversational
    markers and all ate real titles. A word common in ordinary English cannot mark boilerplate by
    itself; only near-exclusively conversational openers qualify.
    """
    assert not is_filler_topic(label)


@pytest.mark.parametrize(
    "label",
    [
        "Direct-to-consumer e-commerce go-to-market strategies",
        "State-of-the-art large-scale machine-learning systems",
    ],
)
def test_hyphenated_compounds_are_not_mistaken_for_truncation(label: str) -> None:
    """The first truncation rule COUNTED slug segments and deleted these.

    ``identity.slugify`` preserves intra-word hyphens, so a legitimately hyphenated label inflates
    the segment count with no truncation at all: a 5-word label became 9 segments and was silently
    dropped from every enrichment surface. Truncation is now detected by COMPARING the id against
    the label, with both sides split on hyphens — a comparison has no such failure mode.
    """
    slug = "topic:" + label.lower().replace(" ", "-")
    assert not is_filler_topic(label, slug)


def test_empty_and_punctuation_only_labels_are_filler() -> None:
    assert is_filler_topic("")
    assert is_filler_topic("...")
    assert is_filler_topic("   ")


def test_the_guard_is_wired_into_the_shared_topic_loader() -> None:
    """Assert the enrichers actually go through it, rather than trusting they do.

    Filtering in ``_loaders.topic_nodes`` is what makes this retro-apply: an already-extracted
    corpus is cleaned by a re-enrich (seconds) instead of a GI re-run over every episode.
    """
    from podcast_scraper.enrichment.enrichers._loaders import topic_nodes

    kg = {
        "nodes": [
            {
                "type": "Topic",
                "id": "topic:welcome-back-to",
                "properties": {"label": "welcome back to"},
            },
            {
                "type": "Topic",
                "id": "topic:ai-regulation",
                "properties": {"label": "ai regulation"},
            },
            {"type": "Person", "id": "person:jane", "properties": {"label": "Jane"}},
        ]
    }
    kept = [n["id"] for n in topic_nodes(kg)]
    assert kept == ["topic:ai-regulation"]


# --- sentences dressed as topics (found on a real DGX pipeline run, 2026-09-03) ---------------


#: Verbatim from a real run over Lenny's Podcast. The LLM emitted 11-13 word PROPOSITIONS as Topic
#: labels; ``_TOPIC_MAX_TOKENS`` then truncated each to six words, so what reaches the KG looks
#: like a tidy topic and is actually a unique mid-sentence fragment that can never match anything
#: in another episode. The run had fallen back to the ollama tier (the DGX vLLM was unreachable),
#: which is precisely when a guard earns its keep — nothing else notices a degraded provider.
_REAL_TRUNCATED_SENTENCES = [
    (
        "topic:product-development-in-frontier-ai-requires-building-for-model-capabilities-two-",
        "Product development in frontier AI requires",
    ),
    (
        "topic:writing-serves-two-distinct-functions-writing-for-thinking-must-remain-a-human-o",
        "Writing serves two distinct functions: writing",
    ),
]


def test_a_truncation_that_happens_to_end_in_a_stopword_is_caught_by_the_label_alone() -> None:
    """One of the eight got caught without the id — luck, not coverage.

    "Ambition must expand because AI tools flatten the" ends in "the", so the fragment rule fires.
    The other seven do not end that conveniently, which is the whole reason the id check exists.
    """
    assert is_filler_topic("Ambition must expand because AI tools flatten the")


@pytest.mark.parametrize("topic_id,label", _REAL_TRUNCATED_SENTENCES)
def test_a_truncated_sentence_is_rejected_not_kept(topic_id: str, label: str) -> None:
    """The LABEL alone cannot reveal this — it has already been cut to a plausible length."""
    assert not is_filler_topic(label), (
        "precondition: post-truncation the label looks like an ordinary topic, which is why the "
        "id has to be consulted"
    )
    assert is_filler_topic(label, topic_id), (
        "an 11-word proposition truncated to six words is a unique fragment; it pollutes "
        "clustering, co-occurrence and trending, and inflates the singleton rate"
    )


@pytest.mark.parametrize(
    "topic_id,label",
    [
        ("topic:open-source-ai-models", "open source ai models"),
        ("topic:ai-regulation", "ai regulation"),
        ("topic:us-china-ai-competition", "us-china ai competition"),
        ("topic:federal-reserve-policy", "federal reserve policy"),
        # The two the 4->6 token cap was widened FOR — they must not be collateral damage.
        ("topic:ai-ethics-and-public-perception", "AI ethics and public perception"),
        ("topic:global-oil-supply-chain", "global oil supply chain"),
        ("topic:international-group-of-p-and-i-clubs", "International Group of P&I Clubs"),
    ],
)
def test_real_topics_survive_the_id_check(topic_id: str, label: str) -> None:
    assert not is_filler_topic(label, topic_id)


def test_the_id_is_optional_and_absence_is_not_evidence() -> None:
    """Callers without an id must not get different answers for the label itself."""
    assert is_filler_topic("welcome back to") == is_filler_topic("welcome back to", None)
    assert is_filler_topic("ai regulation") == is_filler_topic("ai regulation", None)
