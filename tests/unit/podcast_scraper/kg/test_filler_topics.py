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
        "diversify or",
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


def test_a_dangling_conjunction_is_a_fragment() -> None:
    """The rule has to run on the RAW label.

    ``_normalize_topic_label`` strips trailing stopwords, so by the time it is done "diversify or"
    is "diversify" and "without the" is "without" — both indistinguishable from ordinary one-word
    topics. The evidence that they are fragments is the very token normalization removes.
    """
    assert is_filler_topic("diversify or")
    assert is_filler_topic("regulation and")
    assert not is_filler_topic("crime and punishment"), "a conjunction mid-label is fine"


def test_function_words_alone_are_not_a_subject() -> None:
    assert is_filler_topic("this one")
    assert is_filler_topic("back again")
    assert is_filler_topic("the")


def test_a_greeting_needs_no_content_word_to_be_dropped() -> None:
    """But a greeting word FOLLOWED by content is a real topic."""
    assert is_filler_topic("welcome back")
    assert is_filler_topic("thanks")
    assert not is_filler_topic("welcome wagon"), "a content word after the lead makes it a subject"


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
