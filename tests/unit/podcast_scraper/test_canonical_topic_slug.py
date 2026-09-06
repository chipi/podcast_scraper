"""Topic label variants must share one id, or the concept's frequency splits (#1933).

``ai in education`` (16 episodes), ``ai education`` (3) and ``ai and education`` (2) are one topic
written three ways. As separate ids they are structurally unable to co-occur, so the corpus's
strongest signal is diluted across spellings — and co-occurrence feeds theme clustering, the
semantic clusterer's singleton rate, and every "related topic" surface.

Measured on the 2,618-label prod topic set (2026-09-06): **50 families, 109 labels, 229 episodes**
of frequency currently split. All 50 were reviewed by hand; the two known false positives are
pinned below rather than left to be rediscovered.
"""

from __future__ import annotations

import pytest

from podcast_scraper.graph_id_utils import canonical_topic_slug, slugify_label

pytestmark = pytest.mark.unit


def _same(a: str, b: str) -> bool:
    return canonical_topic_slug(a) == canonical_topic_slug(b)


@pytest.mark.parametrize(
    "a,b",
    [
        # Dropped prepositions/articles — the single biggest family in the corpus.
        ("ai in education", "ai education"),
        ("ai in education", "ai and education"),
        ("ai job displacement", "ai and job displacement"),
        ("ai in drug discovery", "ai for drug discovery"),
        ("ai as tool", "ai as a tool"),
        # Word ORDER. This is what token-sorting buys, and it is the aggressive part.
        ("us-china ai competition", "china-us ai competition"),
        ("strait of hormuz disruption", "hormuz strait disruption"),
        ("china-latin america relations", "latin america china relations"),
        ("ai safety and alignment", "ai alignment and safety"),
        ("reinforcement learning vs deep learning", "deep learning vs reinforcement learning"),
        # Hyphenation. Treating "-" as punctuation rather than a separator is exactly what made an
        # earlier measurement conclude only 0.8% of topics collapse.
        ("open-source ai models", "open source ai models"),
        ("human-ai collaboration", "ai and human collaboration"),
        # Plurals.
        ("creator economies", "creator economy"),
        ("dark factories", "dark factory"),
        ("ai agent platforms", "ai agent platform"),
        ("biotech innovation gaps", "biotech innovation gap"),
        # Possessive apostrophe.
        ("iran nuclear program", "irans nuclear program"),
    ],
)
def test_real_prod_variants_collapse(a: str, b: str) -> None:
    assert _same(a, b), f"{a!r} and {b!r} are the same topic but got different ids"


@pytest.mark.parametrize(
    "a,b",
    [
        ("ai safety", "ai regulation"),
        ("ai in education", "ai in healthcare"),
        ("us-china ai competition", "us-china trade war"),
        ("open source ai models", "open source ai risks"),
    ],
)
def test_genuinely_different_topics_stay_apart(a: str, b: str) -> None:
    assert not _same(a, b), f"{a!r} and {b!r} are different topics but collapsed"


@pytest.mark.parametrize(
    "a,b",
    [
        # Investing IN ai, versus using ai FOR investing.
        ("ai investment", "ai in investment"),
        # Agents that ARE scientists, versus agents SERVING scientists.
        ("ai scientist agents", "ai agents for scientists"),
    ],
)
def test_known_false_positives_are_pinned(a: str, b: str) -> None:
    """These two DO wrongly merge, and the trade is accepted knowingly.

    Both differ only by a dropped preposition — the same mechanism that correctly merges
    ``ai in education`` with ``ai education``. You cannot have one without the other under this
    rule. Each is 1+1 episodes, against 229 episodes correctly merged.

    This test asserts the CURRENT behaviour so the trade stays visible. If a future rule separates
    them without breaking the collapse cases above, invert this assertion — that is an improvement,
    not a regression.
    """
    assert _same(a, b), (
        f"{a!r}/{b!r} no longer merge — if that was deliberate and the collapse tests still "
        "pass, this is a genuine improvement: flip this assertion."
    )


def test_display_label_is_not_touched() -> None:
    """Only the ID is canonicalised; a reader must still see the label as written."""
    assert canonical_topic_slug("AI in Education") == canonical_topic_slug("ai education")
    # The function returns a slug, never a display string — the caller keeps `raw` for display.
    assert canonical_topic_slug("AI in Education") == "ai-education"


def test_degenerate_labels() -> None:
    assert canonical_topic_slug("") == "topic"
    assert canonical_topic_slug("   ") == "topic"


def test_an_all_stopword_label_does_not_collapse_into_one_bucket() -> None:
    """``the a of`` has no concept in it. Returning the shared ``topic`` id would merge every
    such label into one phantom topic, which is worse than keeping them distinct."""
    a = canonical_topic_slug("the a of")
    b = canonical_topic_slug("of the and")
    assert a != "topic" and b != "topic"
    assert a != b


def test_episode_ids_are_untouched() -> None:
    """``slugify_label`` mints episode and person ids too — canonicalising it globally would have
    reordered and stripped words from those. This is a separate function for that reason."""
    raw = "The Rest Is History"
    assert slugify_label(raw) == "the-rest-is-history"
    assert canonical_topic_slug(raw) != slugify_label(raw)
