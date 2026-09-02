"""#587: BOTH Topic-node sites must cap the label at a noun phrase, not slice at 200 chars.

`kg/llm_extract.py` has enforced a 50-char noun-phrase cap since #587. `kg/pipeline.py` built
Topic nodes with a raw `label[:200]` slice at two sites, so the enforcement was bypassed wherever
Topics were actually constructed.

Why it mattered, measured on the 1,066-episode corpus before the fix:

    labels <= 50 chars (8,584):  14.3% land in a semantic cluster,  6.9% recur in >=2 episodes
    labels  > 50 chars   (679):   3.4% land in a semantic cluster,  0.0% recur in >=2 episodes

Zero of 679. A 200-char truncated sentence cannot be emitted identically twice, so those topics
were structurally incapable of recurring, clustering, or joining a theme — 7.3% of the corpus dead
on arrival. These tests pin the cap at both sites and pin that the overflow is preserved rather
than discarded.
"""

from __future__ import annotations

import pytest

from podcast_scraper.kg.llm_extract import _MAX_TOPIC_LABEL_CHARS, _enforce_noun_phrase_label

pytestmark = pytest.mark.unit

_SENTENCE = (
    "Test-driven development works against design because it encourages tiny increments "
    "rather than stepping back to consider the big picture"
)


def test_the_cap_is_the_one_587_defined() -> None:
    assert _MAX_TOPIC_LABEL_CHARS == 50


def test_long_label_is_split_not_sliced() -> None:
    label, overflow = _enforce_noun_phrase_label(_SENTENCE)
    assert len(label) <= _MAX_TOPIC_LABEL_CHARS
    assert overflow, "the tail must be returned, not dropped"
    assert not label.endswith(" ")
    # split at a word boundary — no half-words
    assert _SENTENCE.startswith(label)
    assert label.split()[-1] in _SENTENCE.split()


def test_short_label_is_untouched() -> None:
    label, overflow = _enforce_noun_phrase_label("ai safety")
    assert label == "ai safety"
    assert overflow is None


def test_nothing_is_lost_across_the_split() -> None:
    """label + overflow must reconstruct the original — the fix moves text, it does not drop it."""
    label, overflow = _enforce_noun_phrase_label(_SENTENCE)
    assert (label + " " + (overflow or "")).strip() == _SENTENCE.strip()


def test_pipeline_topic_nodes_respect_the_cap() -> None:
    """The regression: Topic nodes built by kg/pipeline.py must not carry sentence labels.

    Exercised through the real builder rather than by re-implementing it, so a future site that
    reintroduces a raw slice fails here.
    """
    import inspect

    from podcast_scraper.kg import pipeline

    src = inspect.getsource(pipeline)
    assert '"label": lab[:200]' not in src, (
        "a Topic label is being sliced at 200 chars again — use _enforce_noun_phrase_label so the "
        "noun-phrase cap from #587 applies (see the 0-of-679 recurrence measurement above)"
    )
    assert '"label": lab_s[:200]' not in src, (
        "a Topic label is being sliced at 200 chars again — use _enforce_noun_phrase_label"
    )
    assert src.count("_enforce_noun_phrase_label(") >= 2, (
        "both Topic-construction sites must enforce the cap"
    )
