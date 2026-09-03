"""A sentence must never be written into a KG as a topic.

The read-time filler guard (``kg.filters.is_filler_topic``) cleans corpora that were extracted
before this check existed. It is not a substitute for this one: catching a bad topic on the way
OUT still means it was written, indexed, slugged, and counted, and every corpus produced from now
on would carry the pollution until someone re-enriched it.

The failure this prevents, from a real DGX pipeline run: the provider emitted 11-13 word
propositions as topic labels, and ``_enforce_noun_phrase_label`` cut each at 50 characters and
moved the tail into ``description``. Truncation does not discard a bad topic — it DISGUISES one.
"Product development in frontier AI requires building for model capabilities two years out"
became "Product development in frontier AI requires": the exact shape of a real topic, unique to
its episode, and permanently unclusterable. All 48 topics across six episodes were this.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.kg.llm_extract import (
    _MAX_TOPIC_LABEL_WORDS,
    _is_proposition_not_a_topic,
    _parse_topic_items,
)

pytestmark = pytest.mark.unit

#: Verbatim provider output from the run that motivated this.
_REAL_PROPOSITIONS = [
    "Product development in frontier AI requires building for model capabilities two years out",
    "Ambition must expand because AI tools flatten the translation layers between functions",
    "Writing serves two distinct functions: writing for thinking must remain a human obligation",
]

_REAL_TOPICS = [
    "ai regulation",
    "open source ai models",
    "AI ethics and public perception",
    "global oil supply chain",
    "Direct-to-consumer e-commerce go-to-market strategies",
    "International Group of P&I Clubs",
]


@pytest.mark.parametrize("label", _REAL_PROPOSITIONS)
def test_a_proposition_is_rejected_not_truncated(label: str) -> None:
    assert _is_proposition_not_a_topic(label)
    assert _parse_topic_items([label]) == [], (
        "the proposition was kept — truncated into a plausible-looking fake topic"
    )


@pytest.mark.parametrize("label", _REAL_TOPICS)
def test_real_topics_are_kept(label: str) -> None:
    """The mirror. A rule that rejects everything passes every test above."""
    assert not _is_proposition_not_a_topic(label)
    rows = _parse_topic_items([label])
    assert len(rows) == 1, f"{label!r} was rejected as a proposition"


def test_the_dict_form_is_guarded_too() -> None:
    """Providers emit both shapes; guarding one and not the other is the usual half-fix."""
    assert _parse_topic_items([{"label": _REAL_PROPOSITIONS[0]}]) == []
    assert len(_parse_topic_items([{"label": "ai regulation", "description": "d"}])) == 1


def test_mixed_input_keeps_exactly_the_real_topics() -> None:
    rows = _parse_topic_items([*_REAL_PROPOSITIONS, "ai regulation", {"label": "gear"}])
    assert [r["label"] for r in rows] == ["ai regulation", "gear"]


def test_the_boundary_is_inclusive() -> None:
    """Exactly at the limit is a topic; one over is a proposition."""
    at_limit = " ".join(["word"] * _MAX_TOPIC_LABEL_WORDS)
    over = " ".join(["word"] * (_MAX_TOPIC_LABEL_WORDS + 1))
    assert not _is_proposition_not_a_topic(at_limit)
    assert _is_proposition_not_a_topic(over)


def test_dropping_is_loud(caplog: pytest.LogCaptureFixture) -> None:
    """Silent dropping would turn a degraded provider into "this episode had no topics".

    The count and a sample must reach the run log, because the signal is about the RUN — a
    fallback tier emitting propositions — not about the episode.
    """
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.llm_extract"):
        _parse_topic_items(list(_REAL_PROPOSITIONS))
    assert any("propositions" in str(r.msg) for r in caplog.records), (
        "topics vanished without a word in the log"
    )
    record = next(r for r in caplog.records if "propositions" in str(r.msg))
    args = record.args
    assert isinstance(args, tuple) and args, "the log must carry the count as an arg"
    assert args[0] == len(_REAL_PROPOSITIONS), "the count must say how many were lost"


def test_a_clean_batch_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """A warning on every healthy run is as useless as no warning at all."""
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.llm_extract"):
        _parse_topic_items(list(_REAL_TOPICS))
    assert not [r for r in caplog.records if "propositions" in str(r.msg)]


# --- every call site must honour the rejection ------------------------------------------------


def test_every_call_site_handles_the_none_return() -> None:
    """THE meta-regression: a partial fix here is invisible until a real ingest.

    ``_enforce_noun_phrase_label`` has FOUR call sites across two modules. The first version of
    this fix guarded the two in ``llm_extract`` and missed the two in ``kg/pipeline`` — which are
    the ones the live provider path uses. A fresh ingest still produced eight truncated
    propositions and the drop never logged; nothing failed, because every test covered the guarded
    half.

    Making the function return ``None`` is what closes it: the type changes, so mypy fails any
    caller that unpacks the result blindly. This test asserts the call-site COUNT so a new one
    cannot be added without a deliberate decision here.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[4] / "src" / "podcast_scraper" / "kg"
    sites: list[tuple[str, str]] = []
    for module in ("llm_extract.py", "pipeline.py"):
        for line in (src / module).read_text(encoding="utf-8").splitlines():
            if "_enforce_noun_phrase_label(" in line and not line.lstrip().startswith("def "):
                sites.append((module, line.strip()))

    assert len(sites) == 4, (
        f"the call-site count changed ({len(sites)}); each one must handle the None return "
        f"(= this label is a proposition, drop it):\n"
        + "\n".join(f"  {m}: {ln}" for m, ln in sites)
    )
    # None of them may unpack directly — that is the shape that silently truncated.
    for module, line in sites:
        assert not re.match(r"^\w+,\s*\w+\s*=\s*_enforce_noun_phrase_label\(", line), (
            f"{module} unpacks the result without checking for None: {line}"
        )


def test_the_live_provider_path_rejects_propositions() -> None:
    """Exercise ``kg/pipeline`` directly — the path a real ingest actually takes.

    Testing only ``llm_extract`` is what let the half-fix ship.
    """
    from podcast_scraper.kg.pipeline import _append_topics_and_entities_from_partial

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    _append_topics_and_entities_from_partial(
        "episode:e",
        {
            "topics": [
                {"label": _REAL_PROPOSITIONS[0]},
                {"label": "ai regulation"},
            ],
            "entities": [],
        },
        nodes,
        edges,
        50,
        50,
    )
    labels = [
        n["properties"]["label"]  # type: ignore[index]
        for n in nodes
        if n.get("type") == "Topic"
    ]
    assert labels == ["ai regulation"], f"the provider path still writes propositions: {labels}"
