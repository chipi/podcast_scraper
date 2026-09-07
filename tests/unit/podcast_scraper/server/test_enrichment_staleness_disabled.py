"""A disabled enricher is not stale work — it is a decision (2026-09-07).

``topic_similarity`` and ``topic_consensus`` are switched off in the prod corpus's
``viewer_operator.yaml``. Because ``compute_enrichment_staleness`` judged every known
manifest purely from on-disk output, both reported ``never_ran`` / ``stale`` forever and
held ``reenrich_recommended`` permanently true.

That is indistinguishable from a genuine backlog, and it cost an investigation that ended
at "they were turned off on purpose". These tests pin the distinction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.enrichment_staleness import (
    _operator_disabled_enricher_ids,
    compute_enrichment_staleness,
)

pytestmark = pytest.mark.unit

_YAML = """\
enrichment:
  enabled: true
  enrichers:
    grounding_rate:
      expected_duration_s: 300
    topic_similarity:
      enabled: false
    topic_consensus:
      enabled: false
"""


def _corpus(tmp_path: Path, yaml_text: str | None) -> Path:
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    if yaml_text is not None:
        (root / "viewer_operator.yaml").write_text(yaml_text, encoding="utf-8")
    return root


def test_reads_the_explicitly_disabled_ids(tmp_path: Path) -> None:
    assert _operator_disabled_enricher_ids(_corpus(tmp_path, _YAML)) == {
        "topic_similarity",
        "topic_consensus",
    }


def test_an_absent_entry_is_not_a_disable(tmp_path: Path) -> None:
    """Only ``enabled: false`` counts. An enricher the operator YAML omits may still be
    enabled by the profile, which this module cannot see — under-reporting a disable leaves
    the row stale (today's behaviour, safe); over-reporting would HIDE real work."""
    ids = _operator_disabled_enricher_ids(_corpus(tmp_path, _YAML))
    assert "grounding_rate" not in ids
    assert "insight_density" not in ids


def test_no_operator_yaml_disables_nothing(tmp_path: Path) -> None:
    assert _operator_disabled_enricher_ids(_corpus(tmp_path, None)) == set()


def test_unparsable_yaml_does_not_break_the_probe(tmp_path: Path) -> None:
    """This runs on a health surface; a broken operator file must not take it down."""
    assert _operator_disabled_enricher_ids(_corpus(tmp_path, "{{{ not yaml")) == set()


def test_disabled_enrichers_are_not_stale_and_do_not_recommend_reenrich(tmp_path: Path) -> None:
    """The point of the change: a switched-off enricher stops looking like a backlog."""
    root = _corpus(tmp_path, _YAML)
    fields = compute_enrichment_staleness(root)
    by_id = {r.enricher_id: r for r in fields.enrichers}

    for eid in ("topic_similarity", "topic_consensus"):
        row = by_id.get(eid)
        assert row is not None, f"{eid} must still be REPORTED, not hidden"
        assert row.disabled is True
        assert row.stale is False, f"{eid} is off by choice — not outstanding work"
        assert row.reasons == []

    # ...and they no longer drag the rollup. Asserted by DIFFERENCE, not by absence:
    # in an empty corpus the other (enabled) enrichers legitimately have never run, so
    # ``never_ran`` in the rollup is correct — it just must not come from these two.
    from podcast_scraper.enrichment.eval.admission import known_enricher_manifests

    all_off = "enrichment:\n  enabled: true\n  enrichers:\n" + "".join(
        f"    {eid}:\n      enabled: false\n" for eid in known_enricher_manifests()
    )
    root_all_off = _corpus(tmp_path / "everything-off", all_off)
    fields_all_off = compute_enrichment_staleness(root_all_off)
    assert fields_all_off.reenrich_recommended is False, (
        "with every enricher disabled there is no outstanding enrichment work, so the "
        "rollup must be quiet"
    )
    assert fields_all_off.reenrich_reasons == []


def test_an_enabled_enricher_with_no_output_is_still_stale(tmp_path: Path) -> None:
    """The guard must not silence genuine never-ran enrichers — that would be worse than
    the bug it replaces."""
    root = _corpus(tmp_path, _YAML)
    by_id = {r.enricher_id: r for r in compute_enrichment_staleness(root).enrichers}
    others = [r for eid, r in by_id.items() if eid not in ("topic_similarity", "topic_consensus")]
    assert others, "expected other known enrichers in the manifest set"
    assert any(r.stale for r in others), "enabled enrichers with no output must remain stale"
