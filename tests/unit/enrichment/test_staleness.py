"""Unit tests for enrichment staleness (#1649).

The property under test is not "does it skip work" — it is **does it skip the RIGHT work**.
An incrementality bug here is worse than no incrementality: a full-corpus pass that skips
everything reports success in seconds and silently leaves the corpus unrepaired, which is
exactly the failure mode #1646 taught us to distrust.

The central test is ``test_upstream_gi_change_invalidates_downstream_enrichment``: keying only
on ``enricher_version``/``schema_version`` — the fields already persisted, and therefore the
tempting choice — would make the entire corpus repair a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.staleness import (
    EnrichmentRunStats,
    envelope_is_current,
    input_fingerprint,
    load_envelope,
    StalenessDecision,
)

pytestmark = [pytest.mark.unit]

VERSION = "1.1.0"
SCHEMA = "1.0"


def _gi(tmp_path: Path, name: str = "ep.gi.json", *, insights: int = 2) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps({"nodes": [{"type": "Insight", "id": f"i{n}"} for n in range(insights)]}),
        encoding="utf-8",
    )
    return path


def _envelope(fingerprint: str | None, **overrides) -> dict:
    payload = {
        "enricher_version": VERSION,
        "schema_version": SCHEMA,
        "computed_at": "2026-08-14T10:00:00Z",
    }
    if fingerprint is not None:
        payload["input_fingerprint"] = fingerprint
    payload.update(overrides)
    return payload


class TestInputFingerprint:
    def test_is_stable_for_identical_content(self, tmp_path: Path) -> None:
        gi = _gi(tmp_path)
        assert input_fingerprint([gi]) == input_fingerprint([gi])

    def test_changes_when_content_changes(self, tmp_path: Path) -> None:
        gi = _gi(tmp_path)
        before = input_fingerprint([gi])
        gi.write_text(json.dumps({"nodes": [{"type": "Insight", "id": "different"}]}), "utf-8")
        assert input_fingerprint([gi]) != before

    def test_is_content_based_not_mtime_based(self, tmp_path: Path) -> None:
        """This pipeline rewrites identical files routinely (C2 in #1654).

        An mtime key would report every rewrite as a change, degrading incrementality to
        "always run" — which is indistinguishable from having none.
        """
        gi = _gi(tmp_path)
        before = input_fingerprint([gi])
        content = gi.read_text(encoding="utf-8")
        gi.write_text(content, encoding="utf-8")  # same bytes, new mtime
        assert input_fingerprint([gi]) == before

    def test_none_paths_are_ignored(self, tmp_path: Path) -> None:
        gi = _gi(tmp_path)
        assert input_fingerprint([gi, None]) == input_fingerprint([gi])

    def test_unreadable_inputs_yield_none_not_a_stable_hash(self, tmp_path: Path) -> None:
        """A stable-looking hash over nothing would let a broken episode be skipped forever."""
        assert input_fingerprint([tmp_path / "does-not-exist.json"]) is None
        assert input_fingerprint([]) is None

    def test_order_does_not_matter(self, tmp_path: Path) -> None:
        gi = _gi(tmp_path, "a.gi.json")
        kg = _gi(tmp_path, "b.kg.json")
        assert input_fingerprint([gi, kg]) == input_fingerprint([kg, gi])


class TestEnvelopeIsCurrent:
    def test_unchanged_inputs_skip(self, tmp_path: Path) -> None:
        fingerprint = input_fingerprint([_gi(tmp_path)])
        decision = envelope_is_current(
            _envelope(fingerprint),
            fingerprint=fingerprint,
            enricher_version=VERSION,
            schema_version=SCHEMA,
        )
        assert decision.should_run is False
        assert decision.reason == "inputs_unchanged"

    def test_upstream_gi_change_invalidates_downstream_enrichment(self, tmp_path: Path) -> None:
        """THE test for this module (#1649).

        Fix speaker attribution upstream, re-run enrichment: the enricher version has not
        changed, so a version-only key would skip all 678 episodes and report a green run
        while repairing nothing. The input fingerprint is what makes the repair actually run.
        """
        gi = _gi(tmp_path)
        before = input_fingerprint([gi])
        old_envelope = _envelope(before)

        # Upstream repair rewrites GI with named speakers.
        gi.write_text(
            json.dumps({"nodes": [{"type": "Insight", "properties": {"speaker": "Simon Last"}}]}),
            encoding="utf-8",
        )
        after = input_fingerprint([gi])

        decision = envelope_is_current(
            old_envelope,
            fingerprint=after,
            enricher_version=VERSION,  # deliberately UNCHANGED
            schema_version=SCHEMA,  # deliberately UNCHANGED
        )
        assert decision.should_run is True
        assert decision.reason == "inputs_changed"

    def test_no_previous_output_runs(self) -> None:
        decision = envelope_is_current(
            None, fingerprint="abc", enricher_version=VERSION, schema_version=SCHEMA
        )
        assert decision.should_run is True
        assert decision.reason == "no_previous_output"

    def test_enricher_version_bump_runs(self, tmp_path: Path) -> None:
        fingerprint = input_fingerprint([_gi(tmp_path)])
        decision = envelope_is_current(
            _envelope(fingerprint),
            fingerprint=fingerprint,
            enricher_version="2.0.0",
            schema_version=SCHEMA,
        )
        assert decision.should_run is True
        assert decision.reason == "enricher_version_changed"

    def test_schema_version_bump_runs(self, tmp_path: Path) -> None:
        fingerprint = input_fingerprint([_gi(tmp_path)])
        decision = envelope_is_current(
            _envelope(fingerprint),
            fingerprint=fingerprint,
            enricher_version=VERSION,
            schema_version="2.0",
        )
        assert decision.should_run is True
        assert decision.reason == "schema_version_changed"

    def test_output_without_a_fingerprint_runs_once_to_acquire_one(self, tmp_path: Path) -> None:
        """Pre-#1649 output must not be frozen as permanently current."""
        decision = envelope_is_current(
            _envelope(None),
            fingerprint="abc",
            enricher_version=VERSION,
            schema_version=SCHEMA,
        )
        assert decision.should_run is True
        assert decision.reason == "no_recorded_fingerprint"

    def test_unreadable_inputs_run_rather_than_skip(self) -> None:
        decision = envelope_is_current(
            _envelope("abc"), fingerprint=None, enricher_version=VERSION, schema_version=SCHEMA
        )
        assert decision.should_run is True
        assert decision.reason == "inputs_unreadable"

    def test_a_non_dict_envelope_is_treated_as_absent(self) -> None:
        decision = envelope_is_current(
            ["not", "a", "dict"],  # type: ignore[arg-type]
            fingerprint="abc",
            enricher_version=VERSION,
            schema_version=SCHEMA,
        )
        assert decision.should_run is True


class TestLoadEnvelope:
    def test_reads_a_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "e.json"
        path.write_text(json.dumps({"enricher_version": "1"}), encoding="utf-8")
        assert load_envelope(path) == {"enricher_version": "1"}

    def test_missing_or_malformed_is_none(self, tmp_path: Path) -> None:
        assert load_envelope(tmp_path / "nope.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_envelope(bad) is None


class TestEnrichmentRunStats:
    def test_counts_and_reasons(self) -> None:
        stats = EnrichmentRunStats()
        stats.record(StalenessDecision(True, "inputs_changed"))
        stats.record(StalenessDecision(False, "inputs_unchanged"))
        stats.record(StalenessDecision(False, "inputs_unchanged"))
        payload = stats.as_dict()
        assert payload["episodes_total"] == 3
        assert payload["episodes_enriched"] == 1
        assert payload["episodes_skipped_unchanged"] == 2
        assert payload["reasons"] == {"inputs_changed": 1, "inputs_unchanged": 2}

    def test_reasons_make_a_useless_skip_visible(self) -> None:
        """ "612 skipped" is fine; "612 skipped: no_recorded_fingerprint" is a broken key."""
        stats = EnrichmentRunStats()
        for _ in range(3):
            stats.record(StalenessDecision(True, "no_recorded_fingerprint"))
        assert stats.as_dict()["reasons"] == {"no_recorded_fingerprint": 3}
