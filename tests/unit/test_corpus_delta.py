"""Unit tests for the shared corpus-delta backbone (RFC-118 PR0).

The reconciliation property that makes delta safe to ship lives here at the
fingerprint layer: same content ⇒ same fingerprint regardless of mtime; any
content change, artifact appearance, or artifact loss ⇒ a different fingerprint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from podcast_scraper.corpus_delta import (
    compute_corpus_delta,
    CorpusDelta,
    episode_derivation_fingerprint,
    FINGERPRINT_SCHEMA_VERSION,
    load_fingerprint_manifest,
    manifest_path,
    write_fingerprint_manifest,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle


def _bundle(tmp_path: Path, eid: str, *, gi: str | None = "{}", kg: str | None = "{}"):
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_dir / f"{eid}.metadata.json"
    meta.write_text(json.dumps({"episode": {"guid": eid}}), encoding="utf-8")
    gi_path = kg_path = None
    if gi is not None:
        gi_path = meta_dir / f"{eid}.gi.json"
        gi_path.write_text(gi, encoding="utf-8")
    if kg is not None:
        kg_path = meta_dir / f"{eid}.kg.json"
        kg_path.write_text(kg, encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=meta,
        gi_path=gi_path,
        kg_path=kg_path,
        bridge_path=None,
        episode_id=eid,
        stem=eid,
    )


class TestEpisodeDerivationFingerprint:
    def test_stable_across_mtime(self, tmp_path):
        b = _bundle(tmp_path, "ep1", gi='{"a": 1}')
        fp1 = episode_derivation_fingerprint(b)
        os.utime(b.gi_path, (1, 1))
        assert episode_derivation_fingerprint(b) == fp1

    def test_changes_on_gi_content_change(self, tmp_path):
        b = _bundle(tmp_path, "ep1", gi='{"a": 1}')
        fp1 = episode_derivation_fingerprint(b)
        b.gi_path.write_text('{"a": 2}', encoding="utf-8")
        assert episode_derivation_fingerprint(b) != fp1

    def test_changes_on_kg_content_change(self, tmp_path):
        b = _bundle(tmp_path, "ep1", kg='{"k": 1}')
        fp1 = episode_derivation_fingerprint(b)
        b.kg_path.write_text('{"k": 2}', encoding="utf-8")
        assert episode_derivation_fingerprint(b) != fp1

    def test_absent_differs_from_empty(self, tmp_path):
        with_empty = _bundle(tmp_path / "a", "ep1", gi="")
        without = _bundle(tmp_path / "b", "ep1", gi=None)
        assert episode_derivation_fingerprint(with_empty) != episode_derivation_fingerprint(without)

    def test_gi_and_kg_are_positionally_distinct(self, tmp_path):
        gi_only = _bundle(tmp_path / "a", "ep1", gi="same", kg=None)
        kg_only = _bundle(tmp_path / "b", "ep1", gi=None, kg="same")
        assert episode_derivation_fingerprint(gi_only) != episode_derivation_fingerprint(kg_only)


class TestManifestRoundTrip:
    def test_round_trip(self, tmp_path):
        write_fingerprint_manifest(tmp_path, {"ep1": "abc", "ep2": "def"})
        assert load_fingerprint_manifest(tmp_path) == {"ep1": "abc", "ep2": "def"}

    def test_missing_manifest_is_empty(self, tmp_path):
        assert load_fingerprint_manifest(tmp_path) == {}

    def test_corrupt_manifest_is_empty(self, tmp_path):
        manifest_path(tmp_path).write_text("not-json{", encoding="utf-8")
        assert load_fingerprint_manifest(tmp_path) == {}

    def test_schema_mismatch_is_empty(self, tmp_path):
        manifest_path(tmp_path).write_text(
            json.dumps({"schema": FINGERPRINT_SCHEMA_VERSION + 1, "fingerprints": {"ep1": "x"}}),
            encoding="utf-8",
        )
        assert load_fingerprint_manifest(tmp_path) == {}

    def test_no_tmp_file_left_behind(self, tmp_path):
        write_fingerprint_manifest(tmp_path, {"ep1": "abc"})
        assert [p.name for p in tmp_path.iterdir()] == ["derivation_fingerprints.json"]


class TestComputeCorpusDelta:
    def test_first_run_everything_changed(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        delta = compute_corpus_delta(tmp_path, bundles)
        assert delta.changed_ids == {"ep1", "ep2"}
        assert delta.removed_ids == frozenset()
        assert not delta.is_empty

    def test_second_run_unchanged_is_empty(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        first = compute_corpus_delta(tmp_path, bundles)
        write_fingerprint_manifest(tmp_path, first.fingerprints)
        second = compute_corpus_delta(tmp_path, bundles)
        assert second.is_empty
        assert second.changed_ids == frozenset()

    def test_one_episode_edit_yields_one_changed(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        write_fingerprint_manifest(tmp_path, compute_corpus_delta(tmp_path, bundles).fingerprints)
        bundles[0].gi_path.write_text('{"edited": true}', encoding="utf-8")
        delta = compute_corpus_delta(tmp_path, bundles)
        assert delta.changed_ids == {"ep1"}
        assert delta.removed_ids == frozenset()

    def test_removed_episode_detected(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        write_fingerprint_manifest(tmp_path, compute_corpus_delta(tmp_path, bundles).fingerprints)
        delta = compute_corpus_delta(tmp_path, [bundles[0]])
        assert delta.changed_ids == frozenset()
        assert delta.removed_ids == {"ep2"}
        assert not delta.is_empty

    def test_force_marks_all_changed(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        write_fingerprint_manifest(tmp_path, compute_corpus_delta(tmp_path, bundles).fingerprints)
        delta = compute_corpus_delta(tmp_path, bundles, force=True)
        assert delta.changed_ids == {"ep1", "ep2"}
        assert delta.forced
        assert not delta.is_empty

    def test_discovers_bundles_when_none_given(self, tmp_path):
        _bundle(tmp_path, "ep1")
        delta = compute_corpus_delta(tmp_path)
        assert delta.changed_ids == {"ep1"}

    def test_fingerprints_cover_all_bundles(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        delta = compute_corpus_delta(tmp_path, bundles)
        assert set(delta.fingerprints) == {"ep1", "ep2"}

    def test_summary_shape(self, tmp_path):
        delta = compute_corpus_delta(tmp_path, [_bundle(tmp_path, "ep1")])
        assert delta.summary() == {"changed": 1, "removed": 0, "total": 1, "forced": False}


class TestCorpusDeltaContract:
    def test_frozen(self):
        delta = CorpusDelta(changed_ids=frozenset(), removed_ids=frozenset(), all_bundles=[])
        with pytest.raises(AttributeError):
            delta.forced = True  # type: ignore[misc]

    def test_changed_metadata_relpaths_maps_ids_to_paths(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        delta = compute_corpus_delta(tmp_path, bundles)
        relpaths = delta.changed_metadata_relpaths(tmp_path)
        assert relpaths == [
            "metadata/ep1.metadata.json",
            "metadata/ep2.metadata.json",
        ]

    def test_changed_metadata_relpaths_only_changed(self, tmp_path):
        bundles = [_bundle(tmp_path, "ep1"), _bundle(tmp_path, "ep2")]
        write_fingerprint_manifest(tmp_path, compute_corpus_delta(tmp_path, bundles).fingerprints)
        bundles[1].kg_path.write_text('{"edited": 1}', encoding="utf-8")
        delta = compute_corpus_delta(tmp_path, bundles)
        assert delta.changed_metadata_relpaths(tmp_path) == ["metadata/ep2.metadata.json"]


class TestComputeCorpusDeltaOnMultiFeedFixture:
    """The committed multi-feed corpus fixture: real feeds/<slug>/run_<id>/metadata layout."""

    FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "app-validation-corpus" / "v3"

    @pytest.fixture()
    def corpus(self, tmp_path):
        import shutil

        dest = tmp_path / "corpus"
        shutil.copytree(self.FIXTURE, dest)
        return dest

    def test_first_run_then_idempotent(self, corpus):
        first = compute_corpus_delta(corpus)
        assert len(first.all_bundles) > 0
        assert first.changed_ids == frozenset(first.fingerprints)
        write_fingerprint_manifest(corpus, first.fingerprints)
        second = compute_corpus_delta(corpus)
        assert second.is_empty, (
            f"delta not idempotent on unchanged fixture corpus: "
            f"changed={sorted(second.changed_ids)[:5]} removed={sorted(second.removed_ids)[:5]}"
        )

    def test_single_episode_gi_edit_scopes_to_one(self, corpus):
        first = compute_corpus_delta(corpus)
        write_fingerprint_manifest(corpus, first.fingerprints)
        target = next(b for b in first.all_bundles if b.gi_path is not None)
        target.gi_path.write_text('{"edited": true}', encoding="utf-8")
        delta = compute_corpus_delta(corpus)
        assert delta.changed_ids == {target.episode_id}
        assert delta.changed_metadata_relpaths(corpus) == [
            target.metadata_path.resolve().relative_to(corpus.resolve()).as_posix()
        ]
