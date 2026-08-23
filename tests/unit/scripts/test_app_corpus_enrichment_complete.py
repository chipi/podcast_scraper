"""The app-validation corpus must carry a COMPLETE enrichment layer.

Why this exists (2026-08-16)
----------------------------
``enrichments/`` has two writers and neither owns the whole directory:

* ``build_app_validation_corpus.py`` authors four files itself — ``temporal_velocity``,
  ``topic_theme_clusters``, ``topic_similarity``, ``topic_consensus`` — deterministically, so the
  fixture has stable, non-empty data for the Trending / Storylines surfaces.
* The other five (``grounding_rate``, ``guest_coappearance``, ``topic_cooccurrence_corpus``, plus
  the executor's own ``run.jsonl`` / ``run_summary.json``) come from the enrichment FRAMEWORK
  (``cli enrich``), which the builder never runs.

A normal rebuild is safe — the builder overwrites its four and leaves the rest alone (verified).
The failure mode is a rebuild that REPLACES the directory rather than writing into it: the five
framework-only files disappear and nothing notices, because no consumer errors on a missing
enrichment — the surfaces just quietly render empty. That happened once during the pipeline
migration and was caught by eye, not by a test.

These tests are that check. They assert presence and shape only, never content, so a legitimate
re-enrichment does not fail them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
ENRICH = CORPUS / "enrichments"

# Authored by build_app_validation_corpus.py.
BUILDER_AUTHORED = {
    "temporal_velocity.json",
    "topic_theme_clusters.json",
    "topic_similarity.json",
    "topic_consensus.json",
}
# Produced only by the enrichment framework (`cli enrich`); the builder never writes these.
FRAMEWORK_ONLY = {
    "grounding_rate.json",
    "guest_coappearance.json",
    "topic_cooccurrence_corpus.json",
    "run_summary.json",
    "run.jsonl",
}
REQUIRED = BUILDER_AUTHORED | FRAMEWORK_ONLY


def _episode_count() -> int:
    return len([p for p in CORPUS.rglob("*.metadata.json") if "enrichments" not in p.parts])


class TestCorpusLevelEnrichment:
    def test_corpus_exists(self) -> None:
        assert CORPUS.is_dir(), f"corpus missing: {CORPUS}"
        assert ENRICH.is_dir(), f"enrichments dir missing: {ENRICH}"

    def test_all_required_enrichments_present(self) -> None:
        present = {p.name for p in ENRICH.iterdir() if p.is_file()}
        missing = sorted(REQUIRED - present)
        assert not missing, (
            f"enrichment sidecars missing from the corpus: {missing}. "
            "A rebuild that REPLACES enrichments/ (rather than writing into it) drops the "
            "framework-produced files; regenerate them with "
            "`cli enrich --output-dir <corpus> --profile homelab_balanced --with-ml` in a "
            "container, since the ML enrichers need the [search] extra."
        )

    def test_framework_only_files_are_not_silently_lost(self) -> None:
        """The specific five a destructive rebuild removes — named so the failure is obvious."""
        present = {p.name for p in ENRICH.iterdir() if p.is_file()}
        lost = sorted(FRAMEWORK_ONLY - present)
        assert not lost, (
            f"framework-produced enrichments are gone: {lost}. The corpus builder does NOT write "
            "these, so they cannot be restored by re-running it."
        )

    def test_enrichment_payloads_parse(self) -> None:
        bad: list[str] = []
        for name in sorted(REQUIRED):
            path = ENRICH / name
            if not path.is_file():
                continue
            try:
                if path.suffix == ".jsonl":
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            json.loads(line)
                else:
                    json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                bad.append(f"{name}: {type(exc).__name__}")
        assert not bad, f"unparseable enrichment sidecars: {bad}"


class TestPerEpisodeEnrichment:
    """insight_density / insight_sentiment are per-episode and feed the player's density band."""

    @pytest.mark.parametrize("kind", ["insight_density", "insight_sentiment"])
    def test_one_sidecar_per_episode(self, kind: str) -> None:
        episodes = _episode_count()
        assert episodes > 0, "no episodes found — corpus layout changed?"
        found = len(list(CORPUS.rglob(f"*.{kind}.json")))
        assert found == episodes, (
            f"{kind}: {found} sidecars for {episodes} episodes. Every episode should have one; "
            "a partial set means enrichment ran over a subset of the corpus."
        )


class TestMatchesProfileMatrix:
    """The corpus should carry what its profile says it runs — one source of truth."""

    def test_no_enricher_in_the_matrix_is_unrepresented(self) -> None:
        from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile

        expected = set(enricher_set_for_profile("homelab_balanced").enabled_enrichers)
        assert expected, "homelab_balanced resolved to an EMPTY enricher set"

        corpus_level = {p.stem for p in ENRICH.iterdir() if p.suffix == ".json"}
        per_episode = {"insight_density", "insight_sentiment"}
        represented = corpus_level | {k for k in per_episode if any(CORPUS.rglob(f"*.{k}.json"))}
        unrepresented = sorted(expected - represented)
        assert not unrepresented, (
            f"the profile's enricher set includes {unrepresented}, but the corpus has no artifact "
            "for them — either enrichment did not run to completion, or the corpus predates them."
        )
