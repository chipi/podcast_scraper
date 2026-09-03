"""Full-chain: run the real corpus enrichers over a CHECKED-IN corpus and inspect the output.

The layer this fills. Unit tests pin the filler predicate; integration tests pin the chokepoints
and the HTTP boundary. All of them use fixtures written to exercise a specific rule. None of them
runs the actual enricher chain over a real corpus and asks the only question that matters at the
end: **is the output usable?**

That gap is not hypothetical. A DGX pipeline run produced 32 Topic nodes and the guard rejected
all 32 — a corpus with zero topics, hence no clustering, no co-occurrence, no trending. Every
existing test still passed, because each asserted its own rule in isolation and none asserted that
something survives. These tests assert BOTH directions over committed data: filler is gone, and
real topics are still there.

Deliberately no ML extras — the deterministic corpus enrichers need none, so this runs anywhere
including CI, which is the point of putting a real-corpus check here rather than in a manual
runbook.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.kg.filters import is_filler_topic

#: NOT marked ``e2e``, deliberately. PR CI runs ``pytest tests/e2e/ -m "e2e and critical_path"``,
#: and ``critical_path`` means the ingest chain (RSS -> transcribe -> NER -> summarize), which this
#: is not. Marked as an e2e test it would run only nightly — and a guardrail against "every topic
#: surface is empty" is worth having BEFORE a merge, not the morning after. It costs ~2s, needs no
#: ML extras and no network, so it belongs in the per-PR integration suite.
pytestmark = pytest.mark.integration

_CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "viewer-validation-corpus" / "v3"

#: Measured on this corpus, 2026-09-03. Asserted rather than described so a fixture change that
#: silently guts the topic set fails here instead of quietly weakening every downstream test.
_EXPECTED_DISTINCT_TOPICS = 13
_EXPECTED_FILLER = {
    "topic:welcome-back-to",
    "topic:great-to-be-back",
    "topic:excited-for-this-one",
    "topic:without-the",
}


@pytest.fixture(scope="module")
def enriched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A writable copy of the committed corpus with the real enrichers run over it."""
    if not _CORPUS.is_dir():
        pytest.skip(f"corpus fixture missing: {_CORPUS}")
    root = tmp_path_factory.mktemp("corpus")
    dest = root / "v3"
    shutil.copytree(_CORPUS, dest)

    # Through the CLI, deliberately: that is the command the runbook tells an operator to run
    # after a scoring change, so this exercises the orchestration and not just the enricher
    # functions. A test that bypasses the entry point cannot catch a wiring break in it.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "podcast_scraper.cli",
            "enrich",
            "--output-dir",
            str(dest),
            "--corpus-only",
            "--only",
            "temporal_velocity,topic_cooccurrence_corpus,topic_theme_clusters",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")},
        timeout=600,
    )
    assert proc.returncode == 0, f"enrichment CLI failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return dest


def _payload(root: Path, name: str) -> dict[str, Any]:
    envelope = json.loads((root / "enrichments" / f"{name}.json").read_text())
    data = envelope["data"]
    assert isinstance(data, dict), f"{name}: envelope has no data object"
    return data


def test_the_corpus_still_has_topics_after_filtering(enriched: Path) -> None:
    """THE regression: a guard that empties the corpus passes every rule-level test."""
    data = _payload(enriched, "temporal_velocity")
    topics = data.get("content_series", {}).get("topics") or []
    assert topics, (
        "the enrichment chain produced ZERO topics over a real corpus. Every downstream surface "
        "is empty: clustering, co-occurrence, trending, storylines. Check the filler guard before "
        "assuming the corpus is at fault."
    )


def test_no_filler_survives_to_any_topic_artifact(enriched: Path) -> None:
    """Sweep every emitted topic id across every artifact, not just one enricher's view."""
    seen: set[str] = set()
    for row in _payload(enriched, "temporal_velocity").get("content_series", {}).get("topics", []):
        seen.add(str(row.get("topic_id")))
    for pair in _payload(enriched, "topic_cooccurrence_corpus").get("pairs", []):
        seen.update({str(pair.get("topic_a_id")), str(pair.get("topic_b_id"))})
    for cluster in _payload(enriched, "topic_theme_clusters").get("clusters", []):
        seen.update(str(m.get("topic_id")) for m in cluster.get("members", []))

    leaked = seen & _EXPECTED_FILLER
    assert not leaked, f"filler reached a topic artifact: {sorted(leaked)}"


def test_the_filler_count_is_reported_by_every_enricher(enriched: Path) -> None:
    """An empty-ish artifact must be attributable. See the no-silent-fail contract."""
    for name in ("temporal_velocity", "topic_cooccurrence_corpus", "topic_theme_clusters"):
        data = _payload(enriched, name)
        assert "topics_filtered_as_filler" in data, f"{name} does not report what it removed"
        assert data["topics_filtered_as_filler"] > 0, (
            f"{name} reported 0 filtered on a corpus known to contain "
            f"{len(_EXPECTED_FILLER)} filler topics — the guard is not running here"
        )


def test_the_fixture_still_contains_what_these_tests_assume(enriched: Path) -> None:
    """Guard the guard. If the corpus is regenerated without filler these tests go vacuous."""
    ids: set[str] = set()
    # rglob: this corpus nests KGs under feeds/<id>/run_*/, not a flat metadata/ dir.
    for kg in enriched.rglob("*.kg.json"):
        for node in json.loads(kg.read_text()).get("nodes", []):
            if node.get("type") == "Topic":
                ids.add(str(node.get("id")))
    assert len(ids) == _EXPECTED_DISTINCT_TOPICS, (
        f"the corpus fixture changed shape ({len(ids)} topics, expected "
        f"{_EXPECTED_DISTINCT_TOPICS}) — re-derive the constants in this module"
    )
    assert _EXPECTED_FILLER <= ids, "the fixture no longer contains the filler these tests need"
    kept = {i for i in ids if not is_filler_topic(i.replace("topic:", "").replace("-", " "), i)}
    assert kept, "every topic in the fixture is filler — nothing left to prove"
