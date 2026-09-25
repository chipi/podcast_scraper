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

#: Measured on this corpus, 2026-09-24. Asserted rather than described so a fixture change that
#: silently guts the topic set fails here instead of quietly weakening every downstream test.
#:
#: Was 13 until the viewer corpus started reading its topics from the v3 ground truth instead of
#: a capitalised-phrase regex over the transcript (#2147). Every episode of a show used to carry
#: the same two or three feed-wide umbrella labels; each now leads with its own authored topic,
#: so the distinct count rose to 51 and the corpus can finally tell two episodes apart.
_EXPECTED_DISTINCT_TOPICS = 51

#: The filler the regex used to promote to Topic nodes. The corpus no longer contains ANY of
#: these — that is the fix, not a regression — so they are now what we inject to prove the guard
#: still runs (see ``enriched_with_filler``) and what we assert never appears in the shipped
#: fixture (see ``test_the_fixture_carries_no_filler_at_source``).
_EXPECTED_FILLER = {
    "topic:welcome-back-to",
    "topic:great-to-be-back",
    "topic:excited-for-this-one",
    "topic:without-the",
}


def _run_enrichers(dest: Path) -> None:
    """Run the three corpus enrichers over ``dest``, through the CLI.

    Through the CLI, deliberately: that is the command the runbook tells an operator to run
    after a scoring change, so this exercises the orchestration and not just the enricher
    functions. A test that bypasses the entry point cannot catch a wiring break in it.
    """
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
    assert (
        proc.returncode == 0
    ), f"enrichment CLI failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"


@pytest.fixture(scope="module")
def enriched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A writable copy of the committed corpus with the real enrichers run over it."""
    if not _CORPUS.is_dir():
        pytest.skip(f"corpus fixture missing: {_CORPUS}")
    dest = tmp_path_factory.mktemp("corpus") / "v3"
    shutil.copytree(_CORPUS, dest)
    _run_enrichers(dest)
    return dest


def _payload(root: Path, name: str) -> dict[str, Any]:
    envelope = json.loads((root / "enrichments" / f"{name}.json").read_text())
    data = envelope["data"]
    assert isinstance(data, dict), f"{name}: envelope has no data object"
    return data


def _topic_ids(root: Path) -> set[str]:
    """Every Topic node id in the corpus.

    rglob: this corpus nests KGs under feeds/<id>/run_*/ as well as a flat metadata/ dir.
    """
    ids: set[str] = set()
    for kg in root.rglob("*.kg.json"):
        for node in json.loads(kg.read_text()).get("nodes", []):
            if node.get("type") == "Topic":
                ids.add(str(node.get("id")))
    return ids


@pytest.fixture(scope="module")
def enriched_with_filler(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same corpus with filler Topic nodes injected, then enriched.

    The guard's "it actually runs" claim used to rest on the shipped fixture containing
    greeting topics. That made a corpus fix look like a test regression and put pressure on
    the wrong thing. Injecting the filler here decouples the two: the fixture can be clean
    and the guard can still be proven to fire.
    """
    if not _CORPUS.is_dir():
        pytest.skip(f"corpus fixture missing: {_CORPUS}")
    dest = tmp_path_factory.mktemp("corpus_filler") / "v3"
    shutil.copytree(_CORPUS, dest)

    targets = sorted(dest.rglob("*.kg.json"))
    assert targets, "no KG artifacts to inject into"
    doc = json.loads(targets[0].read_text())
    episode_node = next((n["id"] for n in doc.get("nodes", []) if n.get("type") == "Episode"), None)
    for topic_id in sorted(_EXPECTED_FILLER):
        label = topic_id.removeprefix("topic:").replace("-", " ")
        doc["nodes"].append(
            {
                "id": topic_id,
                "type": "Topic",
                "properties": {"label": label, "slug": topic_id.removeprefix("topic:")},
            }
        )
        if episode_node:
            doc.setdefault("edges", []).append(
                {"type": "MENTIONS", "from": episode_node, "to": topic_id}
            )
    targets[0].write_text(json.dumps(doc), encoding="utf-8")
    _run_enrichers(dest)
    return dest


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


def test_every_enricher_reports_what_it_removed(enriched: Path) -> None:
    """An empty-ish artifact must be attributable. See the no-silent-fail contract.

    The FIELD is the contract, not a positive value. This used to assert ``> 0``, which held
    only because the shipped corpus contained filler; it now reports 0 because the corpus is
    clean at source. ``test_the_guard_still_fires_when_filler_is_present`` proves the guard
    runs, against a corpus dirtied on purpose — the right place for that claim, since it does
    not need the fixture to stay broken to stay true.
    """
    for name in ("temporal_velocity", "topic_cooccurrence_corpus", "topic_theme_clusters"):
        data = _payload(enriched, name)
        assert "topics_filtered_as_filler" in data, f"{name} does not report what it removed"
        assert isinstance(data["topics_filtered_as_filler"], int), (
            f"{name} reports topics_filtered_as_filler as "
            f"{type(data['topics_filtered_as_filler']).__name__}, not a count"
        )


def test_the_guard_still_fires_when_filler_is_present(enriched_with_filler: Path) -> None:
    """The guard runs and is attributable — proven by injection, not by a dirty fixture."""
    data = _payload(enriched_with_filler, "temporal_velocity")
    assert data["topics_filtered_as_filler"] > 0, (
        "temporal_velocity reported 0 filtered on a corpus with "
        f"{len(_EXPECTED_FILLER)} filler topics injected into it — the guard is not running"
    )
    seen = {str(row.get("topic_id")) for row in data.get("content_series", {}).get("topics", [])}
    leaked = seen & _EXPECTED_FILLER
    assert not leaked, f"injected filler survived the guard: {sorted(leaked)}"


def test_the_fixture_still_contains_what_these_tests_assume(enriched: Path) -> None:
    """Guard the guard — the corpus must stay rich enough to be worth enriching."""
    ids = _topic_ids(enriched)
    assert len(ids) == _EXPECTED_DISTINCT_TOPICS, (
        f"the corpus fixture changed shape ({len(ids)} topics, expected "
        f"{_EXPECTED_DISTINCT_TOPICS}) — re-derive the constants in this module"
    )
    kept = {i for i in ids if not is_filler_topic(i.replace("topic:", "").replace("-", " "), i)}
    assert kept, "every topic in the fixture is filler — nothing left to prove"


def test_the_fixture_carries_no_filler_at_source(enriched: Path) -> None:
    """The shipped corpus must be clean BEFORE any guard runs.

    Filtering filler downstream and shipping a fixture full of it are not the same thing. The
    viewer corpus used to carry ``topic:welcome-back-to`` as a real Topic node, and
    ``cli topic-clusters`` promoted it to ``tc:welcome-back-to`` — a greeting rendered as a
    theme, on a surface no filler guard sits in front of. Fix the source, then assert it.
    """
    leaked = _topic_ids(enriched) & _EXPECTED_FILLER
    assert not leaked, (
        f"the corpus fixture carries filler topics at source: {sorted(leaked)}. "
        "Rebuild it with scripts/build_synthetic_validation_corpus.py."
    )
