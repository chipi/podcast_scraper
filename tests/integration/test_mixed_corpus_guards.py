"""Guards on a mixed corpus (core release §3): another content type changes no podcast reader.

Two copies of the podcast validation corpus (GI 3.1, KG 2.0, 14 feeds with run dirs). Into one, a
batch of the test-only text type is run the way production runs a source, once per entry point:
``cli.main`` with a feeds spec naming the type, and ``service.run`` with a config whose source
entry names it. Its items land under ``sources/test_text_item/`` (RFC-134 §5) with KG 3.0, and the
batch stamps the corpus root, so the caches' corpus token moves.
One text item's KG is then given a podcast guest, a person only it names, and a podcast topic: the
shapes that would join podcast people and topics, or add people to the ledger, if they leaked.

The walks leaving ``sources/`` out is the first layer; the readers' own type checks (catalog,
entity candidates) are the second. Breaking both makes these tests fail; breaking one alone does
not, for the readers that have both.

Every corpus-wide reader must answer the same over both copies: membership, the catalog, the
enrichers' bundles, the migrations' selection, the KG and GI walks, the identity ledger and the
entity map, topic and insight clusters, the CIL bridge walk, and index staleness. A reader that asks
for the text type (catalog ``content_types``, ``include_sources``) sees it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pytest

from podcast_scraper import cli
from podcast_scraper.extensions import use_extensions
from podcast_scraper.utils.corpus_walk import corpus_rglob
from tests.integration.test_text_in_fixture_type import _ext, _transport

pytestmark = [pytest.mark.integration]

_PODCAST = Path(__file__).resolve().parents[1] / "fixtures" / "app-validation-corpus" / "v3"
_GUEST = {"id": "person:liam-verbeek", "type": "Person", "properties": {"name": "Liam Verbeek"}}
_AUTHOR = {"id": "person:text-only-author", "type": "Person", "properties": {"name": "Ana Byline"}}
_TOPIC = {
    "id": "topic:trail-building",
    "type": "Topic",
    "properties": {"label": "trail building", "slug": "trail-building"},
}


_URL = "https://example.com/fixture-source"


def _run_cli(base: Path, corpus: Path) -> None:
    spec = base / "feeds.spec.yaml"
    spec.write_text(
        f"feeds:\n  - url: {_URL}\n    content_type: test_text_item\n", encoding="utf-8"
    )

    def podcast_run(cfg: Any) -> Any:
        raise AssertionError("the podcast pipeline must not run for a text source")

    argv = [
        "--feeds-spec",
        str(spec),
        "--output-dir",
        str(corpus),
        "--no-transcript-cache",
        "--generate-metadata",
        "--generate-kg",
        "--kg-extraction-source",
        "metadata_only",
    ]
    assert cli.main(argv, run_pipeline_fn=podcast_run) == 0


def _run_service(base: Path, corpus: Path) -> None:
    from podcast_scraper import config, service

    cfg = config.Config.model_validate(
        {
            "rss_urls": [{"url": _URL, "content_type": "test_text_item"}],
            "output_dir": str(corpus),
            "single_feed_uses_corpus_layout": True,
            "generate_metadata": True,
            "generate_summaries": False,
            "generate_kg": True,
            "kg_extraction_source": "metadata_only",
        }
    )
    result = service.run(cfg)
    assert result.success, result.error


@pytest.fixture(scope="module", params=[_run_cli, _run_service], ids=["cli", "service"])
def corpora(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    from podcast_scraper.perf_cache import corpus_content_mtime

    base = tmp_path_factory.mktemp("mixed")
    podcast, mixed = base / "podcast", base / "mixed"
    shutil.copytree(_PODCAST, podcast)
    shutil.copytree(_PODCAST, mixed)

    # The fixture has no run summary or manifest at its root, so this is -1. corpus_mtime would
    # fall back to the root dir's own mtime, which creating sources/ moves even when the batch
    # stamps nothing at the root (the service path before 46a7a9a0).
    before = corpus_content_mtime(mixed)
    with use_extensions([_ext(_transport([]))]):
        request.param(base, mixed)
    assert corpus_content_mtime(mixed) > before, "the batch must stamp the corpus root"

    from podcast_scraper.kg.schema import validate_artifact

    text_kgs = sorted((mixed / "sources").rglob("*.kg.json"))
    assert len(text_kgs) == 2, text_kgs
    art = json.loads(text_kgs[0].read_text(encoding="utf-8"))
    art["nodes"].extend([_GUEST, _AUTHOR, _TOPIC])
    validate_artifact(art, strict=True)
    text_kgs[0].write_text(json.dumps(art), encoding="utf-8")
    return podcast, mixed


def _rel(root: Path, paths: Iterable[Optional[Path]]) -> List[str]:
    return sorted(Path(p).resolve().relative_to(root.resolve()).as_posix() for p in paths if p)


def test_the_text_items_are_there_for_a_reader_that_asks(corpora: tuple[Path, Path]) -> None:
    podcast, mixed = corpora
    extra = set(_rel(mixed, corpus_rglob(mixed, "*.kg.json", include_sources=True))) - set(
        _rel(podcast, corpus_rglob(podcast, "*.kg.json", include_sources=True))
    )
    assert len(extra) == 2 and all(p.startswith("sources/test_text_item/") for p in extra)


def test_membership_is_unchanged(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.search.corpus_scope import (
        discover_all_metadata_files,
        discover_metadata_files,
    )

    podcast, mixed = corpora
    for discover in (discover_metadata_files, discover_all_metadata_files):
        assert _rel(mixed, discover(mixed)) == _rel(podcast, discover(podcast))


def test_the_catalog_is_podcast_only_unless_asked(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.server.corpus_catalog import (
        build_catalog_rows,
        build_catalog_rows_cumulative,
    )

    podcast, mixed = corpora
    both = {"podcast_episode", "test_text_item"}
    for build in (build_catalog_rows, build_catalog_rows_cumulative):
        rows = sorted(r.metadata_relative_path for r in build(mixed))
        assert rows == sorted(r.metadata_relative_path for r in build(podcast))
        extra = set(r.metadata_relative_path for r in build(mixed, content_types=both)) - set(rows)
        assert len(extra) == 2 and all(p.startswith("sources/test_text_item/") for p in extra)
        only_text = build(mixed, content_types={"test_text_item"})
        assert {r.metadata_relative_path for r in only_text} == extra


def test_enrichers_and_migrations_select_the_same_episodes(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.enrichment.paths import discover_episode_bundles
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts

    podcast, mixed = corpora

    def bundles(root: Path) -> List[str]:
        return _rel(root, (b.metadata_path for b in discover_episode_bundles(root)))

    assert bundles(mixed) == bundles(podcast)
    for suffix in (".metadata.json", ".gi.json", ".kg.json", ".bridge.json"):
        served_m, superseded_m = select_served_artifacts(mixed, suffix)
        served_p, superseded_p = select_served_artifacts(podcast, suffix)
        assert _rel(mixed, served_m) == _rel(podcast, served_p), suffix
        assert _rel(mixed, superseded_m) == _rel(podcast, superseded_p), suffix


def test_kg_and_gi_walks_are_unchanged(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.gi.explore import scan_artifact_paths
    from podcast_scraper.identity.ledger import corpus_token
    from podcast_scraper.insight_stage.cil_queries import iter_cil_bridge_bundles
    from podcast_scraper.kg.corpus import newest_run_artifact_paths, scan_kg_artifact_paths

    podcast, mixed = corpora
    assert _rel(mixed, scan_kg_artifact_paths(mixed)) == _rel(
        podcast, scan_kg_artifact_paths(podcast)
    )
    assert _rel(mixed, scan_artifact_paths(mixed)) == _rel(podcast, scan_artifact_paths(podcast))
    newest = [
        _rel(r, newest_run_artifact_paths(r, scan_kg_artifact_paths(r), ".kg.json"))
        for r in (mixed, podcast)
    ]
    assert newest[0] == newest[1]
    assert corpus_token(mixed)["newest_kg_count"] == corpus_token(podcast)["newest_kg_count"]

    def bridges(root: Path) -> List[str]:
        return _rel(root, (Path(p) for p, _ in iter_cil_bridge_bundles(str(root), str(root))))

    assert bridges(mixed) == bridges(podcast)


def test_people_and_topics_join_nothing_new(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.insight_stage.insight_clusters import collect_insight_rows_from_corpus
    from podcast_scraper.kg.entity_clusters import build_entity_id_map, collect_entity_candidates
    from podcast_scraper.kg.topic_clustering import gather_corpus_topics

    podcast, mixed = corpora

    def candidates(root: Path) -> dict[str, tuple[Any, ...]]:
        return {
            k: (v.name, sorted(v.episodes), sorted(v.shows), sorted(v.languages), v._name_counts)
            for k, v in collect_entity_candidates(root).items()
        }

    assert candidates(mixed) == candidates(podcast)
    asked = collect_entity_candidates(mixed, content_types={"podcast_episode", "test_text_item"})
    assert "person:text-only-author" in asked
    assert build_entity_id_map(mixed) == build_entity_id_map(podcast)

    def topics(root: Path) -> List[tuple[str, str]]:
        return sorted((t.topic_id, t.episode_id) for t in gather_corpus_topics(root))

    assert topics(mixed) == topics(podcast)
    assert len(collect_insight_rows_from_corpus(mixed)) == len(
        collect_insight_rows_from_corpus(podcast)
    )


def test_the_identity_ledger_is_unchanged(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.identity.ledger import build_ledger

    podcast, mixed = corpora

    def ledger(root: Path) -> str:
        # Not compared: the token's corpus_mtime moves by design (the text batch stamped the
        # root), and built_at is the build's wall clock to the second.
        built = {
            k: v for k, v in build_ledger(root).items() if k not in ("corpus_token", "built_at")
        }
        return json.dumps(built, sort_keys=True, default=str).replace(str(root), "<R>")

    assert ledger(mixed) == ledger(podcast)


def test_index_staleness_ignores_the_text_items(corpora: tuple[Path, Path]) -> None:
    from podcast_scraper.search.index_source_mtime import newest_index_source_mtime_epoch

    podcast, mixed = corpora
    # copytree keeps mtimes, so the text items (written now) would be the newest if counted.
    assert newest_index_source_mtime_epoch(mixed, use_cache=False) == (
        newest_index_source_mtime_epoch(podcast, use_cache=False)
    )
