"""Unit tests for v2.6.1 helpers (#818/#820/#821/#823).

Covers the new functions added by the hotfix:

- ``podcast_scraper.search.corpus_scope.discover_all_metadata_files``
- ``podcast_scraper.server.corpus_catalog.build_catalog_rows_cumulative``
- ``podcast_scraper.workflow.corpus_cost_aggregation.aggregate_corpus_costs``
  (uninstrumented-cost detection branches)

Integration tests in ``tests/integration/test_multi_run_corpus_fixture.py``
exercise the happy path against a generated fixture; these unit tests
focus on edge cases + error paths to lift patch coverage above codecov's
threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.corpus_scope import (
    discover_all_metadata_files,
    discover_metadata_files,
)
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative
from podcast_scraper.workflow.corpus_cost_aggregation import aggregate_corpus_costs

# -----------------------------------------------------------------------------
# discover_all_metadata_files
# -----------------------------------------------------------------------------


def test_discover_all_metadata_files_empty_corpus(tmp_path: Path) -> None:
    """Empty corpus returns an empty list (no metadata/ or feeds/)."""
    assert discover_all_metadata_files(tmp_path) == []


def test_discover_all_metadata_files_nonexistent_root(tmp_path: Path) -> None:
    """Non-existent root returns empty (safe_resolve_directory returns None)."""
    assert discover_all_metadata_files(tmp_path / "does-not-exist") == []


def test_discover_all_metadata_files_flat_layout(tmp_path: Path) -> None:
    """Single-feed flat layout (top-level metadata/) — also discovered."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "ep01.metadata.json").write_text("{}")
    (meta_dir / "ep02.metadata.json").write_text("{}")
    files = discover_all_metadata_files(tmp_path)
    assert len(files) == 2
    assert all(f.name.endswith(".metadata.json") for f in files)


def test_discover_all_metadata_files_keeps_all_runs(tmp_path: Path) -> None:
    """Multiple runs per feed: ALL files are returned (no last-run filter).

    Contrast with :func:`discover_metadata_files` which drops older runs.
    """
    feed = tmp_path / "feeds" / "rss_feed_a"
    for run_name in ("run_001", "run_002", "run_003"):
        meta_dir = feed / run_name / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / f"{run_name}_ep01.metadata.json").write_text("{}")

    all_files = discover_all_metadata_files(tmp_path)
    last_run_only = discover_metadata_files(tmp_path)
    assert len(all_files) == 3, "discover_all sees every run's metadata file"
    assert len(last_run_only) == 1, "discover_metadata_files keeps only latest"


# -----------------------------------------------------------------------------
# build_catalog_rows_cumulative
# -----------------------------------------------------------------------------


def _write_metadata(
    path: Path,
    *,
    feed_id: str,
    feed_title: str,
    episode_id: str,
    episode_title: str,
) -> None:
    """Write a minimal valid metadata.json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "feed": {
                    "feed_id": feed_id,
                    "title": feed_title,
                    "url": f"https://example.invalid/{feed_id}.rss",
                },
                "episode": {
                    "episode_id": episode_id,
                    "guid": episode_id,
                    "title": episode_title,
                    "published_date": "2026-01-01T00:00:00+00:00",
                },
            }
        )
    )


def test_build_catalog_rows_cumulative_empty(tmp_path: Path) -> None:
    """Empty corpus returns empty list (no crash on missing feeds/)."""
    assert build_catalog_rows_cumulative(tmp_path) == []


def test_build_catalog_rows_cumulative_single_run(tmp_path: Path) -> None:
    """Single run, no dedup needed: returns one row per metadata file."""
    feed = tmp_path / "feeds" / "rss_feed_a" / "run_001" / "metadata"
    _write_metadata(
        feed / "ep01.metadata.json",
        feed_id="sha256:feed_a",
        feed_title="Feed A",
        episode_id="ep_001",
        episode_title="Episode 1",
    )
    _write_metadata(
        feed / "ep02.metadata.json",
        feed_id="sha256:feed_a",
        feed_title="Feed A",
        episode_id="ep_002",
        episode_title="Episode 2",
    )
    rows = build_catalog_rows_cumulative(tmp_path)
    assert len(rows) == 2
    eids = {r.episode_id for r in rows}
    assert eids == {"ep_001", "ep_002"}


def test_build_catalog_rows_cumulative_dedupes_across_runs(tmp_path: Path) -> None:
    """Episode in multiple runs: the latest-run row wins, count stays unique."""
    feed = tmp_path / "feeds" / "rss_feed_a"

    # Older run with ep_001 + ep_002.
    older = feed / "run_001" / "metadata"
    _write_metadata(
        older / "ep01.metadata.json",
        feed_id="sha256:feed_a",
        feed_title="Feed A",
        episode_id="ep_001",
        episode_title="Episode 1 OLD",
    )
    _write_metadata(
        older / "ep02.metadata.json",
        feed_id="sha256:feed_a",
        feed_title="Feed A",
        episode_id="ep_002",
        episode_title="Episode 2",
    )

    # Newer run: ep_001 reappears (skip_existing didn't catch it) + new ep_003.
    newer = feed / "run_002" / "metadata"
    _write_metadata(
        newer / "ep01.metadata.json",
        feed_id="sha256:feed_a",
        feed_title="Feed A",
        episode_id="ep_001",
        episode_title="Episode 1 NEW",
    )
    _write_metadata(
        newer / "ep03.metadata.json",
        feed_id="sha256:feed_a",
        feed_title="Feed A",
        episode_id="ep_003",
        episode_title="Episode 3",
    )

    rows = build_catalog_rows_cumulative(tmp_path)
    assert len(rows) == 3, "cumulative-unique = 3 episodes (ep_001 dedup'd)"

    by_id = {r.episode_id: r for r in rows}
    # ep_001's latest-run row should win — its metadata path includes run_002.
    assert "run_002" in by_id["ep_001"].metadata_relative_path
    # The row's episode_title should match the newer version, not OLD.
    assert by_id["ep_001"].episode_title == "Episode 1 NEW"


def test_build_catalog_rows_cumulative_handles_multiple_feeds(tmp_path: Path) -> None:
    """Dedup is scoped per (feed_id, episode_id) — same episode_id across feeds OK."""
    for feed_id, feed_title in [("sha256:feed_a", "Feed A"), ("sha256:feed_b", "Feed B")]:
        feed_dir = tmp_path / "feeds" / feed_id.replace("sha256:", "rss_")
        meta = feed_dir / "run_001" / "metadata"
        _write_metadata(
            meta / "ep01.metadata.json",
            feed_id=feed_id,
            feed_title=feed_title,
            episode_id="ep_001",
            episode_title=f"{feed_title} ep 1",
        )

    rows = build_catalog_rows_cumulative(tmp_path)
    assert len(rows) == 2, "same episode_id under different feeds = 2 distinct rows"


# -----------------------------------------------------------------------------
# aggregate_corpus_costs — uninstrumented-detection branches (#823)
# -----------------------------------------------------------------------------


def _write_run_with_metrics(
    feed_dir: Path,
    run_name: str,
    *,
    cost_fields_present: bool,
    cost_value: float = 0.0,
) -> None:
    """Helper: write a run dir with a metrics.json file."""
    run_dir = feed_dir / run_name
    run_dir.mkdir(parents=True)
    metrics: dict = {"schema_version": "1.0.0", "episodes_scraped_total": 5}
    if cost_fields_present:
        metrics.update(
            {
                "llm_transcription_cost_usd": cost_value,
                "llm_summarization_cost_usd": cost_value,
                "llm_speaker_detection_cost_usd": 0.0,
                "llm_cleaning_cost_usd": 0.0,
                "llm_gi_cost_usd": 0.0,
                "llm_kg_cost_usd": 0.0,
                "llm_bundled_clean_summary_cost_usd": 0.0,
            }
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_aggregate_corpus_costs_empty_corpus(tmp_path: Path) -> None:
    """No feeds/, no runs: all zero, no uninstrumented flag."""
    rollup = aggregate_corpus_costs(tmp_path)
    assert rollup["run_count"] == 0
    assert rollup["total_cost_usd"] == 0.0
    assert rollup["metrics_files_missing_cost_fields"] == 0
    assert (
        rollup["cost_appears_uninstrumented"] is False
    ), "Empty corpus = unknown cost, NOT 'silently dropped data.' Flag stays False."


def test_aggregate_corpus_costs_uninstrumented_pattern(tmp_path: Path) -> None:
    """Cost fields present + all-zero values = the prod bug; flag fires (#823)."""
    feed = tmp_path / "feeds" / "rss_feed_a"
    _write_run_with_metrics(feed, "run_001", cost_fields_present=True, cost_value=0.0)
    _write_run_with_metrics(feed, "run_002", cost_fields_present=True, cost_value=0.0)

    rollup = aggregate_corpus_costs(tmp_path)
    assert rollup["run_count"] == 2
    assert rollup["total_cost_usd"] == 0.0
    assert rollup["metrics_files_missing_cost_fields"] == 0, "Cost fields ARE present."
    assert rollup["cost_appears_uninstrumented"] is True, (
        "When metrics.json files have cost fields PRESENT but the total is "
        "exactly 0.0, that's the silent-drop signature — flag must fire."
    )


def test_aggregate_corpus_costs_real_cost_no_uninstrumented(tmp_path: Path) -> None:
    """When real cost > 0 lands, the flag stays False."""
    feed = tmp_path / "feeds" / "rss_feed_a"
    _write_run_with_metrics(feed, "run_001", cost_fields_present=True, cost_value=0.005)

    rollup = aggregate_corpus_costs(tmp_path)
    assert rollup["total_cost_usd"] > 0.0
    assert rollup["cost_appears_uninstrumented"] is False


def test_aggregate_corpus_costs_missing_fields_no_uninstrumented(tmp_path: Path) -> None:
    """Cost fields entirely absent (pre-#650 artifact) is a different signal."""
    feed = tmp_path / "feeds" / "rss_feed_a"
    _write_run_with_metrics(feed, "run_001", cost_fields_present=False)

    rollup = aggregate_corpus_costs(tmp_path)
    assert (
        rollup["metrics_files_missing_cost_fields"] == 1
    ), "Pre-#650 metrics: counted as 'missing fields' not 'uninstrumented'."
    assert rollup["cost_appears_uninstrumented"] is False, (
        "When NO file has cost fields, it's an artifact-vintage issue, not a "
        "silent-drop bug. Distinct signal."
    )


def test_aggregate_corpus_costs_skips_malformed_metrics(tmp_path: Path) -> None:
    """Unreadable metrics.json files are skipped, run_count reflects valid only."""
    feed = tmp_path / "feeds" / "rss_feed_a"
    bad_run = feed / "run_001"
    bad_run.mkdir(parents=True)
    (bad_run / "metrics.json").write_text("{ not valid json")
    _write_run_with_metrics(feed, "run_002", cost_fields_present=True, cost_value=0.0)

    rollup = aggregate_corpus_costs(tmp_path)
    assert rollup["run_count"] == 1, "Malformed file is skipped; only the valid one counts."
