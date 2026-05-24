"""Integration tests for v2.6.1 hotfix bugs (#818, #819, #820, #821, #823).

Uses the multi-run corpus fixture built by ``scripts/tools/build_multi_run_fixture.py``
which has known counts:

  - 33 metadata files on disk (3 feeds × (1 probe + 5 middle + 5 latest))
  - 21 cumulative-unique episodes (3 feeds × (5 + (5 - 3 overlap)) — uniques per feed)
  - 15 last-run-sum count (3 feeds × 5 episodes in latest run)
  - 9 total runs (3 feeds × 3 runs each)

Each test asserts the value that the **fixed** API should return. They will
fail today against the current code (capturing the bug); they should turn
green as the corresponding fix lands.

Tests are skipped when the fixture hasn't been generated — run
``python scripts/tools/build_multi_run_fixture.py`` first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Pin known fixture counts so the tests document the expected math.
EXPECTED_METADATA_FILES = 33
EXPECTED_CUMULATIVE_UNIQUE = 21
EXPECTED_LAST_RUN_SUM = 15
EXPECTED_TOTAL_RUN_COUNT = 9
EXPECTED_N_FEEDS = 3
EXPECTED_UNIQUE_PER_FEED = 7  # 5 middle + (5 latest - 3 overlap)


FIXTURE_DIR = Path("tests/fixtures/multi-run-corpus")


def _fixture_available() -> bool:
    return FIXTURE_DIR.exists() and (FIXTURE_DIR / "corpus_manifest.json").exists()


fixture_required = pytest.mark.skipif(
    not _fixture_available(),
    reason="multi-run corpus fixture not generated; run scripts/tools/build_multi_run_fixture.py",
)


@fixture_required
def test_fixture_shape_matches_documented_counts() -> None:
    """Sanity check: fixture on disk matches the counts the bug tests rely on."""
    metadata_files = list(FIXTURE_DIR.glob("feeds/*/run_*/metadata/*.metadata.json"))
    assert len(metadata_files) == EXPECTED_METADATA_FILES, (
        f"fixture has {len(metadata_files)} metadata files; expected "
        f"{EXPECTED_METADATA_FILES}. Regenerate the fixture."
    )

    feed_dirs = [p for p in (FIXTURE_DIR / "feeds").iterdir() if p.is_dir()]
    assert len(feed_dirs) == EXPECTED_N_FEEDS

    # Cumulative-unique check via episode GUID across all metadata files.
    guids: set[str] = set()
    for mf in metadata_files:
        data = json.loads(mf.read_text())
        guids.add(data["episode"]["guid"])
    assert len(guids) == EXPECTED_CUMULATIVE_UNIQUE, (
        f"fixture has {len(guids)} unique episode GUIDs across all runs; "
        f"expected {EXPECTED_CUMULATIVE_UNIQUE}. The skip_existing overlap "
        f"between middle + latest runs is what makes cumulative-unique != "
        f"naive-add."
    )


@fixture_required
def test_corpus_manifest_cost_rollup_is_zero_baseline() -> None:
    """Baseline for #823 — fixture deliberately ships all-zero cost rollup.

    This test pins the BUG state. The fix in #823 should land a separate
    test (or modify this one) that asserts the cost rollup is non-zero
    after the fix.
    """
    manifest = json.loads((FIXTURE_DIR / "corpus_manifest.json").read_text())
    assert manifest["cost_rollup"]["total_cost_usd"] == 0.0
    assert manifest["cost_rollup"]["total_llm_cost_usd"] == 0.0


@fixture_required
def test_corpus_run_summary_only_reflects_latest_run_baseline() -> None:
    """Baseline for #820 + #821 — fixture's run_summary has last-run-only data.

    Each feed entry shows ``episodes_processed`` from the LATEST run only,
    matching prod. The fix is to either populate cumulative-unique here,
    or have consumers (Dashboard widget, /api/corpus/stats) read elsewhere.
    """
    summary = json.loads((FIXTURE_DIR / "corpus_run_summary.json").read_text())
    feeds = summary["feeds"]
    assert len(feeds) == EXPECTED_N_FEEDS
    for f in feeds:
        # Each feed reports 5 (latest_episodes from the fixture default).
        # The bug: this becomes the "Dashboard total" + "catalog_episode_count".
        assert f["episodes_processed"] == 5, (
            f"fixture run_summary should report only last-run count (5) per feed; "
            f"got {f['episodes_processed']}. Tests of the bug-fix go elsewhere."
        )


# -----------------------------------------------------------------------------
# The next group of tests EXERCISES THE ROUTES against the fixture. They'll
# be marked xfail today (current code fails them) and flip to passing as
# each bug fix lands. Each test names the issue it belongs to.
# -----------------------------------------------------------------------------


def _build_fastapi_app_against_fixture():
    """Build a FastAPI app instance with output_dir pinned to the fixture.

    Imported lazily so the test file is collectable even when server deps
    aren't installed (CI matrices for unit-only runs).
    """
    from podcast_scraper.server.app import create_app

    app = create_app()
    app.state.output_dir = FIXTURE_DIR.resolve()
    return app


@fixture_required
def test_corpus_stats_returns_cumulative_unique_count_818_821() -> None:
    """#818 + #821 — /api/corpus/stats catalog_episode_count must be cumulative-unique.

    Today returns 15 (last-run-sum). Fix should return 21 (cumulative-unique
    across all runs in the corpus). This is the operator-facing "total" number.
    """
    from fastapi.testclient import TestClient

    app = _build_fastapi_app_against_fixture()
    client = TestClient(app)

    resp = client.get("/api/corpus/stats", params={"path": str(FIXTURE_DIR.resolve())})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["catalog_episode_count"] == EXPECTED_CUMULATIVE_UNIQUE, (
        f"catalog_episode_count returned {payload['catalog_episode_count']}, "
        f"expected {EXPECTED_CUMULATIVE_UNIQUE} (cumulative unique). "
        f"If this is still failing after #821 fix lands, semantics weren't "
        f"changed correctly."
    )


@fixture_required
def test_corpus_episodes_response_includes_total_818() -> None:
    """#818 — /api/corpus/episodes response must include a `total` field.

    Today the response shape lacks `total`/`total_matched`, so the viewer
    cannot show 'X of Y' (#819). The fix adds the field at the response
    layer.
    """
    from fastapi.testclient import TestClient

    app = _build_fastapi_app_against_fixture()
    client = TestClient(app)

    resp = client.get(
        "/api/corpus/episodes",
        params={"path": str(FIXTURE_DIR.resolve()), "limit": 5},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "total" in payload, (
        "CorpusEpisodesResponse must include a `total` field for #819 to "
        "render 'X of Y' counts. Update both the route handler and the "
        "schema model."
    )
    assert payload["total"] == EXPECTED_CUMULATIVE_UNIQUE, (
        f"total returned {payload['total']}, expected {EXPECTED_CUMULATIVE_UNIQUE} "
        f"(cumulative-unique episode count)."
    )


@fixture_required
def test_corpus_episodes_limit_supports_at_least_500_818() -> None:
    """#818 — /api/corpus/episodes must accept limit values up to 500.

    Today capped at 200 (le=200). Prod corpus is 210 cumulative and growing;
    cap must be raised so a single call can fetch the full set.
    """
    from fastapi.testclient import TestClient

    app = _build_fastapi_app_against_fixture()
    client = TestClient(app)

    resp = client.get(
        "/api/corpus/episodes",
        params={"path": str(FIXTURE_DIR.resolve()), "limit": 500},
    )
    assert (
        resp.status_code == 200
    ), f"limit=500 should be accepted; got {resp.status_code}: {resp.text}"
    payload = resp.json()
    assert len(payload["items"]) == EXPECTED_CUMULATIVE_UNIQUE, (
        f"With limit=500, all {EXPECTED_CUMULATIVE_UNIQUE} unique episodes "
        f"should be returned; got {len(payload['items'])}."
    )


@fixture_required
def test_corpus_feeds_per_feed_count_is_cumulative_unique_820() -> None:
    """#820 — Dashboard 'Total Episodes per Feed' widget reads /api/corpus/feeds.

    Each feed's episode_count must reflect cumulative-unique episodes for
    that feed (7 in this fixture), not last-run-only (5).
    """
    from fastapi.testclient import TestClient

    app = _build_fastapi_app_against_fixture()
    client = TestClient(app)

    resp = client.get(
        "/api/corpus/feeds",
        params={"path": str(FIXTURE_DIR.resolve())},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    feeds = payload["feeds"]
    assert len(feeds) == EXPECTED_N_FEEDS
    for feed in feeds:
        assert feed["episode_count"] == EXPECTED_UNIQUE_PER_FEED, (
            f"Feed {feed['display_title']} reports {feed['episode_count']} "
            f"episodes; expected {EXPECTED_UNIQUE_PER_FEED} (cumulative unique)."
        )
