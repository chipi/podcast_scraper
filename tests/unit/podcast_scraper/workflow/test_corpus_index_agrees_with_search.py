"""corpus_metadata_index must resolve an episode the way the rest of the corpus does.

``discover_metadata_files`` is documented as "the CENTRAL corpus-membership rule (single
source of truth)": when an episode is reprocessed into a newer ``run_*`` dir, the newest run
wins, and indexing / digest / topic-clusters / enrichment / catalog / staleness all share it
"so they can never diverge (the 94-vs-106 split-brain)".

``corpus_metadata_index`` did not use it. It ran its own ``sorted(root.glob(...))`` and kept
the FIRST path lexicographically, which is the OLDER run — and, because older run dirs are
``run_<uuid>_<ts>`` while newer ones are ``run_<ts>``, which one wins actually depended on
the first hex character of a UUID rather than on time.

Two consequences, measured on prod 2026-08-31:

1. The index and search disagreed about which artifact is canonical for the same episode
   (already noted in ``gi/integrity.py``), and the disagreement was not rare: 12,608
   duplicate warnings in one batch, a single guid warned 1,120 times.
2. ``corpus_rollback`` locates an episode's run dir through ``by_id``. Keeping the OLDER
   entry means "roll back this episode" deleted the superseded copy and LEFT the newest one
   — the very copy search serves. The user asks to remove an episode and it stays.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.workflow import run_index


def _write_episode(corpus, feed_dir, run_name, episode_id, guid, idx=1, title="Ep"):
    meta_dir = corpus / "feeds" / feed_dir / run_name / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{idx:04d} - {title}_{run_name}.metadata.json"
    path.write_text(
        json.dumps({"episode": {"episode_id": episode_id, "guid": guid, "title": title}}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def corpus(tmp_path):
    run_index.reset_corpus_metadata_index_cache_for_tests()
    yield tmp_path
    run_index.reset_corpus_metadata_index_cache_for_tests()


def test_reprocessed_episode_resolves_to_the_newest_run(corpus):
    """The core defect: a reprocess must supersede, not be shadowed by, the original."""
    _write_episode(corpus, "rss_example", "run_20260814-055303", "ep-1", "guid-1")
    newest = _write_episode(corpus, "rss_example", "run_20260825-054545", "ep-1", "guid-1")

    idx = run_index.corpus_metadata_index(str(corpus))
    assert idx["by_id"]["ep-1"].metadata_rel == str(newest.relative_to(corpus))


def test_uuid_prefixed_run_does_not_beat_a_newer_plain_run(corpus):
    """The nastiest shape: ordering used to hinge on a UUID's first hex character.

    ``run_1ebba1af-..._20260814`` sorts before ``run_20260822`` lexicographically, so the
    August 14 copy won. A UUID starting with 'f' would have produced the opposite result on
    identical data — the winner was effectively random.
    """
    _write_episode(
        corpus,
        "rss_example",
        "run_1ebba1af-527d-4d0c-bfad-d3c08923a83d_20260814-055303",
        "ep-2",
        "guid-2",
    )
    newest = _write_episode(corpus, "rss_example", "run_20260822-222441", "ep-2", "guid-2")

    idx = run_index.corpus_metadata_index(str(corpus))
    assert idx["by_id"]["ep-2"].metadata_rel == str(newest.relative_to(corpus))
    assert idx["by_guid"]["guid-2"].metadata_rel == str(newest.relative_to(corpus))


def test_disjoint_episodes_across_runs_all_survive(corpus):
    """Incremental add must not be mistaken for supersede — both episodes stay."""
    _write_episode(corpus, "rss_example", "run_20260814-055303", "ep-a", "guid-a", idx=1)
    _write_episode(corpus, "rss_example", "run_20260825-054545", "ep-b", "guid-b", idx=2)

    idx = run_index.corpus_metadata_index(str(corpus))
    assert set(idx["by_id"]) == {"ep-a", "ep-b"}


def test_single_run_corpus_is_unaffected(corpus):
    _write_episode(corpus, "rss_example", "run_20260814-055303", "ep-x", "guid-x")
    idx = run_index.corpus_metadata_index(str(corpus))
    assert "ep-x" in idx["by_id"]
    assert "guid-x" in idx["by_guid"]


def test_superseded_copies_do_not_spam_a_warning_per_occurrence(corpus, caplog):
    """12,608 warnings in one batch, one guid warned 1,120 times.

    Once the newest run legitimately supersedes the older, a reprocessed episode is not a
    'duplicate' to complain about — it is the expected shape of a corpus that has been
    reprocessed. Genuine same-run collisions should still surface.
    """
    caplog.set_level("WARNING")
    for run in ("run_20260814-055303", "run_20260820-010101", "run_20260825-054545"):
        _write_episode(corpus, "rss_example", run, "ep-3", "guid-3")

    run_index.corpus_metadata_index(str(corpus))
    dupes = [r for r in caplog.records if "duplicate" in r.getMessage()]
    assert not dupes, f"reprocess supersede should not warn; got {[d.getMessage() for d in dupes]}"
