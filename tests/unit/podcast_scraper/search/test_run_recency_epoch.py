"""Direct coverage for run_recency_epoch's timestamp parsing (2026-08-31).

This function decides which run wins when an episode is reprocessed, and is shared by
search, indexing, digest, enrichment, catalog and staleness. It had NO direct test, and the
anchored pattern ``^run_(\\d{8}-\\d{6})`` never matched production's dir shape
(``run_<run_id>_<ts>_<hash>`` — all 397 dirs on the box), so it had silently been on the
mtime fallback its own docstring calls unsafe for a real corpus.

The trap this pins: `filesystem` PREPENDS run_id, so a spurious timestamp can only appear
BEFORE the real one. The real timestamp is always the LAST match, so the prefix must be
GREEDY. A non-greedy prefix looks identical on prod's UUID run_ids (which contain no ``_``)
and silently picks the run_id's timestamp the moment one contains a date.
"""

from __future__ import annotations

import time

import pytest

from podcast_scraper.search.corpus_scope import _RUN_TS_RE, run_recency_epoch
from podcast_scraper.utils import filesystem


def _epoch(ts: str) -> float:
    return time.mktime(time.strptime(ts, filesystem.TIMESTAMP_FORMAT))


@pytest.mark.parametrize(
    "run_seg,expected",
    [
        ("run_20260825-054545_285e51f2", "20260825-054545"),
        # production's actual shape — every one of the 397 dirs on the box
        ("run_1ebba1af-527d-4d0c-bfad-d3c08923a83d_20260814-055303_285e51f2", "20260814-055303"),
        ("run_e1b90677-bbd3-42c2-81ad-61466098c322_20260830-140720_x", "20260830-140720"),
        # THE TRAP: run_id itself contains a timestamp. The RUN's timestamp is the later one.
        ("run_nightly_20260101-000000_20260830-144405_a1b2c3d4", "20260830-144405"),
    ],
)
def test_extracts_the_runs_own_timestamp(run_seg, expected):
    m = _RUN_TS_RE.match(run_seg)
    assert m is not None, f"no timestamp parsed from {run_seg}"
    assert m.group(1) == expected


@pytest.mark.parametrize("run_seg", ["run_append_abc", "run_", "run_nodate_hash", "notarun"])
def test_timestampless_segments_fall_through_to_mtime(run_seg, tmp_path):
    """`run_append_<hash>` has no timestamp by design; mtime is the correct tiebreak there."""
    f = tmp_path / "x.metadata.json"
    f.write_text("{}", encoding="utf-8")
    assert run_recency_epoch(f, run_seg) == pytest.approx(f.stat().st_mtime, abs=2)


def test_invalid_date_falls_through_rather_than_raising(tmp_path):
    """Shape matches but the date is impossible — must degrade, not explode."""
    f = tmp_path / "x.metadata.json"
    f.write_text("{}", encoding="utf-8")
    got = run_recency_epoch(f, "run_99999999-999999_hash")
    assert got == pytest.approx(f.stat().st_mtime, abs=2)


def test_run_timestamp_beats_mtime(tmp_path):
    """The whole point: ordering must not depend on file mtime, which rsync/restore churns.

    An OLD run whose file was touched recently must still rank older than a NEW run.
    """
    old = tmp_path / "old.metadata.json"
    old.write_text("{}", encoding="utf-8")  # freshest possible mtime
    new = tmp_path / "new.metadata.json"
    new.write_text("{}", encoding="utf-8")

    old_epoch = run_recency_epoch(old, "run_uuid-aaa_20260101-000000_h")
    new_epoch = run_recency_epoch(new, "run_uuid-bbb_20260825-054545_h")
    assert new_epoch > old_epoch


def test_uuid_prefixed_and_plain_runs_are_ordered_by_time_not_lexicography(tmp_path):
    """`run_1ebba...` sorts before `run_2026...` lexicographically; time says otherwise."""
    a = tmp_path / "a.metadata.json"
    a.write_text("{}", encoding="utf-8")
    b = tmp_path / "b.metadata.json"
    b.write_text("{}", encoding="utf-8")

    uuid_run = run_recency_epoch(a, "run_1ebba1af-527d-4d0c-bfad-d3c08923a83d_20260814-055303_h")
    plain_run = run_recency_epoch(b, "run_20260822-222441_h")
    assert plain_run > uuid_run
    assert uuid_run == pytest.approx(_epoch("20260814-055303"), abs=2)
