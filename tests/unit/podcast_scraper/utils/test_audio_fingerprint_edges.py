"""The #1656 fingerprint module's failure discipline: every error path degrades to a no-op.

A money-saving gate that can break ingestion costs more than it saves, so the module's contract
is that IO errors, garbage inputs and missing roots all resolve to "no duplicate found" (or a
silent skip on the write side) with at most a warning. These tests pin each of those branches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.utils import audio_fingerprint

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clean_pending():
    audio_fingerprint.reset_pending_for_tests()
    yield
    audio_fingerprint.reset_pending_for_tests()


def test_sha256_of_a_missing_file_is_none_not_an_exception(tmp_path: Path) -> None:
    assert audio_fingerprint.sha256_file(str(tmp_path / "gone.mp3")) is None


def test_eligibility_of_a_missing_file_is_false_not_an_exception(tmp_path: Path) -> None:
    assert audio_fingerprint.eligible_for_fingerprint(str(tmp_path / "gone.mp3")) is False


def test_a_corrupt_index_reads_as_empty_and_does_not_block(tmp_path: Path) -> None:
    index = tmp_path / audio_fingerprint.INDEX_RELPATH
    index.parent.mkdir(parents=True)
    index.write_text("{not json", encoding="utf-8")
    assert audio_fingerprint.duplicate_of(str(tmp_path), "d" * 64, "guid-a") is None


def test_record_refuses_garbage_without_touching_disk(tmp_path: Path) -> None:
    for digest in (None, "", 12345):
        audio_fingerprint.record(
            str(tmp_path), digest, identity="guid-a", episode_title="Ep"  # type: ignore[arg-type]
        )
    audio_fingerprint.record(None, "d" * 64, identity="guid-a")
    audio_fingerprint.record(str(tmp_path), "d" * 64, identity=None)
    assert not (tmp_path / audio_fingerprint.INDEX_RELPATH).exists()


def test_rootless_lookup_and_claim_are_inert() -> None:
    assert audio_fingerprint.duplicate_of(None, "d" * 64, "guid-a") is None
    audio_fingerprint.claim(None, "d" * 64, "guid-a")  # must not raise or leak state
    assert audio_fingerprint.duplicate_of(None, "d" * 64, "guid-b") is None


def test_claims_do_not_leak_across_corpora(tmp_path: Path) -> None:
    """The e2e-caught defect, pinned at module level: a claim is per-corpus, not per-process."""
    root_a, root_b = str(tmp_path / "a"), str(tmp_path / "b")
    audio_fingerprint.claim(root_a, "d" * 64, "guid-a")
    assert (
        audio_fingerprint.duplicate_of(root_b, "d" * 64, "guid-b") is None
    ), "a digest claimed for one corpus starved another corpus of its job"
    hit = audio_fingerprint.duplicate_of(root_a, "d" * 64, "guid-b")
    assert hit is not None and hit["in_flight"] is True


def test_record_then_lookup_round_trip_and_pending_release(tmp_path: Path) -> None:
    root = str(tmp_path)
    audio_fingerprint.claim(root, "d" * 64, "guid-a")
    audio_fingerprint.record(
        root,
        "d" * 64,
        identity="guid-a",
        feed_id="feed-1",
        episode_title="Ep",
        transcript_path="t/0001.txt",
    )
    on_disk = json.loads((tmp_path / audio_fingerprint.INDEX_RELPATH).read_text())
    assert on_disk["d" * 64]["identity"] == "guid-a"
    # The pending claim was released on record; the persistent entry now answers instead.
    hit = audio_fingerprint.duplicate_of(root, "d" * 64, "guid-b")
    assert hit is not None and "in_flight" not in hit
