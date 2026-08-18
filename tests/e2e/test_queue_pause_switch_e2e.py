"""The operator pause switch, end to end on a real corpus directory (#1653).

WHY THIS MATTERS FOR THE REPAIR SPECIFICALLY
Bringing the API up *is* starting whatever is queued: the startup sweep promotes within seconds
of every boot. That is right for normal operation and wrong for a corpus repair, where the
operator needs the stack up and idle while they look at it.

It also covers a hazard the enqueue rework leaves open: a repair driven as a plain CLI run holds
no registry slot, so nothing stops the sweeper promoting a queued enrichment pass that would read
the very files the repair is rewriting. Two writers, one corpus, no lock between them.

The switch is a FILE, deliberately — an operator can set it over SSH or a volume mount without a
working API, which is the state you are in when you most need it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.queue_sweeper import drain_is_paused, PAUSE_FLAG_RELPATH

pytestmark = [pytest.mark.e2e]


def test_pause_flag_holds_and_releases_the_drain(tmp_path: Path) -> None:
    """Create the file -> paused. Delete it -> running. No API, no restart."""
    corpus = tmp_path / "corpus"
    (corpus / "feeds").mkdir(parents=True)

    assert drain_is_paused(corpus) is False, "a fresh corpus must not start paused"

    flag = corpus / PAUSE_FLAG_RELPATH
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("", encoding="utf-8")
    assert drain_is_paused(corpus) is True, "touching the flag did not hold the drain"

    flag.unlink()
    assert drain_is_paused(corpus) is False, "removing the flag did not release the drain"


def test_an_empty_flag_file_still_counts_as_paused(tmp_path: Path) -> None:
    """Presence is the signal, not contents.

    An operator pausing a production queue types ``touch``; requiring specific contents would
    make the safe action fail open, which on a repair means a second writer in the corpus.
    """
    corpus = tmp_path / "corpus"
    flag = corpus / PAUSE_FLAG_RELPATH
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("", encoding="utf-8")

    assert drain_is_paused(corpus) is True


def test_a_missing_corpus_reads_as_not_paused(tmp_path: Path) -> None:
    """Deliberate: failing CLOSED here would stop the queue for the wrong reason.

    The sweeper is already a no-op against a path that does not exist, so reporting "paused"
    would be an unexplained permanent stop rather than a safety measure.
    """
    assert drain_is_paused(tmp_path / "does-not-exist") is False


def test_the_flag_path_is_the_documented_one() -> None:
    """Pinned because operators type this path by hand from the runbook.

    If the constant moves, every runbook, every SSH one-liner and every volume mount that
    references it silently stops working -- and the failure mode is a queue that drains when
    the operator believes it is held.
    """
    assert PAUSE_FLAG_RELPATH == ".viewer/jobs.paused"
