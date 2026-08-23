"""Selecting a repair set by EXPLICIT episode list (#32).

WHY THIS EXISTS RATHER THAN REUSING --reprocess-source
Measured 2026-08-17 on a corpus carrying #18 unpreprocessed-audio damage: all 9 damaged episodes
had ``transcript_source: whisper_transcription`` — and so did all 6 healthy ones. Selecting by
source would have re-transcribed 6 healthy episodes to reach 9 damaged ones, at real ASR cost.

A detector that can only produce a LIST is useless without a selector that consumes one. That was
exactly the gap that made the placeholder gate a dead end until ``gi-repair`` existed.
"""

# mypy: disable-error-code="call-arg"
# Deliberate in this file: Config(rss_url=...) — the field declares alias="rss", so mypy's pydantic
# plugin
# only knows the alias while populate-by-name accepts either at runtime.
# Constructing the real types would pull in the machinery these tests isolate. The
# annotations on the helpers here are what make mypy check these bodies at all — most
# older test files are unannotated and therefore unchecked.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# defusedxml for the PARSE, matching rss/parser.py and stages/test_scraping.py. Bandit B314
# blacklists stdlib ElementTree parsing regardless of whether the input is trusted, and this
# repo answers that with the safe parser rather than a per-line suppression. ``ET`` stays for
# the ``Element`` type annotation, which B314 does not flag.
from defusedxml.ElementTree import fromstring as safe_fromstring

from podcast_scraper import config
from podcast_scraper.workflow.episode_processor import _force_reprocess_for_source

pytestmark = [pytest.mark.unit]


class _Episode:
    def __init__(self, guid: str, episode_id: str | None = None) -> None:
        self.item = safe_fromstring(f"<item><guid>{guid}</guid></item>")
        self.guid = guid
        self.episode_id = episode_id or guid
        self.title = "An Episode"
        self.title_safe = "An Episode"
        self.idx = 1


def _cfg(root: Path, **kw: Any) -> config.Config:
    return config.Config(
        rss_url="https://example.com/feed.xml",
        output_dir=str(root),
        single_feed_uses_corpus_layout=True,
        **kw,
    )


def _corpus(cfg: config.Config, *, guid: str, episode_id: str, source: str) -> str:
    run = Path(str(cfg.output_dir)) / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    name = "0001 - An Episode"
    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": episode_id, "guid": guid, "title": "An Episode"},
                "content": {"transcript_source": source},
            }
        ),
        encoding="utf-8",
    )
    fresh = Path(str(cfg.output_dir)) / "run_20260816-090000"
    (fresh / "metadata").mkdir(parents=True, exist_ok=True)
    return str(fresh)


def test_an_episode_on_the_list_is_forced(tmp_path):
    cfg = _cfg(tmp_path, reprocess_episode_ids=["ep-damaged"])
    fresh = _corpus(cfg, guid="ep-damaged", episode_id="ep-damaged", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-damaged"), fresh, None, cfg) is True


def test_an_episode_NOT_on_the_list_is_left_alone(tmp_path):
    """The whole point: a healthy episode sharing the same transcript_source must be untouched."""
    cfg = _cfg(tmp_path, reprocess_episode_ids=["ep-damaged"])
    fresh = _corpus(cfg, guid="ep-healthy", episode_id="ep-healthy", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-healthy"), fresh, None, cfg) is False


def test_matching_works_on_the_guid_when_the_list_holds_guids(tmp_path):
    """Detectors emit whatever the artifact carries — guid or episode_id. Matching only one of
    them makes an operator's list silently miss episodes."""
    cfg = _cfg(tmp_path, reprocess_episode_ids=["rss-guid-1"])
    fresh = _corpus(
        cfg, guid="rss-guid-1", episode_id="different-episode-id", source="direct_download"
    )

    assert (
        _force_reprocess_for_source(
            _Episode("rss-guid-1", "different-episode-id"), fresh, None, cfg
        )
        is True
    )


def test_matching_works_on_the_episode_id_when_the_list_holds_episode_ids(tmp_path):
    cfg = _cfg(tmp_path, reprocess_episode_ids=["substack:post:12345"])
    fresh = _corpus(
        cfg, guid="rss-guid-2", episode_id="substack:post:12345", source="direct_download"
    )

    assert (
        _force_reprocess_for_source(_Episode("rss-guid-2", "substack:post:12345"), fresh, None, cfg)
        is True
    )


def test_the_list_does_not_disturb_reprocess_source(tmp_path):
    """Both selectors coexist: an empty list must not suppress the #925 source match."""
    cfg = _cfg(tmp_path, reprocess_source="whisper_transcription")
    fresh = _corpus(cfg, guid="ep-1", episode_id="ep-1", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-1"), fresh, None, cfg) is True


def test_neither_selector_means_no_forcing(tmp_path):
    cfg = _cfg(tmp_path)
    fresh = _corpus(cfg, guid="ep-1", episode_id="ep-1", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-1"), fresh, None, cfg) is False


def test_the_list_forces_even_when_the_source_does_not_match(tmp_path):
    """The list is authoritative — it exists precisely because source cannot express the set."""
    cfg = _cfg(
        tmp_path,
        reprocess_episode_ids=["ep-1"],
        reprocess_source="whisper_transcription",
    )
    fresh = _corpus(cfg, guid="ep-1", episode_id="ep-1", source="direct_download")

    assert _force_reprocess_for_source(_Episode("ep-1"), fresh, None, cfg) is True


def test_an_episode_list_implies_existing_only_scope(tmp_path):
    """The footgun found by the e2e test: the list FORCES episodes but did not RESTRICT the run.

    A one-episode work-list against one feed had preprocessed 12 unrelated episodes before it was
    killed — the run still walked the feed and treated every item not on disk as new work. Over 14
    production feeds that is a large unbudgeted ASR bill from a command that promises the opposite.
    """
    cfg = _cfg(tmp_path, reprocess_episode_ids=["ep-1"])

    assert (
        cfg.reprocess_existing_only is True
    ), "naming explicit episodes must not also pull in whatever else the feed is offering"


def test_no_episode_list_leaves_the_scope_alone(tmp_path):
    """The implication must not silently narrow an ordinary run."""
    cfg = _cfg(tmp_path)

    assert cfg.reprocess_existing_only is False


def test_an_explicit_existing_only_false_is_still_overridden_by_a_list(tmp_path):
    """Explicitness does not rescue the footgun: a list plus feed-walking is never intended."""
    cfg = _cfg(tmp_path, reprocess_episode_ids=["ep-1"], reprocess_existing_only=False)

    assert cfg.reprocess_existing_only is True


# ---------------------------------------------------------------------------------------------
# THE LIST MUST RESTRICT THE RUN, NOT MERELY FORCE ITS MEMBERS.
#
# 2026-08-19 incident. `--reprocess-episode-ids <32 episodes>` re-transcribed ~181 episodes across
# healthy feeds, pulled 15 GB of fresh media (corpus 48 GB -> 63 GB), emptied the operator's
# Deepgram balance, and never reached the 32 it was asked to repair. Deepgram console: 271
# /listen requests, 187.6 hours billed, for a job whose work-list totals 47 hours.
#
# Every piece behaved as documented. The list implies `reprocess_existing_only`, which correctly
# blocks NEW episodes and then makes the episode set the WHOLE on-disk corpus. The list's only
# other job was to force its members past `skip_existing` — which defaults to False, is unset in
# cloud_balanced, and is not passed by reprocess-prod.yml. With nothing being skipped, "force past
# the skip" selects everything.
#
# The ten tests above all assert FORCING. None asserted SCOPE, which is why this shipped.
# ---------------------------------------------------------------------------------------------


def _multi_corpus(root: Path, episodes: list[tuple[str, str]]) -> None:
    """Write an on-disk corpus of (guid, episode_id) pairs."""
    run = root / "run_20260815-120000" / "metadata"
    run.mkdir(parents=True, exist_ok=True)
    for i, (guid, eid) in enumerate(episodes, start=1):
        (run / f"{i:04d} - Episode {i}.metadata.json").write_text(
            json.dumps(
                {
                    "episode": {"episode_id": eid, "guid": guid, "title": f"Episode {i}"},
                    "content": {"transcript_source": "whisper_transcription"},
                }
            ),
            encoding="utf-8",
        )


def _episode_set(root: Path, wanted: list[str]) -> list[Any]:
    from podcast_scraper.models import RssFeed
    from podcast_scraper.workflow.stages.scraping import _reprocess_existing_episodes

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        output_dir=str(root),
        reprocess_episode_ids=wanted,
    )
    feed = RssFeed(title="F", items=[], base_url="https://example.com")
    return _reprocess_existing_episodes(feed, [], cfg, 0)


def test_the_list_RESTRICTS_the_episode_set(tmp_path: Path) -> None:
    """10 on disk, 2 named -> the run must consider 2. Not 10."""
    _multi_corpus(tmp_path, [(f"guid-{i}", f"eid-{i}") for i in range(10)])
    got = _episode_set(tmp_path, ["eid-3", "eid-7"])
    assert len(got) == 2, (
        f"work-list named 2 episodes but the run would process {len(got)} — "
        f"this is the 2026-08-19 overrun (181 transcriptions for a 32-episode job)"
    )
    # _multi_corpus numbers files from 1, so guid-3 -> idx 4 and guid-7 -> idx 8.
    assert {e.idx for e in got} == {4, 8}


def test_restriction_also_matches_on_guid(tmp_path: Path) -> None:
    _multi_corpus(tmp_path, [(f"guid-{i}", f"eid-{i}") for i in range(5)])
    got = _episode_set(tmp_path, ["guid-1"])
    assert len(got) == 1 and got[0].idx == 2  # guid-1 is the 2nd file


def test_a_list_matching_nothing_selects_NOTHING_not_everything(tmp_path: Path) -> None:
    """The dangerous case: a typo'd or stale list must never mean 'the whole corpus'.

    It returns an EMPTY set rather than raising, because in prod's multi-feed topology "none of
    the listed episodes live in this feed" is the normal case for 13 of 14 feeds — see the
    multi-feed test below for why raising here was wrong.
    """
    _multi_corpus(tmp_path, [(f"guid-{i}", f"eid-{i}") for i in range(5)])
    got = _episode_set(tmp_path, ["eid-does-not-exist"])
    assert got == [], "a non-matching work-list must select nothing, never the whole corpus"


def test_multi_feed_a_feed_without_listed_episodes_does_not_fail_the_run(tmp_path: Path) -> None:
    """PROD TOPOLOGY. cli.py loops feeds, each with its own output_dir, and hands every feed the
    WHOLE work-list. A list drawn from feed A matches nothing in feed B — that must be a quiet
    no-op, not a ValueError, which corpus_operations.py classifies "hard" and which would exit
    the batch red with an incident per healthy feed while the real targets repaired fine.
    """
    feed_a = tmp_path / "feeds" / "feed-a"
    feed_b = tmp_path / "feeds" / "feed-b"
    _multi_corpus(feed_a, [("guid-a1", "eid-a1"), ("guid-a2", "eid-a2")])
    _multi_corpus(feed_b, [("guid-b1", "eid-b1"), ("guid-b2", "eid-b2")])

    worklist = ["eid-a1"]  # lives only in feed A

    got_a = _episode_set(feed_a, worklist)
    assert len(got_a) == 1, "feed holding the listed episode must process exactly it"

    got_b = _episode_set(feed_b, worklist)  # must NOT raise
    assert got_b == [], "feed holding none of the listed episodes must be a quiet no-op"


def test_no_list_still_reaches_the_whole_corpus(tmp_path: Path) -> None:
    """The migration mode this path was built for (#876) must be unchanged."""
    _multi_corpus(tmp_path, [(f"guid-{i}", f"eid-{i}") for i in range(6)])
    from podcast_scraper.models import RssFeed
    from podcast_scraper.workflow.stages.scraping import _reprocess_existing_episodes

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        output_dir=str(tmp_path),
        reprocess_existing_only=True,
    )
    got = _reprocess_existing_episodes(
        RssFeed(title="F", items=[], base_url="https://x"), [], cfg, 0
    )
    assert len(got) == 6
