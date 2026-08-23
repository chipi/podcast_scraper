"""Selection -> manifest -> estimate -> refusal, walked through the REAL selection code.

WHY THIS FILE EXISTS. Before it, every cost-cap test in the repo called the enforcer directly
with a hand-built ``Metrics`` object (``test_cost_monitoring.py``, ``test_processing.py:784``) and
``tests/integration`` contained none at all. The mechanism was thoroughly proven and the WIRING
was never tested — and the wiring is what failed on 2026-08-18, letting a 32-episode repair select
678 episodes and spend ~$48 under an active $5 abort cap.

So these tests deliberately do not import the gate. They call
``prepare_episodes_from_feed`` — the function ``run_pipeline`` actually calls — against a real
on-disk corpus, and assert on what comes back. If someone deletes the gate call from the
selection stage, everything in ``test_selection_gate.py`` still passes and everything here fails.

The multi-feed cases are not decoration. The first version of the work-list fix passed its
single-corpus unit tests while being broken for the 14-feed topology prod actually runs.
"""

# mypy: disable-error-code="arg-type"
# _Feed is a deliberate stand-in: selection reads only items/base_url/title/description/
# authors, and constructing a real RssFeed would drag in the parsing machinery these
# tests exist to isolate from.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest
from defusedxml.ElementTree import fromstring as safe_fromstring

from podcast_scraper import config
from podcast_scraper.workflow.cost_monitoring import CostCapExceeded
from podcast_scraper.workflow.run_budget import get_run_budget, reset_run_budget
from podcast_scraper.workflow.stages.scraping import prepare_episodes_from_feed
from podcast_scraper.workflow.worklist_report import (
    get_worklist_report,
    log_worklist_outcome,
    reset_worklist_report,
)

pytestmark = [pytest.mark.integration]

ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_run_budget()
    reset_worklist_report()
    yield
    reset_run_budget()
    reset_worklist_report()


class _Feed:
    """Minimal RssFeed stand-in: selection reads items, base_url, title, description, authors."""

    def __init__(self, items: List[Any], base_url: str = "https://example.com/feed.xml") -> None:
        self.items = items
        self.base_url = base_url
        self.title = "A Show"
        self.description = "About things"
        self.authors: List[str] = []


def _item(guid: str, duration_seconds: int | None, title: str = "An Episode"):
    dur = (
        f'<itunes:duration xmlns:itunes="{ITUNES}">{duration_seconds}</itunes:duration>'
        if duration_seconds is not None
        else ""
    )
    return safe_fromstring(
        f"<item><guid>{guid}</guid><title>{title}</title>{dur}"
        f'<enclosure url="https://example.com/{guid}.mp3" type="audio/mpeg"/></item>'
    )


def _cfg(root: Path, **kw: Any) -> config.Config:
    fields: dict[str, Any] = {
        "rss_url": "https://example.com/feed.xml",
        "output_dir": str(root),
        "single_feed_uses_corpus_layout": True,
        "transcription_provider": "deepgram",
        "deepgram_model": "nova-3",
        # Config validates that a key exists for the chosen provider. Nothing here contacts
        # Deepgram — selection is priced entirely from the local pricing table — so a literal
        # placeholder is correct and keeps the test offline.
        "deepgram_api_key": "not-a-real-key-nothing-is-called",
        "cost_soft_cap_usd_per_run": 5.0,
        "cost_soft_cap_action": "abort",
    }
    fields.update(kw)  # merged, not passed twice: a caller must be able to OVERRIDE the cap
    return config.Config(**fields)


def _write_corpus(cfg: config.Config, episodes: list[tuple[str, str, int]]) -> None:
    """Lay down an on-disk corpus: (guid, episode_id, duration_seconds) per episode.

    Writes under ``cfg.output_dir`` rather than the tmp root: corpus layout rewrites output_dir to
    ``<root>/feeds/<slug>``, so a corpus written at the root is invisible to the reprocess scan.
    """
    run = Path(str(cfg.output_dir)) / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    for i, (guid, episode_id, duration) in enumerate(episodes, start=1):
        (run / "metadata" / f"{i:04d} - Episode {i}.metadata.json").write_text(
            json.dumps(
                {
                    "episode": {
                        "episode_id": episode_id,
                        "guid": guid,
                        "title": f"Episode {i}",
                        "duration_seconds": duration,
                        "published_date": "Mon, 01 Jan 2026 00:00:00 +0000",
                    },
                    "content": {"transcript_source": "whisper_transcription"},
                }
            ),
            encoding="utf-8",
        )


# -- the normal (live-feed) selection path ----------------------------------------------------


def test_an_affordable_selection_proceeds_and_reports_its_manifest(tmp_path, caplog) -> None:
    feed = _Feed([_item(f"g{i}", 600) for i in range(10)])  # 10 x 10min = 1.67h
    with caplog.at_level("INFO"):
        episodes = prepare_episodes_from_feed(feed, _cfg(tmp_path))
    assert len(episodes) == 10
    assert "selection: 10 of 10 episodes" in caplog.text
    assert "audio-hours" in caplog.text and "est. $" in caplog.text


def test_an_unaffordable_selection_is_REFUSED_by_the_real_selection_call(tmp_path) -> None:
    """1200 hours of audio is ~$387. Selection must raise, having spent nothing."""
    feed = _Feed([_item(f"g{i}", 3600) for i in range(1200)])
    with pytest.raises(CostCapExceeded):
        prepare_episodes_from_feed(feed, _cfg(tmp_path))
    assert get_run_budget().spent_usd == 0.0


def test_max_episodes_narrows_the_selection_and_makes_it_affordable(tmp_path, caplog) -> None:
    """The estimate must reflect the selection AFTER filters, not the whole feed."""
    feed = _Feed([_item(f"g{i}", 3600) for i in range(1200)])
    with caplog.at_level("INFO"):
        episodes = prepare_episodes_from_feed(feed, _cfg(tmp_path, max_episodes=10))
    assert len(episodes) == 10
    assert "selection: 10 of 1200 episodes" in caplog.text, "the denominator stays the full feed"


# -- the reprocess path, which is what the repair actually runs --------------------------------


def test_a_worklist_reprocess_prices_only_the_listed_episodes(tmp_path, caplog) -> None:
    """THE INCIDENT, inverted: 3 named episodes out of 40 on disk, priced as 3."""
    cfg = _cfg(tmp_path, reprocess_existing_only=True, reprocess_episode_ids=["ep1", "ep2", "ep3"])
    _write_corpus(cfg, [(f"g{i}", f"ep{i}", 3600) for i in range(40)])
    feed = _Feed([_item(f"g{i}", 3600) for i in range(40)])

    with caplog.at_level("INFO"):
        episodes = prepare_episodes_from_feed(feed, cfg)

    assert len(episodes) == 3
    assert "selection: 3 of 40 episodes" in caplog.text
    assert "3.0 audio-hours" in caplog.text


def test_a_reprocess_that_would_take_the_WHOLE_corpus_is_refused(tmp_path) -> None:
    """Exactly the shape of the incident: no work-list, so the set is the entire corpus."""
    cfg = _cfg(tmp_path, reprocess_existing_only=True)
    _write_corpus(cfg, [(f"g{i}", f"ep{i}", 3600) for i in range(600)])
    feed = _Feed([_item(f"g{i}", 3600) for i in range(600)])

    with pytest.raises(CostCapExceeded):
        prepare_episodes_from_feed(feed, cfg)
    assert get_run_budget().spent_usd == 0.0


def test_an_AGED_OUT_episode_is_priced_from_its_on_disk_duration(tmp_path, caplog) -> None:
    """An episode reconstructed from disk must not read as unknown-duration.

    The synthesized <item> previously carried no duration at all, so every aged-out episode was
    invisible to the estimate — which on a corpus that has scrolled past its feed window is most
    of the corpus.
    """
    cfg = _cfg(tmp_path, reprocess_existing_only=True)
    _write_corpus(cfg, [(f"g{i}", f"ep{i}", 1800) for i in range(4)])
    feed = _Feed([])  # nothing served live: all four must be reconstructed

    with caplog.at_level("INFO"):
        episodes = prepare_episodes_from_feed(feed, cfg)

    assert len(episodes) == 4
    assert "2.0 audio-hours" in caplog.text, "4 x 30min must be priced, not counted as unknown"
    assert "NO known duration" not in caplog.text


# -- multi-feed: the topology prod actually runs -----------------------------------------------


def test_the_budget_is_CUMULATIVE_across_feeds(tmp_path) -> None:
    """Each feed is individually affordable; the batch is not.

    cli calls run_pipeline once per feed, so this is the shape that turned a $5 cap into a $70
    ceiling. The refusal must land on a later feed even though no single feed is large.
    """
    feeds = []
    for f in range(12):
        root = tmp_path / f"feed{f}"
        root.mkdir()
        feeds.append((root, _Feed([_item(f"f{f}g{i}", 1800) for i in range(4)])))  # 2h per feed

    refused_at = None
    for i, (root, feed) in enumerate(feeds):
        cfg = _cfg(root)
        try:
            prepare_episodes_from_feed(feed, cfg)
        except CostCapExceeded:
            refused_at = i
            break
        # the feed then actually spends roughly what it was quoted
        get_run_budget().record(2 * 60 * 0.0043)

    assert refused_at is not None, "a 12-feed batch of 2h feeds must be stopped somewhere"
    assert refused_at < 12


def test_a_feed_holding_none_of_the_worklist_is_a_no_op_not_a_refusal(tmp_path) -> None:
    """The multi-feed normal case: 13 of 14 feeds match nothing and must pass through quietly."""
    cfg = _cfg(
        tmp_path,
        reprocess_existing_only=True,
        reprocess_episode_ids=["not-in-this-feed-at-all"],
    )
    _write_corpus(cfg, [(f"g{i}", f"ep{i}", 3600) for i in range(50)])
    feed = _Feed([_item(f"g{i}", 3600) for i in range(50)])
    assert prepare_episodes_from_feed(feed, cfg) == []


# -- the escape hatches ------------------------------------------------------------------------


def test_observe_action_reports_without_stopping(tmp_path, caplog) -> None:
    feed = _Feed([_item(f"g{i}", 3600) for i in range(1200)])
    cfg = _cfg(tmp_path, cost_soft_cap_action="observe")
    with caplog.at_level("INFO"):
        episodes = prepare_episodes_from_feed(feed, cfg)
    assert len(episodes) == 1200
    assert "selection: 1200 of 1200 episodes" in caplog.text


def test_no_cap_configured_never_refuses(tmp_path) -> None:
    feed = _Feed([_item(f"g{i}", 3600) for i in range(1200)])
    cfg = _cfg(tmp_path, cost_soft_cap_usd_per_run=None)
    assert len(prepare_episodes_from_feed(feed, cfg)) == 1200


def test_episodes_with_no_duration_are_reported_as_uncosted(tmp_path, caplog) -> None:
    feed = _Feed([_item(f"g{i}", None) for i in range(5)])
    with caplog.at_level("INFO"):
        episodes = prepare_episodes_from_feed(feed, _cfg(tmp_path))
    assert len(episodes) == 5
    assert "5 episode(s) have NO known duration" in caplog.text
    assert "the real cost is higher" in caplog.text


# -- the work-list outcome report, driven through real selection ------------------------------


def test_selection_registers_the_ask_and_what_it_FOUND(tmp_path) -> None:
    """The report must be populated by the real selection call, not by a test helper."""
    cfg = _cfg(
        tmp_path,
        reprocess_existing_only=True,
        reprocess_episode_ids=["ep1", "ep2", "no-such-episode"],
    )
    _write_corpus(cfg, [(f"g{i}", f"ep{i}", 600) for i in range(10)])
    feed = _Feed([_item(f"g{i}", 600) for i in range(10)])

    prepare_episodes_from_feed(feed, cfg)

    report = get_worklist_report()
    assert report.active is True
    assert report.unmatched == ["no-such-episode"]


def test_THE_INCIDENT_a_run_that_matched_NOTHING_says_so(tmp_path, caplog) -> None:
    """32 requested, 0 found — the state the 2026-08-18 run was in, now stated in its own log."""
    cfg = _cfg(
        tmp_path,
        reprocess_existing_only=True,
        reprocess_episode_ids=[f"missing{i}" for i in range(32)],
    )
    _write_corpus(cfg, [(f"g{i}", f"ep{i}", 600) for i in range(40)])
    feed = _Feed([_item(f"g{i}", 600) for i in range(40)])

    assert prepare_episodes_from_feed(feed, cfg) == []

    with caplog.at_level("ERROR"):
        line = log_worklist_outcome()
    assert line is not None
    assert "repaired 0/32" in caplog.text
    assert "NOT FOUND" in caplog.text


def test_a_run_with_NO_worklist_reports_nothing(tmp_path) -> None:
    """An ordinary ingest must not grow a spurious repair line."""
    feed = _Feed([_item(f"g{i}", 600) for i in range(3)])
    prepare_episodes_from_feed(feed, _cfg(tmp_path))
    assert get_worklist_report().active is False
    assert log_worklist_outcome() is None


def test_matches_ACCUMULATE_across_feeds_so_no_healthy_feed_looks_like_a_failure(tmp_path) -> None:
    """Every feed's config carries the whole list; each holds only part of it."""
    ids = ["ep1", "ep7"]
    for f, held in ((0, 1), (1, 7)):
        root = tmp_path / f"feed{f}"
        root.mkdir()
        cfg = _cfg(root, reprocess_existing_only=True, reprocess_episode_ids=ids)
        _write_corpus(cfg, [(f"f{f}g{held}", f"ep{held}", 600)])
        prepare_episodes_from_feed(_Feed([]), cfg)

    report = get_worklist_report()
    # active + matched counts FIRST: `unmatched == []` is trivially true for an empty report, so
    # asserting only that would pass even if selection registered nothing at all.
    assert report.active is True, "selection never registered the ask"
    assert len(report.requested) == 2
    assert len(report.matched) >= 2, f"each feed's hit must accumulate: {report.as_dict()}"
    assert report.unmatched == [], f"both ids were found across the two feeds: {report.as_dict()}"


def test_INCIDENT_3_worklist_ids_absent_from_EVERY_configured_feed(tmp_path, caplog) -> None:
    """The 2026-08-19 incident exactly: 32 wanted ids, 8 feeds, ZERO overlap.

    Forensics from the prod box: the work-list held 32 `substack:post` ids; the run processed 127
    episodes across 8 mainstream feeds (megaphone x3, simplecast x2, npr, acast, flightcast); the
    intersection was 0. So the ~$40 of Deepgram bought episodes nobody asked for, and the 32
    targets were never touched.

    Two properties must hold together, and only together are they the fix:
      * NOTHING is selected in any feed — so the run costs nothing rather than $40;
      * the run SAYS "0/32, not found" — so the operator learns it in the log rather than from
        a corpus audit two days later.

    A run that quietly does nothing is not much better than one that loudly does the wrong thing.
    """
    wanted = [f"substack:post:{i}" for i in range(32)]
    total_selected = 0

    for f in range(8):
        root = tmp_path / f"feed{f}"
        root.mkdir()
        cfg = _cfg(root, reprocess_existing_only=True, reprocess_episode_ids=wanted)
        # each feed's corpus holds only its OWN episodes — none of them substack
        _write_corpus(cfg, [(f"f{f}g{i}", f"f{f}ep{i}", 1800) for i in range(15)])
        selected = prepare_episodes_from_feed(
            _Feed([_item(f"f{f}g{i}", 1800) for i in range(15)]), cfg
        )
        total_selected += len(selected)

    assert total_selected == 0, (
        f"{total_selected} episodes would have been transcribed for a work-list that matches "
        "nothing — this is the bug that cost ~$40"
    )
    assert get_run_budget().spent_usd == 0.0

    with caplog.at_level("ERROR"):
        line = log_worklist_outcome()
    assert line is not None
    assert "repaired 0/32" in caplog.text
    assert "32 NOT FOUND in any feed's corpus" in caplog.text
    assert get_worklist_report().unmatched == sorted(wanted)
