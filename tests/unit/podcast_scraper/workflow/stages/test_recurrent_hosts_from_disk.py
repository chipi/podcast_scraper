"""The feed's own transcripts say who presents it, week after week (#2075).

A guest appears once; a host appears every week. No per-episode rule can see that, and on a show
whose RSS author tag is an organisation it is the only host signal there is. These tests pin what
the disk scan must read — and what it must never pool together.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from podcast_scraper.workflow.stages.processing import (
    _newest_run_transcripts,
    _recurrent_hosts_from_disk,
)

INTRO = "Hello and welcome to the show. I'm {name}, and today we are talking about markets. "


def _episode(run_dir: Path, idx: int, body: str, *, suffix: str = ".adfree.txt") -> None:
    stem = f"{idx:04d} - an episode_{run_dir.name}"
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcripts" / f"{stem}{suffix}").write_text(body, encoding="utf-8")
    (run_dir / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": f"ep{idx}"}, "feed": {"title": "A Show"}}),
        encoding="utf-8",
    )


def _feed(title: str = "A Show") -> SimpleNamespace:
    return SimpleNamespace(title=title)


class TestItFindsTheRecurringPresenter:
    def test_a_weekly_self_introduction_becomes_a_host_candidate(self, tmp_path: Path) -> None:
        run = tmp_path / "run_abc_20260101-000000"
        for i in range(4):
            _episode(run, i, INTRO.format(name="Russ Roberts"))
        assert _recurrent_hosts_from_disk(str(tmp_path), _feed()) == {"Russ Roberts"}

    def test_a_one_off_guest_does_not(self, tmp_path: Path) -> None:
        run = tmp_path / "run_abc_20260101-000000"
        for i in range(4):
            _episode(run, i, INTRO.format(name="Russ Roberts"))
        _episode(run, 9, INTRO.format(name="Ada Lovelace"))
        assert _recurrent_hosts_from_disk(str(tmp_path), _feed()) == {"Russ Roberts"}


class TestItCountsEachEpisodeOnce:
    def test_the_ad_free_body_is_preferred_over_the_cleaned_one(self, tmp_path: Path) -> None:
        # Both bodies exist for every episode. Reading both would double every tally; reading the
        # cleaned one would put the host's introduction behind a pre-roll.
        run = tmp_path / "run_abc_20260101-000000"
        for i in range(4):
            _episode(run, i, INTRO.format(name="Russ Roberts"))
            _episode(run, i, "A word from our sponsor. " * 400, suffix=".cleaned.txt")
        assert len(_newest_run_transcripts(tmp_path)) == 4
        assert _recurrent_hosts_from_disk(str(tmp_path), _feed()) == {"Russ Roberts"}

    def test_a_superseded_run_does_not_vote_twice(self, tmp_path: Path) -> None:
        # The same three episodes, reprocessed. Counted twice the denominator doubles with the
        # numerator and nothing changes; counted once the answer is honest about how much
        # evidence there actually is.
        old = tmp_path / "run_old_20260101-000000"
        new = tmp_path / "run_new_20260201-000000"
        for i in range(3):
            _episode(old, i, INTRO.format(name="Russ Roberts"))
            _episode(new, i, INTRO.format(name="Russ Roberts"))
        assert len(_newest_run_transcripts(tmp_path)) == 3


class TestItStaysInsideOneFeed:
    def test_another_shows_transcripts_are_not_pooled_in(self, tmp_path: Path) -> None:
        # Pointed at a CORPUS root by mistake, this must find nothing rather than make one show's
        # host a candidate on every other show. Transcripts live at <feed>/run_*/transcripts.
        for slug in ("rss_a_1111", "rss_b_2222"):
            run = tmp_path / "feeds" / slug / "run_abc_20260101-000000"
            for i in range(4):
                _episode(run, i, INTRO.format(name="Russ Roberts"))
        assert _newest_run_transcripts(tmp_path) == []
        assert _recurrent_hosts_from_disk(str(tmp_path), _feed()) == set()


class TestItIsSilentWithNothingToRead:
    def test_a_feeds_first_ever_run_contributes_nothing(self, tmp_path: Path) -> None:
        assert _recurrent_hosts_from_disk(str(tmp_path), _feed()) == set()

    def test_a_missing_output_dir_is_not_an_error(self) -> None:
        assert _recurrent_hosts_from_disk("/nonexistent/path/xyz", _feed()) == set()

    def test_no_output_dir_at_all_is_not_an_error(self) -> None:
        assert _recurrent_hosts_from_disk(None, _feed()) == set()
