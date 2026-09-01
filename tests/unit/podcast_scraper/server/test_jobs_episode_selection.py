"""``POST /api/jobs?episode_selection=`` — a per-REQUEST selection mode (follow-up to #1898 E1).

WHY THIS IS A REQUEST PARAMETER AND NOT A CONFIG KEY. The nightly and a manual backfill want
OPPOSITE values, and before this parameter the only override point was the corpus-wide operator
YAML, which cannot hold both:

* The nightly is a newest-N **window** (``max_episodes: 10`` applied to feed POSITIONS). Its
  whole safety property is that the back catalogue is unreachable — the operator YAML comment
  calls it "the ultimate control against back-catalog ingestion: a nightly can never reach past
  the newest 10, however large the gap."
* A deliberate catch-up wants ``unprocessed``, where already-ingested episodes are dropped by
  guid FIRST so the cap counts episodes of actual work.

Setting ``episode_selection: unprocessed`` in the corpus YAML to serve a backfill therefore
converts EVERY nightly into a back-catalogue crawler: once the newest N are all on disk, the
newest *un-ingested* item is deep in the archive, so the nightly ingests 10 old episodes a night,
per feed, until the feed is exhausted. On a 1000-episode feed that is ~100 unattended nights of
download + GI spend. That is the bug this parameter exists to make impossible, and it is why
OMISSION must stay meaningful: no parameter -> emit no flag -> inherit the YAML/profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.jobs import (
    build_pipeline_argv,
    EPISODE_SELECTION_ALLOWED,
    normalize_episode_selection,
)

pytestmark = pytest.mark.unit


class TestNormalizer:
    @pytest.mark.parametrize("value", ["position", "unprocessed"])
    def test_allowed_modes_pass_through(self, value: str) -> None:
        assert normalize_episode_selection(value) == value

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert normalize_episode_selection("  unprocessed  ") == "unprocessed"

    @pytest.mark.parametrize("value", [None, ""])
    def test_absent_stays_absent(self, value: str | None) -> None:
        """OMISSION IS MEANINGFUL: it must inherit the YAML/profile, not pick a default here."""
        assert normalize_episode_selection(value) is None

    @pytest.mark.parametrize("value", ["bogus", "UNPROCESSED", "Position", "newest", "1"])
    def test_unknown_modes_are_dropped_not_forwarded(self, value: str) -> None:
        """Never forward an unvalidated value: it reaches a subprocess argv.

        Case matters — the CLI's argparse ``choices`` are lower-case, so ``UNPROCESSED`` would
        crash the child rather than mean anything.
        """
        assert normalize_episode_selection(value) is None

    def test_allowlist_matches_the_config_literal(self) -> None:
        """The API must not be able to select a mode ``Config`` does not model."""
        import typing

        from podcast_scraper.config import Config

        literal = Config.model_fields["episode_selection"].annotation
        assert set(typing.get_args(literal)) == EPISODE_SELECTION_ALLOWED


class TestArgv:
    def _argv(self, tmp_path: Path, **kw: object) -> list[str]:
        return build_pipeline_argv(
            tmp_path,
            tmp_path / "viewer_operator.yaml",
            feed_url="https://example.com/feed.xml",
            **kw,  # type: ignore[arg-type]
        )

    def test_requested_mode_reaches_argv(self, tmp_path: Path) -> None:
        argv = self._argv(tmp_path, episode_selection="unprocessed")
        assert "--episode-selection" in argv
        assert argv[argv.index("--episode-selection") + 1] == "unprocessed"

    def test_omitted_emits_no_flag(self, tmp_path: Path) -> None:
        """THE REGRESSION GUARD.

        Emitting a default here would silently override the operator YAML for every caller that
        did not ask — the same class of bug as #1888, where a silent omission inherited the file
        and the caller's choice vanished. Here the danger runs the other way: a hardcoded default
        would make the API, not the operator, decide the nightly's selection mode.
        """
        assert "--episode-selection" not in self._argv(tmp_path)

    def test_explicit_position_is_emitted(self, tmp_path: Path) -> None:
        """``position`` is not the same as omission — it PINS positional over the YAML."""
        argv = self._argv(tmp_path, episode_selection="position")
        assert argv[argv.index("--episode-selection") + 1] == "position"

    def test_rejected_mode_never_reaches_argv(self, tmp_path: Path) -> None:
        assert "--episode-selection" not in self._argv(tmp_path, episode_selection="bogus")

    def test_whole_batch_mode_ignores_it(self, tmp_path: Path) -> None:
        """Per-feed knob only — a whole-batch run takes its selection from feeds.spec/YAML."""
        argv = build_pipeline_argv(
            tmp_path, tmp_path / "viewer_operator.yaml", episode_selection="unprocessed"
        )
        assert "--episode-selection" not in argv

    def test_it_does_not_disturb_the_other_selection_flags(self, tmp_path: Path) -> None:
        argv = self._argv(
            tmp_path, max_episodes=10, episode_order="newest", episode_selection="unprocessed"
        )
        assert argv[argv.index("--max-episodes") + 1] == "10"
        assert argv[argv.index("--episode-order") + 1] == "newest"
        assert argv[argv.index("--episode-selection") + 1] == "unprocessed"

    def test_no_offset_is_injected_alongside(self, tmp_path: Path) -> None:
        """An offset under ``unprocessed`` skips episodes never ingested (config.py warns)."""
        assert "--episode-offset" not in self._argv(tmp_path, episode_selection="unprocessed")
