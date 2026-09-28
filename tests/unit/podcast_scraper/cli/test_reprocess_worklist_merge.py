"""A configured repair work-list must survive the CLI merge — an absent flag may not erase it.

THE INCIDENT THIS PINS (prod, 2026-09-28). ``reprocess_episode_ids`` is a real ``Config`` field, so
a corpus/operator YAML may carry one, and ``_load_and_merge_config`` duly puts it on ``args`` via
``parser.set_defaults(**model_dump)``. But ``_build_config`` called
``_load_reprocess_episode_ids(args.reprocess_episode_ids_file)`` UNCONDITIONALLY, and that returns
``[]`` when no file was passed. So a YAML work-list was parsed, validated, placed on ``args`` — then
overwritten with empty.

The blast radius is the inverse of what the loader's own docstring guards against, and worse.
``reprocess_episode_ids`` implies ``reprocess_existing_only``, whose episode set is *the whole
on-disk corpus for the feed*. A work-list naming ONE episode therefore selected 50, which were
rewritten in place. Nothing raised, nothing warned: the repair simply had no scope.

Why the seam and not the layers. ``workflow/test_reprocess_episode_ids.py`` covers the config layer
thoroughly and passed throughout; the CLI flag parsing was fine too. The defect lived precisely
BETWEEN them, in the handoff — so these tests drive the REAL ``_load_and_merge_config`` →
``_build_config`` chain. Per the sibling suite's hard-won lesson: a mirror of the logic cannot
disagree with the logic.
"""

from __future__ import annotations

import pytest
import yaml as _yaml

from podcast_scraper import cli

_YAML_BASE = {
    "rss_url": "https://example.com/feed.xml",
    "transcription_provider": "whisper",
}


def _cfg(yaml_data, argv, tmp_path):
    """Drive the WHOLE real chain: a real ``--config`` YAML through the real parser.

    Deliberately not a hand-built parser plus a monkeypatched loader. The first version of this
    file did that and every test errored on a missing argparse default, because
    ``_add_common_arguments`` does not define every attribute ``_build_config`` reads — the harness
    diverged from the production path in exactly the region under test.
    """
    cfg_path = tmp_path / "operator.yaml"
    cfg_path.write_text(_yaml.safe_dump(dict(yaml_data)), encoding="utf-8")
    args = cli.parse_args(["--config", str(cfg_path), *argv])
    return cli._build_config(args)


class TestConfiguredWorklistSurvivesAnAbsentFlag:
    def test_THE_INCIDENT_yaml_worklist_is_not_erased(self, tmp_path):
        """The regression itself: one named episode must stay one named episode."""
        cfg = _cfg(
            {**_YAML_BASE, "reprocess_episode_ids": ["7ed31c1e-3cf5-467d-a302-b47b014db8c5"]},
            ["--pipeline-stage", "retranscript_only"],
            tmp_path,
        )
        assert cfg.reprocess_episode_ids == ["7ed31c1e-3cf5-467d-a302-b47b014db8c5"]

    def test_the_surviving_worklist_still_implies_existing_only(self, tmp_path):
        """Scope restriction is the whole point — the implication must not be lost in the merge."""
        cfg = _cfg(
            {**_YAML_BASE, "reprocess_episode_ids": ["ep-1", "ep-2"]},
            ["--pipeline-stage", "retranscript_only"],
            tmp_path,
        )
        assert cfg.reprocess_existing_only is True

    def test_multiple_ids_all_survive(self, tmp_path):
        ids = [f"ep-{i}" for i in range(5)]
        cfg = _cfg({**_YAML_BASE, "reprocess_episode_ids": ids}, [], tmp_path)
        assert cfg.reprocess_episode_ids == ids

    def test_whitespace_only_entries_are_dropped_not_kept_as_empty_ids(self, tmp_path):
        """An empty id would match nothing yet still count as "a work-list was given"."""
        cfg = _cfg({**_YAML_BASE, "reprocess_episode_ids": ["ep-1", "   ", "ep-2"]}, [], tmp_path)
        assert cfg.reprocess_episode_ids == ["ep-1", "ep-2"]


class TestTheExplicitFlagStillWins:
    def test_a_file_overrides_the_yaml_worklist(self, tmp_path):
        """An operator naming a file means THAT file, even when the YAML carries its own list."""
        wl = tmp_path / "worklist.txt"
        wl.write_text("from-file-1\nfrom-file-2\n", encoding="utf-8")
        cfg = _cfg(
            {**_YAML_BASE, "reprocess_episode_ids": ["from-yaml"]},
            ["--reprocess-episode-ids", str(wl)],
            tmp_path,
        )
        assert cfg.reprocess_episode_ids == ["from-file-1", "from-file-2"]

    def test_comments_and_blanks_in_the_file_are_ignored(self, tmp_path):
        wl = tmp_path / "worklist.txt"
        wl.write_text("# repair batch\n\nep-a\nep-b  # trailing\n", encoding="utf-8")
        cfg = _cfg(_YAML_BASE, ["--reprocess-episode-ids", str(wl)], tmp_path)
        assert cfg.reprocess_episode_ids == ["ep-a", "ep-b"]


class TestNoWorklistIsStillNoWorklist:
    def test_neither_source_yields_an_empty_list(self, tmp_path):
        """The fix must not invent a work-list: a normal run stays unscoped."""
        cfg = _cfg(_YAML_BASE, [], tmp_path)
        assert cfg.reprocess_episode_ids == []
        assert cfg.reprocess_existing_only is False

    def test_an_empty_file_still_fails_loudly(self, tmp_path):
        """The sibling guard must be preserved — it must NOT now fall back to the YAML."""
        wl = tmp_path / "empty.txt"
        wl.write_text("# nothing but a comment\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no episode ids"):
            _cfg(
                {**_YAML_BASE, "reprocess_episode_ids": ["from-yaml"]},
                ["--reprocess-episode-ids", str(wl)],
                tmp_path,
            )

    def test_a_missing_file_still_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="no such file"):
            _cfg(_YAML_BASE, ["--reprocess-episode-ids", str(tmp_path / "nope.txt")], tmp_path)
