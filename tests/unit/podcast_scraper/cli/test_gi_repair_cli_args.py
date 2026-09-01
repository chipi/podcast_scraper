"""The ``gi-repair`` CLI surface: ids-file parsing and the exit codes it maps to.

The work-list file is the operator's only expression of intent for a paid re-derivation, so
every way it can be mis-read is a way to spend money on the wrong episodes. Exit codes matter
for the same reason: this command is meant to be scripted, and 0-on-nothing-happened is how a
reprocess silently does not reprocess.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper import cli
from podcast_scraper.gi import repair as repair_mod


def _run(tmp_path, ids_path, monkeypatch):
    """Drive the REAL ``gi-repair`` argv parser, stubbing only the paid repair itself.

    Deliberately goes through ``_parse_gi_repair_argv`` rather than hand-building a Namespace:
    the parser is half of what is under test (flag names, dest names, defaults), and a
    hand-built Namespace would keep passing after a flag was renamed out from under it.
    """
    captured = {}

    class _Report:
        repaired: list = []
        failed: list = []
        requested_not_found: list = []
        ok = True

        def format(self):
            return "stub"

    def _fake_repair(corpus_root, cfg, **kwargs):
        captured.update(kwargs)
        captured["_corpus_root"] = corpus_root
        return _Report()

    # The handler imports it inline (``from .gi.repair import ...``) on every call, so the
    # binding to replace lives on the source module, not on ``cli``.
    monkeypatch.setattr(repair_mod, "repair_placeholder_artifacts", _fake_repair)

    argv = ["--output-dir", str(tmp_path), "--dry-run"]
    if ids_path is not None:
        argv += ["--episode-ids", str(ids_path)]
    args = cli._parse_gi_repair_argv(argv)
    rc = cli._run_gi_repair_cli(args, logging.getLogger("test.gi_repair"))
    return rc, captured


class TestIdsFileParsing:
    @pytest.mark.parametrize(
        "content,expected",
        [
            ("a\nb\nc\n", ["a", "b", "c"]),
            ("a\r\nb\r\n", ["a", "b"]),  # CRLF work-lists come from Windows/pasted output
            ("# a comment\na\n\n  \nb\n", ["a", "b"]),
            ("  a  \n\tb\t\n", ["a", "b"]),  # ids pasted with surrounding whitespace
            ("a\nb\na\n", ["a", "b"]),  # duplicate collapsed, order preserved
            ("b\na\nb\n", ["b", "a"]),  # dedupe keeps FIRST occurrence, not sorted order
        ],
        ids=["plain", "crlf", "comments-and-blanks", "whitespace", "duplicate", "order"],
    )
    def test_parses(self, tmp_path, monkeypatch, content, expected):
        ids = tmp_path / "ids.txt"
        ids.write_text(content, encoding="utf-8")
        rc, captured = _run(tmp_path, ids, monkeypatch)
        assert rc == 0
        assert captured["episode_ids"] == expected

    def test_duplicates_are_reported_not_silently_collapsed(self, tmp_path, monkeypatch, caplog):
        """A duplicate is money: each id is a paid LLM re-derivation."""
        ids = tmp_path / "ids.txt"
        ids.write_text("a\nb\na\na\n", encoding="utf-8")
        caplog.set_level(logging.WARNING)
        rc, captured = _run(tmp_path, ids, monkeypatch)
        assert rc == 0
        assert captured["episode_ids"] == ["a", "b"]
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "duplicate" in msg and "2" in msg

    def test_a_comment_only_file_is_an_error_not_a_corpus_sweep(self, tmp_path, monkeypatch):
        """THE DANGEROUS CASE.

        ``episode_ids=None`` means "sweep every legacy-placeholder artifact in the corpus".
        If a file that parsed to nothing fell through to None, a typo'd work-list would launch
        a corpus-wide paid re-derivation instead of the handful of episodes intended.
        """
        ids = tmp_path / "ids.txt"
        ids.write_text("# nothing but comments\n\n   \n", encoding="utf-8")
        rc, captured = _run(tmp_path, ids, monkeypatch)
        assert rc == 2
        assert captured == {}, "the repair must not have been reached at all"

    def test_missing_file_exits_2_without_repairing(self, tmp_path, monkeypatch):
        rc, captured = _run(tmp_path, tmp_path / "nope.txt", monkeypatch)
        assert rc == 2
        assert captured == {}

    def test_no_ids_flag_means_the_placeholder_sweep(self, tmp_path, monkeypatch):
        """Omitting the flag is the only way to ask for a corpus-wide sweep."""
        rc, captured = _run(tmp_path, None, monkeypatch)
        assert rc == 0
        assert captured["episode_ids"] is None
