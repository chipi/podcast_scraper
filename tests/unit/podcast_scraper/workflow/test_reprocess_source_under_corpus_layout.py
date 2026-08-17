"""``--reprocess-source`` must find the episode's metadata in a PRIOR run dir (#925 / #33).

THE BUG THIS PINS, verified live 2026-08-16
``_episode_existing_transcript_source`` resolved the metadata with
``_determine_metadata_path(episode, effective_output_dir, run_suffix, cfg)`` — a path inside THIS
run's directory. Under ``--single-feed-uses-corpus-layout`` every run creates a fresh
``run_<timestamp>/``, so the prior metadata is in a different directory, the open raises OSError,
the function returns None, and ``_force_reprocess_for_source`` decides the episode does not match
the requested source.

Consequence: the "#925 forcing re-transcription + diarization" branch was UNREACHABLE on a
corpus. Every episode fell through to "transcript already exists; skipping (--skip-existing)".
Reproduced on a corpus copy whose metadata declared ``transcript_source:
whisper_transcription`` while ``--reprocess-source whisper_transcription`` was passed: the
forcing log line never appeared, result was ``episodes=1 ok=0``, ``Episodes skipped: 1``.

``make redo-diarization`` is built entirely on that flag, so it would report success and
re-diarize nothing.

The transcript lookup fixed exactly this in D7 (``existing_transcript_path_in_corpus``); the
metadata lookup did not get the same treatment. These tests hold both halves together.
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
from podcast_scraper.workflow.episode_processor import (
    _episode_existing_transcript_source,
    _force_reprocess_for_source,
)

pytestmark = [pytest.mark.unit]


class _Episode:
    """Duck-typed episode matching what the resolvers actually read.

    ``run_index._episode_guid`` calls ``episode.item.find("guid").text`` — ``item`` is the RSS
    XML element, not a plain object. A fixture with a ``.guid`` attribute silently resolves to
    None and makes every assertion here pass for the wrong reason.
    """

    def __init__(self, guid: str, title: str = "An Episode", idx: int = 1) -> None:
        self.item = safe_fromstring(f"<item><guid>{guid}</guid></item>")
        self.guid = guid
        self.title = title
        self.title_safe = title  # _determine_metadata_path (legacy, same-run path)
        self.idx = idx
        self.episode_id = guid


def _cfg(root: Path, **kw: Any) -> config.Config:
    """Corpus-layout config.

    NOTE ``_apply_single_feed_corpus_layout`` REWRITES ``output_dir`` to
    ``<root>/feeds/<slug>/`` when the flag is set, so ``cfg.output_dir`` is the FEED dir, not the
    corpus root. ``corpus_metadata_index`` is built for exactly that: it globs both
    ``run_*/metadata/*`` (feed dir) and ``feeds/*/run_*/metadata/*`` (corpus root). A fixture
    that writes to a hand-picked feed slug will not be found.
    """
    return config.Config(
        rss_url="https://example.com/feed.xml",
        output_dir=str(root),
        single_feed_uses_corpus_layout=True,
        **kw,
    )


def _corpus_with_prior_run(
    cfg: config.Config,
    *,
    guid: str = "guid-abc",
    transcript_source: str = "whisper_transcription",
) -> None:
    """A corpus whose ONLY copy of the episode lives in an older run dir under the FEED dir."""
    run = Path(str(cfg.output_dir)) / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    name = "0001 - An Episode"
    (run / "transcripts" / f"{name}.txt").write_text("words", encoding="utf-8")
    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": guid, "guid": guid, "title": "An Episode"},
                "content": {
                    "transcript_file_path": f"transcripts/{name}.txt",
                    "transcript_source": transcript_source,
                },
            }
        ),
        encoding="utf-8",
    )


def _fresh_run_dir(cfg: config.Config) -> str:
    """This run's (empty) output dir — a DIFFERENT run_* from the one holding the metadata."""
    d = Path(str(cfg.output_dir)) / "run_20260816-090000"
    (d / "metadata").mkdir(parents=True, exist_ok=True)
    return str(d)


def test_the_source_is_found_in_a_prior_run_dir(tmp_path):
    """THE regression: this returned None, which made --reprocess-source a no-op."""
    cfg = _cfg(tmp_path)
    _corpus_with_prior_run(cfg)

    src = _episode_existing_transcript_source(_Episode("guid-abc"), _fresh_run_dir(cfg), None, cfg)

    assert src == "whisper_transcription", (
        "the episode's metadata is in an older run dir; resolving against THIS run's dir finds "
        "nothing and silently disables --reprocess-source"
    )


def test_force_reprocess_fires_when_the_source_matches(tmp_path):
    """The predicate the whole flag hangs on — false here means the episode is skipped."""
    cfg = _cfg(tmp_path, reprocess_source="whisper_transcription")
    _corpus_with_prior_run(cfg)

    assert _force_reprocess_for_source(_Episode("guid-abc"), _fresh_run_dir(cfg), None, cfg) is True


def test_force_reprocess_does_not_fire_for_a_different_source(tmp_path):
    """Scoped reprocess must stay scoped: direct_download episodes are left alone (#925)."""
    cfg = _cfg(tmp_path, reprocess_source="whisper_transcription")
    _corpus_with_prior_run(cfg, transcript_source="direct_download")

    assert (
        _force_reprocess_for_source(_Episode("guid-abc"), _fresh_run_dir(cfg), None, cfg) is False
    )


def test_no_reprocess_source_means_no_forcing(tmp_path):
    cfg = _cfg(tmp_path)
    _corpus_with_prior_run(cfg)

    assert (
        _force_reprocess_for_source(_Episode("guid-abc"), _fresh_run_dir(cfg), None, cfg) is False
    )


def test_an_episode_absent_from_the_corpus_resolves_to_none(tmp_path):
    cfg = _cfg(tmp_path)
    _corpus_with_prior_run(cfg, guid="guid-abc")

    src = _episode_existing_transcript_source(
        _Episode("guid-SOMETHING-ELSE"), _fresh_run_dir(cfg), None, cfg
    )

    assert src is None


def test_the_legacy_non_corpus_path_still_works(tmp_path):
    """Without corpus layout the old same-run resolution must be untouched."""
    run = tmp_path / "run_x"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    name = "0001 - An Episode"
    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "guid-abc", "guid": "guid-abc", "title": "An Episode"},
                "content": {"transcript_source": "direct_download"},
            }
        ),
        encoding="utf-8",
    )
    cfg = config.Config(rss_url="https://example.com/feed.xml", output_dir=str(tmp_path))

    src = _episode_existing_transcript_source(_Episode("guid-abc"), str(run), None, cfg)

    assert src == "direct_download"
