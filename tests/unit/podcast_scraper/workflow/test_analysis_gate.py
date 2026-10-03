"""The consumer half of RFC-124 §5.3's completeness gate.

The producer half withholds `.en.*` when a translation is incomplete. This is the half that
stops the English stages reading the SOURCE anyway — without it a pending translation falls
straight through the resolver's precedence to `.adfree.txt`/`.txt` and runs English prompts
over Spanish text, which §5.2 measured as confidently wrong rather than blind.

BOTH DIRECTIONS MATTER EQUALLY HERE. A gate that blocks too eagerly would stall the 678
English episodes in the corpus, which is a worse outcome than the one it prevents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper import config
from podcast_scraper.workflow.translation_stage import analysis_blocked_reason

pytestmark = pytest.mark.unit

REL = "transcripts/ep.txt"


def _cfg(**kw: Any) -> config.Config:
    return config.Config(rss="https://example.com/f.xml", **kw)


def _english_set(root: Path, *, present: bool = True, analysis_body: bool = True) -> None:
    """Lay down a SWAPPED episode, or an un-swapped one (D-44).

    `present` is now "did the atomic swap happen": the canonical body holds the translation and the
    source is kept at its language-tagged name. The gate reads the presence of that tagged file,
    because only the swap creates it — so this helper's job is to create or omit it.

    `analysis_body` controls `<base>.adfree.txt`, the body ANALYSIS actually resolves. An earlier
    version of this helper omitted it, which is why the gate test passed while the gate had a hole.
    """
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    if present:
        # swapped: canonical holds English, the source keeps its tag
        (root / REL).write_text("Maya: Hello.\n", encoding="utf-8")
        (root / "transcripts" / "ep.es.txt").write_text("Maya: Hola.\n", encoding="utf-8")
        (root / "transcripts" / "ep.segments.json").write_text("[]", encoding="utf-8")
        if analysis_body:
            (root / "transcripts" / "ep.adfree.txt").write_text("Maya: Hello.\n", encoding="utf-8")
    else:
        # un-swapped: the canonical body is still the source language, no tagged sibling
        (root / REL).write_text("Maya: Hola.\n", encoding="utf-8")


def _ledger(root: Path, *, failed: int, total: int) -> None:
    units: list[dict[str, Any]] = [
        {
            "unit_id": f"t000{i}.u01",
            "turn_id": f"t000{i}",
            "content_key": f"k{i}",
            "status": "failed" if i < failed else "ok",
            "alignment": "sentence",
            "sentences": [],
        }
        for i in range(total)
    ]
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / "transcripts" / "ep.translation.json").write_text(
        json.dumps({"version": "1.0", "units": units}), encoding="utf-8"
    )


class TestEnglishIsNeverBlocked:
    def test_an_english_episode_passes_immediately(self, tmp_path: Path) -> None:
        """One language comparison and return. A gate that could stall the English pipeline in
        exchange for protecting the Spanish one would not be worth having."""
        assert (
            analysis_blocked_reason(
                _cfg(language="en"),
                transcript_relpath=REL,
                effective_output_dir=str(tmp_path),
            )
            is None
        )

    def test_an_english_episode_with_no_files_at_all_still_passes(self, tmp_path: Path) -> None:
        """The check must not depend on artifacts an English episode never has."""
        assert (
            analysis_blocked_reason(
                _cfg(language="en"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
            )
            is None
        )

    def test_an_unknown_language_is_not_blocked(self, tmp_path: Path) -> None:
        """Most of the corpus predates language resolution. Blocking on "no language" would
        stop ingesting the English corpus that works today — the same reasoning
        `_unsupported_language_skip_reason` records for the transcription guard."""
        cfg = _cfg()
        object.__setattr__(cfg, "language", None) if hasattr(cfg, "language") else None
        assert analysis_blocked_reason(
            cfg, transcript_relpath=REL, effective_output_dir=str(tmp_path)
        ) in (None,)


class TestNonEnglishIsBlockedUnlessComplete:
    def test_a_complete_english_set_lets_analysis_run(self, tmp_path: Path) -> None:
        _english_set(tmp_path, present=True)
        assert (
            analysis_blocked_reason(
                _cfg(language="es"),
                transcript_relpath=REL,
                effective_output_dir=str(tmp_path),
            )
            is None
        )

    def test_a_missing_english_set_blocks_and_says_no_translation_was_attempted(
        self, tmp_path: Path
    ) -> None:
        _english_set(tmp_path, present=False)
        reason = analysis_blocked_reason(
            _cfg(language="es"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert reason and "no translation was attempted" in reason
        assert "SKIPPED" in reason

    def test_failed_units_block_and_the_reason_counts_them(self, tmp_path: Path) -> None:
        """The operator reading any one of the log line, the manifest or the metric learns the
        same thing — how many units failed — rather than only that something was skipped."""
        _english_set(tmp_path, present=False)
        _ledger(tmp_path, failed=2, total=7)
        reason = analysis_blocked_reason(
            _cfg(language="es"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert reason and "2 of 7 units failed" in reason

    def test_a_MISSING_analysis_base_no_longer_blocks_and_that_is_correct(
        self, tmp_path: Path
    ) -> None:
        """The hole these two tests plugged cannot exist under D-44, so they assert the new
        property.

        THEY USED TO BE: `.en.txt` plus `.en.segments.json` existed, so the old predicate
        passed and
        `translation_status` said `translated` — while ANALYSIS, whose first English candidate was
        `.en.adfree.txt`, fell through it and the deliberately-absent source `.adfree.txt` to the
        SPANISH `.txt`. English prompts over Spanish, behind a gate reporting success. The predicate
        therefore had to name all three files.

        WHY IT IS GONE. The ANALYSIS candidates are now `[<base>.adfree.txt, <base>.txt]` and, after
        the swap, BOTH are English. A missing ad-free base falls back to the canonical body,
        which is
        the analysis language — so nothing reads the wrong language and there is nothing to gate.
        A translation is complete or it is not; the swap is indivisible, so there is no half-written
        English set to detect.

        What a missing ad-free base now costs is ads left in the analysis text, which is a quality
        matter for S2.5 and not a correctness one.
        """
        from podcast_scraper.workflow.transcript_resolution import (
            resolve_text_path,
            TranscriptPurpose,
        )

        _english_set(tmp_path, present=True, analysis_body=False)
        reason = analysis_blocked_reason(
            _cfg(language="es"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert reason is None, "the swap happened, so analysis may run"
        # And what analysis reads is English, which is the whole point.
        resolved = resolve_text_path(tmp_path, REL, purpose=TranscriptPurpose.ANALYSIS)
        assert resolved == tmp_path / REL
        assert "Hello" in resolved.read_text(encoding="utf-8")

    def test_an_UNSWAPPED_episode_blocks(self, tmp_path: Path) -> None:
        """The condition that replaces all of the partial-set checks: the tagged source is absent,
        so the swap never happened and the canonical body is still the source language."""
        _english_set(tmp_path, present=False)
        assert not (tmp_path / "transcripts" / "ep.es.txt").exists()
        reason = analysis_blocked_reason(
            _cfg(language="es"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert reason is not None

    def test_the_reason_tells_the_operator_what_to_do(self, tmp_path: Path) -> None:
        """A skip nobody can act on is a dead end. The sentence names the repair and states
        that nothing expensive was lost."""
        _english_set(tmp_path, present=False)
        reason = analysis_blocked_reason(
            _cfg(language="es"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert reason and "Repair the translation and reprocess" in reason
        assert "source transcript and every successful unit are on disk" in reason

    def test_an_override_language_is_honoured(self, tmp_path: Path) -> None:
        """The override is the remedy for a feed whose declared tag is wrong, so it has to win
        here too — otherwise a mis-tagged English show would be permanently blocked."""
        _english_set(tmp_path, present=False)
        assert (
            analysis_blocked_reason(
                _cfg(language="es", language_override="en"),
                transcript_relpath=REL,
                effective_output_dir=str(tmp_path),
            )
            is None
        )

    def test_the_feeds_declared_language_reaches_the_gate(self, tmp_path: Path) -> None:
        _english_set(tmp_path, present=False)
        reason = analysis_blocked_reason(
            _cfg(language="en"),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            feed_language="es-ES",
        )
        assert reason and "'es'" in reason


class TestItCannotCrashTheEpisode:
    def test_no_transcript_path_does_not_block(self, tmp_path: Path) -> None:
        assert (
            analysis_blocked_reason(
                _cfg(language="es"), transcript_relpath=None, effective_output_dir=str(tmp_path)
            )
            is None
        )

    def test_a_corrupt_ledger_still_blocks_rather_than_raising(self, tmp_path: Path) -> None:
        """A ledger that cannot be parsed is not evidence that the translation completed."""
        _english_set(tmp_path, present=False)
        (tmp_path / "transcripts" / "ep.translation.json").write_text("{not json", "utf-8")
        reason = analysis_blocked_reason(
            _cfg(language="es"), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert reason is not None
