"""D-44's atomic swap: both bodies move, or neither does (#2254).

WHY ATOMICITY IS THE WHOLE POINT. Under D-44 every generic reader opens `<base>.txt` without
asking what language it holds, because that file is the analysis language by construction. A
half-applied swap is therefore not a degraded episode — it is an episode that LIES, and
summary/GI/KG/search would run English prompts over source-language text. §5.2 measured that
shape: recall holds while precision falls 67% -> 18%, so the failure is inventing people rather
than finding none.

These tests force each failure point rather than trusting the code reads correctly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.translation.artifacts import _swap_in_translation

pytestmark = pytest.mark.unit

REL = "transcripts/01 - ep.txt"
SOURCE_TEXT = "Lucía Herrera: Bienvenidos de nuevo."
TARGET_TEXT = "Lucía Herrera: Welcome back."
SOURCE_SEGS: List[Dict[str, Any]] = [{"id": 0, "start": 0.0, "end": 2.0, "text": "Bienvenidos"}]
TARGET_SEGS: List[Dict[str, Any]] = [{"id": 0, "start": 0.0, "end": 2.0, "text": "Welcome back"}]


def _episode(root: Path, *, with_segments: bool = True) -> None:
    tr = root / "transcripts"
    tr.mkdir(parents=True, exist_ok=True)
    (tr / "01 - ep.txt").write_text(SOURCE_TEXT, encoding="utf-8")
    if with_segments:
        (tr / "01 - ep.segments.json").write_text(json.dumps(SOURCE_SEGS), encoding="utf-8")


def _swap(root: Path, language: str = "es") -> Any:
    return _swap_in_translation(
        REL,
        str(root),
        language,
        target_text=TARGET_TEXT,
        target_segments=TARGET_SEGS,
        unit_count=1,
    )


class TestTheHappyPath:
    def test_the_canonical_body_becomes_the_TRANSLATION(self, tmp_path: Path) -> None:
        _episode(tmp_path)
        assert _swap(tmp_path) == REL
        tr = tmp_path / "transcripts"
        assert (tr / "01 - ep.txt").read_text(encoding="utf-8") == TARGET_TEXT

    def test_the_SOURCE_is_kept_at_its_language_tagged_name(self, tmp_path: Path) -> None:
        """D-2: the record of what was actually said is not destroyed, it is renamed."""
        _episode(tmp_path)
        _swap(tmp_path)
        tr = tmp_path / "transcripts"
        assert (tr / "01 - ep.es.txt").read_text(encoding="utf-8") == SOURCE_TEXT

    def test_both_sidecars_move_with_their_bodies(self, tmp_path: Path) -> None:
        """A body and a sidecar from different languages is the displacement bug in its newest
        shape: English text sliced at source-language offsets."""
        _episode(tmp_path)
        _swap(tmp_path)
        tr = tmp_path / "transcripts"
        assert json.loads((tr / "01 - ep.segments.json").read_text()) == TARGET_SEGS
        assert json.loads((tr / "01 - ep.es.segments.json").read_text()) == SOURCE_SEGS

    def test_no_temp_files_survive(self, tmp_path: Path) -> None:
        _episode(tmp_path)
        _swap(tmp_path)
        leftover = sorted(p.name for p in (tmp_path / "transcripts").iterdir() if ".tmp" in p.name)
        assert not leftover, leftover

    def test_an_episode_with_no_sidecar_still_swaps(self, tmp_path: Path) -> None:
        """The sidecar is optional; the body is not."""
        _episode(tmp_path, with_segments=False)
        assert _swap(tmp_path) == REL
        tr = tmp_path / "transcripts"
        assert (tr / "01 - ep.txt").read_text(encoding="utf-8") == TARGET_TEXT
        assert (tr / "01 - ep.es.txt").read_text(encoding="utf-8") == SOURCE_TEXT


class TestItRefusesWhatItCannotDo:
    @pytest.mark.parametrize("language", ["en", "EN", "en-US", "", None])
    def test_english_and_blank_are_refused(self, tmp_path: Path, language: Any) -> None:
        """An English episode has no source variant — the canonical file already holds English — and
        generating `<base>.en.txt` is exactly the suffix D-44 removed."""
        _episode(tmp_path)
        assert _swap(tmp_path, language) is None
        # untouched
        assert (tmp_path / "transcripts" / "01 - ep.txt").read_text(encoding="utf-8") == SOURCE_TEXT
        assert not (tmp_path / "transcripts" / "01 - ep.en.txt").exists()

    def test_a_regional_tag_is_normalized_to_the_primary_subtag(self, tmp_path: Path) -> None:
        """`es-ES` must produce `.es.txt`: the toggle asks for a predictable name."""
        _episode(tmp_path)
        assert _swap(tmp_path, "es-ES") == REL
        assert (tmp_path / "transcripts" / "01 - ep.es.txt").is_file()
        assert not (tmp_path / "transcripts" / "01 - ep.es-ES.txt").exists()


class TestItIsIdempotent:
    def test_a_second_swap_is_a_no_op(self, tmp_path: Path) -> None:
        """Without the guard, a re-run moves the TRANSLATION aside as though it were the source —
        hiding English behind a language tag and leaving the canonical path holding nothing."""
        _episode(tmp_path)
        assert _swap(tmp_path) == REL
        assert _swap(tmp_path) == REL  # returns success, changes nothing

        tr = tmp_path / "transcripts"
        assert (tr / "01 - ep.txt").read_text(encoding="utf-8") == TARGET_TEXT
        assert (tr / "01 - ep.es.txt").read_text(encoding="utf-8") == SOURCE_TEXT
        assert not (tr / "01 - ep.es.es.txt").exists()


class TestRollback:
    """Each failure point forced, because "it rolls back" is a claim until the process dies."""

    def test_a_failure_renaming_the_source_leaves_the_episode_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _episode(tmp_path)
        real = os.replace

        def fail_on_first(src: Any, dst: Any) -> None:
            raise OSError("forced: renaming the source aside")

        monkeypatch.setattr(os, "replace", fail_on_first)
        assert _swap(tmp_path) is None
        monkeypatch.setattr(os, "replace", real)

        tr = tmp_path / "transcripts"
        assert (tr / "01 - ep.txt").read_text(encoding="utf-8") == SOURCE_TEXT
        assert json.loads((tr / "01 - ep.segments.json").read_text()) == SOURCE_SEGS
        assert not (tr / "01 - ep.es.txt").exists()
        assert not [p for p in tr.iterdir() if ".tmp" in p.name]

    def test_a_failure_PART_WAY_through_restores_the_source_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dangerous one: the source has already been renamed away when the failure hits, so the
        canonical path does not exist. Rollback has to put it back."""
        _episode(tmp_path)
        real = os.replace
        calls = {"n": 0}

        def fail_on_third(src: Any, dst: Any) -> None:
            calls["n"] += 1
            if calls["n"] == 3:  # source body moved, sidecar moved, now the staged body
                raise OSError("forced: installing the translation")
            real(src, dst)

        monkeypatch.setattr(os, "replace", fail_on_third)
        assert _swap(tmp_path) is None
        monkeypatch.setattr(os, "replace", real)

        tr = tmp_path / "transcripts"
        assert (tr / "01 - ep.txt").is_file(), "the canonical body was not restored"
        assert (tr / "01 - ep.txt").read_text(encoding="utf-8") == SOURCE_TEXT
        assert json.loads((tr / "01 - ep.segments.json").read_text()) == SOURCE_SEGS
        assert not (tr / "01 - ep.es.txt").exists()
        assert not [p for p in tr.iterdir() if ".tmp" in p.name]

    def test_the_gate_reads_false_after_a_rolled_back_swap(self, tmp_path: Path) -> None:
        """The completeness signal is the presence of the tagged SOURCE, so a rolled-back swap must
        leave it absent — otherwise a failed translation reads as a complete one."""
        from podcast_scraper.translation.artifacts import translation_swap_happened

        _episode(tmp_path)
        assert not translation_swap_happened(REL, str(tmp_path), "es")
        _swap(tmp_path)
        assert translation_swap_happened(REL, str(tmp_path), "es")

    def test_the_gate_is_true_for_english_without_any_swap(self, tmp_path: Path) -> None:
        """An English episode needs no swap; its canonical body is already the analysis language."""
        from podcast_scraper.translation.artifacts import translation_swap_happened

        _episode(tmp_path)
        assert translation_swap_happened(REL, str(tmp_path), "en")
