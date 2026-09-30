"""S2.7: the save sites stop producing artifacts the patterns cannot see, and stale `.en.*` dies.

TWO DEFECTS THIS CLOSES. A source-language ad-free base for a non-English episode is an IDENTITY
artifact — a file asserting ads were removed when the English `_AD_PATTERNS` matched nothing
(measured: zero hits on the Spanish source against two on its English render). And a stale
`.en.txt` outliving the source it was translated from is served as CURRENT, because the resolver
keys on presence with no status check and D-38 makes it the default — with offsets that now
index different text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.workflow.episode_processor import _produce_transcript_sidecars

pytestmark = pytest.mark.unit

REL = "transcripts/ep.txt"

_PREROLL = (
    "Ramp understands no one wants to chase receipts. Ramp saves companies 5 percent. "
    "Check out ramp dot com slash invest. They all use WorkOS for SSO and SCIM and RBAC. "
    "Visit WorkOS dot com to get started. Learn more at rogo dot ai slash Felix. "
)


def _segments() -> List[Dict[str, Any]]:
    body = ("Hello and welcome everyone I am the host. Today we discuss the boom. ") * 12
    segs: List[Dict[str, Any]] = [
        {"start": 0.0, "end": 1.0, "text": _PREROLL, "speaker_label": "Announcer"}
    ]
    clock = 1.0
    for i, sentence in enumerate(body.split(". ")):
        sentence = sentence.strip()
        if not sentence:
            continue
        segs.append(
            {
                "start": clock,
                "end": clock + 1.0,
                "text": sentence if sentence.endswith(".") else sentence + ".",
                "speaker_label": "Maya" if (i // 2) % 2 == 0 else "Liam",
            }
        )
        clock += 1.0
    return segs


def _lay_down(root: Path) -> tuple:
    segs = _segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / REL).write_text(text, encoding="utf-8")
    return text, segs


def _cfg(**kw: Any) -> config.Config:
    base: Dict[str, Any] = {"rss": "https://e.com/f.xml", "save_adfree_transcript": True}
    base.update(kw)
    return config.Config(**base)


class TestTheSourceAdFreeBaseIsEnglishOnly:
    def test_an_english_episode_still_gets_its_adfree_base(self, tmp_path: Path) -> None:
        """The path that must not regress: 678 episodes depend on it."""
        text, segs = _lay_down(tmp_path)
        _produce_transcript_sidecars(_cfg(language="en"), text, segs, REL, str(tmp_path))
        assert (tmp_path / "transcripts" / "ep.adfree.txt").is_file()

    def test_an_episode_with_no_resolved_language_still_gets_one(self, tmp_path: Path) -> None:
        """Most of the corpus predates language resolution. Withholding from those would stop
        producing the ad-free base the English corpus already relies on."""
        text, segs = _lay_down(tmp_path)
        cfg = _cfg()
        _produce_transcript_sidecars(cfg, text, segs, REL, str(tmp_path))
        assert (tmp_path / "transcripts" / "ep.adfree.txt").is_file()

    def test_a_SPANISH_episode_gets_NO_source_adfree_base(self, tmp_path: Path) -> None:
        """The identity artifact. On Spanish the English patterns match nothing, so the file
        would claim ads were removed when none could be seen."""
        text, segs = _lay_down(tmp_path)
        _produce_transcript_sidecars(_cfg(language="es"), text, segs, REL, str(tmp_path))
        assert not (tmp_path / "transcripts" / "ep.adfree.txt").exists()
        assert not (tmp_path / "transcripts" / "ep.adfree.segments.json").exists()

    def test_the_skip_is_logged_with_its_reason(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silence would read as "the flag is off". An operator has to be able to tell the two
        apart from the log alone."""
        import logging

        caplog.set_level(logging.INFO)
        text, segs = _lay_down(tmp_path)
        _produce_transcript_sidecars(_cfg(language="es"), text, segs, REL, str(tmp_path))
        assert any("identity artifact" in r.message for r in caplog.records)

    def test_the_SOURCE_turns_are_still_written_for_a_spanish_episode(self, tmp_path: Path) -> None:
        """Turns are language-neutral: sentence splitting is punctuation-only, and turns are
        the input translation needs. Withholding them would break the next stage."""
        text, segs = _lay_down(tmp_path)
        _produce_transcript_sidecars(_cfg(language="es"), text, segs, REL, str(tmp_path))
        assert (tmp_path / "transcripts" / "ep.turns.json").is_file()
        assert not (tmp_path / "transcripts" / "ep.adfree.turns.json").exists()


class TestStaleEnglishIsInvalidated:
    @staticmethod
    def _plant_english(root: Path) -> List[Path]:
        planted = []
        for name in (
            "ep.en.txt",
            "ep.en.segments.json",
            "ep.en.adfree.txt",
            "ep.en.adfree.segments.json",
            "ep.en.adfree.admap.json",
        ):
            path = root / "transcripts" / name
            path.write_text("stale", encoding="utf-8")
            planted.append(path)
        return planted

    def test_rewriting_the_source_deletes_every_english_derivative(self, tmp_path: Path) -> None:
        """A stale English body outlives the source it was translated from and is served as
        current, with offsets that now index different text — the displacement bug with a time
        axis."""
        text, segs = _lay_down(tmp_path)
        planted = self._plant_english(tmp_path)
        _produce_transcript_sidecars(_cfg(language="es"), text, segs, REL, str(tmp_path))
        assert not any(p.exists() for p in planted)

    def test_invalidation_happens_even_with_the_adfree_flag_OFF(self, tmp_path: Path) -> None:
        """The trigger is "a non-English source was rewritten", not "the ad-free branch ran".

        A review found invalidation living inside the `elif save_adfree_transcript` branch, so
        with the flag off a rewrite left stale `.en.*`: the gate passes on presence, TIMELINE
        serves the old English with old cue times, and provenance resolves new spans against old
        segments — while the ledger says the translation failed.
        """
        text, segs = _lay_down(tmp_path)
        planted = self._plant_english(tmp_path)
        cfg = _cfg(language="es", save_adfree_transcript=False)
        _produce_transcript_sidecars(cfg, text, segs, REL, str(tmp_path))
        assert not any(p.exists() for p in planted)

    def test_translation_json_is_KEPT(self, tmp_path: Path) -> None:
        """It is the content-keyed translation memory (D-33). A relabel changes every offset but
        no unit's text, so the ledger still answers for most units and the re-render costs no
        GPU. Deleting it would turn the most common repair in this corpus into a full
        re-translation."""
        text, segs = _lay_down(tmp_path)
        ledger = tmp_path / "transcripts" / "ep.translation.json"
        ledger.write_text('{"version": "1.0", "units": []}', encoding="utf-8")
        _produce_transcript_sidecars(_cfg(language="es"), text, segs, REL, str(tmp_path))
        assert ledger.is_file(), "the translation memory must survive a source rewrite"

    def test_an_english_episode_invalidates_nothing(self, tmp_path: Path) -> None:
        """An English episode has no `.en.*` to invalidate, and must not touch files that
        happen to share the prefix."""
        text, segs = _lay_down(tmp_path)
        planted = self._plant_english(tmp_path)
        _produce_transcript_sidecars(_cfg(language="en"), text, segs, REL, str(tmp_path))
        assert all(p.exists() for p in planted)

    def test_missing_english_files_are_not_an_error(self, tmp_path: Path) -> None:
        text, segs = _lay_down(tmp_path)
        _produce_transcript_sidecars(_cfg(language="es"), text, segs, REL, str(tmp_path))
        assert (tmp_path / "transcripts" / "ep.turns.json").is_file()
