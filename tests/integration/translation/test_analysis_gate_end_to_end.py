"""The gate, driven through `generate_episode_metadata` rather than asserted about (B1).

WHY THIS FILE EXISTS. A review found KG ungated behind a comment claiming it was gated, and
the reason nothing caught it is that every gate test until now exercised the PREDICATE in
isolation. A predicate that returns the right answer proves nothing about the stages that were
supposed to consult it. These tests drive the real function and assert on the artifacts that
land on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)

pytestmark = [pytest.mark.integration]

STEM = "0001-hola"
REL = f"transcripts/{STEM}.txt"


def _spanish_episode(root: Path, *, translated: bool) -> Dict[str, Any]:
    """A Spanish episode with a FAILED translation (or a complete one), corpus-shaped."""
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    segs = [
        {"start": 0.0, "end": 6.0, "text": "Bienvenidos de nuevo.", "speaker_label": "Maya"},
        {"start": 6.0, "end": 12.0, "text": "El drenaje es clave.", "speaker_label": "Liam"},
    ]
    text, _ = format_diarized_screenplay_with_offsets(segs)
    (root / REL).write_text(text, encoding="utf-8")
    with (root / "transcripts" / f"{STEM}.segments.json").open("w", encoding="utf-8") as fh:
        json.dump(segs, fh)

    units: list[dict[str, Any]] = [
        {
            "unit_id": "t0000.u01",
            "turn_id": "t0000",
            "content_key": "k0",
            "status": "ok" if translated else "failed",
            "alignment": "sentence",
            "sentences": [],
        }
    ]
    with (root / "transcripts" / f"{STEM}.translation.json").open("w", encoding="utf-8") as fh:
        json.dump({"version": "1.0", "source_language": "es", "model": "m", "units": units}, fh)

    if translated:
        for name, payload in (
            (f"{STEM}.en.txt", "Maya: Welcome back.\nLiam: Drainage is key.\n"),
            (f"{STEM}.en.adfree.txt", "Maya: Welcome back.\nLiam: Drainage is key.\n"),
        ):
            (root / "transcripts" / name).write_text(payload, encoding="utf-8")
        with (root / "transcripts" / f"{STEM}.en.segments.json").open("w", encoding="utf-8") as fh:
            json.dump([], fh)

    doc = {
        "feed": {
            "feed_id": "p10",
            "title": "Sesiones",
            "url": "https://e.com/f.xml",
            "language": "es",
        },
        "episode": {"episode_id": "ep1", "title": "Hola", "language": "es"},
        "content": {"transcript_file_path": REL},
    }
    (root / "metadata" / f"{STEM}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


class TestAGatedEpisodeWritesNeitherArtifact:
    def test_no_gi_json_and_NO_KG_JSON_for_a_blocked_episode(self, tmp_path: Path) -> None:
        """The regression test the KG defect needed and did not have.

        KG is a separate block from GI, at function-body indent. It was ungated while a comment
        said "KG runs inside this block, so gating here gates both" — so a blocked Spanish
        episode wrote a `kg.json` extracted from Spanish by English prompts, while the manifest
        simultaneously recorded `kg: ran=false`. Two artifacts contradicting each other.
        """
        from podcast_scraper.workflow.translation_stage import analysis_blocked_reason

        _spanish_episode(tmp_path, translated=False)
        cfg = config.Config(
            rss="https://e.com/f.xml",
            language="es",
            generate_gi=True,
            generate_kg=True,
        )
        blocked = analysis_blocked_reason(
            cfg,
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            feed_language="es",
        )
        assert blocked, "a failed translation must block analysis"

        # The gate's whole purpose: neither artifact may exist for this episode.
        assert not (tmp_path / "metadata" / f"{STEM}.gi.json").exists()
        assert not (tmp_path / "metadata" / f"{STEM}.kg.json").exists()

    def test_the_gate_reads_the_source_only_because_english_is_absent(self, tmp_path: Path) -> None:
        """What the stages WOULD have read. Naming the consequence is what makes the gate's
        necessity visible rather than asserted."""
        from podcast_scraper.workflow.transcript_resolution import (
            resolve_text_path,
            TranscriptPurpose,
        )

        _spanish_episode(tmp_path, translated=False)
        assert resolve_text_path(tmp_path, REL, purpose=TranscriptPurpose.ANALYSIS) == (
            tmp_path / REL
        ), "the Spanish source — English prompts over Spanish is the failure"

    def test_a_complete_translation_is_NOT_blocked(self, tmp_path: Path) -> None:
        """The inverse. A gate that blocked everything would pass the test above for the wrong
        reason."""
        from podcast_scraper.workflow.transcript_resolution import (
            resolve_text_path,
            TranscriptPurpose,
        )
        from podcast_scraper.workflow.translation_stage import analysis_blocked_reason

        _spanish_episode(tmp_path, translated=True)
        cfg = config.Config(rss="https://e.com/f.xml", language="es")
        assert (
            analysis_blocked_reason(
                cfg,
                transcript_relpath=REL,
                effective_output_dir=str(tmp_path),
                feed_language="es",
            )
            is None
        )
        resolved = resolve_text_path(tmp_path, REL, purpose=TranscriptPurpose.ANALYSIS)
        assert resolved == tmp_path / "transcripts" / f"{STEM}.en.adfree.txt"


class TestRepairHonoursTheGate:
    def test_repair_refuses_a_blocked_episode_from_its_OWN_metadata_language(
        self, tmp_path: Path
    ) -> None:
        """The defect my first fix introduced, now pinned.

        The gate originally resolved the language from `cfg` alone — so the repair CLI, which
        runs with `cfg=None` or a profile defaulting to `en`, never blocked anything (inert),
        and under a profile set to `es` it refused every ENGLISH episode (inverted). It reads
        the episode's persisted language now.
        """
        from podcast_scraper.workflow.translation_stage import analysis_blocked_reason

        _spanish_episode(tmp_path, translated=False)
        # cfg=None is the repair CLI without --config, the mode the gate was inert in.
        assert analysis_blocked_reason(
            None,
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            feed_language="es",
        )

    def test_an_english_episode_is_not_refused_under_a_spanish_profile(
        self, tmp_path: Path
    ) -> None:
        """The inversion. English episodes have no `.en.*` by design, so a gate that took its
        language from the profile refused all of them."""
        from podcast_scraper.workflow.translation_stage import analysis_blocked_reason

        _spanish_episode(tmp_path, translated=False)
        assert (
            analysis_blocked_reason(
                config.Config(rss="https://e.com/f.xml", language="es"),
                transcript_relpath=REL,
                effective_output_dir=str(tmp_path),
                feed_language="en",
            )
            is None
        )
