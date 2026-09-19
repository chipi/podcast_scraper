"""m0009 writes no speaking role for a person no voice was matched to (#2075).

m0009 writes Person roles into kg.json irreversibly on production. It already refused to DEMOTE on
a guessed roster (#2070), but it still PROMOTED from one, on the reasoning that promotion "adds
information and is safe whatever the roster omits". A guessed roster is the pre-listening hint —
names from the feed and show notes that no voice was matched to — so promoting from it writes a
speaking role the audio never supported. 136 production episodes carry a guess as their roster.

Each case drives the real `apply()` over a corpus on disk, including the segments sidecar that
decides whether a roster is a measurement or a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import (
    BackfillSpeakerRolesMigration,
)


def _corpus(
    root: Path,
    *,
    speakers: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    persons: List[Tuple[str, str, str]],
    speakers_source: Optional[str] = None,
) -> Path:
    meta_dir, tx_dir = root / "metadata", root / "transcripts"
    meta_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)
    content: Dict[str, Any] = {
        "speakers": speakers,
        "transcript_file_path": "transcripts/e1.txt",
        "diarization_num_speakers": len({s["speaker"] for s in segments if "speaker" in s}) or None,
    }
    if speakers_source:
        content["speakers_source"] = speakers_source
    (meta_dir / "e1.metadata.json").write_text(
        json.dumps({"feed": {"title": "A Show"}, "content": content}), encoding="utf-8"
    )
    (tx_dir / "e1.segments.json").write_text(json.dumps(segments), encoding="utf-8")
    kg = {
        "schema_version": "2.1",
        "nodes": [
            {"id": pid, "type": "Person", "properties": {"name": name, "role": role}}
            for pid, name, role in persons
        ],
        "edges": [],
    }
    path = meta_dir / "e1.kg.json"
    path.write_text(json.dumps(kg), encoding="utf-8")
    return path


def _roles(kg_path: Path) -> Dict[str, str]:
    doc = json.loads(kg_path.read_text(encoding="utf-8"))
    return {n["properties"]["name"]: n["properties"]["role"] for n in doc["nodes"]}


def _apply(root: Path) -> Dict[str, Any]:
    res = BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=root))
    return dict(res.details)


class TestAPreRecordArtifact:
    def test_a_measured_roster_still_promotes(self, tmp_path: Path) -> None:
        """Golden: the path that was right stays unchanged. A voice is labelled with the name."""
        kg = _corpus(
            tmp_path,
            speakers=[{"id": "host", "name": "Kevin Roose", "role": "host"}],
            segments=[{"speaker": "SPEAKER_00", "speaker_label": "Kevin Roose"}],
            persons=[("person:kevin-roose", "Kevin Roose", "mentioned")],
        )
        details = _apply(tmp_path)
        assert details["persons_promoted"] == 1
        assert details["guess_rosters_skipped"] == 0
        assert _roles(kg) == {"Kevin Roose": "host"}

    def test_a_guessed_roster_promotes_nobody(self, tmp_path: Path) -> None:
        """Diarization heard voices and named none: the roster is the feed's guess."""
        kg = _corpus(
            tmp_path,
            speakers=[{"id": "host", "name": "Tracy Alloway", "role": "host"}],
            segments=[{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}],
            persons=[("person:tracy-alloway", "Tracy Alloway", "mentioned")],
        )
        details = _apply(tmp_path)
        assert details["persons_promoted"] == 0
        assert details["guess_rosters_skipped"] == 1
        assert details["already_correct"] == 0, "a skipped guess was never checked"
        assert _roles(kg) == {"Tracy Alloway": "mentioned"}


class TestASpeakerRecord:
    def test_only_the_placed_person_is_promoted(self, tmp_path: Path) -> None:
        kg = _corpus(
            tmp_path,
            speakers=[
                {
                    "id": "guest",
                    "name": "Matina Stevis-Gridneff",
                    "role": "guest",
                    "placed": True,
                    "voices": ["SPEAKER_00"],
                },
                {"id": "unplaced_1", "name": "Natalie Kitroeff", "role": "host", "placed": False},
            ],
            segments=[{"speaker": "SPEAKER_00", "speaker_label": "Matina Stevis-Gridneff"}],
            persons=[
                ("person:matina-stevis-gridneff", "Matina Stevis-Gridneff", "mentioned"),
                ("person:natalie-kitroeff", "Natalie Kitroeff", "mentioned"),
            ],
            speakers_source="diarized",
        )
        details = _apply(tmp_path)
        assert details["persons_promoted"] == 1
        assert _roles(kg) == {"Matina Stevis-Gridneff": "guest", "Natalie Kitroeff": "mentioned"}

    def test_a_record_with_nobody_placed_promotes_nobody(self, tmp_path: Path) -> None:
        """The a16z shape: diarized, nobody named, every guessed host kept as placed: false."""
        kg = _corpus(
            tmp_path,
            speakers=[
                {"id": "unplaced_1", "name": "Garry Tan", "role": "host", "placed": False},
                {"id": "unplaced_2", "name": "Sriram Krishnan", "role": "host", "placed": False},
            ],
            segments=[{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}],
            persons=[
                ("person:garry-tan", "Garry Tan", "mentioned"),
                ("person:sriram-krishnan", "Sriram Krishnan", "mentioned"),
            ],
            speakers_source="hint",
        )
        details = _apply(tmp_path)
        assert details["persons_promoted"] == 0
        assert _roles(kg) == {"Garry Tan": "mentioned", "Sriram Krishnan": "mentioned"}


class TestANeverDiarizedEpisode:
    def test_a_publisher_transcript_with_no_voice_ids_promotes_nobody(self, tmp_path: Path) -> None:
        """Odd Lots from a publisher transcript: segments exist, but carry no voice at all.

        The guess check used to require raw voice ids, so this episode — never diarized, its roster
        the hint by construction — was treated as a measurement. On the production snapshot that
        let m0009 write 105 promotions on 71 such episodes (Odd Lots 69, In Moscow's Shadows 33).
        Operator decision 2026-09-17: an episode with no voices casts nobody.
        """
        kg = _corpus(
            tmp_path,
            speakers=[
                {"id": "host", "name": "Tracy Alloway", "role": "host"},
                {"id": "host_2", "name": "Joe Weisenthal", "role": "host"},
            ],
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello and welcome to Odd Lots."}],
            persons=[
                ("person:tracy-alloway", "Tracy Alloway", "mentioned"),
                ("person:joe-weisenthal", "Joe Weisenthal", "mentioned"),
            ],
        )
        details = _apply(tmp_path)
        assert details["persons_promoted"] == 0
        assert details["guess_rosters_skipped"] == 1
        assert _roles(kg) == {"Tracy Alloway": "mentioned", "Joe Weisenthal": "mentioned"}
