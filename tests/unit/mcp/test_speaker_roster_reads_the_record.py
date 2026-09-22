"""The MCP speaker roster answers "who spoke" from the speaker RECORD (#2075).

`episode_speaker_roster` used to return the speakers diagnostics wholesale, including its `tried`
block — the raw candidate lists the roster was given, among them the pre-listening guess. An AI
client reading that was handed a second, contradictory answer to "who is on this episode". Now the
roster is the record: every person, whether a voice was matched to them, and talk share from the
voices matched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import enrichment

META = "metadata/ep.metadata.json"
DIAG = {
    "voices": [
        {
            "voice": "SPEAKER_02",
            "resolved_name": "Michael Barbaro",
            "role": "host",
            "named": True,
            "source": "self_intro",
            "voice_type": "person",
            "talk_time_s": 600.0,
        },
        {
            "voice": "SPEAKER_00",
            "resolved_name": "Matina Stevis-Gridneff",
            "role": "guest",
            "named": True,
            "source": "llm_resolution",
            "voice_type": "person",
            "talk_time_s": 300.0,
        },
        {
            "voice": "SPEAKER_01",
            "resolved_name": "SPEAKER_01",
            "role": "unknown",
            "named": False,
            "source": "raw",
            "voice_type": "commercial",
            "talk_time_s": 100.0,
        },
    ],
    "summary": {"num_speakers": 3},
    "tried": {"known_hosts": ["Michael Barbaro", "Natalie Kitroeff", "Rachel Abrams"]},
}


def _corpus(tmp: Path, speakers: List[Dict[str, Any]]) -> CorpusContext:
    (tmp / "metadata").mkdir()
    (tmp / "transcripts").mkdir()
    (tmp / "transcripts" / "ep.txt").write_text("x", encoding="utf-8")
    (tmp / "transcripts" / "ep.speakers.diagnostics.json").write_text(
        json.dumps(DIAG), encoding="utf-8"
    )
    (tmp / META).write_text(
        json.dumps(
            {"content": {"transcript_file_path": "transcripts/ep.txt", "speakers": speakers}}
        ),
        encoding="utf-8",
    )
    return CorpusContext(corpus_dir=tmp)


RECORD = [
    {
        "id": "host",
        "name": "Michael Barbaro",
        "role": "host",
        "placed": True,
        "voices": ["SPEAKER_02"],
        "source": "self_intro",
    },
    {
        "id": "guest",
        "name": "Matina Stevis-Gridneff",
        "role": "guest",
        "placed": True,
        "voices": ["SPEAKER_00"],
        "source": "llm_resolution",
    },
    {
        "id": "unplaced_1",
        "name": "Natalie Kitroeff",
        "role": "host",
        "placed": False,
        "voices": [],
        "source": "feed_statement",
    },
]


class TestTheRosterIsTheRecord:
    def test_placed_people_carry_their_talk_share(self, tmp_path: Path) -> None:
        out = enrichment.episode_speaker_roster(_corpus(tmp_path, RECORD), META)
        by = {s["name"]: s for s in out["speakers"]}
        assert (by["Michael Barbaro"]["placed"], by["Michael Barbaro"]["talk_share"]) == (True, 0.6)
        assert by["Matina Stevis-Gridneff"]["talk_share"] == 0.3

    def test_a_person_only_named_is_listed_but_never_given_talk_time(self, tmp_path: Path) -> None:
        out = enrichment.episode_speaker_roster(_corpus(tmp_path, RECORD), META)
        kitroeff = {s["name"]: s for s in out["speakers"]}["Natalie Kitroeff"]
        assert kitroeff["placed"] is False
        assert (kitroeff["talk_time_s"], kitroeff["voices"]) == (0.0, [])

    def test_the_candidate_lists_are_not_returned(self, tmp_path: Path) -> None:
        out = enrichment.episode_speaker_roster(_corpus(tmp_path, RECORD), META)
        assert "tried" not in out["diagnostics"]
        assert out["diagnostics"]["voices"], "the explanation itself is still returned"

    def test_a_pre_record_artifact_matches_voices_by_name(self, tmp_path: Path) -> None:
        legacy = [{"id": "host", "name": "Michael Barbaro", "role": "host"}]
        out = enrichment.episode_speaker_roster(_corpus(tmp_path, legacy), META)
        (barbaro,) = out["speakers"]
        assert (barbaro["placed"], barbaro["voices"], barbaro["talk_share"]) == (
            None,
            ["SPEAKER_02"],
            0.6,
        )

    def test_the_episode_digest_uses_the_record(self, tmp_path: Path) -> None:
        from podcast_scraper.mcp.tools import composites

        ctx = _corpus(tmp_path, RECORD)
        roster = composites.episode_digest(ctx, META)["speaker_roster"]
        assert [s["name"] for s in roster] == [
            "Michael Barbaro",
            "Matina Stevis-Gridneff",
            "Natalie Kitroeff",
        ]


# TestTheDiarizationQualityEval moved to chipi/podcast-scraper-eval-data
# (tests/unit/podcast_scraper_eval/test_diarization_quality_ignores_unplaced.py)
# with diarization_quality.py itself. The #2075 fix it guards was ported there;
# everything above stays here, because it asserts the MCP roster — runtime.
