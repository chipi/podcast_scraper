"""The voice rung reads the feed's other episodes from the directory a real run passes it.

``pipeline.apply_diarization_to_result`` hands the rung ``cfg.output_dir`` as the feed workspace.
The service sets every feed's ``output_dir`` to ``corpus_feed_output_dir(parent, url)``
(``<corpus>/feeds/<feed>/``), which holds that feed's ``run_*/metadata`` -- what
``voice_identity.feed_history`` globs. This runs the whole diarization stage on that layout, with
two fingerprinted sibling episodes that name the host, and an episode whose host never says who
he is: the rung must name him from the siblings. Synthetic fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.ml.diarization import pipeline as P
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.utils import correlation, filesystem

pytestmark = pytest.mark.unit

MODEL = "pyannote/speaker-diarization-community-1"
HOST = [1.0, 0.0, 0.0]
GUEST = [0.0, 1.0, 0.0]
URL = "https://example.com/wren.xml"


def _sibling(feed: Path, run: str, eid: str) -> None:
    (feed / run / "metadata").mkdir(parents=True)
    (feed / run / "transcripts").mkdir()
    (feed / run / "metadata" / f"{eid}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": eid},
                "content": {"transcript_file_path": f"transcripts/{eid}.txt"},
            }
        )
    )
    voices: List[Dict[str, Any]] = [
        {
            "voice": "SPEAKER_00",
            "resolved_name": "Tobias Wren",
            "named": True,
            "role": "host",
            "source": "self_intro",
            "talk_time_s": 900,
        }
    ]
    (feed / run / "transcripts" / f"{eid}.speakers.diagnostics.json").write_text(
        json.dumps(
            {
                "voices": voices,
                "voice_embeddings": {
                    "model_name": MODEL,
                    "dimension": 3,
                    "by_voice": {"SPEAKER_00": HOST},
                },
            }
        )
    )


def test_a_real_runs_output_dir_is_the_feed_workspace_the_rung_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    feed = Path(filesystem.corpus_feed_output_dir(str(tmp_path), URL))
    _sibling(feed, "run_20261001-000000_aaaaaaaa", "ep1")
    _sibling(feed, "run_20261008-000000_bbbbbbbb", "ep2")
    monkeypatch.setattr(P, "_resolve_voices_via_llm", lambda *a, **k: ({}, {}))
    P._copresence_cache.clear()
    correlation.set_episode_id("ep3")
    words = " ".join(["indeed"] * 60)
    cfg = Config(
        output_dir=str(feed),
        speaker_resolution_llm=False,
        voice_identity_rung=True,
    )
    assert Path(cfg.output_dir) == feed
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 120, "SPEAKER_00"),
            DiarizationSegment(120, 240, "SPEAKER_01"),
        ],
        num_speakers=2,
        model_name=MODEL,
        speaker_embeddings={"SPEAKER_00": HOST, "SPEAKER_01": GUEST},
    )
    out = P.apply_diarization_to_result(
        {
            "text": "x",
            "segments": [
                {"start": 0, "end": 120, "text": "So, where were we. " + words},
                {"start": 120, "end": 240, "text": "Right, the harvest. " + words},
            ],
        },
        "",
        cfg,
        [],
        precomputed_diarization=diar,
        episode_title="Ep 3",
    )
    correlation.set_episode_id(None)
    by_voice = {v["voice"]: v for v in out["speaker_diagnostics"]["voices"]}
    assert by_voice["SPEAKER_00"]["resolved_name"] == "Tobias Wren", by_voice
    assert by_voice["SPEAKER_00"]["source"] == "voice_identity"
    assert not by_voice["SPEAKER_01"]["named"]
