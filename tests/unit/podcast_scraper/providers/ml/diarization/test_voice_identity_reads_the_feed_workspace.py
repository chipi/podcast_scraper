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
from typing import Any, Dict, List, Literal

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.ml.diarization import pipeline as P, voice_identity as vi
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


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: Literal["off", "shadow", "publish"],
    fingerprints: bool = True,
) -> Dict[str, Any]:
    feed = Path(filesystem.corpus_feed_output_dir(str(tmp_path), URL))
    if not feed.exists():
        _sibling(feed, "run_20261001-000000_aaaaaaaa", "ep1")
        _sibling(feed, "run_20261008-000000_bbbbbbbb", "ep2")
    monkeypatch.setattr(P, "_resolve_voices_via_llm", lambda *a, **k: ({}, {}))
    P._copresence_cache.clear()
    correlation.set_episode_id("ep3")
    words = " ".join(["indeed"] * 60)
    cfg = Config(output_dir=str(feed), speaker_resolution_llm=False, voice_identity_rung=mode)
    assert Path(str(cfg.output_dir)) == feed
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 120, "SPEAKER_00"),
            DiarizationSegment(120, 240, "SPEAKER_01"),
        ],
        num_speakers=2,
        model_name=MODEL,
        speaker_embeddings={"SPEAKER_00": HOST, "SPEAKER_01": GUEST} if fingerprints else {},
    )
    try:
        out: Dict[str, Any] = P.apply_diarization_to_result(
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
    finally:
        correlation.set_episode_id(None)
    return out


def _voices(out: Dict[str, Any]) -> Dict[str, Any]:
    return {v["voice"]: v for v in out["speaker_diagnostics"]["voices"]}


def test_a_real_runs_output_dir_is_the_feed_workspace_the_rung_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    by_voice = _voices(_run(monkeypatch, tmp_path, "publish"))
    assert by_voice["SPEAKER_00"]["resolved_name"] == "Tobias Wren", by_voice
    assert by_voice["SPEAKER_00"]["source"] == "voice_identity"
    assert not by_voice["SPEAKER_01"]["named"]


def _published(out: Dict[str, Any]) -> str:
    view = dict(out)
    view["speaker_diagnostics"] = {
        k: v for k, v in out["speaker_diagnostics"].items() if k != vi.SIDECAR_KEY
    }
    return json.dumps(view, sort_keys=True, default=str)


def test_off_writes_no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert vi.SIDECAR_KEY not in _run(monkeypatch, tmp_path, "off")["speaker_diagnostics"]


def test_shadow_publishes_what_off_publishes_and_records_the_fingerprints_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off = _run(monkeypatch, tmp_path, "off")
    shadow = _run(monkeypatch, tmp_path, "shadow")
    assert _published(shadow) == _published(off)
    assert not _voices(shadow)["SPEAKER_00"]["named"]
    side = shadow["speaker_diagnostics"][vi.SIDECAR_KEY]
    assert side["version"] == vi.SIDECAR_VERSION and side["mode"] == "shadow"
    assert side["report"]["history_voices"] == 2 and side["report"]["groups"] == 1
    host = side["voices"]["SPEAKER_00"]
    assert host["ladder"]["name"] is None
    assert host["fingerprint_result"]["name"] == "Tobias Wren"
    assert host["outcome"] == "adds"
    assert host["match"]["group"] == "Tobias Wren" and host["match"]["similarity"] == 1.0
    assert side["voices"]["SPEAKER_01"]["outcome"] == "agree"
    assert side["outcomes"] == {"agree": 1, "adds": 1, "removes": 0, "changes": 0}


def test_publish_records_the_ladders_name_beside_the_published_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    side = _run(monkeypatch, tmp_path, "publish")["speaker_diagnostics"][vi.SIDECAR_KEY]
    assert side["mode"] == "publish"
    assert side["voices"]["SPEAKER_00"]["ladder"]["name"] is None
    assert side["voices"]["SPEAKER_00"]["fingerprint_result"]["name"] == "Tobias Wren"


def test_an_episode_without_fingerprints_says_why_the_rung_abstained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    side = _run(monkeypatch, tmp_path, "shadow", fingerprints=False)["speaker_diagnostics"][
        vi.SIDECAR_KEY
    ]
    assert side["report"]["abstained"] == "no fingerprints for this episode"
    assert side["outcomes"]["agree"] == 2
