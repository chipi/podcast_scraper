"""A person diarization split over two voices is ONE person on every surface (#2075).

Validation run, In Our Time: `Misha Glenny` host on one voice, `Misha Glennie` guest on another.
The transcript,
the diagnostics, the record and the graph each carried two people for one human, and the sync audit
reported the record and diagnostics disagreeing about his role.

Drives the real roster (`resolve_speaker_roster` with a canned identification, no model), the real
diagnostics builder, segments labelled the way the diarization pipeline labels them, the real
`generate_episode_metadata` with the real KG builder (`metadata_only`), then the corpus sync audit.
Synthetic names and text; the shape is the measured one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.providers.ml.diarization import roster as roster_mod
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.workflow import metadata_generation as metadata

pytestmark = [pytest.mark.integration]

_tests_dir = Path(__file__).parent.parent.parent
_repo = _tests_dir.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)
_aspec = importlib.util.spec_from_file_location(
    "speaker_sync_audit", _repo / "scripts" / "audit" / "speaker_sync_audit.py"
)
assert _aspec is not None and _aspec.loader is not None
audit_mod = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(audit_mod)

REL = "transcripts/0001 - Episode_Title.txt"
TURNS = [
    ("SPEAKER_00", "Welcome to the show. I'm Sarah Guo, and today we talk about model routing."),
    ("SPEAKER_01", "Thanks. The short version is that routing is a pricing problem first."),
    ("SPEAKER_00", "Say more about that."),
    ("SPEAKER_02", "Sure. Every request has a cost ceiling, and the router works under it."),
]


def _diarization() -> DiarizationResult:
    segs, t = [], 30.0
    for voice, _ in TURNS:
        segs.append(DiarizationSegment(start=t, end=t + 300.0, speaker=voice))
        t += 300.0
    return DiarizationResult(segments=segs, num_speakers=3)


def _write_episode(out: Path) -> None:
    diar = _diarization()
    voice_texts: Dict[str, str] = {}
    for v, text in TURNS:
        voice_texts[v] = (voice_texts.get(v, "") + " " + text).strip()
    roster = roster_mod.resolve_speaker_roster(
        diar,
        None,
        known_hosts=["Sarah Guo"],
        voice_texts=voice_texts,
        ordered_turns=TURNS,
        # The measured split: the identifier named the two halves of one guest differently and
        # gave them different roles.
        llm_voice_names={
            "SPEAKER_00": "Sarah Guo",
            "SPEAKER_01": "Elad Gilman",
            "SPEAKER_02": "Elad Gilmann",
        },
        llm_voice_roles={"SPEAKER_00": "host", "SPEAKER_01": "guest", "SPEAKER_02": "host"},
    )
    diagnostics = roster_mod.build_speaker_diagnostics(
        diar, roster, voice_texts=voice_texts, known_hosts=["Sarah Guo"]
    )
    (out / "transcripts").mkdir(parents=True)
    (out / REL).write_text(
        "\n".join(f"{roster.label_for(v)}: {text}" for v, text in TURNS) + "\n", encoding="utf-8"
    )
    segs: List[Dict[str, Any]] = [
        {
            "start": float(i),
            "end": float(i) + 1.0,
            "text": text,
            "speaker": v,
            "speaker_label": roster.label_for(v),
            "speaker_role": roster.by_voice[v].role,
        }
        for i, (v, text) in enumerate(TURNS)
    ]
    base = str(out / REL)[: -len(".txt")]
    Path(base + ".segments.json").write_text(json.dumps(segs), encoding="utf-8")
    Path(base + ".speakers.diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    cfg = _pc.create_test_config(
        output_dir=str(out),
        generate_metadata=True,
        metadata_format="json",
        generate_kg=True,
        kg_extraction_source="metadata_only",
    )
    path = metadata.generate_episode_metadata(
        feed=_pc.create_test_feed(),
        episode=_pc.create_test_episode(),
        feed_url=_pc.TEST_FEED_URL,
        cfg=cfg,
        output_dir=str(out),
        run_suffix=None,
        transcript_file_path=REL,
        transcript_source="whisper_transcription",
        whisper_model="base",
        detected_hosts=["Sarah Guo"],
        detected_guests=["Elad Gilman"],
    )
    assert path and Path(path).is_file()


@pytest.fixture()
def written(tmp_path: Path) -> Path:
    out = tmp_path / "corpus"
    _write_episode(out)
    return out


def test_the_record_holds_one_entry_for_the_split_person(written: Path) -> None:
    meta = json.loads(next(written.rglob("*.metadata.json")).read_text(encoding="utf-8"))
    placed = [s for s in meta["content"]["speakers"] if s.get("placed") is True]
    elads = [s for s in placed if "elad" in s["name"].lower()]
    assert len(elads) == 1, placed
    assert elads[0]["name"] == "Elad Gilman"
    assert elads[0]["role"] == "guest"
    assert sorted(elads[0]["voices"]) == ["SPEAKER_01", "SPEAKER_02"]


def test_the_graph_casts_him_once_as_a_guest(written: Path) -> None:
    kg = json.loads(next(written.rglob("*.kg.json")).read_text(encoding="utf-8"))
    people = [
        (n.get("properties") or {})
        for n in kg["nodes"]
        if n.get("type") == "Person"
        and "elad" in str((n.get("properties") or {}).get("name")).lower()
    ]
    assert [(p.get("name"), p.get("role")) for p in people] == [("Elad Gilman", "guest")]


def test_every_surface_agrees(written: Path) -> None:
    findings, counts = audit_mod.audit(written)
    assert counts.get("examined") == 1
    assert findings == [], findings
