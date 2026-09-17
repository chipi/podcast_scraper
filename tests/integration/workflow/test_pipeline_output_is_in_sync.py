"""What the pipeline writes is in sync with its own speaker record — and drift is caught (#2075).

Drives the real `generate_episode_metadata`, with the real knowledge-graph builder (no LLM:
`kg_extraction_source=metadata_only`), over a real segments sidecar and speakers diagnostics, then
runs the corpus audit script over the output directory. The unit tests pin each rule; this proves
the pipeline and the rules agree, and that the audit fails loudly on the drift it exists to catch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

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


def _seg(label: str, voice: str, role: Any, text: str) -> Dict[str, Any]:
    return {
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "speaker": voice,
        "speaker_label": label,
        "speaker_role": role,
    }


@pytest.fixture()
def written(tmp_path: Path) -> Path:
    out = tmp_path / "corpus"
    (out / "transcripts").mkdir(parents=True)
    (out / REL).write_text(
        "Michael Barbaro: From The New York Times, this is The Daily.\n"
        "Matina Stevis-Gridneff: Canada walked away from the trade talks.\n",
        encoding="utf-8",
    )
    base = str(out / REL)[: -len(".txt")]
    segs = [
        _seg("Michael Barbaro", "SPEAKER_02", "host", "From The New York Times."),
        _seg("Matina Stevis-Gridneff", "SPEAKER_00", "guest", "Canada walked away."),
    ]
    Path(base + ".segments.json").write_text(json.dumps(segs), encoding="utf-8")
    Path(base + ".speakers.diagnostics.json").write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice": "SPEAKER_02",
                        "resolved_name": "Michael Barbaro",
                        "role": "host",
                        "named": True,
                        "source": "self_intro",
                    },
                    {
                        "voice": "SPEAKER_00",
                        "resolved_name": "Matina Stevis-Gridneff",
                        "role": "guest",
                        "named": True,
                        "source": "llm_resolution",
                    },
                ],
                "tried": {"known_hosts": ["Michael Barbaro", "Natalie Kitroeff", "Rachel Abrams"]},
            }
        ),
        encoding="utf-8",
    )
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
        detected_hosts=["Michael Barbaro", "Natalie Kitroeff"],
        detected_guests=["Matina Stevis-Gridneff"],
    )
    assert path and Path(path).is_file()
    kg_files = list(out.rglob("*.kg.json"))
    assert kg_files, "the real KG builder wrote no kg.json — the test missed the seam"
    return out


def test_the_pipeline_output_is_in_sync(written: Path) -> None:
    findings, counts = audit_mod.audit(written)
    assert counts.get("examined") == 1
    assert findings == [], findings
    assert audit_mod.main(["--corpus-dir", str(written), "--quiet-ok"]) == 0


def test_the_record_it_wrote_is_what_the_audit_judged(written: Path) -> None:
    meta = json.loads(next(written.rglob("*.metadata.json")).read_text(encoding="utf-8"))
    by = {s["name"]: s for s in meta["content"]["speakers"]}
    assert by["Michael Barbaro"]["placed"] is True
    assert by["Natalie Kitroeff"]["placed"] is False  # a feed host no voice was matched to
    kg = json.loads(next(written.rglob("*.kg.json")).read_text(encoding="utf-8"))
    roles = {
        (n.get("properties") or {}).get("name"): (n.get("properties") or {}).get("role")
        for n in kg["nodes"]
        if n.get("type") == "Person"
    }
    assert roles.get("Michael Barbaro") == "host"
    assert roles.get("Matina Stevis-Gridneff") == "guest"
    assert roles.get("Natalie Kitroeff") not in ("host", "guest")


def test_drift_is_caught_and_fails_the_gate(written: Path) -> None:
    kg_path = next(written.rglob("*.kg.json"))
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    kg["nodes"].append(
        {
            "id": "person:natalie-kitroeff",
            "type": "Person",
            "properties": {"name": "Natalie Kitroeff", "role": "host"},
        }
    )
    kg_path.write_text(json.dumps(kg), encoding="utf-8")
    findings, counts = audit_mod.audit(written)
    assert counts["out_of_sync"] == 1
    assert any(v.startswith("UNPLACED_CAST") for v in findings[0]["violations"])
    assert audit_mod.main(["--corpus-dir", str(written), "--quiet-ok"]) == 1


def test_the_context_digest_it_wrote_is_checked_and_its_drift_caught(written: Path) -> None:
    ctx_files = list(written.rglob("*.context.json"))
    assert ctx_files, "the pipeline wrote no context.json — the context rule was never exercised"
    ctx = json.loads(ctx_files[0].read_text(encoding="utf-8"))
    assert ctx["basic"]["hosts"] == ["Michael Barbaro"]
    ctx["basic"]["hosts"] = ["Michael Barbaro", "Natalie Kitroeff"]  # the old guess-shaped copy
    ctx_files[0].write_text(json.dumps(ctx), encoding="utf-8")
    findings, _counts = audit_mod.audit(written)
    assert any(v.startswith("CONTEXT_VS_RECORD") for f in findings for v in f["violations"])
