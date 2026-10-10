"""Unit tests for scripts/measure/naming_coverage.py — absolute naming coverage (#2304)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "naming_coverage_under_test", ROOT / "scripts" / "measure" / "naming_coverage.py"
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["naming_coverage_under_test"] = _mod
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _episode(
    root: Path,
    name: str,
    labels: list,
    quote_speakers: list,
    *,
    diagnostics: bool = True,
    segments: bool = True,
    language: str = "en",
) -> Path:
    run = root / "feeds" / "f1" / f"run_{name}"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    meta = run / "metadata" / f"0001 - {name}.metadata.json"
    meta.write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f1", "title": "Show", "language": language},
                "episode": {"episode_id": name, "title": name},
                "content": {"transcript_file_path": f"transcripts/0001 - {name}.txt"},
            }
        ),
        encoding="utf-8",
    )
    nodes = [
        {"id": f"quote:{i}", "type": "Quote", "properties": {"speaker_id": sid}}
        for i, sid in enumerate(quote_speakers)
    ]
    (run / "metadata" / f"0001 - {name}.gi.json").write_text(
        json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8"
    )
    if segments:
        (run / "transcripts" / f"0001 - {name}.segments.json").write_text(
            json.dumps(
                [{"speaker": f"SPEAKER_0{i}", "speaker_label": lab} for i, lab in enumerate(labels)]
            ),
            encoding="utf-8",
        )
    if diagnostics:
        (run / "transcripts" / f"0001 - {name}.speakers.diagnostics.json").write_text(
            "{}", encoding="utf-8"
        )
    return meta


def test_a_named_voice_and_named_quotes_count(tmp_path: Path) -> None:
    meta = _episode(tmp_path, "a", ["Tom Holland", "SPEAKER_01"], ["person:tom-holland", None])
    row = _mod.scan_episode(meta, "en")
    assert row["group"] == "named" and row["named_voices"] == 1
    assert (row["quotes"], row["named_quotes"]) == (2, 1)


def test_only_diarization_labels_is_unnamed_and_replayable(tmp_path: Path) -> None:
    meta = _episode(tmp_path, "b", ["SPEAKER_00", "SPEAKER_01", "SPEAKER"], [None])
    assert _mod.scan_episode(meta, "en")["group"] == "stored_unnamed"


def test_without_diagnostics_it_cannot_be_replayed(tmp_path: Path) -> None:
    meta = _episode(tmp_path, "c", ["SPEAKER_00"], [None], diagnostics=False)
    assert _mod.scan_episode(meta, "en")["group"] == "no_diagnostics"


def test_without_segments(tmp_path: Path) -> None:
    meta = _episode(tmp_path, "d", [], [None], segments=False)
    assert _mod.scan_episode(meta, "en")["group"] == "no_segments"


def test_another_language_is_out_of_scope(tmp_path: Path) -> None:
    meta = _episode(tmp_path, "e", ["SPEAKER_00"], [None], language="pt")
    assert _mod.scan_episode(meta, "en") is None
    assert _mod.scan_episode(meta, None)["group"] == "stored_unnamed"


def test_the_gate_fails_above_the_limit(tmp_path: Path) -> None:
    _episode(tmp_path, "a", ["Tom Holland"], ["person:tom-holland"])
    _episode(tmp_path, "b", ["SPEAKER_00"], [None])
    assert _mod.main(["--corpus", str(tmp_path), "--max-unnamed", "1"]) == 0
    assert _mod.main(["--corpus", str(tmp_path), "--max-unnamed", "0"]) == 1
