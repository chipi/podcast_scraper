"""Unit tests for scripts/tools/corpus_quality_report.py (#1647).

The reader is a thin layer over ``podcast_scraper.quality.attribution``, so these tests focus
on the part that is easy to get quietly wrong: turning missing or malformed on-disk artifacts
into recorded NOTES rather than into zeros that read as health.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "corpus_quality_report_under_test",
    ROOT / "scripts" / "tools" / "corpus_quality_report.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["corpus_quality_report_under_test"] = _mod
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _write_episode(
    run_dir: Path,
    idx: int,
    *,
    ledger: Dict[str, Any] | None = None,
    insights: List[Dict[str, Any]] | None = None,
    write_gi: bool = True,
) -> Path:
    """Lay down one episode's metadata (+ optional GI) the way the pipeline does."""
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{idx:04d} - Episode"
    processing: Dict[str, Any] = {"run_id": "r1"}
    if ledger is not None:
        processing["stage_ledger"] = ledger
    metadata_path = meta_dir / f"{stem}.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "episode": {"episode_id": f"ep-{idx}", "duration_seconds": 1800},
                "feed": {"title": "Test Feed"},
                "processing": processing,
            }
        ),
        encoding="utf-8",
    )
    if write_gi:
        (meta_dir / f"{stem}.gi.json").write_text(
            json.dumps({"nodes": insights if insights is not None else []}), encoding="utf-8"
        )
    return metadata_path


def _insight(speaker: str, surfaceable: bool | None = None) -> Dict[str, Any]:
    props: Dict[str, Any] = {"speaker": speaker}
    if surfaceable is not None:
        props["surfaceable"] = surfaceable
    return {"type": "Insight", "properties": props}


class TestEpisodeReading:
    def test_reads_ledger_and_attribution(self, tmp_path: Path) -> None:
        path = _write_episode(
            tmp_path / "run_a",
            1,
            ledger={"speaker_detection": {"outcome": "ran"}},
            insights=[_insight("Casey Newton"), _insight("Kevin Roose")],
        )
        record = _mod._episode_from_metadata(path)
        assert record.stage_ledger["speaker_detection"]["outcome"] == "ran"
        assert record.insights_total == 2
        assert record.insights_surfaceable == 2
        assert record.voices_named == 2
        assert record.notes == []

    def test_absent_surfaceable_flag_counts_as_surfaceable(self, tmp_path: Path) -> None:
        """Mirrors is_surfaceable_insight(): only an explicit False excludes."""
        path = _write_episode(tmp_path / "run_a", 1, ledger={}, insights=[_insight("A")])
        assert _mod._episode_from_metadata(path).insights_surfaceable == 1

    def test_explicit_false_is_unsurfaceable(self, tmp_path: Path) -> None:
        path = _write_episode(
            tmp_path / "run_a",
            1,
            ledger={},
            insights=[_insight("SPEAKER_00", surfaceable=False)],
        )
        record = _mod._episode_from_metadata(path)
        assert record.insights_total == 1
        assert record.insights_surfaceable == 0

    def test_raw_diarization_labels_are_not_counted_as_named_voices(self, tmp_path: Path) -> None:
        """SPEAKER_00 is the machine label — treating it as a name would hide the whole defect."""
        path = _write_episode(
            tmp_path / "run_a",
            1,
            ledger={},
            insights=[_insight("SPEAKER_00", False), _insight("SPEAKER_01", False)],
        )
        record = _mod._episode_from_metadata(path)
        assert record.voices_total == 2
        assert record.voices_named == 0

    def test_missing_gi_becomes_a_note_not_a_zero(self, tmp_path: Path) -> None:
        path = _write_episode(tmp_path / "run_a", 1, ledger={}, write_gi=False)
        record = _mod._episode_from_metadata(path)
        assert "gi_unreadable" in record.notes
        assert record.insights_total is None  # unknown, NOT zero

    def test_unreadable_metadata_becomes_a_note(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.metadata.json"
        bad.write_text("{not json", encoding="utf-8")
        assert "metadata_unreadable" in _mod._episode_from_metadata(bad).notes

    def test_missing_ledger_is_recorded_as_pre_1647(self, tmp_path: Path) -> None:
        path = _write_episode(tmp_path / "run_a", 1, ledger=None, insights=[])
        record = _mod._episode_from_metadata(path)
        assert any("no_stage_ledger" in note for note in record.notes)


class TestCollectAndMain:
    def test_collects_across_runs_and_scopes_to_one(self, tmp_path: Path) -> None:
        _write_episode(tmp_path / "run_a", 1, ledger={}, insights=[])
        _write_episode(tmp_path / "run_b", 2, ledger={}, insights=[])
        assert len(_mod.collect(tmp_path, None)) == 2
        assert len(_mod.collect(tmp_path, "run_a")) == 1

    def test_main_reports_and_writes_json(self, tmp_path: Path, capsys) -> None:
        _write_episode(
            tmp_path / "run_a",
            1,
            ledger={"speaker_detection": {"outcome": "skipped", "reason": "media_over_size_limit"}},
            insights=[_insight("SPEAKER_00", False)],
        )
        out = tmp_path / "report.json"
        rc = _mod.main(["--corpus", str(tmp_path), "--json", str(out)])
        assert rc == 0
        printed = capsys.readouterr().out
        assert "media_over_size_limit" in printed
        assert "NOT MEASURED" in printed
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["attribution"]["episodes_fully_zeroed"] == 1

    def test_main_exits_nonzero_when_there_is_nothing_to_report(self, tmp_path: Path) -> None:
        """Empty must not print a clean report — that would be the friendliest possible lie."""
        assert _mod.main(["--corpus", str(tmp_path)]) == 1

    def test_main_rejects_a_bad_corpus_path(self, tmp_path: Path) -> None:
        assert _mod.main(["--corpus", str(tmp_path / "nope")]) == 2
