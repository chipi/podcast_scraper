"""scripts/measure/verdict_lines_check.py: a verdict row's deciding line is checked against the
served segment; a changed or missing line fails the check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.reviewer_verdicts import line_sha256

pytestmark = pytest.mark.unit

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "measure" / "verdict_lines_check.py"
_spec = importlib.util.spec_from_file_location("verdict_lines_check", _PATH)
assert _spec and _spec.loader
vlc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlc)

META = "feeds/f1/run_x/metadata/ep.metadata.json"
HEADER = "episode_id\tmeta\tvoice\tname\tverdict\tevidence\tbasis\tsegment_idx\tline_sha256"


def _corpus(root: Path, texts: list) -> Path:
    run = root / "feeds" / "f1" / "run_x"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    (run / "metadata" / "ep.metadata.json").write_text(
        json.dumps({"content": {"transcript_file_path": "transcripts/ep.txt"}}), encoding="utf-8"
    )
    (run / "transcripts" / "ep.segments.json").write_text(
        json.dumps([{"text": t, "speaker": "SPEAKER_00"} for t in texts]), encoding="utf-8"
    )
    return root


def _verdicts(d: Path, rows: list) -> Path:
    d.mkdir()
    lines = ["# migration: 9999_x", HEADER]
    for idx, sha in rows:
        lines.append(f"e1\t{META}\tSPEAKER_00\tN\tCORRECT\tx\tself_intro\t{idx}\t{sha}")
    (d / "9999_x.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_a_matching_line_passes(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "c", ["Hi, I'm the host.", "Welcome."])
    vd = _verdicts(tmp_path / "v", [(0, line_sha256("Hi, I'm the host."))])
    assert vlc.main(["--corpus", str(corpus), "--verdicts-dir", str(vd)]) == 0


def test_a_changed_line_and_a_missing_index_fail(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "c", ["Hi, I'm someone else now.", "Welcome."])
    vd = _verdicts(
        tmp_path / "v", [(0, line_sha256("Hi, I'm the host.")), (7, line_sha256("Welcome."))]
    )
    report = vlc.check(corpus, vd / "9999_x.tsv")
    assert report["checked"] == 2 and report["match"] == 0
    assert [k["segment_idx"] for k in report["mismatch"]] == [0]
    assert [k["segment_idx"] for k in report["missing"]] == [7]
    assert vlc.main(["--corpus", str(corpus), "--verdicts-dir", str(vd)]) == 1


def test_a_migration_with_no_verdicts_file_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "c", ["Welcome."])
    vd = _verdicts(tmp_path / "v", [(0, line_sha256("Welcome."))])
    assert (
        vlc.main(["--corpus", str(corpus), "--verdicts-dir", str(vd), "--migration", "typo"]) == 1
    )


def test_no_row_checked_fails(tmp_path: Path) -> None:
    """A file whose rows carry no deciding line proves nothing: zero checked is not a pass."""
    corpus = _corpus(tmp_path / "c", ["Welcome."])
    vd = _verdicts(tmp_path / "v", [("", "")])
    assert vlc.check(corpus, vd / "9999_x.tsv")["checked"] == 0
    assert vlc.main(["--corpus", str(corpus), "--verdicts-dir", str(vd)]) == 1


@pytest.mark.parametrize(("idx", "sha"), [("x", line_sha256("Welcome.")), (0, "abc123"), (0, "")])
def test_a_malformed_deciding_line_is_counted_bad(tmp_path: Path, idx: object, sha: str) -> None:
    corpus = _corpus(tmp_path / "c", ["Welcome."])
    vd = _verdicts(tmp_path / "v", [(0, line_sha256("Welcome.")), (idx, sha)])
    report = vlc.check(corpus, vd / "9999_x.tsv")
    assert report["match"] == 1 and len(report["malformed"]) == 1
    assert vlc.main(["--corpus", str(corpus), "--verdicts-dir", str(vd)]) == 1
