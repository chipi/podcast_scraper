"""The audit must find the damage, classify it, and stay silent on a clean corpus (#2082)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit.transcript_pairing_audit import audit, main


def _episode(
    run: Path,
    idx: int,
    title: str,
    guid: str,
    *,
    points_at: str | None = None,
    speakers=(),
    transcript_lines=("Jane Doe: hello there",),
    segments=True,
) -> None:
    stem = f"{idx:04d} - {title}"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    (run / "transcripts" / f"{stem}.txt").write_text("\n".join(transcript_lines), encoding="utf-8")
    if segments:
        (run / "transcripts" / f"{stem}.segments.json").write_text("[]", encoding="utf-8")
    (run / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"guid": guid, "title": title},
                "feed": {"title": "A Show"},
                "content": {
                    "transcript_file_path": points_at or f"transcripts/{stem}.txt",
                    "speakers": [{"name": n, "role": "guest"} for n in speakers],
                },
            }
        ),
        encoding="utf-8",
    )


class TestACleanCorpus:
    def test_no_findings_and_exit_zero(self, tmp_path: Path, capsys) -> None:
        run = tmp_path / "feeds" / "rss_x" / "run_a_20260101-000000"
        _episode(run, 1, "First", "g1")
        _episode(run, 2, "Second", "g2")
        findings, total = audit(tmp_path)
        assert findings == []
        assert total == 2
        assert main(["--corpus-dir", str(tmp_path)]) == 0


class TestItFindsTheMisattribution:
    def test_a_roster_from_the_wrong_transcript_is_confirmed(self, tmp_path: Path) -> None:
        run = tmp_path / "feeds" / "rss_x" / "run_a_20260101-000000"
        # "Second" points at First's transcript AND carries First's speaker.
        _episode(run, 1, "First", "g1", transcript_lines=("Grady Booch: hello",))
        _episode(
            run,
            2,
            "Second",
            "g2",
            points_at="transcripts/0001 - First.txt",
            speakers=("Grady Booch",),
            transcript_lines=("Kent Beck: hello",),
        )
        findings, _ = audit(tmp_path)
        assert len(findings) == 1
        f = findings[0]
        assert f["verdict"] == "roster_matches_wrong"
        assert f["episode"] == "Second"
        assert f["speakers"] == ["Grady Booch"]

    def test_exit_is_one_so_it_can_gate_a_deploy(self, tmp_path: Path) -> None:
        run = tmp_path / "feeds" / "rss_x" / "run_a_20260101-000000"
        _episode(run, 1, "First", "g1")
        _episode(run, 2, "Second", "g2", points_at="transcripts/0001 - First.txt")
        assert main(["--corpus-dir", str(tmp_path)]) == 1


class TestTheRepairRoutes:
    def test_an_episode_keeping_its_own_transcript_is_repairable(self, tmp_path: Path) -> None:
        run = tmp_path / "feeds" / "rss_x" / "run_a_20260101-000000"
        _episode(run, 1, "First", "g1")
        _episode(run, 2, "Second", "g2", points_at="transcripts/0001 - First.txt")
        findings, _ = audit(tmp_path)
        assert findings[0]["repairable"] is True

    def test_without_its_own_transcript_it_needs_a_reingest(self, tmp_path: Path) -> None:
        # relabel_only OVERWRITES the file it picks, so the victim's own transcript is often gone.
        run = tmp_path / "feeds" / "rss_x" / "run_a_20260101-000000"
        _episode(run, 1, "First", "g1")
        _episode(run, 2, "Second", "g2", points_at="transcripts/0001 - First.txt")
        (run / "transcripts" / "0002 - Second.txt").unlink()
        findings, _ = audit(tmp_path)
        assert findings[0]["repairable"] is False

    def test_the_transcript_is_found_in_another_run_dir(self, tmp_path: Path) -> None:
        # The episode's own transcript may live in a different run than its metadata.
        feed = tmp_path / "feeds" / "rss_x"
        _episode(
            feed / "run_a_20260101-000000",
            2,
            "Second",
            "g2",
            points_at="transcripts/0001 - Elsewhere.txt",
        )
        _episode(feed / "run_b_20260201-000000", 2, "Second", "g2")
        findings, _ = audit(tmp_path)
        assert any(f["repairable"] for f in findings)


class TestUnclassifiedIsNotSafe:
    def test_a_roster_in_neither_transcript_is_inconclusive(self, tmp_path: Path) -> None:
        run = tmp_path / "feeds" / "rss_x" / "run_a_20260101-000000"
        _episode(run, 1, "First", "g1", transcript_lines=("Alice: hi",))
        _episode(
            run,
            2,
            "Second",
            "g2",
            points_at="transcripts/0001 - First.txt",
            speakers=("Someone Else",),
            transcript_lines=("Bob: hi",),
        )
        findings, _ = audit(tmp_path)
        assert findings[0]["verdict"] == "inconclusive"
