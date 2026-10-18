"""The step-8 verdicts reader: one rule for every list migration, every bad file refused by name."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[4]
NAMES = {("ep1", "SPEAKER_00"): "Ann Lee", ("ep2", "SPEAKER_01"): "Bo Chen"}
HEAD = (
    f"# migration: 0099_x\n# reviewed_commit: abc\n# list_sha256: {rv.list_sha256(NAMES)}\n"
    "# reviewer: r\n"
    "# date: 2026-10-18\n" + "\t".join(rv.COLUMNS) + "\n"
)


def _row(
    eid: str = "ep1", voice: str = "SPEAKER_00", name: str = "Ann Lee", verdict: str = "CORRECT"
):
    return (
        "\t".join((eid, f"feeds/f/{eid}.metadata.json", voice, name, verdict, "says 'hi'")) + "\n"
    )


def test_a_valid_file_reads_every_verdict_by_episode_and_voice() -> None:
    v = rv.parse(
        HEAD + _row() + _row("ep2", "SPEAKER_01", "Bo Chen", "WRONG_PERSON"), "0099_x", NAMES
    )
    a, b = v.get("ep1", "SPEAKER_00"), v.get("ep2", "SPEAKER_01")
    assert a is not None and a.verdict == "CORRECT"
    assert b is not None and b.verdict == "WRONG_PERSON"
    assert v.get("ep2", "SPEAKER_00") is None  # same episode, other voice: no verdict
    assert v.get("ep1", "SPEAKER_01") is None  # same voice id, other episode: no verdict
    assert len(v.sha256) == 64 and v.header["reviewer"] == "r"


def test_a_file_with_no_rows_is_valid_and_decides_nothing() -> None:
    assert rv.parse(HEAD, "0099_x", NAMES).rows == {}


@pytest.mark.parametrize(
    "text, why",
    [
        (HEAD + _row(verdict="WRONG"), "unknown verdict 'WRONG'"),
        (HEAD + _row(verdict="WRONG PERSON"), "unknown verdict"),
        (HEAD.replace("# reviewer: r\n", ""), "header lacks reviewer"),
        (HEAD.replace("\tevidence", ""), "columns are"),
        (HEAD + _row() + _row(verdict="WRONG_PERSON"), "duplicate key"),
        (HEAD + _row() + _row(), "duplicate key"),
        (HEAD.replace("0099_x", "0031_other"), "file is for '0031_other'"),
        (HEAD + _row(name="Anne Lee"), "stale"),
        (HEAD + _row(eid="ep9"), "no entry of 0099_x"),
        ("# migration: 0099_x\n", "header lacks"),
        (HEAD.replace(rv.list_sha256(NAMES), "0" * 64), "list_sha256 is not the current list"),
    ],
)
def test_every_unusable_file_is_refused_with_its_reason(text: str, why: str) -> None:
    with pytest.raises(rv.VerdictError, match=why):
        rv.parse(text, "0099_x", NAMES)


def test_a_missing_file_is_refused_and_differs_from_an_empty_one(tmp_path: Path) -> None:
    with pytest.raises(rv.VerdictError, match="no verdicts file 0099_x.tsv"):
        rv.load("0099_x", NAMES, tmp_path)
    (tmp_path / "0099_x.tsv").write_text(HEAD, encoding="utf-8")
    assert rv.load("0099_x", NAMES, tmp_path).rows == {}


def test_the_list_fingerprint_moves_with_the_list() -> None:
    a = rv.list_sha256({("e", "v"): ("A", "guest")})
    assert a == rv.list_sha256({("e", "v"): ("A", "guest")})
    assert a != rv.list_sha256({("e", "v"): ("B", "guest")})


def test_the_verdicts_ship_in_the_wheel() -> None:
    """``pip install .`` (the images) keeps only declared package data: a verdicts file missing from
    the wheel makes the migration refuse on prod (the known_models.yaml lesson in pyproject)."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    globs = data["tool"]["setuptools"]["package-data"]["podcast_scraper"]
    assert "upgrade/verdicts/*.tsv" in globs
    assert (REPO / "src/podcast_scraper/upgrade/verdicts/0031_named_voices_shown.tsv").is_file()


def test_the_reader_names_no_entry_of_any_migration() -> None:
    """The rule is generic: no episode id, voice id or person name in the reader's code."""
    code = (REPO / "src/podcast_scraper/upgrade/reviewer_verdicts.py").read_text(encoding="utf-8")
    for needle in ("5555e937", "SPEAKER_0", "Justin Norman", "Espinoza", "Kamen"):
        assert needle not in code


def test_the_conversion_script_maps_against_the_migrations_own_list(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rvt", REPO / "scripts/measure/review_verdicts_to_tsv.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Fake:
        id = "0099_x"

        def review_names(self):
            return dict(NAMES)

    monkeypatch.setattr(mod, "get_migrations", lambda: [Fake()])
    assert mod._entries("0099_x") == NAMES
    with pytest.raises(SystemExit, match="not registered"):
        mod._entries("0098_y")
