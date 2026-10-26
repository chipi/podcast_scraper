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


# The basis columns (0034's revision, operator 2026-10-26: keep the name unless the text shows
# someone else). Every refusal below fails without its check.
HEAD_B = HEAD.replace("\t".join(rv.COLUMNS), "\t".join(rv.COLUMNS_WITH_BASIS))
LINE = rv.line_sha256("Hi, I'm Ann Lee.")


def _row_b(verdict: str = "CORRECT", basis: str = "self_intro", idx: str = "12", line: str = LINE):
    return _row(verdict=verdict).rstrip("\n") + "\t" + "\t".join((basis, idx, line)) + "\n"


def test_a_file_with_the_basis_columns_reads_each_rows_basis_and_line() -> None:
    v = rv.parse(HEAD_B + _row_b(), "0099_x", NAMES, require_basis=True)
    row = v.get("ep1", "SPEAKER_00")
    assert row is not None
    assert (row.basis, row.segment_idx, row.line_sha256) == ("self_intro", 12, LINE)


def test_an_undecidable_keep_needs_no_deciding_line() -> None:
    v = rv.parse(HEAD_B + _row_b(basis="undecidable", idx="", line=""), "0099_x", NAMES)
    row = v.get("ep1", "SPEAKER_00")
    assert row is not None and row.verdict == "CORRECT" and row.segment_idx is None


@pytest.mark.parametrize(
    "row, why",
    [
        (_row_b(basis=""), "no basis"),
        (_row_b(basis="vibes"), "basis 'vibes' is not one CORRECT allows"),
        (_row_b(verdict="WRONG_PERSON", basis="undecidable"), "not one WRONG_PERSON allows"),
        (_row_b(verdict="WRONG_PERSON", basis="self_intro"), "not one WRONG_PERSON allows"),
        (_row_b(verdict="CORRECT", basis="another_person"), "not one CORRECT allows"),
        (_row_b(verdict="WRONG_PERSON", basis="another_person", idx=""), "needs a segment_idx"),
        (_row_b(verdict="WRONG_PERSON", basis="another_person", line=""), "needs a segment_idx"),
        (_row_b(idx="twelve"), "is not a segment index"),
        (_row_b(line="abc"), "is not a sha256"),
        (_row_b(basis="undecidable", line="abc"), "is not a sha256"),
        (_row_b(basis="undecidable", idx=""), "needs a segment_idx"),
    ],
)
def test_every_unusable_basis_is_refused_with_its_reason(row: str, why: str) -> None:
    with pytest.raises(rv.VerdictError, match=why):
        rv.parse(HEAD_B + row, "0099_x", NAMES)


def test_a_migration_that_requires_a_basis_refuses_a_file_without_the_columns() -> None:
    with pytest.raises(rv.VerdictError, match="columns are"):
        rv.parse(HEAD + _row(), "0099_x", NAMES, require_basis=True)
    # Files written before the columns still read for migrations that do not require them.
    assert rv.parse(HEAD + _row(), "0099_x", NAMES).get("ep1", "SPEAKER_00") is not None


def test_a_migration_may_supply_its_own_basis_set_for_its_load_only() -> None:
    row = _row_b(verdict="WRONG_PERSON", basis="undecidable", idx="", line="")
    with pytest.raises(rv.VerdictError, match="not one WRONG_PERSON allows"):
        rv.parse(HEAD_B + row, "0099_x", NAMES)
    own = {**rv.BASES, "WRONG_PERSON": rv.BASES["WRONG_PERSON"] + (rv.UNDECIDABLE,)}
    v = rv.parse(HEAD_B + row, "0099_x", NAMES, bases=own)
    got = v.get("ep1", "SPEAKER_00")
    assert got is not None and got.basis == "undecidable"
    assert "undecidable" not in rv.BASES["WRONG_PERSON"]  # the default is untouched
    with pytest.raises(rv.VerdictError, match="not one CORRECT allows"):
        rv.parse(HEAD_B + _row_b(basis="vibes"), "0099_x", NAMES, bases=own)


@pytest.mark.parametrize("basis", ["self_intro", "host_read", "gold"])
def test_only_an_undecidable_keep_may_leave_its_deciding_line_empty(basis: str) -> None:
    # Both columns empty is the undecidable shortcut; any other basis states a deciding line.
    with pytest.raises(rv.VerdictError, match="needs a segment_idx and a line_sha256"):
        rv.parse(HEAD_B + _row_b(basis=basis, idx="", line=""), "0099_x", NAMES)
    with pytest.raises(rv.VerdictError, match="needs a segment_idx and a line_sha256"):
        rv.parse(
            HEAD_B + _row_b(verdict="WRONG_PERSON", basis="another_person", idx="", line=""),
            "0099_x",
            NAMES,
        )
