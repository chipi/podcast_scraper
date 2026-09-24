"""The fixture docs' numbers must match the fixture tree.

Every defect found in the 2026-09-24 doc audit was a TRUE STATEMENT THAT STOPPED
BEING TRUE: "5 podcasts, each with 3 episodes" (it is 9, with 4-6), "23 episodes"
(36), "4 per show" (4/5/6), "p07 — 1 extra-long episode, 14,471 words" (that was
v1; v3's longest is 1,309). Each was accurate when written.

markdownlint, the spell checker and ``mkdocs build --strict`` pass on all of them,
every day, because none of them is a formatting problem. Prose cannot be linted
into agreement with a directory. So the counts are asserted here instead, against
the filesystem, and the docs become the thing that breaks when the corpus grows.

If you are here because this test failed: the corpus changed and a doc did not.
Update the doc. Do not relax the assertion — that turns the number back into prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_V3 = _FIXTURES / "transcripts" / "v3"
_SPEC = _FIXTURES / "FIXTURES_SPEC.md"
_README = _FIXTURES / "README.md"

#: Files in ``transcripts/v3/`` that look like episodes and are not: a 60-second cut
#: of p01_e01, and five multi-feed connectivity stubs.
_NOT_EPISODES = re.compile(r"(_fast$|_multi_e\d+$)")


def _canonical_episodes() -> list[str]:
    return sorted(
        p.stem
        for p in _V3.glob("*.txt")
        if re.fullmatch(r"p\d{2}_e\d{2}", p.stem) and not _NOT_EPISODES.search(p.stem)
    )


def _documented_counts() -> dict[int, str]:
    """The bolded counts from the spec's reconciliation table, in order."""
    body = _SPEC.read_text(encoding="utf-8")
    start = body.index("Four different episode counts, all correct")
    end = body.index("The special episodes")
    rows = re.findall(r"^\|\s*\*\*(\d+)\*\*\s*\|\s*([^|]+)\|", body[start:end], re.M)
    return {int(n): desc.strip() for n, desc in rows}


def test_spec_reconciles_exactly_four_counts() -> None:
    counts = _documented_counts()
    assert sorted(counts) == [36, 38, 40, 46], (
        "the spec's count table changed shape; it should reconcile 46/40/38/36 "
        f"and it now lists {sorted(counts)}"
    )


def test_documented_file_count_matches_disk() -> None:
    """Read the number OUT of the doc, so editing the doc cannot satisfy the test."""
    counts = _documented_counts()
    files = len(list(_V3.glob("*.txt")))
    documented = next(n for n, desc in counts.items() if ".txt" in desc)
    assert files == documented, (
        f"transcripts/v3 holds {files} .txt files; FIXTURES_SPEC.md says {documented}"
    )
    assert f"{files} files" in _README.read_text("utf-8"), (
        f"README.md's '46 files - these 6 = 40 episodes' line disagrees with disk ({files})"
    )


def test_documented_episode_count_matches_disk() -> None:
    counts = _documented_counts()
    episodes = len(_canonical_episodes())
    documented = next(n for n, desc in counts.items() if "real episodes" in desc)
    assert episodes == documented, (
        f"disk has {episodes} canonical pNN_eNN episodes; FIXTURES_SPEC.md says {documented}"
    )
    assert f"**{episodes} episodes across 9 shows.**" in _README.read_text("utf-8"), (
        f"README.md's headline episode count disagrees with disk ({episodes})"
    )


def test_generator_defines_the_documented_number_of_episodes() -> None:
    """The 38 is what ``build_v3_fixtures.py`` produces, counted from the script."""
    src = (Path(__file__).resolve().parents[2] / "scripts" / "build_v3_fixtures.py").read_text(
        encoding="utf-8"
    )
    total = 0
    for m in re.finditer(r"episodes=\[", src):
        i = m.end() - 1
        depth, j, commas = 0, i, 0
        while j < len(src):
            c = src[j]
            if c in "[(":
                depth += 1
            elif c in "])":
                depth -= 1
                if depth == 0:
                    break
            elif c == "," and depth == 1:
                commas += 1
            j += 1
        block = src[i:j]
        total += commas + (0 if block.rstrip().rstrip("]").rstrip().endswith(",") else 1)
    assert total == 38, f"the v3 generator now defines {total} episodes, not the documented 38"


def test_built_corpus_matches_what_its_readme_claims() -> None:
    corpus = _FIXTURES / "app-validation-corpus" / "v3"
    built = len(list(corpus.rglob("*.metadata.json")))
    readme = (corpus.parent / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*Why (\d+) and not (\d+):\*\*", readme)
    assert m, "app-validation-corpus/README.md no longer states the built-vs-on-disk counts"
    assert built == int(m.group(1)), (
        f"the committed corpus holds {built} episodes; its README says {m.group(1)}"
    )
    assert len(_canonical_episodes()) == int(m.group(2))


def test_readme_per_show_table_matches_disk() -> None:
    """The per-show episode ranges in README.md, row by row."""
    rows = re.findall(
        r"^\|\s*`(p\d{2})`\s*\|[^|]*\|\s*e(\d{2})[-–]e(\d{2})\s*\|", _README.read_text("utf-8"), re.M
    )
    assert len(rows) == 9, f"expected 9 show rows in README.md, found {len(rows)}"
    on_disk: dict[str, list[str]] = {}
    for ep in _canonical_episodes():
        show, num = ep.split("_")
        on_disk.setdefault(show, []).append(num)
    for show, first, last in rows:
        eps = sorted(on_disk.get(show, []))
        assert eps, f"README.md documents {show}, which has no episodes on disk"
        assert eps[0] == f"e{first}" and eps[-1] == f"e{last}", (
            f"README.md says {show} is e{first}-e{last}; disk says "
            f"{eps[0]}-{eps[-1]} ({len(eps)} episodes)"
        )


def test_every_episode_has_audio() -> None:
    """audio/README.md's file count is only meaningful if the pairing holds."""
    audio = _FIXTURES / "audio" / "v3"
    missing = [e for e in _canonical_episodes() if not (audio / f"{e}.mp3").is_file()]
    assert not missing, f"episodes with a transcript and no audio: {', '.join(missing)}"


def test_no_doc_claims_the_v1_long_episodes_are_current() -> None:
    """14,471 and 19,251 are v1 word counts. They may be cited as history, not as v3."""
    for doc in sorted(_FIXTURES.rglob("*.md")):
        text = doc.read_text(encoding="utf-8")
        for stale in ("14,471", "19,251"):
            if stale not in text:
                continue
            assert re.search(r"(?i)\bv1\b|HISTORY|historical", text), (
                f"{doc.relative_to(_FIXTURES)} cites {stale} words without saying it is v1. "
                "v3's longest transcript is 1,309 words."
            )
