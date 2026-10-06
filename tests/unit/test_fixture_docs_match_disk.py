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


def _feed_language(show: str) -> str:
    """The show's language, from its corpus RSS feed — the pipeline's own source of truth.

    Read from the feed rather than inferred from the show id, because "p10 and up is non-English"
    is a coincidence of the order languages were added, not a rule.
    """
    feed = _FIXTURES / "rss" / f"{show}_corpus.xml"
    if not feed.is_file():
        return ""
    m = re.search(r"<language>([^<]+)</language>", feed.read_text(encoding="utf-8"))
    return m.group(1).split("-")[0].strip().lower() if m else ""


def _built_episode_labels(corpus: Path) -> set[str]:
    """``{pNN_eNN}`` actually present in the built corpus.

    Derived from the TRANSCRIPT filenames rather than the metadata ids, because the metadata
    carries content-hash episode ids (``ep-7072d2e8...``) that cannot be compared to disk labels.
    """
    return {
        p.stem for p in corpus.rglob("transcripts/*.txt") if re.fullmatch(r"p\d{2}_e\d{2}", p.stem)
    }


def _documented_counts() -> list[tuple[int, str]]:
    """The bolded counts from the spec's reconciliation table, in order.

    A LIST, not a dict keyed by the count, because two rows may legitimately share a number: on
    2026-10-03 the built corpus caught up with disk and both rows became 55. Keyed by count, the
    second row overwrote the first and the "real episodes" lookup below raised `StopIteration` —
    the table was right and the reader was wrong.
    """
    body = _SPEC.read_text(encoding="utf-8")
    start = body.index("Four episode counts, all correct")
    end = body.index("The special episodes")
    rows = re.findall(r"^\|\s*\*\*(\d+)\*\*\s*\|\s*([^|]+)\|", body[start:end], re.M)
    return [(int(n), desc.strip()) for n, desc in rows]


def test_spec_reconciles_exactly_three_counts() -> None:
    """51 files / 45 episodes / 38 generated.

    There used to be a fourth, 36, which was the built corpus under a default
    ``--max-episodes-per-feed 4``. Removing that default collapsed it into 40, so a
    table that still lists four counts is describing a flag that no longer exists.

    46/40 became 47/41 on 2026-09-30 when `p10_e01` landed — the Spanish counterpart of `p01`.
    The **38 did not move**: that row counts what `build_v3_fixtures.py` generates, and p10's
    transcript is hand-written like `p06_e05` and `p06_e06`, so it subtracts from 41 rather than
    adding to 38.

    47/41 became 51/45 on 2026-10-01 with `p11`..`p14` (it, fr, de, pt) — four more hand-written
    counterparts of p01, so the 38 does not move for exactly the same reason.

    51/45 became 61/55 on 2026-10-03 when the ten non-English `e02`/`e03` episodes got their
    translations. That ALSO collapsed a distinction: the built app-validation corpus had been 45
    while disk was 55, because the ten had no captured English render and
    `_drop_untranslated_non_english` excluded them (D-44). The capture run closed the gap, so the
    built-corpus row and the real-episode row are both 55 and the table reconciles THREE distinct
    numbers, not four — the same kind of collapse as the 36 that disappeared into 40.
    """
    counts = _documented_counts()
    distinct = sorted({n for n, _ in counts})
    assert len(counts) == 4, (
        f"the spec's table should still have four ROWS (61 / 55 disk / 55 built / 38 generated); "
        f"it has {len(counts)}"
    )
    assert distinct == [38, 55, 61], (
        "the spec's count table changed shape; it should reconcile 61/55/55/38 "
        f"and it now lists {distinct}"
    )


def test_documented_file_count_matches_disk() -> None:
    """Read the number OUT of the doc, so editing the doc cannot satisfy the test."""
    counts = _documented_counts()
    files = len(list(_V3.glob("*.txt")))
    documented = next(n for n, desc in counts if ".txt" in desc)
    assert (
        files == documented
    ), f"transcripts/v3 holds {files} .txt files; FIXTURES_SPEC.md says {documented}"
    assert f"{files} files" in _README.read_text(
        "utf-8"
    ), f"README.md's '46 files - these 6 = 40 episodes' line disagrees with disk ({files})"


def test_documented_episode_count_matches_disk() -> None:
    counts = _documented_counts()
    episodes = len(_canonical_episodes())
    documented = next(n for n, desc in counts if "real episodes" in desc)
    assert (
        episodes == documented
    ), f"disk has {episodes} canonical pNN_eNN episodes; FIXTURES_SPEC.md says {documented}"
    # The SHOW count is derived too, not hardcoded. It was a literal `9` and went stale the
    # moment `p10` landed — which is the same failure this whole file exists to catch, one level
    # up: a number written into a test instead of read from disk.
    shows = len({e.split("_", 1)[0] for e in _canonical_episodes()})
    assert f"**{episodes} episodes across {shows} shows.**" in _README.read_text(
        "utf-8"
    ), f"README.md's headline disagrees with disk ({episodes} episodes across {shows} shows)"


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
    assert m, "app-validation-corpus/README.md no longer states its episode count"
    assert built == int(
        m.group(1)
    ), f"the committed corpus holds {built} episodes; its README says {m.group(1)}"
    # THE BUILT CORPUS IS SMALLER THAN DISK AGAIN, DELIBERATELY — and the point of this
    # assertion is that the gap is exactly the explainable set and nothing else.
    #
    # It used to assert `built == on disk`, true once the per-feed cap lost its default. Since
    # 2026-10-02 the ten non-English `e02`/`e03` episodes are excluded by
    # `build_app_validation_corpus.py::_drop_untranslated_non_english`: D-44 makes the canonical
    # body the ANALYSIS language, the English body is replayed from a captured translation run,
    # and no capture exists for them — so admitting them would put source-language text at the
    # path every generic reader treats as English.
    #
    # Asserting equality against that EXPECTED set rather than relaxing to `<=` is what keeps the
    # original protection: an episode dropped for any OTHER reason still fails here, which is how
    # the silent 36 happened.
    on_disk = set(_canonical_episodes())
    captures = _FIXTURES / "pipeline-renders" / "v3"
    captured = {p.stem for p in captures.glob("*.json")} if captures.is_dir() else set()
    corpus_feeds = {p.name.split("_")[0] for p in (_FIXTURES / "rss").glob("p*_corpus.xml")}
    non_english = {
        e
        for e in on_disk
        if e.split("_")[0] in corpus_feeds and _feed_language(e.split("_")[0]) not in ("", "en")
    }
    expected = on_disk - (non_english - captured)
    missing = expected - _built_episode_labels(corpus)
    extra = _built_episode_labels(corpus) - expected
    assert not missing and not extra, (
        f"the built corpus is not the expected set. Missing: {sorted(missing)}. "
        f"Unexpected: {sorted(extra)}. Expected = every canonical episode on disk minus the "
        "non-English ones with no captured translation in pipeline-renders/v3."
    )


def test_readme_per_show_table_matches_disk() -> None:
    """The per-show episode ranges in README.md, row by row."""
    rows = re.findall(
        r"^\|\s*`(p\d{2})`\s*\|[^|]*\|\s*e(\d{2})[-–]e(\d{2})\s*\|",
        _README.read_text("utf-8"),
        re.M,
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
