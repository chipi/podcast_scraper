"""ADR/RFC/PRD numbers must be unique — and the only check that sees a collision compares branches.

WHY THIS IS NOT A SAME-BRANCH CHECK. Two branches that each add ADR-156 under a DIFFERENT
filename merge with **zero conflict markers**: git sees two unrelated new files. The result is
two ADR-156s, each cited by different code, and every same-branch check passes on both sides —
both files exist, both are real prose, every link resolves.

Measured 2026-09-30: `feat/multilingual-ingest` carried
`ADR-156-translation-model-and-serving.md` while `origin/main` had
`ADR-156-topics-come-only-from-the-extractor.md`. Nothing in the repo could see it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "tools" / "check_doc_structure.py"


def _load() -> Any:
    """Import the script by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("check_doc_structure_under_test", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load()


class TestTheNumberPattern:
    def test_it_reads_family_and_number(self, mod: Any) -> None:
        m = mod.NUMBERED_DOC.match("ADR-156-translation-model-and-serving.md")
        assert m is not None
        assert m.group(1) == "ADR-156"
        assert m.group(2) == "156"

    @pytest.mark.parametrize("name", ["ADR-157-x.md", "RFC-124-y.md", "PRD-042-z.md"])
    def test_all_three_families(self, mod: Any, name: str) -> None:
        assert mod.NUMBERED_DOC.match(name) is not None

    @pytest.mark.parametrize("name", ["README.md", "index.md", "ADR-README.md", "notes-156.md"])
    def test_it_ignores_unnumbered_docs(self, mod: Any, name: str) -> None:
        assert mod.NUMBERED_DOC.match(name) is None


class TestTheRealRepoIsClean:
    def test_this_branch_collides_with_nothing_on_the_merge_target(self, mod: Any) -> None:
        """The assertion that would have blocked the PR. Skips rather than passes when the merge
        target is unavailable — an unrunnable check must never read as a green one."""
        if mod._numbered_docs_on(mod.MERGE_TARGET) is None:
            pytest.skip(f"{mod.MERGE_TARGET} unavailable; run `git fetch origin main`")
        assert mod.check_doc_numbers() == []

    def test_the_translation_adr_is_157_not_156(self, mod: Any) -> None:
        """Pinned by name because the rename is the fix, and a revert is silent otherwise:
        `main` owns ADR-156 for the topics extractor."""
        adr = Path(__file__).resolve().parents[3] / "docs" / "adr"
        assert (adr / "ADR-157-translation-model-and-serving.md").is_file()
        assert not (adr / "ADR-156-translation-model-and-serving.md").exists()

    def test_no_number_is_used_twice_in_one_directory(self, mod: Any) -> None:
        for rel in mod.NUMBERED_DOC_DIRS:
            directory = Path(mod.REPO_ROOT) / rel
            if not directory.is_dir():
                continue
            seen: dict[str, str] = {}
            for path in sorted(directory.glob("*.md")):
                m = mod.NUMBERED_DOC.match(path.name)
                if not m:
                    continue
                key = m.group(1)
                assert key not in seen, f"{rel}: {key} used by both {seen[key]} and {path.name}"
                seen[key] = path.name


class TestAnUnavailableTargetIsNotAPass:
    def test_a_bogus_ref_returns_None_rather_than_an_empty_mapping(self, mod: Any) -> None:
        """An empty mapping would mean "no number is taken anywhere", turning an offline run or a
        shallow clone into a check that always passes."""
        assert mod._numbered_docs_on("refs/heads/definitely-not-a-real-ref-xyz") is None

    def test_the_merge_target_mapping_is_non_trivial_when_available(self, mod: Any) -> None:
        theirs = mod._numbered_docs_on(mod.MERGE_TARGET)
        if theirs is None:
            pytest.skip(f"{mod.MERGE_TARGET} unavailable")
        assert len(theirs) > 50, "a nearly-empty mapping means the ls-tree paths are wrong"
        assert ("ADR", "156") in theirs, "main's ADR-156 is the collision this check exists for"
