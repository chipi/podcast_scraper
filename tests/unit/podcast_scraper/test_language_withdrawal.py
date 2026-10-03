"""S3.2: what `enabled: false` means once episodes exist, and how to withdraw one on purpose.

The arc left this undefined, and "undefined" here did not mean broken — it meant a one-line YAML
edit did nothing to published content, silently. These tests pin BOTH halves of the decision taken
2026-10-01: `enabled` stays an INGEST gate, and withdrawal is a separate deliberate command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.language_withdrawal import format_plan, plan_withdrawal

pytestmark = pytest.mark.unit

# MOVED 2026-10-03 — TestApplyingItRemovesTheIndexRowsAndNothingElse (3 tests) now lives in
# tests/integration/search/test_language_withdrawal_index.py.
#
# Those tests build a real LanceDB index and count rows in it, which a unit test may not: the Unit
# Testing Guide requires unit tests to run with no ML packages installed, and the policy checker's
# rule U1 forbids the import-or-skip helper here. Mocking would not preserve them — "the Italian
# row survives and the English one does too" is a statement about a real table.
#
# What stays here needs no backend: the plan, its formatting, and the source-tree check that no
# read path consults `is_language_enabled`.


def _episode(root: Path, feed: str, ep: str, language: str) -> None:
    meta_dir = root / "feeds" / feed / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "feed": {"feed_id": feed, "language": language, "language_source": "rss"},
        "episode": {"episode_id": ep, "language": language, "language_source": "rss"},
        "content": {"transcript_file_path": f"transcripts/{ep}.txt"},
    }
    (meta_dir / f"{ep}.metadata.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _episode(root, "p01", "ep-en-1", "en")
    _episode(root, "p01", "ep-en-2", "en")
    _episode(root, "p10", "ep-es-1", "es")
    _episode(root, "p10", "ep-es-2", "es")
    _episode(root, "p11", "ep-it-1", "it")
    return root


class TestDisablingALanguageDoesNotTouchPublishedEpisodes:
    """The decision, asserted where it can regress: `enabled` is an INGEST gate and only that.

    If someone later adds an `is_language_enabled` check to a read path, this is the test that
    should fail — the read side consulting the flag is precisely what was rejected, because a
    YAML edit would then become a content outage with no confirmation step and no log line.
    """

    def test_no_read_path_consults_the_enabled_flag(self) -> None:
        """Enforced by inspection of the source tree, because the alternative is asserting the
        absence of a behaviour across every route and query path."""
        import subprocess

        repo = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            [
                "grep",
                "-rn",
                "is_language_enabled",
                str(repo / "src" / "podcast_scraper" / "server"),
                str(repo / "src" / "podcast_scraper" / "search"),
            ],
            capture_output=True,
            text=True,
        )
        hits = [ln for ln in out.stdout.splitlines() if ln.strip()]
        assert not hits, (
            "a read path now consults `is_language_enabled`, which makes a one-line "
            "`enabled: false` edit hide published episodes with no confirmation step. "
            "Withdrawal is `language_withdrawal.apply_withdrawal`, run on purpose:\n  "
            + "\n  ".join(hits)
        )

    def test_the_ingest_gate_still_refuses_a_disabled_language(self) -> None:
        """The other half: disabling must still stop NEW episodes, or the flag does nothing."""
        from podcast_scraper import config
        from podcast_scraper.languages import language_registry
        from podcast_scraper.workflow.episode_processor import (
            _unsupported_language_skip_reason,
        )

        disabled = sorted(
            c for c, e in language_registry().items() if not getattr(e, "enabled", False)
        )
        assert disabled, "no disabled language in the registry to exercise the gate with"
        # `rss`, not `rss_url`: the alias the Config field actually declares (mypy caught it).
        cfg = config.Config(rss="https://example.invalid/rss").model_copy(  # type: ignore[arg-type]
            update={"feed_declared_language": disabled[0]}
        )
        assert _unsupported_language_skip_reason(cfg) is not None


class TestThePlanIsReadOnlyAndSaysWhatItWouldDo:
    def test_it_finds_exactly_that_language_s_episodes(self, corpus: Path) -> None:
        plan = plan_withdrawal(corpus, "es")
        assert plan.episodes == ["ep-es-1", "ep-es-2"]
        assert plan.feeds == ["p10"]

    def test_a_regional_tag_still_matches_the_stored_language(self, corpus: Path) -> None:
        """`es-ES` must find the episodes stored as `es`. A withdrawal that reports zero because
        of a tag mismatch is the worst outcome here — it reads as "nothing to withdraw"."""
        assert plan_withdrawal(corpus, "es-ES").episodes == ["ep-es-1", "ep-es-2"]

    def test_english_is_refused(self, corpus: Path) -> None:
        """English is the default on every surface (D-38), so withdrawing it would empty the
        product. Refused here rather than handled."""
        plan = plan_withdrawal(corpus, "en")
        assert plan.error and "refusing to withdraw English" in plan.error
        assert "REFUSED" in format_plan(plan)

    def test_an_unusable_tag_is_refused_rather_than_matching_nothing(self, corpus: Path) -> None:
        plan = plan_withdrawal(corpus, "!!")
        assert plan.error and "not a usable language tag" in plan.error

    def test_a_tag_the_REGISTRY_never_declared_is_refused_too(self, corpus: Path) -> None:
        """`normalize_language_tag` strips the region and nothing more, so a typo survives it and
        would match zero episodes. Reporting "nothing to withdraw" for a typo is the worst outcome
        this tool has, because it reads as "there was nothing there"."""
        for typo in ("zzz", "nto", "not-a-language-!!"):
            plan = plan_withdrawal(corpus, typo)
            assert plan.error and "not declared in" in plan.error, f"{typo!r} was accepted"

    def test_a_language_with_no_episodes_says_so(self, corpus: Path) -> None:
        plan = plan_withdrawal(corpus, "de")
        assert plan.empty
        assert "nothing to withdraw" in format_plan(plan)

    def test_the_plan_writes_nothing(self, corpus: Path) -> None:
        before = sorted(p.relative_to(corpus) for p in corpus.rglob("*"))
        plan_withdrawal(corpus, "es")
        assert sorted(p.relative_to(corpus) for p in corpus.rglob("*")) == before

    def test_the_dry_run_names_the_reversal(self, corpus: Path) -> None:
        """An operator reading the plan must be told how to undo it before they commit."""
        text = format_plan(plan_withdrawal(corpus, "es"))
        assert "index-two-tier" in text
        assert "corpus is untouched" in text
