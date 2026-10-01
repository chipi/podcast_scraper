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

from podcast_scraper.language_withdrawal import (
    apply_withdrawal,
    format_plan,
    plan_withdrawal,
)

pytestmark = pytest.mark.unit


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


class TestApplyingItRemovesTheIndexRowsAndNothingElse:
    def _index(self, corpus: Path) -> Any:
        """A real index holding one row per episode, in the tier its language routes to."""
        from podcast_scraper.search.backend import SegmentDocument
        from podcast_scraper.search.backends.lancedb_backend import (
            DEFAULT_EMBED_DIM,
            LanceDBBackend,
        )

        index_dir = corpus / "search" / "lance_index"
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        backend = LanceDBBackend(str(index_dir), embed_dim=DEFAULT_EMBED_DIM)
        backend.replace_segments(
            [
                SegmentDocument(
                    id=f"chunk:{ep}:0",
                    text="t",
                    show_id=feed,
                    episode_id=ep,
                    start_time=0.0,
                    end_time=1.0,
                    embedding=[0.1] * DEFAULT_EMBED_DIM,
                    language=lang,
                )
                for feed, ep, lang in (
                    ("p01", "ep-en-1", "en"),
                    ("p10", "ep-es-1", "es"),
                    ("p10", "ep-es-2", "es"),
                    ("p11", "ep-it-1", "it"),
                )
            ]
        )
        return backend

    @staticmethod
    def _rows(backend: Any, tier: str) -> int:
        table = backend._open_if_exists(tier)
        return 0 if table is None else int(table.count_rows())

    def test_it_removes_the_language_s_rows_from_the_non_english_tier(self, corpus: Path) -> None:
        backend = self._index(corpus)
        assert self._rows(backend, "segment_nonen") == 3  # 2 es + 1 it

        result = apply_withdrawal(corpus, "es")

        assert result.error is None
        assert result.episodes == ["ep-es-1", "ep-es-2"]
        assert result.rows_removed.get("segment_nonen") == 2
        # Reopened, because the assertion is about what is ON DISK after the call.
        from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

        after = LanceDBBackend(str(corpus / "search" / "lance_index"))
        assert self._rows(after, "segment_nonen") == 1  # the Italian row survives
        assert self._rows(after, "segment") == 1  # and so does the English one

    def test_the_CORPUS_is_left_alone_so_a_reindex_undoes_it(self, corpus: Path) -> None:
        """The reversal claim, asserted rather than asserted-in-prose: every artifact the index
        is derived FROM is still there, byte for byte."""
        self._index(corpus)
        metas = sorted(corpus.glob("feeds/**/metadata/*.metadata.json"))
        before = {p: p.read_bytes() for p in metas}

        apply_withdrawal(corpus, "es")

        assert {
            p: p.read_bytes() for p in sorted(corpus.glob("feeds/**/metadata/*.json"))
        } == before

    def test_running_it_twice_removes_nothing_the_second_time(self, corpus: Path) -> None:
        self._index(corpus)
        first = apply_withdrawal(corpus, "es")
        second = apply_withdrawal(corpus, "es")
        assert first.total_rows == 2
        assert second.total_rows == 0
        assert second.error is None

    def test_no_index_on_disk_is_not_an_error(self, corpus: Path) -> None:
        """A corpus that was never indexed has nothing to withdraw from, which is a no-op rather
        than a failure — the operator asked for an outcome that is already true."""
        result = apply_withdrawal(corpus, "es")
        assert result.error is None
        assert result.total_rows == 0
