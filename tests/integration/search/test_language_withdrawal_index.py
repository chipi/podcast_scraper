"""Withdrawing a language removes its INDEX rows and nothing else (S3.2).

MOVED OUT OF `tests/unit/podcast_scraper/test_language_withdrawal.py` on 2026-10-03. The unit file
keeps everything that can be asserted without a backend — the plan, its formatting, and the
source-tree check that no read path consults `is_language_enabled`. These three tests build a REAL
LanceDB index and count rows in it, which a unit test may not do: the Unit Testing Guide requires
unit tests to run with no ML packages installed, and the policy checker's rule U1 forbids
`pytest.importorskip()` in `tests/unit/`. Mocking would not preserve them either — "the Italian row
survives and the English one does too" is a statement about a real table.

WHY THE ASSERTIONS ARE SHAPED AROUND REOPENING THE INDEX. `apply_withdrawal` returns a count, and a
count is what a buggy implementation can also return. Each test reopens the index from disk
afterwards, so what is asserted is the state an operator would find, not the function's own report.

THE DECISION THESE PIN (2026-10-01): `enabled` in `config/languages.yaml` is an INGEST gate and
only that. Withdrawal is a separate, deliberate command with a dry run, because a one-line YAML
edit must not be able to cause a content outage with no confirmation step. Everything the index is
derived FROM stays on disk, which is what makes an ordinary reindex the undo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.language_withdrawal import apply_withdrawal  # noqa: E402


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
    """Five episodes across three languages, so withdrawing one must leave the other two."""
    root = tmp_path / "corpus"
    _episode(root, "p01", "ep-en-1", "en")
    _episode(root, "p01", "ep-en-2", "en")
    _episode(root, "p10", "ep-es-1", "es")
    _episode(root, "p10", "ep-es-2", "es")
    _episode(root, "p11", "ep-it-1", "it")
    return root


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
