"""Importing the scorer must not touch the network (regression for a 95-second import).

``scorer.py`` used to call ``nltk.download("punkt")`` and ``nltk.download("punkt_tab")`` at MODULE
IMPORT, inside a ``try/except Exception: pass``. That swallowed the error but not the wait: every
import of ``podcast_scraper.evaluation`` made two blocking HTTP requests to a third-party server,
whether or not the caller ever computed BLEU.

How it surfaced: two tests in ``test_score_rejudge.py`` shell out to
``autoresearch/.../eval/score.py`` with a 180s subprocess timeout, and started failing. Measured,
the script took **1m35s at 1% CPU** to print an argument error it could have printed instantly —
the process sat in ``ssl.read`` under ``nltk.downloader`` while the server answered HTTP 429. After
the fix the same command takes **1.6s**, and this whole test directory went from 541s to 18s.

Worth stating plainly: ``pytest-socket`` did NOT catch this. The call happens in a SUBPROCESS,
outside the ban — so the repo's network guard is not the backstop it looks like for anything that
shells out. These tests are the backstop instead.

``nltk`` is imported directly, not via ``importorskip``: it is a ``[dev]`` dependency
(``pyproject.toml``), and U1 bans skipping in the unit tier.
"""

from __future__ import annotations

import importlib
import socket

import nltk
import pytest

from podcast_scraper.evaluation import scorer

pytestmark = [pytest.mark.unit]


def _raise_lookup(path: str) -> None:
    raise LookupError(path)


def test_importing_the_scorer_downloads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh import must not reach ``nltk.download`` at all.

    Re-imports the module with the downloader booby-trapped — the only way to observe import-time
    behaviour from inside a process that has already imported it once.
    """

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("nltk.download() was called at import time")

    monkeypatch.setattr(nltk, "download", explode)
    importlib.reload(scorer)


def test_the_check_has_not_run_merely_because_the_module_is_imported() -> None:
    """The laziness itself: nothing is ensured until something needs a tokenizer."""
    fresh = importlib.reload(scorer)
    assert fresh._nltk_data_checked is False


class TestEnsureTokenizerData:
    @pytest.fixture(autouse=True)
    def _no_leaked_socket_timeout(self):
        """Restore the process-global socket timeout around every test in this class.

        Not hygiene — correctness. Without it ``test_a_download_attempt_is_time_bounded`` was
        ORDER-DEPENDENT and proved nothing: an earlier test in the class had already leaked the
        15s timeout, so its ``before`` snapshot was 15, the sabotaged "never restore" version left
        it at 15, and the assertion passed. Caught by sabotage — removing the ``finally`` restore
        from the scorer left all 7 tests green.
        """
        previous = socket.getdefaulttimeout()
        try:
            yield
        finally:
            socket.setdefaulttimeout(previous)

    @staticmethod
    def _reset() -> None:
        scorer._nltk_data_checked = False

    def test_present_data_is_never_re_downloaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The common case — data already on disk means zero network requests.

        This is the half that makes the fix cheap rather than merely deferred: a developer who has
        run BLEU once never pays again, and a CI image with the corpora baked in never pays at all.
        """
        self._reset()
        monkeypatch.setattr(nltk.data, "find", lambda path: f"/fake/{path}")
        monkeypatch.setattr(
            nltk, "download", lambda *a, **k: pytest.fail("downloaded data that was already there")
        )
        scorer._ensure_nltk_tokenizer_data()

    def test_only_the_missing_resource_is_fetched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._reset()
        asked: list[str] = []

        def find(path: str) -> str:
            if "punkt_tab" in path:
                raise LookupError(path)
            return f"/fake/{path}"

        monkeypatch.setattr(nltk.data, "find", find)
        monkeypatch.setattr(nltk, "download", lambda r, **k: asked.append(r))
        scorer._ensure_nltk_tokenizer_data()
        assert asked == ["punkt_tab"], asked

    def test_a_download_attempt_is_time_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A third-party server that never answers must not hang the caller for ever.

        The original code had no bound at all, which is why a slow server became a 95-second
        import. Also asserts the timeout is RESTORED afterwards, so this cannot leak a global
        socket timeout into whatever runs next.
        """
        self._reset()
        seen: list[float | None] = []
        monkeypatch.setattr(nltk.data, "find", _raise_lookup)
        monkeypatch.setattr(
            nltk, "download", lambda r, **k: seen.append(socket.getdefaulttimeout())
        )

        # A SENTINEL, not whatever the process happens to hold. Snapshotting the ambient value made
        # this test pass against a scorer that never restored anything (see the fixture above).
        sentinel = 3.0
        socket.setdefaulttimeout(sentinel)
        scorer._ensure_nltk_tokenizer_data()

        assert seen, "no download was attempted, so the bound was never exercised"
        assert all(t is not None and 0 < t <= 60 for t in seen), seen
        assert all(t != sentinel for t in seen), "the download ran on the ambient timeout, not ours"
        assert (
            socket.getdefaulttimeout() == sentinel
        ), "the download's timeout leaked into the rest of the process"

    def test_a_failing_download_degrades_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline is not an error here — BLEU is optional, the pipeline around it is not."""
        self._reset()
        monkeypatch.setattr(nltk.data, "find", _raise_lookup)

        def boom(*_a: object, **_k: object) -> None:
            raise OSError("no route to host")

        monkeypatch.setattr(nltk, "download", boom)
        scorer._ensure_nltk_tokenizer_data()  # must not raise

    def test_it_gives_up_after_one_attempt_per_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise every BLEU call retries a dead server — the 95s wait, once per episode."""
        self._reset()
        calls: list[str] = []
        monkeypatch.setattr(nltk.data, "find", _raise_lookup)
        monkeypatch.setattr(nltk, "download", lambda r, **k: calls.append(r))

        scorer._ensure_nltk_tokenizer_data()
        first = len(calls)
        for _ in range(5):
            scorer._ensure_nltk_tokenizer_data()
        assert len(calls) == first, f"retried the download {len(calls) - first} extra times"
