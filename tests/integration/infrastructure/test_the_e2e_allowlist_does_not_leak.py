"""The E2E handler's allowlist is CLASS-level state, and it must not outlive one test.

WHY THIS EXISTS. `E2EHTTPRequestHandler._allowed_podcasts` and `_use_fast_fixtures` are class
attributes shared by every handler instance in the process. `tests/e2e/conftest.py`'s autouse
fixture sets them per test mode and, until this was fixed, deliberately did NOT restore the
allowlist — the comment read "don't reset allowed_podcasts - next test will set them".

That assumption holds only for tests the autouse fixture applies to, which is `tests/e2e/` alone.
`tests/integration/infrastructure/` imports the `e2e_server` fixture directly and this conftest
does not apply, so those tests inherited whatever the last e2e test left behind. Measured: three
tests whose subject is "is the RSS feed served" passed in isolation (class default `None` = allow
everything) and returned **404** on `/feeds/podcast1/feed.xml` in any run where an e2e test went
first, because the default mode's allowlist does not contain `podcast1`.

An ordering-dependent 404 points at a broken server, not at a leaked fixture, which is what makes
this worth a test rather than a comment.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

try:
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler
except ImportError:  # pragma: no cover - mirrors this directory's conftest fallback
    E2EHTTPRequestHandler = None  # type: ignore[assignment,misc]


requires_handler = pytest.mark.skipif(
    E2EHTTPRequestHandler is None, reason="E2E handler unavailable"
)


@requires_handler
class TestTheStateIsReadableAndRestorable:
    """A fixture can only restore what it can read. `_use_fast_fixtures` had a setter and no
    getter, so the old teardown reset it to a hardcoded `True` — correct by luck, since `True`
    is also the class default, and wrong the moment a caller wanted otherwise."""

    def test_both_flags_have_a_getter(self) -> None:
        assert callable(E2EHTTPRequestHandler.get_allowed_podcasts)
        assert callable(E2EHTTPRequestHandler.get_use_fast_fixtures)

    def test_a_round_trip_returns_what_was_set(self) -> None:
        before_allowed = E2EHTTPRequestHandler.get_allowed_podcasts()
        before_fast = E2EHTTPRequestHandler.get_use_fast_fixtures()
        try:
            E2EHTTPRequestHandler.set_allowed_podcasts({"podcast_xyz"})
            E2EHTTPRequestHandler.set_use_fast_fixtures(not before_fast)

            assert E2EHTTPRequestHandler.get_allowed_podcasts() == {"podcast_xyz"}
            assert E2EHTTPRequestHandler.get_use_fast_fixtures() is (not before_fast)
        finally:
            E2EHTTPRequestHandler.set_allowed_podcasts(before_allowed)
            E2EHTTPRequestHandler.set_use_fast_fixtures(before_fast)

        assert E2EHTTPRequestHandler.get_allowed_podcasts() == before_allowed
        assert E2EHTTPRequestHandler.get_use_fast_fixtures() is before_fast


@requires_handler
class TestThisDirectoryIsNotLEFTRestricted:
    """The observable consequence, asserted where it bit.

    Not "the allowlist is None" — a legitimate run may narrow it — but "podcast1 is reachable",
    which is what the three tests in this directory actually need and what they silently lost.
    """

    def test_podcast1_is_serveable_here(self) -> None:
        allowed = E2EHTTPRequestHandler.get_allowed_podcasts()
        assert allowed is None or "podcast1" in allowed, (
            "an e2e test leaked its allowlist into tests/integration/; "
            f"podcast1 is not reachable (allowed={allowed!r}) and the RSS-feed tests in this "
            "directory will 404 for a reason that has nothing to do with the server"
        )

    def test_the_e2e_conftest_restores_rather_than_re_sets(self) -> None:
        """Read from the source, because the fixture cannot be invoked from outside its own
        directory and the failure mode is a teardown that silently does nothing."""
        from pathlib import Path

        conftest = (Path(__file__).resolve().parents[2] / "e2e" / "conftest.py").read_text(
            encoding="utf-8"
        )
        # Both restore calls, not the absence of the old comment: the snapshot comment above
        # the fixture QUOTES the old "don't reset allowed_podcasts" line as the record of what
        # it cost, so an absence check would fail on the explanation rather than the behaviour.
        # A revert removes these two lines, which is what this catches.
        assert "set_allowed_podcasts(_prior_allowed)" in conftest
        assert "set_use_fast_fixtures(_prior_fast)" in conftest
