"""The in-memory fake must honour the same contract the real backend is held to.

Half of a pair. The other half, `tests/integration/search/test_lancedb_backend_contract.py`, runs
the identical checks against `LanceDBBackend`. Running them here — with no `[search]` extra — is
what makes the fake trustworthy enough for the unit files that use a stand-in: without it, a unit
suite built on fakes is agreeing with itself rather than checking anything.

`prepare` is a no-op because the fake has no index concept. That asymmetry with the real runner is
the contract being explicit about a precondition, not a shortcut — see the contract module.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from tests._fake_search_backend import FakeSearchBackend
from tests.search_backend_contract import CONTRACT_CHECKS, CONTRACT_IDS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("check", [c for _, c in CONTRACT_CHECKS], ids=list(CONTRACT_IDS))
def test_the_fake_backend_honours_the_contract(
    check: Callable[[Callable[[], Any], Callable[[Any], None]], None],
) -> None:
    check(FakeSearchBackend, lambda _backend: None)
