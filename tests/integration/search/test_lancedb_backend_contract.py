"""The real backend must honour the contract the fake is held to.

Half of a pair with `tests/unit/search/test_fake_backend_contract.py`. The checks are written
strictly against the `SearchBackend` port, so the only difference between the two files is which
implementation is constructed and how rows are made answerable — which is exactly what turns a
disagreement between fake and reality into a failing test instead of a latent bug. It found two on
the day it was written; see the contract module's docstring.

Needs a real LanceDB directory, so it lives here: storage on a real temp filesystem, which this
layer specifies as real. No embedding model is involved — vectors are literals.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from tests.search_backend_contract import (  # noqa: E402
    CONTRACT_CHECKS,
    CONTRACT_IDS,
    DIM,
)


@pytest.mark.parametrize("check", [c for _, c in CONTRACT_CHECKS], ids=list(CONTRACT_IDS))
def test_lancedb_honours_the_contract(
    check: Callable[[Callable[[], Any], Callable[[Any], None]], None], tmp_path: Any
) -> None:
    def make_backend() -> LanceDBBackend:
        return LanceDBBackend(str(tmp_path / "lance_index"), embed_dim=DIM)

    def prepare(backend: LanceDBBackend) -> None:
        """Build the INVERTED index, without which LanceDB answers no full-text query.

        `create_indices()` is what the production indexer runs at the end of a build, so this is
        the contract's documented "make the rows answerable" step, not a test fixup.
        """
        backend.create_indices()

    check(make_backend, prepare)
