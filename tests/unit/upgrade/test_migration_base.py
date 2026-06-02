"""Coverage for the base Migration default verify/plan (#862)."""

from __future__ import annotations

import pytest

from podcast_scraper.upgrade.migration import Migration, MigrationContext, MigrationResult

pytestmark = pytest.mark.unit


class _Bare(Migration):
    id = "0000_bare"
    to_version = "2.7.0"
    description = "bare step"

    def apply(self, ctx):
        return MigrationResult(self.id, applied=True)


def test_base_verify_and_plan_defaults(tmp_path):
    m = _Bare()
    ctx = MigrationContext(corpus_root=tmp_path)
    ok, msg = m.verify(ctx)
    assert ok and "no verification" in msg  # default verify
    assert m.plan(ctx) == "bare step"  # default plan = description
