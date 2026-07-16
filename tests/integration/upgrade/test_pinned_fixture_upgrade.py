"""Prod-deploy gap test: the framework can move a prod-shape corpus to HEAD (#1176, CI net C).

The pinned fixture at ``tests/fixtures/upgrade/corpus_at_last_prod_release/``
represents the on-disk state of the production corpus after the most recent
successful prod deploy. The migrations currently in the registry that are NOT
in ``config/last_deployed_prod_version.json``'s ``applied_migrations`` list are
the ones that will run on the next prod deploy — that is the ONLY gap that
matters.

This test:

1. Computes the pending set = current registry − prod marker.
2. Copies the fixture to a tmp corpus.
3. Runs ``podcast upgrade run --yes`` and asserts the pending set applied.
4. Asserts idempotency (a second run is a no-op, ledger unchanged).
5. Asserts nothing regressed for migrations already in the prod ledger.

When the pending set is EMPTY (main and prod are in sync on migrations), the
test still validates that the framework treats the fixture as up-to-date
without touching anything — that catches "someone added migration side-effects
that fire even when the ledger says applied" drift.

We do NOT test the historical gap between v2.6 and prod. That is history. If
we ever roll back prod to an older version, this fixture and marker get
updated together to reflect that new prod state.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path

import pytest

from podcast_scraper.search import two_tier_indexer
from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv, run_upgrade_cli
from podcast_scraper.upgrade.registry import get_migrations
from podcast_scraper.upgrade.state import FilesystemStateStore

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "upgrade" / "corpus_at_last_prod_release"
PROD_MARKER = REPO_ROOT / "config" / "last_deployed_prod_version.json"


def _copy_fixture(dst: Path) -> Path:
    """Copy the pinned fixture into a fresh directory the test can mutate."""
    corpus = dst / "corpus"
    shutil.copytree(FIXTURE, corpus)
    # README is fixture doc, not corpus data.
    (corpus / "README.md").unlink()
    return corpus


def _stub_lancedb_build(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace ``build_two_tier_index`` with a marker-file writer so the test
    doesn't require LanceDB. Only actually called if the pending set includes
    m0002 (i.e. a fresh prod deploy that reintroduces the native reindex).
    """
    calls: dict = {"built": False}

    def _fake(corpus_dir, lance_dir, **kwargs):
        calls["built"] = True
        Path(lance_dir).mkdir(parents=True, exist_ok=True)
        (Path(lance_dir).parent / "metadata.json").write_text(
            json.dumps({"stub": True}), encoding="utf-8"
        )
        return two_tier_indexer.TwoTierIndexStats(episodes=1, segments=1, insights=2)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    return calls


def _run(corpus: Path, *argv: str) -> int:
    args = parse_upgrade_argv([argv[0], "--corpus-dir", str(corpus), *argv[1:]])
    return run_upgrade_cli(args, logging.getLogger("test"))


def _pending_migration_ids() -> set[str]:
    """Migrations on HEAD that were NOT in the last prod deploy."""
    marker = json.loads(PROD_MARKER.read_text(encoding="utf-8"))
    prod_applied = set(marker["applied_migrations"])
    registry_ids = {m.id for m in get_migrations()}
    return registry_ids - prod_applied


def test_gap_between_prod_and_head_applies_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migrations added since the last prod deploy must all apply cleanly
    to a prod-shape corpus. This is the only gap the next deploy will hit.
    """
    corpus = _copy_fixture(tmp_path)
    _stub_lancedb_build(monkeypatch)

    pending = _pending_migration_ids()

    if not pending:
        # No new migrations since the last prod deploy — the framework should
        # treat the fixture as up-to-date and touch nothing.
        assert _run(corpus, "status") == 0, (
            "no pending migrations but status still reports pending — "
            "framework/marker/fixture drift"
        )
        # A run should still succeed (no-op) without mutating the ledger.
        ledger_before = json.loads((corpus / "upgrade_ledger.json").read_text())
        assert _run(corpus, "run", "--yes") == 0
        ledger_after = json.loads((corpus / "upgrade_ledger.json").read_text())
        assert (
            ledger_before == ledger_after
        ), "framework mutated the ledger despite no pending migrations"
        return

    # Pending migrations exist. Snapshot the prod-applied set for the
    # "did-not-regress" check.
    prod_applied_before = FilesystemStateStore(corpus).applied_migration_ids()

    # Status BEFORE: pending → exit 2.
    assert _run(corpus, "status") == 2, f"expected 'pending' status with pending={sorted(pending)}"

    # Apply the gap.
    assert _run(corpus, "run", "--yes") == 0, "run --yes failed to close the prod gap"

    # Every pending migration is now in the ledger.
    store = FilesystemStateStore(corpus)
    applied_after = store.applied_migration_ids()
    for mid in pending:
        assert (
            mid in applied_after
        ), f"pending migration {mid!r} did not get recorded after run --yes"

    # Nothing that was previously in the prod ledger disappeared.
    assert prod_applied_before <= applied_after, (
        f"framework dropped previously-applied migrations: "
        f"{sorted(prod_applied_before - applied_after)}"
    )

    # Status AFTER: no pending.
    assert _run(corpus, "status") == 0

    # Idempotency: a second run mutates nothing.
    ledger_before_rerun = json.loads((corpus / "upgrade_ledger.json").read_text())
    assert _run(corpus, "run", "--yes") == 0
    ledger_after_rerun = json.loads((corpus / "upgrade_ledger.json").read_text())
    assert (
        ledger_before_rerun == ledger_after_rerun
    ), "re-running upgrade after closing the gap mutated the ledger"


def test_prod_shape_gi_is_transform_stable_under_current_migrations(
    tmp_path: Path,
) -> None:
    """The fixture's post-prod .gi.json should be a fixed point under
    ``migrate_gi_document_v3`` (it already ran on prod). If a new migration
    changes what m0003 produces, this test tells the author "you moved the
    goalpost — bump the marker / regenerate the fixture, or add m0004".
    """
    from podcast_scraper.migrations.gil_kg_identity_migrations import (
        migrate_gi_document_v3,
    )

    marker = json.loads(PROD_MARKER.read_text(encoding="utf-8"))
    if "0003_gi_v3_typed_mentions" not in marker["applied_migrations"]:
        pytest.skip("m0003 is not in the prod marker — nothing to check here")

    before = json.loads((FIXTURE / "metadata" / "ep1.gi.json").read_text(encoding="utf-8"))
    after = migrate_gi_document_v3(copy.deepcopy(before))
    assert before == after, (
        "prod-shape fixture .gi.json changes under migrate_gi_document_v3. "
        "Either m0003 changed and the fixture needs updating, or a downstream "
        "migration is needed. See tests/fixtures/upgrade/corpus_at_last_prod_release/README.md."
    )
