"""Corpus migrations on rows (RFC-135): receipts, the ledger, dry runs, undo and the lease."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db import leases
from podcast_scraper.db.schema import (
    corpus_metadata,
    corpus_migrations,
    migration_receipts,
    migration_runs,
)
from podcast_scraper.db.settings import normalize_url
from podcast_scraper.upgrade.rows import (
    migrator_engine,
    RowContext,
    RowMigration,
    RowResult,
    RowsRefused,
    run_migration,
    undo_migration,
    UndoRefused,
)
from tests._database import worker_database_url

ITEMS = corpus_metadata.tables["corpus.content_items"]
CARDS = corpus_metadata.tables["corpus.item_cards"]
SOURCES = corpus_metadata.tables["corpus.sources"]


@pytest.fixture
def engine() -> Iterator[Engine]:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    made = migrator_engine(url)
    yield made
    made.dispose()
    cleaner = create_engine(normalize_url(url))
    with cleaner.begin() as conn:
        for table in reversed(corpus_metadata.sorted_tables):
            conn.execute(table.delete())
    cleaner.dispose()


@pytest.fixture
def items(engine: Engine) -> dict[str, int]:
    """Three items; the first and third have cards."""
    with engine.begin() as conn:
        source_id = conn.execute(
            insert(SOURCES)
            .values(natural_id="source:a", content_type="podcast", title="A")
            .returning(SOURCES.c.source_id)
        ).scalar_one()
        made: dict[str, int] = {}
        for n in ("1", "2", "3"):
            made[n] = conn.execute(
                insert(ITEMS)
                .values(
                    natural_id=f"item:{n}",
                    slug=f"slug-{n}",
                    source_id=source_id,
                    external_item_id=f"ext-{n}",
                    content_type="podcast",
                    title=f"Episode {n}",
                )
                .returning(ITEMS.c.item_id)
            ).scalar_one()
        for n in ("1", "3"):
            conn.execute(
                insert(CARDS).values(item_id=made[n], title=f"Episode {n}", source_title="A")
            )
    return made


class Retitle(RowMigration):
    """Retitles item 1, gives item 2 a card and removes item 3's: one update, insert and delete."""

    id = "9001_test_retitle"
    to_version = "9.0.1"

    def __init__(self, items: dict[str, int], fail: bool = False) -> None:
        self.items = items
        self.fail = fail

    def apply_rows(self, ctx: RowContext) -> RowResult:
        one, two, three = self.items["1"], self.items["2"], self.items["3"]
        ctx.rows.update("content_items", {"item_id": one}, {"title": "Episode One"})
        # Unchanged: no receipt.
        ctx.rows.update("content_items", {"item_id": two}, {"title": "Episode 2"})
        ctx.rows.insert("item_cards", {"item_id": two, "title": "Episode 2", "source_title": "A"})
        ctx.rows.delete("item_cards", {"item_id": three})
        if self.fail:
            raise RuntimeError("boom")
        return RowResult(item_ids={one, two, three}, message="retitled", counts={"items": 3})

    def verify_rows(self, conn: Connection) -> tuple[bool, str]:
        title = conn.execute(
            select(ITEMS.c.title).where(ITEMS.c.item_id == self.items["1"])
        ).scalar_one()
        return title == "Episode One", f"item 1 title {title!r}"


def _snapshot(engine: Engine) -> dict[tuple[str, int], Any]:
    """Every content item and card, as ``to_jsonb``."""
    with engine.connect() as conn:
        out: dict[tuple[str, int], Any] = {}
        for name in ("content_items", "item_cards"):
            for item_id, state in conn.execute(
                text(f"SELECT item_id, to_jsonb(t) FROM corpus.{name} AS t")
            ):
                out[(name, item_id)] = state
        return out


def _runs(engine: Engine) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        return [
            (r.kind, r.status)
            for r in conn.execute(select(migration_runs).order_by(migration_runs.c.run_id))
        ]


def _ledger(engine: Engine, natural_id: str) -> Any:
    with engine.connect() as conn:
        return conn.execute(
            select(corpus_migrations).where(corpus_migrations.c.natural_id == natural_id)
        ).one_or_none()


def _receipts(engine: Engine) -> list[Any]:
    with engine.connect() as conn:
        return list(conn.execute(select(migration_receipts).order_by(migration_receipts.c.seq)))


def test_apply_then_undo_restores_every_row(engine: Engine, items: dict[str, int]) -> None:
    before = _snapshot(engine)
    applied = run_migration(engine, Retitle(items), code_digest="abc")
    assert (applied.kind, applied.status, applied.receipts) == ("apply", "ok", 3)
    assert applied.result["verify"] == {"ok": True, "message": "item 1 title 'Episode One'"}
    assert applied.result["corpus_revision_bumped"] is False
    ledger = _ledger(engine, Retitle.id)
    assert (ledger.status, ledger.to_version, ledger.last_run_id) == (
        "applied",
        "9.0.1",
        applied.run_id,
    )
    assert _snapshot(engine) != before

    undone = undo_migration(engine, Retitle.id)
    assert (undone.kind, undone.status) == ("undo", "ok")
    assert undone.result["restored"] == 3
    assert _snapshot(engine) == before
    ledger = _ledger(engine, Retitle.id)
    assert (ledger.status, ledger.last_run_id) == ("undone", undone.run_id)
    assert _runs(engine) == [("apply", "ok"), ("undo", "ok")]


def test_every_changed_row_has_a_receipt(engine: Engine, items: dict[str, int]) -> None:
    before = _snapshot(engine)
    run_migration(engine, Retitle(items))
    after = _snapshot(engine)
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    receipts = _receipts(engine)
    keyed = {(r.target_table, json.loads(r.target_key)[0]) for r in receipts}
    assert keyed == changed
    for r in receipts:
        k = (r.target_table, json.loads(r.target_key)[0])
        assert (r.before, r.after) == (before.get(k), after.get(k))
    assert [r.seq for r in receipts] == [1, 2, 3]


def test_a_dry_run_changes_nothing_and_is_recorded(engine: Engine, items: dict[str, int]) -> None:
    before = _snapshot(engine)
    outcome = run_migration(engine, Retitle(items), dry_run=True)
    assert (outcome.kind, outcome.status, outcome.receipts) == ("dry_run", "ok", 3)
    assert _snapshot(engine) == before
    assert _receipts(engine) == []
    assert _ledger(engine, Retitle.id) is None
    assert _runs(engine) == [("dry_run", "ok")]


def test_a_second_apply_is_refused(engine: Engine, items: dict[str, int]) -> None:
    run_migration(engine, Retitle(items))
    after = _snapshot(engine)
    with pytest.raises(RowsRefused, match="applied already"):
        run_migration(engine, Retitle(items))
    assert _snapshot(engine) == after
    assert _runs(engine) == [("apply", "ok"), ("apply", "refused")]


def test_undo_refuses_and_restores_nothing_when_a_row_changed(
    engine: Engine, items: dict[str, int]
) -> None:
    run_migration(engine, Retitle(items))
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE corpus.content_items SET title = 'Edited' WHERE item_id = :i"),
            {"i": items["1"]},
        )
    changed = _snapshot(engine)
    with pytest.raises(UndoRefused) as caught:
        undo_migration(engine, Retitle.id)
    assert caught.value.refused == [f"content_items [{items['1']}]"]
    assert _snapshot(engine) == changed
    assert _ledger(engine, Retitle.id).status == "applied"
    assert _runs(engine) == [("apply", "ok"), ("undo", "refused")]


def test_a_failure_inside_apply_rows_leaves_nothing(engine: Engine, items: dict[str, int]) -> None:
    before = _snapshot(engine)
    with pytest.raises(RuntimeError, match="boom"):
        run_migration(engine, Retitle(items, fail=True))
    assert _snapshot(engine) == before
    assert _receipts(engine) == []
    assert _ledger(engine, Retitle.id) is None
    assert _runs(engine) == [("apply", "failed")]


def test_undo_of_a_migration_not_applied_is_refused(engine: Engine) -> None:
    with pytest.raises(RowsRefused, match="not applied"):
        undo_migration(engine, Retitle.id)
    assert _runs(engine) == [("undo", "refused")]


def test_a_dry_run_undo_restores_nothing(engine: Engine, items: dict[str, int]) -> None:
    run_migration(engine, Retitle(items))
    after = _snapshot(engine)
    outcome = undo_migration(engine, Retitle.id, dry_run=True)
    assert (outcome.kind, outcome.result["restored"]) == ("dry_run", 3)
    assert _snapshot(engine) == after
    assert _ledger(engine, Retitle.id).status == "applied"
    assert _runs(engine) == [("apply", "ok"), ("dry_run", "ok")]


def test_undo_updates_in_place_so_children_survive(engine: Engine, items: dict[str, int]) -> None:
    run_migration(engine, Retitle(items))
    undo_migration(engine, Retitle.id)
    with engine.connect() as conn:
        cards = set(conn.execute(select(CARDS.c.item_id)).scalars())
    # Item 1's title was restored by an update: a delete and insert would have cascaded its card.
    assert cards == {items["1"], items["3"]}


def test_the_corpus_lease_refuses_a_run_while_another_holds_it(
    engine: Engine, items: dict[str, int]
) -> None:
    other = create_engine(normalize_url(worker_database_url() or ""))
    try:
        leases.acquire(other, [leases.LeaseRequest(leases.CORPUS, leases.SHARED)], "api:test")
        before = _snapshot(engine)
        with pytest.raises(RowsRefused):
            run_migration(engine, Retitle(items))
        assert _snapshot(engine) == before
        assert _runs(engine) == [("apply", "refused")]
        leases.release(other, "api:test")
        assert run_migration(engine, Retitle(items)).status == "ok"
    finally:
        other.dispose()


def test_the_revision_bump_is_a_no_op_without_its_table(engine: Engine) -> None:
    from podcast_scraper.upgrade.rows import bump_corpus_revision

    with engine.begin() as conn:
        found = conn.execute(text("SELECT to_regclass('corpus.corpus_revision')")).scalar()
        assert found is None
        assert bump_corpus_revision(conn) is False


def test_apply_through_the_file_runner_is_refused(items: dict[str, int]) -> None:
    with pytest.raises(TypeError, match="run_migration"):
        Retitle(items).apply(None)  # type: ignore[arg-type]
