"""Corpus migrations on rows (RFC-135): receipts, the ledger, dry runs, undo and the lease."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import cast, create_engine, func, insert, select, Text, text
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db import leases
from podcast_scraper.db.schema import (
    corpus_metadata,
    corpus_migrations,
    corpus_revision,
    migration_receipts,
    migration_runs,
)
from podcast_scraper.db.settings import ENV_MIGRATOR_URL, normalize_url
from podcast_scraper.upgrade import cli_handlers, outcome_hook, rows as rows_mod
from podcast_scraper.upgrade.migration import Migration
from podcast_scraper.upgrade.rows import (
    DbStateStore,
    migrator_engine,
    PlanStopped,
    prepare_regime,
    rerun_after_commit,
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
            if table is corpus_revision:
                # Reset, not deleted, as tests/_database.py does: corpus_0007 seeds its one row.
                conn.execute(corpus_revision.update().values(revision=0))
            else:
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
        for t in (ITEMS, CARDS):
            for item_id, state in conn.execute(
                select(t.c.item_id, func.to_jsonb(t.table_valued()))
            ):
                out[(t.name, item_id)] = state
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


def _revision(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(corpus_revision.c.revision)).scalar_one())


def _receipts(engine: Engine) -> list[Any]:
    with engine.connect() as conn:
        return list(conn.execute(select(migration_receipts).order_by(migration_receipts.c.seq)))


def test_apply_then_undo_restores_every_row(engine: Engine, items: dict[str, int]) -> None:
    before = _snapshot(engine)
    start = _revision(engine)
    applied = run_migration(engine, Retitle(items))
    assert (applied.kind, applied.status, applied.receipts) == ("apply", "ok", 3)
    assert applied.result["verify"] == {"ok": True, "message": "item 1 title 'Episode One'"}
    assert applied.result["corpus_revision_bumped"] is True
    assert _revision(engine) == start + 1
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
    assert undone.result["corpus_revision_bumped"] is True
    assert _revision(engine) == start + 2
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

    with engine.connect() as conn, conn.begin() as tx:
        # A database not yet at corpus_0007: its table dropped here and rolled back.
        conn.execute(text("DROP TABLE corpus.corpus_revision"))
        assert bump_corpus_revision(conn) is False
        tx.rollback()
    with engine.connect() as conn:
        restored = conn.execute(text("SELECT to_regclass('corpus.corpus_revision')")).scalar()
    assert restored is not None


def test_apply_through_the_file_runner_is_refused(items: dict[str, int]) -> None:
    with pytest.raises(TypeError, match="run_migration"):
        Retitle(items).apply(None)  # type: ignore[arg-type]


def test_the_file_runner_leaves_migrations_on_rows_out(
    items: dict[str, int], tmp_path: Path
) -> None:
    from podcast_scraper.upgrade.runner import UpgradeRunner
    from podcast_scraper.upgrade.state import FilesystemStateStore

    runner = UpgradeRunner(FilesystemStateStore(tmp_path), migrations=[Retitle(items)])
    assert (runner.migrations, runner.pending(), runner.status().pending) == ([], [], [])


def _every_corpus_row(engine: Engine) -> dict[str, list[str]]:
    """Every corpus table but the run history, as sorted row hashes."""
    with engine.connect() as conn:
        return {
            t.name: sorted(
                conn.execute(select(func.md5(cast(func.to_jsonb(t.table_valued()), Text))))
                .scalars()
                .all()
            )
            for t in corpus_metadata.sorted_tables
            if t.name != "migration_runs"
        }


def test_a_dry_run_leaves_every_corpus_table_as_it_was(
    engine: Engine, items: dict[str, int]
) -> None:
    before = _every_corpus_row(engine)
    outcome = run_migration(engine, Retitle(items), dry_run=True)
    assert _every_corpus_row(engine) == before
    with engine.connect() as conn:
        plan = conn.execute(
            select(migration_runs.c.plan).where(migration_runs.c.run_id == outcome.run_id)
        ).scalar_one()
    # Keyed by the items' natural ids (the fixture's are item:1, item:2, item:3).
    one, two, three = "item:1", "item:2", "item:3"
    assert [p["meta"] for p in plan] == [one, two, three]
    by_meta = {p["meta"]: p for p in plan}
    assert by_meta[one]["counts"] == {"content_items": 1}
    assert by_meta[two]["counts"] == {"item_cards": 1}
    assert by_meta[three]["counts"] == {"item_cards": 1}
    # Item 2's row was read and left as it was: an input, no receipt.
    assert set(by_meta[two]["inputs"]) == {f"content_items:[{items['2']}]"}
    assert set(by_meta[three]["inputs"]) == {f"item_cards:[{items['3']}]"}


def _untimed(snapshot: dict[tuple[str, int], Any]) -> dict[tuple[str, int], Any]:
    return {
        k: {c: v for c, v in s.items() if c not in ("created_at", "updated_at")}
        for k, s in snapshot.items()
    }


def test_apply_undo_apply_gives_the_rows_of_one_apply(
    engine: Engine, items: dict[str, int]
) -> None:
    run_migration(engine, Retitle(items))
    once = _snapshot(engine)
    assert DbStateStore(engine).applied_migration_ids() == {Retitle.id}
    assert DbStateStore(engine).current_version() == "9.0.1"
    undo_migration(engine, Retitle.id)
    assert DbStateStore(engine).applied_migration_ids() == set()
    again = run_migration(engine, Retitle(items))
    assert again.receipts == 3
    # The same rows but for the times the database stamps: the re-inserted card's are new.
    assert _untimed(_snapshot(engine)) == _untimed(once)
    assert _ledger(engine, Retitle.id).last_run_id == again.run_id


def test_an_in_transaction_surface_sees_the_changes_and_its_failure_undoes_them(
    engine: Engine, items: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[frozenset[int], str]] = []

    def refresh(conn: Connection, changed: frozenset[int]) -> None:
        title = conn.execute(
            select(ITEMS.c.title).where(ITEMS.c.item_id == items["1"])
        ).scalar_one()
        seen.append((changed, title))

    monkeypatch.setattr(rows_mod, "IN_TRANSACTION_SURFACES", [("cards", refresh)])
    run_migration(engine, Retitle(items))
    assert seen == [(frozenset(items.values()), "Episode One")]
    undo_migration(engine, Retitle.id)

    def broken(conn: Connection, changed: frozenset[int]) -> None:
        raise RuntimeError("card build failed")

    monkeypatch.setattr(rows_mod, "IN_TRANSACTION_SURFACES", [("cards", broken)])
    before = _snapshot(engine)
    receipts = len(_receipts(engine))
    with pytest.raises(RuntimeError, match="card build failed"):
        run_migration(engine, Retitle(items))
    assert _snapshot(engine) == before
    assert len(_receipts(engine)) == receipts
    assert _ledger(engine, Retitle.id).status == "undone"


def test_an_after_commit_failure_keeps_the_apply_and_its_rerun_completes_it(
    engine: Engine, items: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[frozenset[int]] = []
    fail = [True]

    def lance(_engine: Engine, changed: frozenset[int]) -> None:
        calls.append(changed)
        if fail[0]:
            raise RuntimeError("lance unreachable")

    monkeypatch.setattr(rows_mod, "AFTER_COMMIT_SURFACES", [("lance", lance)])
    outcome = run_migration(engine, Retitle(items))
    assert outcome.after_commit_failed == ["lance"]
    assert _ledger(engine, Retitle.id).status == "applied"
    with engine.connect() as conn:
        result = conn.execute(
            select(migration_runs.c.result).where(migration_runs.c.run_id == outcome.run_id)
        ).scalar_one()
    assert result["after_commit"] == {"lance": "failed: RuntimeError: lance unreachable"}

    fail[0] = False
    again = rerun_after_commit(engine, Retitle.id)
    assert (again.kind, again.after_commit_failed) == ("rerun", [])
    assert again.result["for_run"] == outcome.run_id
    assert calls == [frozenset(items.values())] * 2
    assert _runs(engine) == [("apply", "ok"), ("rerun", "ok")]


def test_an_ingest_is_refused_while_a_migration_holds_the_lease(
    engine: Engine, items: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = create_engine(normalize_url(worker_database_url() or ""))
    refused: list[str] = []

    def ingest_tries(conn: Connection, changed: frozenset[int]) -> None:
        try:
            leases.acquire(other, [leases.LeaseRequest(leases.CORPUS, leases.SHARED)], "ingest:t")
        except leases.LeaseRefused as exc:
            refused.append(str(exc))

    monkeypatch.setattr(rows_mod, "IN_TRANSACTION_SURFACES", [("probe", ingest_tries)])
    try:
        run_migration(engine, Retitle(items))
    finally:
        other.dispose()
    assert len(refused) == 1
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM corpus.job_leases")).scalar_one() == 0


def test_the_revision_bump_advances_the_counter_when_its_table_exists(engine: Engine) -> None:
    from podcast_scraper.upgrade.rows import bump_corpus_revision

    with engine.begin() as conn:
        conn.execute(corpus_revision.update().values(revision=41))
        assert bump_corpus_revision(conn) is True
    assert _revision(engine) == 42


def test_every_run_is_an_event(
    engine: Engine, items: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        rows_mod, "emit_event", lambda event_type, **f: events.append({"type": event_type, **f})
    )
    run_migration(engine, Retitle(items), dry_run=True)
    run_migration(engine, Retitle(items))
    undo_migration(engine, Retitle.id)
    assert [(e["type"], e["kind"], e["receipts"]) for e in events] == [
        ("corpus_migration.run", "dry_run", 3),
        ("corpus_migration.run", "apply", 3),
        ("corpus_migration.run", "undo", 3),
    ]


# ---------------------------------------------------------------- the regime (RFC-135 §5)


def outcome_for(migration: RowMigration, plan: list[dict[str, Any]], **over: Any) -> dict:
    doc = {
        "migration": migration.id,
        "snapshot": "test",
        "applied_before": [],
        "code_digest": outcome_hook.migration_digest(migration),
        "code_dirty": False,
        "verdicts_sha256": None,
        "scoreboard_before": {},
        "scoreboard_after": {},
        "episode_deltas": {},
        "plan": plan,
        "computed_class": "light",
        "verdict": "GO",
        "conditions": [],
    }
    doc.update(over)
    return doc


def _clone_plan(engine: Engine, migration: RowMigration) -> list[dict[str, Any]]:
    """The plan the gate measures on its clone: a dry run's (here, of the same database)."""
    outcome = run_migration(engine, migration, dry_run=True)
    with engine.begin() as conn:
        plan: list[dict[str, Any]] = conn.execute(
            select(migration_runs.c.plan).where(migration_runs.c.run_id == outcome.run_id)
        ).scalar_one()
        conn.execute(migration_runs.delete())
    return plan


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_the_regime_applies_when_the_plan_is_the_clones(
    engine: Engine, items: dict[str, int], tmp_path: Path
) -> None:
    mig = Retitle(items)
    doc = outcome_for(mig, _clone_plan(engine, mig))
    regime = prepare_regime(mig, set(), [mig.id], outcome_file=_write(tmp_path, "o.json", doc))
    outcome = run_migration(engine, mig, regime=regime)
    assert outcome.status == "ok"
    with engine.connect() as conn:
        run = conn.execute(
            select(migration_runs).where(migration_runs.c.run_id == outcome.run_id)
        ).one()
    assert (run.code_digest, run.outcome_sha256) == (regime.code_digest, regime.outcome_sha)


@pytest.mark.parametrize(
    "over, match",
    [
        ({"code_digest": "0" * 64}, "not the code that was measured"),
        ({"applied_before": ["0055_other"]}, "applied migrations differ"),
        ({"verdict": "NO GO"}, "not GO"),
        ({"code_dirty": True}, "uncommitted code"),
    ],
)
def test_the_regime_refuses_before_any_write(
    engine: Engine, items: dict[str, int], tmp_path: Path, over: dict, match: str
) -> None:
    mig = Retitle(items)
    path = _write(tmp_path, "o.json", outcome_for(mig, [], **over))
    with pytest.raises(outcome_hook.HookRefused, match=match) as caught:
        prepare_regime(mig, set(), [mig.id], outcome_file=path)
    assert caught.value.code == 3


def test_the_regime_refuses_without_an_outcome(items: dict[str, int], tmp_path: Path) -> None:
    with pytest.raises(outcome_hook.HookRefused, match="no shipped outcome"):
        prepare_regime(Retitle(items), set(), [], outcome_file=tmp_path / "absent.json")


def test_a_plan_that_differs_stops_the_run_and_writes_nothing(
    engine: Engine, items: dict[str, int], tmp_path: Path
) -> None:
    mig = Retitle(items)
    plan = _clone_plan(engine, mig)
    # Prod's item 2 changed after the clone was measured: its input sha differs.
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE corpus.content_items SET language = 'en' WHERE item_id = :i"),
            {"i": items["2"]},
        )
    before = _snapshot(engine)
    regime = prepare_regime(
        mig, set(), [mig.id], outcome_file=_write(tmp_path, "o.json", outcome_for(mig, plan))
    )
    with pytest.raises(PlanStopped) as caught:
        run_migration(engine, mig, regime=regime)
    assert caught.value.code == 4
    assert caught.value.diff == [{"meta": "item:2", "kind": "differs", "fields": ["inputs"]}]
    assert _snapshot(engine) == before
    assert _receipts(engine) == []
    assert _ledger(engine, mig.id) is None
    assert _runs(engine) == [("apply", "refused")]

    accept = {
        "migration": mig.id,
        "outcome_sha256": regime.outcome_sha,
        "operator_go": "test 2026-11-26",
        "entries": [
            dict(e, cause="item 2's language was set after the gate") for e in caught.value.diff
        ],
    }
    accepted = prepare_regime(
        mig,
        set(),
        [mig.id],
        outcome_file=_write(tmp_path, "o.json", outcome_for(mig, plan)),
        accept_file=_write(tmp_path, "accept.json", accept),
    )
    assert run_migration(engine, mig, regime=accepted).status == "ok"


def test_an_accept_file_for_another_difference_stops_the_run(
    engine: Engine, items: dict[str, int], tmp_path: Path
) -> None:
    mig = Retitle(items)
    plan = [p for p in _clone_plan(engine, mig) if p["meta"] != "item:3"]
    doc = outcome_for(mig, plan)
    sha = hashlib.sha256(json.dumps(doc).encode("utf-8")).hexdigest()
    accept = {
        "migration": mig.id,
        "outcome_sha256": sha,
        "operator_go": "test",
        "entries": [{"meta": "item:0", "kind": "prod_only", "cause": "another"}],
    }
    regime = prepare_regime(
        mig,
        set(),
        [mig.id],
        outcome_file=_write(tmp_path, "o.json", doc),
        accept_file=_write(tmp_path, "a.json", accept),
    )
    before = _snapshot(engine)
    with pytest.raises(outcome_hook.HookStopped, match="not the accepted one"):
        run_migration(engine, mig, regime=regime)
    assert _snapshot(engine) == before


# ---------------------------------------------------------------- the measure (exit 5)


def _titles(conn: Connection) -> dict[str, dict[str, int]]:
    """A test scoreboard: per item, whether it is titled 'Episode One', and no inconsistency."""
    return {
        natural_id: {"titled_one": int(title == "Episode One"), "S1c_inconsistent": 0}
        for natural_id, title in conn.execute(select(ITEMS.c.natural_id, ITEMS.c.title))
    }


@pytest.fixture
def scoreboard(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def score(conn: Connection) -> dict[str, dict[str, int]]:
        calls.append(1)
        return _titles(conn)

    board = rows_mod.RowScoreboard(score_items=score, gating_metrics=lambda: ["titled_one"])
    monkeypatch.setattr(rows_mod, "SCOREBOARD", board)
    return calls


def test_without_a_scoreboard_the_measure_is_recorded_skipped(
    engine: Engine, items: dict[str, int]
) -> None:
    outcome = run_migration(engine, Retitle(items))
    assert "gated on the plan only" in outcome.result["measure"]["skipped"]


def test_the_measure_passes_when_the_deltas_are_the_clones(
    engine: Engine, items: dict[str, int], tmp_path: Path, scoreboard: list[int]
) -> None:
    mig = Retitle(items)
    doc = outcome_for(mig, _clone_plan(engine, mig), episode_deltas={"item:1": {"titled_one": 1}})
    regime = prepare_regime(mig, set(), [mig.id], outcome_file=_write(tmp_path, "o.json", doc))
    outcome = run_migration(engine, mig, regime=regime)
    assert outcome.outcome_failed == []
    measured = outcome.result["measure"]
    assert measured["prod_deltas"] == {"item:1": {"titled_one": 1}}
    assert (measured["totals_before"], measured["totals_after"]) == (
        {"titled_one": 0},
        {"titled_one": 1},
    )


def test_a_measure_that_differs_keeps_the_apply_and_names_the_failure(
    engine: Engine, items: dict[str, int], tmp_path: Path, scoreboard: list[int]
) -> None:
    mig = Retitle(items)
    doc = outcome_for(mig, _clone_plan(engine, mig), episode_deltas={})
    regime = prepare_regime(mig, set(), [mig.id], outcome_file=_write(tmp_path, "o.json", doc))
    outcome = run_migration(engine, mig, regime=regime)
    assert outcome.outcome_failed == ["item:1: titled_one changed +1, the clone's +0"]
    assert _ledger(engine, mig.id).status == "applied"
    with engine.connect() as conn:
        result = conn.execute(
            select(migration_runs.c.result).where(migration_runs.c.run_id == outcome.run_id)
        ).scalar_one()
    assert result["measure"]["failures"] == outcome.outcome_failed


def test_a_scoreboard_that_fails_after_the_apply_rolls_it_back(
    engine: Engine, items: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def score(conn: Connection) -> dict[str, dict[str, int]]:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("scoreboard down")
        return _titles(conn)

    monkeypatch.setattr(
        rows_mod, "SCOREBOARD", rows_mod.RowScoreboard(score, lambda: ["titled_one"])
    )
    before = _snapshot(engine)
    with pytest.raises(RuntimeError, match="scoreboard down"):
        run_migration(engine, Retitle(items))
    assert _snapshot(engine) == before
    assert _ledger(engine, Retitle.id) is None
    assert _runs(engine) == [("apply", "failed")]


# ---------------------------------------------------------------- the gate (--database-url)

#: What ``scripts/measure/build_outcome.py`` copies from a gate report into the outcome.
BUILD_OUTCOME_KEYS = {
    "snapshot",
    "applied_before",
    "code_digest",
    "verdicts_sha256",
    "scoreboard_before",
    "scoreboard_after",
    "episode_deltas",
    "plan",
    "plan_sha256",
    "computed_class",
}


def test_the_gate_measures_and_writes_nothing(
    engine: Engine, items: dict[str, int], scoreboard: list[int]
) -> None:
    before = _every_corpus_row(engine)
    report = rows_mod.gate_report(engine, Retitle(items), snapshot="snap-test")
    assert _every_corpus_row(engine) == before
    assert _runs(engine) == []
    assert BUILD_OUTCOME_KEYS <= set(report)
    assert (report["computed_class"], report["rows_changed"]) == ("full", 3)
    assert report["rows_changed_by_table"] == {"content_items": 1, "item_cards": 2}
    assert report["episode_deltas"] == {"item:1": {"titled_one": 1}}
    assert report["gating_moves"] == {"titled_one": 1}
    assert [p["meta"] for p in report["plan"]] == ["item:1", "item:2", "item:3"]


def test_an_outcome_from_the_gate_lets_the_run_through(
    engine: Engine, items: dict[str, int], tmp_path: Path, scoreboard: list[int]
) -> None:
    mig = Retitle(items)
    report = rows_mod.gate_report(engine, mig, snapshot="snap-test")
    doc = outcome_for(mig, report["plan"], episode_deltas=report["episode_deltas"])
    doc.update({k: report[k] for k in BUILD_OUTCOME_KEYS})
    regime = prepare_regime(mig, set(), [mig.id], outcome_file=_write(tmp_path, "o.json", doc))
    outcome = run_migration(engine, mig, regime=regime)
    assert (outcome.status, outcome.outcome_failed) == ("ok", [])


def test_the_gate_script_writes_the_report_for_a_migration_on_rows(
    engine: Engine, items: dict[str, int], tmp_path: Path
) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[4] / "scripts" / "measure" / "corpus_change_gate.py"
    spec = importlib.util.spec_from_file_location("corpus_change_gate_rows", script)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    out = tmp_path / "gate.json"
    url = worker_database_url() or ""
    assert gate.rows_gate(Retitle(items), url, "snap-test", out) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert BUILD_OUTCOME_KEYS | {"code", "measured_at", "clone"} <= set(report)
    assert "podcast:***@" in report["clone"]
    assert "gated on the plan only" in " ".join(report["why"])

    class OnFiles(Migration):
        id = "9002_test_files"

        def apply(self, ctx: Any) -> Any:
            raise AssertionError("not applied")

    assert gate.rows_gate(OnFiles(), url, "snap-test", tmp_path / "files.json") == 2
    assert not (tmp_path / "files.json").exists()


# ---------------------------------------------------------------- the CLI (row mode)

LOG = logging.getLogger("test.upgrade.rows.cli")


def cli(*argv: str) -> int:
    return cli_handlers.run_upgrade_cli(cli_handlers.parse_upgrade_argv(list(argv)), LOG)


@pytest.fixture
def shipped(
    engine: Engine, items: dict[str, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Retitle:
    """Retitle registered, its outcome shipped (measured on this database), migrator configured."""
    mig = Retitle(items)
    outcome = tmp_path / f"{mig.id}.json"
    outcome.write_text(json.dumps(outcome_for(mig, _clone_plan(engine, mig))), encoding="utf-8")
    monkeypatch.setenv(ENV_MIGRATOR_URL, worker_database_url() or "")
    monkeypatch.setattr(cli_handlers, "_row_migrations", lambda: [mig])
    monkeypatch.setattr(outcome_hook, "outcome_path", lambda mid: tmp_path / f"{mid}.json")
    return mig


def test_cli_without_a_migrator_url_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_MIGRATOR_URL, raising=False)
    assert cli("run", "--database", "--yes") == 1
    assert cli("undo", "--migration", "9001_test_retitle", "--yes") == 1


def test_cli_without_yes_off_a_terminal_refuses(engine: Engine, shipped: Retitle) -> None:
    before = _snapshot(engine)
    assert cli("run", "--database") == 1
    assert cli("undo", "--migration", shipped.id) == 1
    assert _snapshot(engine) == before
    assert _runs(engine) == []


def test_cli_dry_run_apply_then_undo(
    engine: Engine, shipped: Retitle, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(engine)
    assert cli("run", "--database", "--dry-run") == 0
    assert _snapshot(engine) == before
    capsys.readouterr()
    assert cli("run", "--database", "--yes", "--json") == 0
    printed: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert (printed["kind"], printed["status"], printed["receipts"]) == ("apply", "ok", 3)
    assert _ledger(engine, shipped.id).status == "applied"
    assert cli("run", "--database", "--yes") == 0
    assert "Up to date" in capsys.readouterr().out
    assert cli("undo", "--migration", shipped.id, "--dry-run") == 0
    assert _ledger(engine, shipped.id).status == "applied"
    assert cli("undo", "--migration", shipped.id, "--yes") == 0
    assert _snapshot(engine) == before
    assert _runs(engine) == [("dry_run", "ok"), ("apply", "ok"), ("dry_run", "ok"), ("undo", "ok")]


def test_cli_after_commit_failure_exits_6_and_names_the_rerun(
    engine: Engine,
    shipped: Retitle,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fail = [True]

    def lance(_engine: Engine, _items: frozenset[int]) -> None:
        if fail[0]:
            raise RuntimeError("lance unreachable")

    monkeypatch.setattr(rows_mod, "AFTER_COMMIT_SURFACES", [("lance", lance)])
    assert cli("run", "--database", "--yes") == 6
    err = capsys.readouterr().err.strip().splitlines()
    assert err[-1] == (
        "after-commit step(s) failed (lance); the migration stays committed; re-run them: "
        f"podcast_scraper upgrade surfaces --migration {shipped.id}"
    )
    assert _ledger(engine, shipped.id).status == "applied"
    assert cli("surfaces", "--migration", shipped.id) == 6
    fail[0] = False
    assert cli("surfaces", "--migration", shipped.id) == 0


def test_cli_regime_exit_codes(
    engine: Engine,
    shipped: Retitle,
    items: dict[str, int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = (tmp_path / f"{shipped.id}.json").read_text(encoding="utf-8")
    doc = json.loads(good)
    (tmp_path / f"{shipped.id}.json").write_text(
        json.dumps({**doc, "code_digest": "0" * 64}), encoding="utf-8"
    )
    assert cli("run", "--database", "--yes") == 3
    (tmp_path / f"{shipped.id}.json").write_text(good, encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE corpus.content_items SET language = 'en' WHERE item_id = :i"),
            {"i": items["2"]},
        )
    before = _snapshot(engine)
    assert cli("run", "--database", "--yes") == 4
    assert "prod's plan differs from the clone's in 1 item(s)" in capsys.readouterr().err
    assert _snapshot(engine) == before
    assert _ledger(engine, shipped.id) is None


def test_cli_a_measure_that_differs_exits_5_and_names_the_undo(
    engine: Engine,
    shipped: Retitle,
    scoreboard: list[int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The shipped outcome's deltas are the fixture's empty ones: the apply's +1 is not measured.
    assert cli("run", "--database", "--yes") == 5
    err = capsys.readouterr().err
    assert "applied, and the outcome is not the measured one (1)" in err
    assert f"upgrade undo --migration {shipped.id}" in err
    assert _ledger(engine, shipped.id).status == "applied"


def test_cli_a_dry_run_whose_measure_differs_exits_5_and_applies_nothing(
    engine: Engine,
    shipped: Retitle,
    scoreboard: list[int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _snapshot(engine)
    assert cli("run", "--database", "--dry-run") == 5
    err = capsys.readouterr().err
    assert "dry run: the outcome would not be the measured one (1)" in err
    assert _snapshot(engine) == before
    assert _ledger(engine, shipped.id) is None


def test_cli_a_verify_that_fails_exits_1_and_applies_nothing(
    engine: Engine,
    shipped: Retitle,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(shipped, "verify_rows", lambda conn: (False, "deliberately wrong"))
    before = _snapshot(engine)
    assert cli("run", "--database", "--yes") == 1
    assert "VERIFY FAILED, rolled back" in capsys.readouterr().err
    assert _snapshot(engine) == before
    assert _ledger(engine, shipped.id) is None


def test_the_gate_script_refuses_a_migration_whose_verify_fails(
    engine: Engine, items: dict[str, int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[4] / "scripts" / "measure" / "corpus_change_gate.py"
    spec = importlib.util.spec_from_file_location("corpus_change_gate_rows_verify", script)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    out = tmp_path / "gate.json"
    assert gate.rows_gate(WrongVerify(items), worker_database_url() or "", "snap-test", out) == 1
    assert "deliberately wrong" in capsys.readouterr().err
    assert not out.exists()


def test_cli_undo_refused_exits_1_and_lists_the_rows(
    engine: Engine,
    shipped: Retitle,
    items: dict[str, int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli("run", "--database", "--yes") == 0
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE corpus.content_items SET title = 'Edited' WHERE item_id = :i"),
            {"i": items["1"]},
        )
    capsys.readouterr()
    assert cli("undo", "--migration", shipped.id, "--yes") == 1
    err = capsys.readouterr().err
    assert f"changed since the apply: content_items [{items['1']}]" in err
    assert _ledger(engine, shipped.id).status == "applied"


def test_cli_file_commands_are_unchanged(tmp_path: Path) -> None:
    (tmp_path / "feeds").mkdir()
    assert cli("list", "--corpus-dir", str(tmp_path)) == 0
    args = cli_handlers.parse_upgrade_argv(["run", "--corpus-dir", str(tmp_path)])
    assert args.database is False


class WrongVerify(Retitle):
    """Retitle whose verify says no."""

    id = "9002_test_wrong_verify"

    def verify_rows(self, conn: Connection) -> tuple[bool, str]:
        return False, "deliberately wrong"


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_verify_that_fails_rolls_the_apply_back_and_is_a_failed_run(
    engine: Engine, items: dict[str, int], dry_run: bool
) -> None:
    before, revision = _snapshot(engine), _revision(engine)
    with pytest.raises(rows_mod.VerifyFailed, match="deliberately wrong"):
        run_migration(engine, WrongVerify(items), dry_run=dry_run)
    assert _snapshot(engine) == before
    assert _receipts(engine) == []
    assert _ledger(engine, WrongVerify.id) is None
    assert _revision(engine) == revision
    assert _runs(engine) == [("dry_run" if dry_run else "apply", "failed")]


class DropItem(RowMigration):
    """Deletes item 1, whose card cascades from it."""

    id = "9003_test_drop_item"
    to_version = "9.0.3"

    def __init__(self, items: dict[str, int], children_first: bool = False) -> None:
        self.items = items
        self.children_first = children_first

    def apply_rows(self, ctx: RowContext) -> RowResult:
        one = self.items["1"]
        if self.children_first:
            ctx.rows.delete("item_cards", {"item_id": one})
        ctx.rows.delete("content_items", {"item_id": one})
        return RowResult(item_ids={one}, message="dropped", counts={"items": 1})

    def verify_rows(self, conn: Connection) -> tuple[bool, str]:
        return True, "dropped"


def test_a_delete_that_would_cascade_is_refused_and_writes_nothing(
    engine: Engine, items: dict[str, int]
) -> None:
    before = _snapshot(engine)
    with pytest.raises(RowsRefused, match="item_cards"):
        run_migration(engine, DropItem(items))
    assert _snapshot(engine) == before
    assert _receipts(engine) == []


def test_a_delete_after_its_children_undoes_whole(engine: Engine, items: dict[str, int]) -> None:
    before = _snapshot(engine)
    run_migration(engine, DropItem(items, children_first=True))
    assert ("item_cards", items["1"]) not in _snapshot(engine)
    undo_migration(engine, DropItem.id)
    assert _snapshot(engine) == before


class NoOp(RowMigration):
    """Reads a row and writes nothing."""

    id = "9004_test_no_op"
    to_version = "9.0.4"

    def __init__(self, items: dict[str, int]) -> None:
        self.items = items

    def apply_rows(self, ctx: RowContext) -> RowResult:
        ctx.rows.update("content_items", {"item_id": self.items["2"]}, {"title": "Episode 2"})
        return RowResult(item_ids=set(), message="nothing to do", counts={})

    def verify_rows(self, conn: Connection) -> tuple[bool, str]:
        return True, "nothing to do"


def test_an_apply_that_changes_nothing_does_not_bump_the_revision(
    engine: Engine, items: dict[str, int]
) -> None:
    revision = _revision(engine)
    out = run_migration(engine, NoOp(items))
    assert out.receipts == 0
    assert out.result["corpus_revision_bumped"] is False
    assert _revision(engine) == revision
