"""The delivery poll runs the retention DELETE every few minutes, not on every poll.

Prod (2026-11-06): 1,228 purge DELETEs in 5.4 h at 3.2 ms each, each deleting 0 rows. The filters
read ``envelope ->> 'type'``, which no index serves, and the delivery worker polls every few
seconds. Retention is a day at the shortest, so a purge a few minutes late changes nothing a person
sees.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, text, update

from podcast_scraper.db import ro_engine, rw_engine
from podcast_scraper.db.schema import outbox
from podcast_scraper.server import app_outbox_store
from tests._database import captured_statements, register_user

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"
_NO_DIR = Path(".")


@pytest.fixture(autouse=True)
def _account() -> None:
    register_user(_UID)
    app_outbox_store.reset_purge_clock()


def _envelope(eid: str, **over: Any) -> dict[str, Any]:
    env = {
        "id": eid,
        "user_id": _UID,
        "type": "digest",
        "channel": "email",
        "recipient": {"email": "u@x.com"},
    }
    env.update(over)
    return env


def _purges(seen: list[tuple[str, bool]]) -> int:
    return sum(1 for sql, _ in seen if sql.startswith("DELETE FROM") and "outbox" in sql)


def _age(envelope_id: str, days: float) -> None:
    with rw_engine().begin() as conn:
        conn.execute(
            update(outbox)
            .where(outbox.c.envelope_id == envelope_id)
            .values(updated_at=func.now() - text(f"interval '{days} days'"))
        )


def _ids() -> set[str]:
    with ro_engine().connect() as conn:
        return {r[0] for r in conn.execute(select(outbox.c.envelope_id))}


def test_polls_inside_the_interval_do_not_purge() -> None:
    every = app_outbox_store._PURGE_EVERY_S
    with captured_statements() as seen:
        for i in range(5):
            app_outbox_store.list_pending(_NO_DIR, channel="email", now=1_000_000 + i)
    assert _purges(seen) == 1
    with captured_statements() as seen:
        app_outbox_store.list_pending(_NO_DIR, channel="email", now=1_000_000 + every)
    assert _purges(seen) == 1


def test_a_clock_that_moves_back_purges_again() -> None:
    app_outbox_store.list_pending(_NO_DIR, channel="email", now=2_000_000)
    with captured_statements() as seen:
        app_outbox_store.list_pending(_NO_DIR, channel="email", now=1_000_000)
    assert _purges(seen) == 1


def test_a_failed_purge_is_retried_by_the_next_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(conn: Any, now: int) -> None:
        raise RuntimeError("purge failed")

    monkeypatch.setattr(app_outbox_store, "_purge_past_retention", boom)
    with pytest.raises(RuntimeError):
        app_outbox_store.list_pending(_NO_DIR, channel="email", now=1_000_000)
    monkeypatch.undo()
    with captured_statements() as seen:
        app_outbox_store.list_pending(_NO_DIR, channel="email", now=1_000_001)
    assert _purges(seen) == 1


def test_retention_still_removes_finished_envelopes_on_the_poll() -> None:
    app_outbox_store.enqueue(_NO_DIR, _envelope("old"))
    app_outbox_store.enqueue(_NO_DIR, _envelope("fresh"))
    app_outbox_store.enqueue(_NO_DIR, _envelope("pending-old"))
    app_outbox_store.enqueue(_NO_DIR, _envelope("link", type="auth_link"))
    for eid in ("old", "fresh", "link"):
        app_outbox_store.record_status(_NO_DIR, eid, "delivered")
    _age("old", 31)
    _age("fresh", 20)
    _age("pending-old", 90)  # only FINISHED envelopes age out
    _age("link", 2)  # a sign-in envelope goes after a day
    app_outbox_store.list_pending(_NO_DIR, channel="email", now=int(time.time()))
    assert _ids() == {"fresh", "pending-old"}


def test_an_envelope_that_ages_out_between_purges_goes_at_the_next_one() -> None:
    now = int(time.time())
    app_outbox_store.list_pending(_NO_DIR, channel="email", now=now)
    app_outbox_store.enqueue(_NO_DIR, _envelope("old"))
    app_outbox_store.record_status(_NO_DIR, "old", "delivered")
    _age("old", 31)
    app_outbox_store.list_pending(_NO_DIR, channel="email", now=now + 5)
    assert _ids() == {"old"}
    app_outbox_store.list_pending(
        _NO_DIR, channel="email", now=now + app_outbox_store._PURGE_EVERY_S
    )
    assert _ids() == set()
