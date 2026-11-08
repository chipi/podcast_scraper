"""Deleting one account leaves another account's records alone when the two share an email (two
providers, one address). Before: the outbox purge and the throttle removal matched by address
alone, so they also took the other account's envelopes and its sign-in throttle."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from podcast_scraper.db import rw_engine
from podcast_scraper.db.schema import magic_link_requests, outbox
from podcast_scraper.server import app_account_deletion, app_magic_link, app_outbox_store
from podcast_scraper.server.app_user_store import User
from tests._database import register_user

ADDRESS = "shared@example.com"
GONE = "u_" + "a" * 24
STAYS = "u_" + "b" * 24


def _envelope(envelope_id: str, user_id: str | None) -> dict:
    env = {"id": envelope_id, "channel": "email", "recipient": {"email": ADDRESS}}
    if user_id:
        env["user_id"] = user_id
    return env


def _envelope_ids() -> set[str]:
    with rw_engine().connect() as conn:
        return {r for (r,) in conn.execute(select(outbox.c.envelope_id))}


def _throttled_hashes() -> set[str]:
    with rw_engine().connect() as conn:
        return {r for (r,) in conn.execute(select(magic_link_requests.c.email_hash))}


def _user(user_id: str) -> User:
    return User(user_id=user_id, email=ADDRESS, name="", provider="test", subject=user_id)


def _seed(tmp_path: Path) -> None:
    for uid in (GONE, STAYS):
        register_user(uid, email=ADDRESS)
    app_outbox_store.enqueue(tmp_path, _envelope("gone-digest", GONE))
    app_outbox_store.enqueue(tmp_path, _envelope("stays-digest", STAYS))
    app_outbox_store.enqueue(tmp_path, _envelope("sign-in", None))
    assert not app_magic_link.request_throttled(ADDRESS, now=10, min_interval_s=60)


def test_the_other_accounts_envelopes_and_throttle_stay(tmp_path: Path) -> None:
    _seed(tmp_path)
    report = app_account_deletion.delete_account(tmp_path, _user(GONE))
    assert _envelope_ids() == {"stays-digest", "sign-in"}
    assert _throttled_hashes() == {app_magic_link.email_hash(ADDRESS)}
    assert report.outbox_envelopes == 1
    assert report.throttle_markers == 0


def test_the_last_account_on_an_address_takes_its_sign_in_records(tmp_path: Path) -> None:
    _seed(tmp_path)
    app_account_deletion.delete_account(tmp_path, _user(STAYS))
    report = app_account_deletion.delete_account(tmp_path, _user(GONE))
    assert _envelope_ids() == set()
    assert _throttled_hashes() == set()
    assert report.outbox_envelopes == 2  # its own digest and the address's sign-in envelope
    assert report.throttle_markers == 1
