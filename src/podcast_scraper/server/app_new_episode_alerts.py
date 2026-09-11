"""New-episode alerts (wave-J) — the ``new_episodes`` notification type.

"A new episode dropped in a show you follow." The delta already exists: the digest's
:func:`app_digest_sections.new_in_follows_items` returns recent UNHEARD episodes in the user's
followed feeds, newest-first, graph-carrying. This module turns that delta into in-app
notifications, deduped per episode so a re-run never double-posts.

**Computed on-demand, not on ingest** (RFC-114 read-time-projection spirit): the sweep runs lazily
when the user loads their inbox, so there is no ingest hook to keep in sync. Idempotency is the
notification store's ``dedupe_key`` (``newep:<slug>``) — the sweep is safe to run on every load.

Channels: the **in-app** channel is delivered here (gated on the user's ``new_episodes`` ×
``in_app`` matrix cell via :func:`app_notifications_store.emit`). Email already reaches the user
through the weekly digest's *new-in-follows* section; a standalone push for this type would need
its own delivery envelope/template and is a separate follow-up.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_digest_sections, app_notifications_store

#: How many new-episode alerts to surface per sweep (deduped across sweeps).
_MAX_ALERTS = 10


def sweep_for_user(
    root: Path, data_dir: Path, user_id: str, *, limit: int = _MAX_ALERTS, now: int | None = None
) -> list[str]:
    """Emit an in-app ``new_episodes`` notification per recent unheard episode in followed shows.

    Idempotent via the store's ``dedupe_key`` and gated on the user's in-app consent for the type
    (``emit`` returns None when off / already-posted). Returns the ids newly emitted. Best-effort:
    a bad user id or empty follows simply yields nothing.
    """
    items = app_digest_sections.new_in_follows_items(root, data_dir, user_id, limit=limit)
    emitted: list[str] = []
    for item in items:
        slug = str(item.get("episode_slug") or "")
        if not slug:
            continue
        rec = app_notifications_store.emit(
            data_dir,
            user_id,
            ntype="new_episodes",
            title=str(item.get("episode_title") or "New episode"),
            body="New episode in a show you follow.",
            deep_link=item.get("deep_link"),
            meta={"episode_slug": slug},
            dedupe_key=f"newep:{slug}",
            now=now,
        )
        if rec is not None:
            emitted.append(str(rec["id"]))
    return emitted
