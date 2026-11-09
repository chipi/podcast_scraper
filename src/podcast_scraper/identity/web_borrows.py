"""Which member row of a joined person's group lends its bio or hosted photo to the others (moved
verbatim from ``server.relational_view``, E04 S3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _group_borrows(root: Path, rows: list[Any]) -> tuple[dict[str, str], dict[str, str]]:
    """``(bio_from, photo_from)``: for each id of a joined person's group whose own row has no bio
    (no hosted photo), the member id whose row it may borrow (record s6f_person_web).

    A row is borrowed only when its ``name`` is the group's spelling: the enricher looked each id up
    by its own spelling, so a member's row under another name may describe someone else. The
    member with the most episodes wins. No valid ledger: nothing is borrowed (today's behaviour)."""
    from podcast_scraper.identity.ledger import _fold, read_ledger

    ledger, _state = read_ledger(root)
    if ledger is None:
        return {}, {}
    by_id = {str(r.get("person_id")): r for r in rows if isinstance(r, dict) and r.get("person_id")}

    def has_bio(pid: str) -> bool:
        bio = (by_id.get(pid) or {}).get("bio")
        return isinstance(bio, str) and bool(bio.strip())

    def has_photo(pid: str) -> bool:
        return bool((by_id.get(pid) or {}).get("image_hosted"))

    bio_from: dict[str, str] = {}
    photo_from: dict[str, str] = {}
    for cluster in (ledger.get("clusters_payload") or {}).get("clusters") or []:
        canonical = str(cluster.get("canonical_id") or "")
        shown = cluster.get("display_name") or cluster.get("canonical_name")
        if not canonical.startswith("person:") or not shown:
            continue
        members = sorted(
            (m for m in cluster.get("members") or [] if m.get("id")),
            key=lambda m: (-int(m.get("episode_count") or 0), str(m["id"])),
        )
        ids = [canonical] + [str(m["id"]) for m in members if m["id"] != canonical]
        named = [
            i for i in ids if _fold(str((by_id.get(i) or {}).get("name") or "")) == _fold(shown)
        ]
        for kind, has, out in (("bio", has_bio, bio_from), ("photo", has_photo, photo_from)):
            source = next((i for i in named if has(i)), None)
            if source is None:
                continue
            for i in ids:
                if i != source and not has(i):
                    out[i] = source
    return bio_from, photo_from
