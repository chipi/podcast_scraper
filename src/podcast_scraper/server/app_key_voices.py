"""Per-user "key voices" (wave-G) — the people most present in a user's own corpus.

The operator reframe of "cluster people" → **key voices**: a prominence surface, not community
detection. This is the PER-USER flavor — rank the people across the episodes the user has
heard∪captured (their `scope=mine` corpus) by how many of those episodes they appear in. Pure
surfacing over the shipped KG (deterministic, corpus-internal, zero external calls); the per-topic
flavor is served by the topic card's people + perspectives.

Bounded: the KG scan is capped at the user's most-RECENT episodes so a heavy listener can't make
the rail slow (resolving slugs to rows is O(1) each via the cached slug index, so the whole heard
set is cheaply ranked by recency before the cap; only the expensive KG load is bounded).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_relational_view import hosted_photo_urls
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_corpus import user_episode_set

#: Cap the per-episode KG scan (mirrors app_digest_sections' heard-scan bound).
_MAX_SCAN = 200


def key_voices_for_user(
    root: Path, data_dir: Path, user_id: str, *, limit: int = 8
) -> list[dict[str, Any]]:
    """The user's key voices, most-present first: ``[{id, kind, label, episode_count, image_url}]``.

    Empty when the user has heard nothing graph-carrying — a rail is a claim, so no data → no rail.
    ``image_url`` is the served hosted-photo route when the web enricher has one, else None.
    """
    if limit <= 0:
        return []
    heard = user_episode_set(root, data_dir, user_id)
    # Rank the scan by recency (newest-first ``sort_key``) rather than alphabetically by slug, so a
    # heavy listener's capped sample is their most-recent listening — not an arbitrary A–Z slice.
    rows = [(s, row) for s in heard if (row := resolve_slug(root, s)) is not None]
    rows.sort(key=lambda sr: sr[1].sort_key())
    photos = hosted_photo_urls(root)
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for _slug, row in rows[:_MAX_SCAN]:
        if not row.has_kg:
            continue
        persons, _orgs, _topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        for person in persons:
            counts[person.id] += 1
            labels.setdefault(person.id, person.label)
    return [
        {
            "id": pid,
            "kind": "person",
            "label": labels[pid],
            "episode_count": n,
            "image_url": photos.get(pid),
        }
        for pid, n in counts.most_common(limit)
    ]
