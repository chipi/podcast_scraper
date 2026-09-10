"""Per-user "key voices" (wave-G) — the people most present in a user's own corpus.

The operator reframe of "cluster people" → **key voices**: a prominence surface, not community
detection. This is the PER-USER flavor — rank the people across the episodes the user has
heard∪captured (their `scope=mine` corpus) by how many of those episodes they appear in. Pure
surfacing over the shipped KG (deterministic, corpus-internal, zero external calls); the per-topic
flavor is served by the topic card's people + perspectives.

Bounded: the heard set is capped before the per-episode KG scan so a heavy listener can't make the
rail slow.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_corpus import user_episode_set

#: Cap the per-episode KG scan (mirrors app_digest_sections' heard-scan bound).
_MAX_SCAN = 200


def key_voices_for_user(
    root: Path, data_dir: Path, user_id: str, *, limit: int = 8
) -> list[dict[str, Any]]:
    """The user's key voices, most-present first: ``[{id, kind, label, episode_count}]``.

    Empty when the user has heard nothing graph-carrying — a rail is a claim, so no data → no rail.
    """
    if limit <= 0:
        return []
    heard = user_episode_set(root, data_dir, user_id)
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for slug in sorted(heard)[:_MAX_SCAN]:
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        persons, _orgs, _topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        for person in persons:
            counts[person.id] += 1
            labels.setdefault(person.id, person.label)
    return [
        {"id": pid, "kind": "person", "label": labels[pid], "episode_count": n}
        for pid, n in counts.most_common(limit)
    ]
