"""Ranking-experiment telemetry (#11) — append-only impression + click log per user.

Records what the discovery feed **showed** (episode slugs in rank order + the ranking variant in
force) and what the user **clicked** (which slug, at which shown position), so experiments can
compare the configured rank against actual clicks and A/B-test ranking variants with real users.

One JSON line per event in ``<data_dir>/users/<id>/ranking_events.jsonl`` — append-only, never
rewrites history (same contract as the listen log); aggregation reads the whole small file.
Variant assignment is a **stable hash of the user id**, so a user stays in one bucket across
sessions rather than flip-flopping per request. Pairs with the ranking-signal registry
(``app_ranking_config``): a variant name maps to a :class:`RankingConfig`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _events_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / "ranking_events.jsonl"


def assign_variant(user_id: str, variants: Sequence[str]) -> str:
    """Deterministically bucket *user_id* into one of *variants* by a stable hash.

    Same user → same variant across sessions (A/B is per-user, not per-request). An empty or
    all-blank ``variants`` yields ``"default"``.
    """
    names = [v for v in variants if v and v.strip()]
    if not names:
        return "default"
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return names[int(digest, 16) % len(names)]


def _append(data_dir: Path, user_id: str, record: dict[str, Any]) -> None:
    path = _events_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_impressions(
    data_dir: Path, user_id: str, *, shown: Sequence[str], variant: str, ts: int
) -> None:
    """Log the ranked episode slugs the feed showed (list index = rank) under *variant*."""
    _append(
        data_dir,
        user_id,
        {
            "kind": "impression",
            "shown": [str(s) for s in shown],
            "variant": str(variant),
            "ts": int(ts),
        },
    )


def record_click(
    data_dir: Path, user_id: str, *, slug: str, position: int, variant: str, ts: int
) -> None:
    """Log a click on *slug* shown at *position* (0-based rank) under *variant*."""
    _append(
        data_dir,
        user_id,
        {
            "kind": "click",
            "slug": str(slug),
            "position": int(position),
            "variant": str(variant),
            "ts": int(ts),
        },
    )


def read_events(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All of one user's ranking events (chronological as written); skips corrupt lines."""
    path = _events_path(data_dir, user_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("kind"):
            out.append(rec)
    return out


__all__ = [
    "assign_variant",
    "record_impressions",
    "record_click",
    "read_events",
]
