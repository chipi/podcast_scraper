"""The audio dedup gate's fingerprint files as ``audio_fingerprints`` rows (tracker s-fingerprints).

The gate (``utils.audio_fingerprint``) keeps one index per corpus root it runs on: a feed's
``.podcast_scraper/audio-fingerprints.json``, digest to the episode transcribed from those bytes and
the identities skipped as its copies. A file's source is the source of the items under the same
feed directory; a file whose feed directory holds no item, or items of two sources, is refused and
listed, never guessed. An entry's ``item_id`` is the item of that source whose external id is the
entry's identity (a GUID, as the gate keys it), else null: an identity that is a media URL, or an
episode the corpus holds no row for.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from podcast_scraper.db.corpus_rows import _instant
from podcast_scraper.db.schema import audio_fingerprints, content_items
from podcast_scraper.utils.audio_fingerprint import INDEX_RELPATH

_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass
class FingerprintsWritten:
    """What :func:`import_fingerprints` wrote and refused."""

    rows: int = 0
    linked: int = 0
    duplicates: int = 0
    refused: list[tuple[str, str]] = field(default_factory=list)


def _feed_dir(relpath: str) -> Optional[str]:
    parts = relpath.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 and parts[0] == "feeds" else None


def _text(entry: Mapping[str, Any], key: str) -> Optional[str]:
    value = entry.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def import_fingerprints(
    conn: Connection, corpus_root: Path, sources_by_relpath: Mapping[str, int]
) -> FingerprintsWritten:
    """Write every feed's fingerprint file under ``corpus_root``; ``sources_by_relpath`` is each
    imported item's ``metadata_relpath`` to its ``source_id``."""
    written = FingerprintsWritten()
    by_feed: dict[str, set[int]] = {}
    for relpath, source_id in sources_by_relpath.items():
        feed = _feed_dir(relpath)
        if feed is not None:
            by_feed.setdefault(feed, set()).add(source_id)
    for path in sorted(Path(corpus_root).glob(f"feeds/*/{INDEX_RELPATH}")):
        relpath = path.relative_to(corpus_root).as_posix()
        feed = _feed_dir(relpath)
        found = by_feed.get(feed or "", set())
        if len(found) != 1:
            written.refused.append(
                (relpath, f"its feed directory holds items of {len(found)} sources")
            )
            continue
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            written.refused.append((relpath, f"unreadable: {exc}"))
            continue
        if not isinstance(index, dict):
            written.refused.append((relpath, "not an object"))
            continue
        (source_id,) = found
        items = dict(
            conn.execute(
                select(content_items.c.external_item_id, content_items.c.item_id).where(
                    content_items.c.source_id == source_id
                )
            ).all()
        )
        for digest, entry in sorted(index.items()):
            if not _DIGEST.fullmatch(digest):
                written.refused.append((f"{relpath}#{digest}", "the key is not a sha256 digest"))
                continue
            identity = _text(entry, "identity") if isinstance(entry, dict) else None
            if identity is None:
                written.refused.append((f"{relpath}#{digest}", "the entry names no identity"))
                continue
            duplicates = sorted(
                {str(i) for i in entry.get("duplicates") or [] if isinstance(i, str) and i}
            )
            item_id = items.get(identity)
            inserted = conn.execute(
                insert(audio_fingerprints)
                .values(
                    source_id=source_id,
                    digest=digest,
                    identity=identity,
                    item_id=item_id,
                    episode_title=_text(entry, "episode_title"),
                    transcript_path=_text(entry, "transcript_path"),
                    recorded_at=_instant(entry.get("recorded_at")),
                    duplicates=duplicates,
                )
                .on_conflict_do_nothing(index_elements=["source_id", "digest"])
                .returning(audio_fingerprints.c.digest)
            ).first()
            if inserted is None:
                continue
            written.rows += 1
            written.linked += int(item_id is not None)
            written.duplicates += len(duplicates)
    return written
