"""Content-level duplicate detection for episode audio (#1656).

``skip_existing`` is GUID-keyed, so a feed that republishes the same content under a new GUID —
a re-run, a "best of", a corrected upload, a feed migration — used to produce a second full
episode: a second transcription bill, a second set of GI/KG artifacts, and two corpus entries a
consumer sees as distinct. This module is the content-level check that identity keys cannot do:
the sha256 of the downloaded audio bytes, checked AFTER download (bandwidth is already spent;
the fingerprint needs the bytes) and BEFORE transcription (where the money is).

Two layers, because duplicates arrive two ways:

- **Persistent index** (``.podcast_scraper/audio-fingerprints.json`` at the corpus root):
  digest → the episode that already produced a transcript from these exact bytes. Entries are
  recorded only AFTER a transcript is successfully saved, so a failed transcription leaves no
  claim behind and the retry proceeds normally.
- **In-process pending set**: two identical episodes in the SAME run's work queue would both
  miss the persistent index (neither has finished transcribing). The first to pass the gate
  claims the digest in memory; the second skips. If the first then fails, the second stays
  skipped for THIS run only — it was never marked done, so the next run retries it against the
  (still-empty) persistent index and transcribes. Self-healing beats cross-thread bookkeeping.

A duplicate is the same digest under a DIFFERENT identity (GUID, falling back to media URL).
The same identity re-encountering its own digest is a retry and must proceed.

Every failure path degrades to "no duplicate found" with a warning: this gate exists to save
money, and a gate that can break ingestion costs more than it saves. Cross-PROCESS writers are
not coordinated (last writer wins) — prod serializes corpus touchers via the ``prod-corpus``
concurrency group, and a lost index entry only means one duplicate goes undetected later.

KNOWN NOT COVERED: episodes ingested via publisher transcript URLs (no ASR bill to protect) are
neither fingerprinted nor gated, and the pre-existing corpus has no fingerprints (its audio is
in cold storage; hashing it would mean pulling every byte back down).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

INDEX_RELPATH = os.path.join(".podcast_scraper", "audio-fingerprints.json")

_LOCK = threading.Lock()
#: (corpus root, digest) -> identity of the in-flight episode that claimed it (this process
#: only). Root-scoped: a digest claimed for one corpus says nothing about another.
_PENDING: Dict[Tuple[str, str], str] = {}

#: Files below this size are never fingerprinted. Content identity only means something for a
#: payload that could plausibly BE an episode; below this, identical bytes are far more likely a
#: shared failure artifact — a CDN error page, a truncated fragment (#1834) — and dup-skipping on
#: those would wrongly mark a real episode as already-transcribed because another episode's
#: download failed the same way. 256 KiB ≈ 16 seconds of 128 kbps audio; every real episode
#: clears it by orders of magnitude.
MIN_FINGERPRINT_BYTES = 256 * 1024


def eligible_for_fingerprint(path: str) -> bool:
    """Is this file big enough that its content hash identifies an EPISODE, not an artifact?"""
    try:
        return os.path.getsize(path) >= MIN_FINGERPRINT_BYTES
    except OSError:
        return False


def resolve_index_root(cfg: Any, effective_output_dir: Optional[str]) -> Optional[str]:
    """Corpus root for the index — same precedence as the audio cache (corpus-wide > run dir)."""
    root = getattr(cfg, "output_dir", None) or effective_output_dir
    return str(root) if root else None


def episode_identity(guid: Optional[str], media_url: Optional[str]) -> Optional[str]:
    """The stable identity a duplicate is judged AGAINST — guid first, media URL as fallback."""
    for candidate in (guid, media_url):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def sha256_file(path: str) -> Optional[str]:
    """Streaming sha256 of a file; None (never an exception) when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        logger.warning("audio fingerprint: cannot hash %s: %s", path, exc)
        return None


def _index_path(root: str) -> str:
    return os.path.join(root, INDEX_RELPATH)


def _load(root: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(_index_path(root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        # An unreadable index must not block ingestion — but say so, because every duplicate
        # goes undetected until it is repaired.
        logger.warning("audio fingerprint: index unreadable at %s: %s", _index_path(root), exc)
        return {}


def duplicate_of(root: Optional[str], digest: str, identity: str) -> Optional[Dict[str, Any]]:
    """The already-known episode holding these exact bytes, or None.

    Checks the in-process pending set first (same-run duplicates), then the persistent index.
    A hit under the SAME identity is a retry, not a duplicate. The pending claim is scoped to
    the CORPUS, not the process: identical bytes in two different corpora are two different
    episodes (one process serving multiple corpora is a test-suite reality, and a claim leaking
    across corpora starved fresh corpora of jobs).
    """
    if root is None:
        return None
    with _LOCK:
        pending_owner = _PENDING.get((root, digest))
    if pending_owner is not None and pending_owner != identity:
        return {"identity": pending_owner, "in_flight": True}
    entry = _load(root).get(digest)
    if entry is None or entry.get("identity") == identity:
        return None
    return entry


def claim(root: Optional[str], digest: str, identity: str) -> None:
    """Mark ``digest`` in-flight for this corpus so a same-run twin skips instead of billing."""
    if root is None:
        return
    with _LOCK:
        _PENDING.setdefault((root, digest), identity)


def record(
    root: Optional[str],
    digest: Optional[str],
    *,
    identity: Optional[str],
    feed_id: Optional[str] = None,
    episode_title: Optional[str] = None,
    transcript_path: Optional[str] = None,
) -> None:
    """Persist ``digest`` → episode once a transcript actually exists. Atomic, best-effort."""
    if not root or not isinstance(digest, str) or not digest or not identity:
        return
    entry = {
        "identity": identity,
        "feed_id": feed_id,
        "episode_title": episode_title,
        "transcript_path": transcript_path,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _LOCK:
        try:
            index = _load(root)
            index[digest] = entry
            path = _index_path(root)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".fingerprints-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(index, fh, ensure_ascii=False, indent=0, sort_keys=True)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("audio fingerprint: failed to record %s: %s", digest[:12], exc)
        finally:
            _PENDING.pop((root, digest), None)


def reset_pending_for_tests() -> None:
    """Clear the in-process pending set (tests only — production never needs it)."""
    with _LOCK:
        _PENDING.clear()
