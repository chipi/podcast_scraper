"""End-of-run local-audio eviction after cold offload (#1787, epic #1788).

With ``audio_storage_backend='remote'`` each episode's raw audio is uploaded to cold
storage as it is processed (``episode_processor`` -> ``store_via``, #1199). The audio
archive is INTERNAL-ONLY (offline analysis + reproducible reprocessing); it is never
served (the player is bridge-only), so the local ``media/`` copy is a disposable working
copy. This module reclaims that copy once it is safely in cold — turning "local audio
grows every run" into "local audio trends to ~0 at rest".

Safety contract (this DELETES corpus audio — read before touching):

* **Confirmed-in-cold gate.** A media file is evicted ONLY when its episode's audio is
  confirmed present in the cold backend (``already_archived`` by GUID). A file whose GUID
  we cannot resolve, or whose audio is not (yet) in cold, is KEPT. Eviction never destroys
  the only copy of an episode's audio.
* **``media/`` only.** Only files under ``<run_dir>/media/`` are ever removed, resolved
  from each episode's own ``content.audio_relpath`` and re-checked to sit under
  ``media/``. Transcripts, derivations, ``.podcast_scraper/corpus-art/`` and
  ``.podcast_scraper/audio-archive-provenance.jsonl`` are never candidates.
* **Best-effort, non-fatal.** A backend hiccup logs at ERROR and keeps the file; eviction
  is a finalize-time reclaim and must never break a run.

A ``dry_run`` pass reports what *would* be evicted without deleting anything.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

CORPUS_MEDIA_DIR = "media"
_METADATA_SUFFIX = ".metadata.json"


@dataclass
class EvictReport:
    """What an eviction pass did (or would do, under ``dry_run``)."""

    evicted: int = 0
    bytes_freed: int = 0
    kept_not_in_cold: int = 0
    kept_no_guid: int = 0
    dry_run: bool = False

    def merge(self, other: "EvictReport") -> None:
        self.evicted += other.evicted
        self.bytes_freed += other.bytes_freed
        self.kept_not_in_cold += other.kept_not_in_cold
        self.kept_no_guid += other.kept_no_guid

    def summary(self) -> str:
        verb = "would evict" if self.dry_run else "evicted"
        gb = self.bytes_freed / 1e9
        return (
            f"audio eviction: {verb} {self.evicted} file(s) ({gb:.2f} GB); "
            f"kept {self.kept_not_in_cold} not-yet-in-cold, "
            f"{self.kept_no_guid} without a resolvable GUID"
        )


def _metadata_dir(run_dir: str) -> Path:
    return Path(run_dir) / "metadata"


def _media_root(run_dir: str) -> Path:
    return Path(run_dir) / CORPUS_MEDIA_DIR


def _safe_media_path(run_dir: str, audio_relpath: str) -> Optional[Path]:
    """Resolve ``audio_relpath`` under the run's ``media/`` — or None if it escapes it.

    ``audio_relpath`` is corpus-relative (``media/<stem>.<ext>``). The realpath must sit
    inside ``<run_dir>/media/`` so a crafted or malformed relpath can never point the
    delete at a transcript, the provenance file, or anywhere outside media/.
    """
    if not audio_relpath:
        return None
    rel = str(audio_relpath).strip().replace("\\", "/").lstrip("/")
    candidate = Path(run_dir) / rel
    try:
        real = os.path.realpath(candidate)
        media_root_real = os.path.realpath(_media_root(run_dir))
    except OSError:
        return None
    if real != media_root_real and not real.startswith(media_root_real + os.sep):
        return None
    return Path(real)


def _iter_run_episode_audio(run_dir: str) -> Iterator[Tuple[str, Path]]:
    """Yield ``(guid, media_abs_path)`` for each episode record whose media file exists.

    Reads the authoritative ``metadata/*.metadata.json`` records rather than scanning
    ``media/`` blindly, so eviction only ever considers audio the run actually produced.
    """
    md = _metadata_dir(run_dir)
    if not md.is_dir():
        return
    for meta_path in sorted(md.glob(f"*{_METADATA_SUFFIX}")):
        try:
            doc: Any = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("audio eviction: unreadable metadata %s: %s", meta_path, exc)
            continue
        if not isinstance(doc, dict):
            continue
        episode = doc.get("episode") or {}
        content = doc.get("content") or {}
        guid = str(episode.get("guid") or "").strip()
        audio_relpath = str(content.get("audio_relpath") or "").strip()
        if not guid or not audio_relpath:
            continue
        media_abs = _safe_media_path(run_dir, audio_relpath)
        if media_abs is None or not media_abs.is_file():
            continue
        yield guid, media_abs


def evict_run_dir(run_dir: str, backend: Any, *, dry_run: bool = False) -> EvictReport:
    """Evict local ``media/`` audio for one run dir, gated on cold-storage presence.

    For each episode with a resolvable GUID and an existing media file: if the cold
    backend already holds the audio (``already_archived``), delete the local copy (or,
    under ``dry_run``, only count it). Anything unconfirmed is kept.
    """
    from .backfill import already_archived

    report = EvictReport(dry_run=dry_run)
    if backend is None:
        return report

    for guid, media_abs in _iter_run_episode_audio(run_dir):
        try:
            in_cold = already_archived(backend, guid) is not None
        except Exception as exc:  # noqa: BLE001 - a backend hiccup must never delete or crash
            logger.error("audio eviction: cold-presence check failed guid=%s: %s", guid, exc)
            report.kept_not_in_cold += 1
            continue

        if not in_cold:
            report.kept_not_in_cold += 1
            continue

        try:
            size = media_abs.stat().st_size
        except OSError:
            size = 0

        if dry_run:
            report.evicted += 1
            report.bytes_freed += size
            continue

        try:
            media_abs.unlink()
            report.evicted += 1
            report.bytes_freed += size
            logger.debug("audio eviction: removed %s (guid=%s, in cold)", media_abs, guid)
        except OSError as exc:
            logger.error("audio eviction: failed to remove %s: %s", media_abs, exc)
            report.kept_not_in_cold += 1

    if report.evicted or report.kept_not_in_cold:
        logger.info("audio eviction (%s): %s", os.path.basename(run_dir), report.summary())
    return report


def _find_run_dirs(output_dir: str) -> List[str]:
    """Every run dir under ``output_dir`` — a dir that has both ``metadata/`` and ``media/``."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    seen: List[str] = []
    for media_dir in root.glob("**/media"):
        if not media_dir.is_dir():
            continue
        run_dir = media_dir.parent
        if (run_dir / "metadata").is_dir():
            seen.append(str(run_dir))
    return sorted(set(seen))


def sweep_corpus(output_dir: str, backend: Any, *, dry_run: bool = False) -> EvictReport:
    """Start-of-run orphan sweep: evict local audio already in cold across every run dir.

    Covers audio a crashed / killed run left behind (its finalize eviction never ran — the
    2026-08-18 incident shape). Idempotent: a file already gone or not-in-cold is a no-op.
    """
    total = EvictReport(dry_run=dry_run)
    if backend is None:
        return total
    for run_dir in _find_run_dirs(output_dir):
        total.merge(evict_run_dir(run_dir, backend, dry_run=dry_run))
    if total.evicted or total.kept_not_in_cold:
        logger.info("audio eviction sweep over %s: %s", output_dir, total.summary())
    return total
