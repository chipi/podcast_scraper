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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

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
    kept_size_mismatch: int = 0
    kept_unlink_failed: int = 0
    # Orphan pass (#1834): media files NO metadata references. The metadata-driven iterator
    # cannot see them by design, which left 330 files (~19 GB) invisible to every sweep until
    # measured 2026-08-25. Counted separately — an orphan is a different animal from a
    # referenced file, and merging the counters would hide which pass reclaimed what.
    orphans_evicted: int = 0
    orphan_bytes_freed: int = 0
    orphans_kept: int = 0
    dry_run: bool = False
    # Wall-clock of the whole corpus sweep. Set once by ``sweep_corpus`` (per-run reports leave
    # it 0); ``merge`` deliberately does NOT sum it. #1808 — the sweep never timed itself, so a
    # killed run was indistinguishable from a slow one and nobody knew if it ever completes.
    duration_s: float = 0.0

    def merge(self, other: "EvictReport") -> None:
        """Accumulate another report's counters into this one (used by the corpus sweep)."""
        self.evicted += other.evicted
        self.bytes_freed += other.bytes_freed
        self.kept_not_in_cold += other.kept_not_in_cold
        self.kept_no_guid += other.kept_no_guid
        self.kept_size_mismatch += other.kept_size_mismatch
        self.kept_unlink_failed += other.kept_unlink_failed
        self.orphans_evicted += other.orphans_evicted
        self.orphan_bytes_freed += other.orphan_bytes_freed
        self.orphans_kept += other.orphans_kept

    @property
    def candidates(self) -> int:
        """Local media files examined = episodes still holding a local ``media/`` copy (#1808).

        This is the sweep's cost multiplier — one cold-backend round trip per candidate.
        Every examined file increments exactly one of the counters below (the iterator only
        yields episodes with a resolvable GUID and an existing media file), so their sum is the
        candidate count. It was never measured; a dry-run sweep now reports it directly.
        """
        return (
            self.evicted
            + self.kept_not_in_cold
            + self.kept_no_guid
            + self.kept_size_mismatch
            + self.kept_unlink_failed
        )

    def summary(self) -> str:
        """One-line human summary of what was evicted and what was kept (and why)."""
        verb = "would evict" if self.dry_run else "evicted"
        gb = self.bytes_freed / 1e9
        took = f" in {self.duration_s:.1f}s" if self.duration_s else ""
        return (
            f"audio eviction: examined {self.candidates} local media file(s); "
            f"{verb} {self.evicted} ({gb:.2f} GB); "
            f"kept {self.kept_not_in_cold} not-yet-in-cold, "
            f"{self.kept_no_guid} without a resolvable GUID, "
            f"{self.kept_size_mismatch} size-mismatch vs cold, "
            f"{self.kept_unlink_failed} unlink-failed"
            f"; orphans: {verb} {self.orphans_evicted} "
            f"({self.orphan_bytes_freed / 1e9:.2f} GB), kept {self.orphans_kept}{took}"
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
    """Evict local ``media/`` audio for one run dir, gated on cold-storage presence + SIZE.

    For each episode with a resolvable GUID and an existing media file, the local copy is
    deleted ONLY when the cold backend holds the audio (``already_archived``) AND the cold
    object's byte size equals the local file's size. The size match is the load-bearing guard
    (advisor H1): ``rclone`` ``upload`` dedupes by existence, so a re-download that produced
    DIFFERENT bytes (dynamic-ad re-encode) leaves cold holding the OLD object under this GUID —
    without the size check the delete would destroy the only copy of the bytes that made this
    run's transcript. On a mismatch (or an unknowable cold size) the file is KEPT + ERROR-logged.
    """
    from .backfill import already_archived

    report = EvictReport(dry_run=dry_run)
    if backend is None:
        return report

    for guid, media_abs in _iter_run_episode_audio(run_dir):
        try:
            cold_key = already_archived(backend, guid)
        except Exception as exc:  # noqa: BLE001 - a backend hiccup must never delete or crash
            logger.error("audio eviction: cold-presence check failed guid=%s: %s", guid, exc)
            report.kept_not_in_cold += 1
            continue

        if cold_key is None:
            report.kept_not_in_cold += 1
            continue

        try:
            local_size = media_abs.stat().st_size
        except OSError:
            local_size = 0

        # SIZE GUARD (H1): confirm the cold object is the SAME bytes as the local file before
        # deleting. size() returns None on any transport failure / absent object -> treat as
        # "cannot confirm" -> keep. A positive mismatch means cold holds a different encode.
        try:
            cold_size = backend.size(cold_key)
        except Exception as exc:  # noqa: BLE001 - never let a size probe delete or crash
            logger.error("audio eviction: cold size probe failed guid=%s: %s", guid, exc)
            cold_size = None

        if cold_size is None or cold_size != local_size:
            logger.error(
                "audio eviction: KEEP %s (guid=%s) — cold size %s != local %s (not byte-identical)",
                media_abs,
                guid,
                cold_size,
                local_size,
            )
            report.kept_size_mismatch += 1
            continue

        if dry_run:
            report.evicted += 1
            report.bytes_freed += local_size
            continue

        try:
            media_abs.unlink()
            report.evicted += 1
            report.bytes_freed += local_size
            logger.debug(
                "audio eviction: removed %s (guid=%s, size-matched in cold)", media_abs, guid
            )
        except OSError as exc:
            logger.error("audio eviction: failed to remove %s: %s", media_abs, exc)
            report.kept_unlink_failed += 1

    # Log the summary whenever the pass DID anything — including an all-unlink-failed run, so a
    # run dir where every eviction failed is not silent at the summary level (each failure also
    # ERROR-logs individually above). Genuinely empty run dirs (0 candidates) stay quiet.
    if report.candidates:
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


def _episode_guid_by_idx_prefix(run_dir: str, media_name: str) -> Optional[str]:
    """The episode GUID for an unreferenced media file, via its NNNN-index metadata sibling."""
    prefix = media_name.split(" - ", 1)[0].strip()
    if not prefix.isdigit():
        return None
    for meta_path in _metadata_dir(run_dir).glob(f"{prefix} - *{_METADATA_SUFFIX}"):
        try:
            doc: Any = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            guid = str((doc.get("episode") or {}).get("guid") or "").strip()
            if guid:
                return guid
    return None


def evict_orphan_media(run_dir: str, backend: Any, *, dry_run: bool = False) -> EvictReport:
    """Evict media files NO metadata references, when their EPISODE is archived in cold (#1834).

    The referenced-file pass (``evict_run_dir``) reads metadata and so can never see a file
    metadata does not point at — leftover download variants from episodes processed in multiple
    runs, or a killed run's debris. Measured 2026-08-25: 330 such files (~19 GB) invisible to
    every sweep.

    The criterion is deliberately the EPISODE, not the bytes: an orphan is an alternate
    ad-stitch of an archived episode, so its exact bytes exist nowhere else but carry no
    provenance value (no transcript derives from an unreferenced file). Anything whose episode
    cannot be resolved (no NNNN-prefixed metadata sibling) or whose GUID is not in cold is KEPT
    and logged — same error-and-keep philosophy as the H1 size guard.
    """
    from .backfill import already_archived

    report = EvictReport(dry_run=dry_run)
    if backend is None:
        return report
    referenced = {str(media_abs) for _guid, media_abs in _iter_run_episode_audio(run_dir)}
    media_root = _media_root(run_dir)
    if not media_root.is_dir():
        return report
    for media_abs in sorted(p for p in media_root.iterdir() if p.is_file()):
        if str(media_abs) in referenced:
            continue
        guid = _episode_guid_by_idx_prefix(run_dir, media_abs.name)
        cold_key = None
        if guid:
            try:
                cold_key = already_archived(backend, guid)
            except Exception as exc:  # noqa: BLE001 - a backend hiccup must never delete
                logger.error("orphan eviction: cold check failed guid=%s: %s", guid, exc)
        if not cold_key:
            report.orphans_kept += 1
            logger.info(
                "orphan eviction: KEEP %s (%s)",
                media_abs,
                "no resolvable episode" if not guid else "episode not in cold",
            )
            continue
        try:
            size = media_abs.stat().st_size
        except OSError:
            size = 0
        if dry_run:
            report.orphans_evicted += 1
            report.orphan_bytes_freed += size
            continue
        try:
            media_abs.unlink()
            report.orphans_evicted += 1
            report.orphan_bytes_freed += size
            logger.debug("orphan eviction: removed %s (guid=%s archived)", media_abs, guid)
        except OSError as exc:
            report.orphans_kept += 1
            logger.error("orphan eviction: failed to remove %s: %s", media_abs, exc)
    return report


def sweep_corpus(
    output_dir: str,
    backend: Any,
    *,
    dry_run: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> EvictReport:
    """On-demand orphan sweep: evict local audio already in cold across every run dir.

    Covers audio a crashed / killed run left behind (its finalize eviction never ran — the
    2026-08-18 incident shape). Idempotent: a file already gone or not-in-cold is a no-op.

    ON DEMAND, NOT ON THE RUN PATH. Until 2026-08-21 this was called at the start of every
    ``run_pipeline``, before the run applied its episode work-list, so a one-episode repair paid
    a whole-corpus cost. Cost is inherent here — one backend round trip per episode — which is
    exactly why it belongs behind an operator's deliberate dispatch rather than in front of one.

    ``on_progress`` is called once per run dir with a human-readable line. The in-run version
    reported NOTHING until it finished, so sixteen minutes of work were indistinguishable from a
    wedged process; a pass measured in minutes must say where it is while it runs.
    """
    total = EvictReport(dry_run=dry_run)
    if backend is None:
        return total
    started = time.monotonic()
    run_dirs = _find_run_dirs(output_dir)
    if on_progress:
        on_progress(f"{len(run_dirs)} run dir(s) under {output_dir}")
    for i, run_dir in enumerate(run_dirs, 1):
        one = evict_run_dir(run_dir, backend, dry_run=dry_run)
        one.merge(evict_orphan_media(run_dir, backend, dry_run=dry_run))
        total.merge(one)
        if on_progress:
            on_progress(f"[{i}/{len(run_dirs)}] {os.path.basename(run_dir)}: {one.summary()}")
    total.duration_s = time.monotonic() - started
    # Always log — even a 0-candidate sweep. #1808: the sweep used to emit NOTHING unless it
    # evicted, so its cost (candidate count) + wall-clock were invisible and a killed run looked
    # the same as a slow one. The measurement is the point of the log line now.
    logger.info("audio eviction sweep over %s: %s", output_dir, total.summary())
    return total
