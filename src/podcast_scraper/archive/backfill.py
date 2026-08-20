"""``archive backfill`` — reconcile episode audio into the cold archive (#1631, v2 #55).

A **local-first reconcile**, not a blind re-download. For each episode, in order:

* **already in cold** (``already_archived`` by GUID) → nothing to fetch; the cleanup sweep
  reclaims any lingering local ``media/`` copy.
* **a local original exists** (the run's ``media/`` via ``audio_relpath``, or the GUID-keyed
  ``.podcast_scraper/audio-cache``) → **harvest** it: upload the *original* bytes to cold with
  no download. These are byte-identical to the audio that produced the transcript.
* **neither** → **download** from the publisher's enclosure URL, with bounded retry + backoff.

Why the local-first order matters: prod already holds hundreds of episodes' worth of local
audio from past runs (~65 GB). Re-downloading would throw those *originals* away for dynamic-ad
*re-encodes* (and lose anything rolled off), and — because the #1787 eviction only deletes what
is confirmed in cold — that local audio is otherwise stranded, neither preserved nor reclaimable.
Harvesting is the prerequisite for reclaiming it.

Two properties of the world this encodes rather than discovers at runtime:

* **Coverage is partial by nature.** Publishers truncate their feeds at wildly different
  depths, so an episode older than the window is simply not fetchable. That is ``rolled_off``
  — a reported outcome, not a failure and not something to retry.
* **Downloaded audio is not the original bytes.** Dynamic-ad-insertion feeds re-encode per
  request, so what a download returns is *not* the file that produced the existing transcript.
  Good enough to re-transcribe from; wrong for any WER-vs-original comparison. Downloaded
  (``stored``) entries are stamped ``byte_identical=False``; harvested originals, ``True``.

No transcription, no enrichment, no LLM calls — this is download/upload/evict only and must not
touch the pipeline's cost accounting.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: Identifies us to publishers. A bare urllib default on hundreds of requests across a handful
#: of hosts reads as a scraper and risks a block on feeds the corpus depends on.
USER_AGENT = "podcast-scraper-archive-backfill/1.0 (+https://github.com/chipi/podcast_scraper)"

#: Minimum seconds between requests to the SAME host. Backfill walks a whole corpus, which is
#: concentrated on very few CDNs, so the polite unit is per-host rather than global.
DEFAULT_RATE_LIMIT_S = 1.0

#: Per-episode fetch ceiling. A hung CDN connection must not stall the whole pass.
DEFAULT_TIMEOUT_S = 300

#: Download resilience (the compromise, not the pipeline's httpx stack — see the v2 spec).
#: A bounded retry with exponential backoff around the urllib fetch, which KEEPS the exc.code
#: the rolled-off vs failed classification depends on. 404/410 are never retried (they're gone).
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 2.0

#: HTTP statuses that mean "this episode is gone from the feed window", not "try again".
#: 410 Gone is explicit; 404 is what most CDNs actually return for an expired enclosure.
_ROLLED_OFF_STATUSES = frozenset({404, 410})

# Outcome vocabulary — closed, so a report can group by it.
STORED = "stored"  # downloaded from the publisher (re-encode; NOT byte-identical)
HARVESTED = "harvested"  # uploaded an existing LOCAL original (byte-identical; no download)
ALREADY_PRESENT = "already_present"
ROLLED_OFF = "rolled_off"
FETCH_FAILED = "fetch_failed"
NO_MEDIA_URL = "no_media_url"


@dataclass
class EpisodeOutcome:
    """What happened to one episode, and enough identity to act on it."""

    guid: str
    title: str
    feed_title: str
    outcome: str
    rel_key: Optional[str] = None
    bytes_stored: int = 0
    detail: Optional[str] = None


@dataclass
class BackfillReport:
    """Aggregate result. ``by_feed`` is what an operator actually reads."""

    outcomes: List[EpisodeOutcome] = field(default_factory=list)
    estimated_bytes: int = 0

    def counts(self) -> Dict[str, int]:
        """Outcome slug -> how many episodes ended in it, across every feed."""
        out: Dict[str, int] = {}
        for o in self.outcomes:
            out[o.outcome] = out.get(o.outcome, 0) + 1
        return out

    def by_feed(self) -> Dict[str, Dict[str, int]]:
        """Feed title -> its own outcome counts.

        The per-feed split is the operator-facing view: a single failing host shows up here as
        one feed's column going red, which a corpus-wide total would average away.
        """
        feeds: Dict[str, Dict[str, int]] = {}
        for o in self.outcomes:
            feeds.setdefault(o.feed_title, {})
            feeds[o.feed_title][o.outcome] = feeds[o.feed_title].get(o.outcome, 0) + 1
        return feeds

    @property
    def stored_bytes(self) -> int:
        return sum(o.bytes_stored for o in self.outcomes)


class HostRateLimiter:
    """Sleep just enough that consecutive hits on one host stay ``min_interval`` apart.

    Per-host rather than global: a corpus of 14 feeds resolves to a handful of CDNs, so a
    global limiter would either crawl needlessly across distinct hosts or hammer one of them.
    """

    def __init__(
        self,
        min_interval_s: float = DEFAULT_RATE_LIMIT_S,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._sleep = sleep
        self._clock = clock
        self._last: Dict[str, float] = {}

    def wait(self, url: str) -> None:
        """Block until this URL's host may be hit again; returns immediately for a new host."""
        if self.min_interval_s <= 0:
            return
        host = (urlsplit(url).hostname or "").lower()
        if not host:
            return
        previous = self._last.get(host)
        now = self._clock()
        if previous is not None:
            remaining = self.min_interval_s - (now - previous)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last[host] = now


def _provenance_path(corpus_dir: str) -> str:
    """Where recovered-audio provenance is recorded, beside the cache it describes."""
    return os.path.join(corpus_dir, ".podcast_scraper", "audio-archive-provenance.jsonl")


def _append_provenance(corpus_dir: str, row: Dict[str, Any]) -> None:
    """Append one provenance row to the corpus's audio-archive-provenance.jsonl."""
    path = _provenance_path(corpus_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_provenance(corpus_dir: str, outcome: EpisodeOutcome, *, source_url: str) -> None:
    """Append a breadcrumb marking this episode's audio as RE-FETCHED, not original.

    Load-bearing rather than decorative: dynamic-ad feeds re-encode per request, so a later
    reprocess comparing WER against the original transcript would be measuring the re-encode.
    Anything reading archived audio needs to be able to tell the two apart.
    """
    _append_provenance(
        corpus_dir,
        {
            "guid": outcome.guid,
            "rel_key": outcome.rel_key,
            "source_url": source_url,
            "bytes": outcome.bytes_stored,
            "origin": "backfill_refetch",
            "byte_identical_to_transcribed_audio": False,
            "note": (
                "Re-fetched from the publisher after the original was lost. Dynamic-ad feeds "
                "re-encode per request, so these bytes may differ from the audio that produced "
                "the existing transcript."
            ),
        },
    )


def record_pipeline_provenance(
    corpus_dir: str,
    *,
    guid: str,
    rel_key: Optional[str],
    source_url: str,
    byte_identical: bool = True,
) -> None:
    """Stamp audio the PIPELINE freshly downloaded + archived this run (#1789).

    The provenance writer used to fire only from ``archive backfill``, so the normal
    download/reprocess path recorded nothing and a corpus had ZERO provenance despite every
    episode's audio being archived. Recording here — at the download choke point, after a
    successful ``store_via`` — closes that gap: every pipeline-archived episode gets a breadcrumb.

    ``byte_identical`` MUST be False when the archive already held an object for this GUID and
    ``store_via`` therefore DEDUPED rather than uploading these bytes (advisor H1): in that case
    the cold object may be a different (dynamic-ad re-encoded) copy, so we cannot claim the
    archived bytes are the ones this run transcribed. Only a genuine fresh upload is byte-identical.
    """
    if not guid:
        return
    if byte_identical:
        origin = "pipeline_download"
        note = (
            "Downloaded from the publisher by the pipeline and transcribed from these "
            "bytes (the original audio for this run's transcript)."
        )
    else:
        origin = "pipeline_download_deduped"
        note = (
            "The pipeline downloaded + transcribed these bytes this run, but the archive already "
            "held an object for this GUID, so store_via deduped instead of uploading. The cold "
            "object may be a different (re-encoded) copy — NOT confirmed byte-identical."
        )
    _append_provenance(
        corpus_dir,
        {
            "guid": guid,
            "rel_key": rel_key,
            "source_url": source_url,
            "origin": origin,
            "byte_identical_to_transcribed_audio": bool(byte_identical),
            "note": note,
        },
    )


def record_harvest_provenance(
    corpus_dir: str, outcome: EpisodeOutcome, *, source_path: str
) -> None:
    """Stamp audio HARVESTED from a local original — byte-identical, NOT a re-fetch.

    The harvest path only runs when the cold backend is a MISS for this GUID, so ``store_via``
    genuinely uploads (no dedup collision) and the archived bytes ARE the local original that
    produced the transcript. That is the opposite of ``record_provenance`` (re-fetch, re-encoded)
    and worth recording distinctly so a later reprocess can trust these bytes for WER comparison.
    """
    _append_provenance(
        corpus_dir,
        {
            "guid": outcome.guid,
            "rel_key": outcome.rel_key,
            "source_path": source_path,
            "bytes": outcome.bytes_stored,
            "origin": "backfill_harvest_local",
            "byte_identical_to_transcribed_audio": True,
            "note": (
                "Uploaded the original local copy (audio-cache / run media/) that produced this "
                "transcript — byte-identical to the transcribed audio, not a publisher re-fetch."
            ),
        },
    )


def build_local_lookup(corpus_dir: str) -> Callable[[str], Optional[str]]:
    """Return ``guid -> local original audio path`` (or None), indexing the corpus once.

    Two sources, both holding the ORIGINAL downloaded bytes (unlike a publisher re-fetch):

    * the per-run ``media/`` files (via each record's ``content.audio_relpath``), which the
      cleanup sweep later reclaims once they are in cold; and
    * the GUID-keyed ``.podcast_scraper/audio-cache`` (#947), which is run-independent and
      survives ``media/`` eviction — the more durable fallback.

    ``media/`` is preferred when present (it is what the sweep reclaims), the cache is the
    fallback. Both are verified to exist and be non-empty before being offered as a source.
    """
    from pathlib import Path

    from ..utils.audio_cache import IN_CORPUS_AUDIO_CACHE_REL, lookup_by_guid
    from .offload import _find_run_dirs, _iter_run_episode_audio

    media_index: Dict[str, str] = {}
    for run_dir in _find_run_dirs(corpus_dir):
        for guid, media_abs in _iter_run_episode_audio(run_dir):
            media_index.setdefault(guid, str(media_abs))

    cache_root = Path(corpus_dir) / IN_CORPUS_AUDIO_CACHE_REL

    def _lookup(guid: str) -> Optional[str]:
        path = media_index.get(guid)
        if path:
            try:
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    return path
            except OSError:
                pass
        # lookup_by_guid already verifies the cache entry exists and is non-empty.
        return lookup_by_guid(cache_root, guid)

    return _lookup


def already_archived(backend: Any, guid: str) -> Optional[str]:
    """The archive key already holding this episode, or None.

    Probes the same extension list the pipeline writes with, so an episode archived as
    ``.m4a`` is not re-fetched just because we guessed ``.mp3`` from the URL.
    """
    from ..utils.audio_cache import _LOOKUP_EXTENSIONS, rel_key_for_guid

    for ext in _LOOKUP_EXTENSIONS:
        rel = rel_key_for_guid(guid, ext)
        if rel is None:
            return None
        try:
            if backend.exists(rel):
                return rel
        except Exception as exc:  # noqa: BLE001 — one bad probe must not abort the pass
            logger.warning("archive backfill: exists() failed guid=%s ext=%s: %s", guid, ext, exc)
            return None
    return None


def _download(url: str, dest_path: str, *, timeout_s: int, opener: Any = None) -> int:
    """Fetch ``url`` to ``dest_path``. Returns bytes written. Raises on HTTP/transport error."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    open_fn = opener or urllib.request.urlopen
    written = 0
    with open_fn(req, timeout=timeout_s) as resp, open(dest_path, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
    return written


def _download_with_retry(
    url: str,
    dest_path: str,
    *,
    timeout_s: int,
    opener: Any = None,
    max_attempts: int = DEFAULT_MAX_RETRIES,
    backoff_base_s: float = DEFAULT_RETRY_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """``_download`` with bounded exponential-backoff retry. Raises the last error on give-up.

    Deliberately NOT the pipeline's httpx downloader: this keeps the urllib error (whose
    ``.code`` drives the rolled-off vs failed split). A 404/410 is re-raised immediately — the
    episode is gone from the window, retrying only wastes a request. Everything else (5xx,
    timeout, transport error) is transient and retried with a ``backoff_base * 2**n`` pause.
    """
    attempts = max(1, int(max_attempts))
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return _download(url, dest_path, timeout_s=timeout_s, opener=opener)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — decide retry vs give-up, then re-raise to classify
            status = getattr(exc, "code", None) or getattr(exc, "status", None)
            if status in _ROLLED_OFF_STATUSES:
                raise  # gone, not flaky — let the caller classify it as rolled_off
            last_exc = exc
            if attempt < attempts:
                sleep(backoff_base_s * (2 ** (attempt - 1)))
    assert last_exc is not None  # loop ran >=1 time and never returned -> an exception was caught
    raise last_exc


def backfill_episode(
    episode: Dict[str, Any],
    backend: Any,
    *,
    corpus_dir: str,
    force: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    limiter: Optional[HostRateLimiter] = None,
    opener: Any = None,
    local_lookup: Optional[Callable[[str], Optional[str]]] = None,
    max_attempts: int = DEFAULT_MAX_RETRIES,
) -> EpisodeOutcome:
    """Reconcile one episode's audio into cold. Never raises — always a classified outcome.

    Order (local-first): already in cold -> nothing to fetch; else a local original exists ->
    HARVEST it (upload the byte-identical bytes, no download); else DOWNLOAD from the publisher
    (with retry/backoff). The cleanup sweep reclaims local copies confirmed in cold afterwards.
    """
    from ..utils.audio_cache import store_via

    guid = str(episode.get("guid") or "")
    title = str(episode.get("title") or "")
    feed_title = str(episode.get("feed_title") or "unknown feed")
    media_url = str(episode.get("media_url") or "")

    def _out(outcome: str, **kw: Any) -> EpisodeOutcome:
        return EpisodeOutcome(guid=guid, title=title, feed_title=feed_title, outcome=outcome, **kw)

    if not force:
        existing = already_archived(backend, guid)
        if existing is not None:
            # In cold already; its lingering local media/ copy (if any) is reclaimed by the sweep.
            return _out(ALREADY_PRESENT, rel_key=existing)

    # HARVEST: a local original beats a re-download — original bytes, no network, no rolled-off.
    local_path = local_lookup(guid) if (local_lookup is not None and guid) else None
    if local_path:
        try:
            local_size = os.path.getsize(local_path)
        except OSError:
            local_size = 0
        if local_size > 0:
            rel = store_via(backend, guid, local_path)
            if rel is not None:
                outcome = _out(HARVESTED, rel_key=rel, bytes_stored=local_size)
                record_harvest_provenance(corpus_dir, outcome, source_path=local_path)
                return outcome
            # Upload of the local original failed — fall through to a publisher download.
            logger.warning("archive backfill: harvest upload failed guid=%s; trying download", guid)

    if not media_url:
        return _out(NO_MEDIA_URL, detail="episode metadata carries no media_url")

    if limiter is not None:
        limiter.wait(media_url)

    from .cli_handlers import _ext_from

    ext = _ext_from(media_url, str(episode.get("media_type") or ""))
    tmp_dir = tempfile.mkdtemp(prefix="ps-backfill-")
    tmp_path = os.path.join(tmp_dir, f"audio{ext}")
    try:
        try:
            written = _download_with_retry(
                media_url, tmp_path, timeout_s=timeout_s, opener=opener, max_attempts=max_attempts
            )
        except Exception as exc:  # noqa: BLE001 — classify, never propagate
            status = getattr(exc, "code", None) or getattr(exc, "status", None)
            if status in _ROLLED_OFF_STATUSES:
                # Expected and normal: the episode aged out of the publisher's window.
                return _out(ROLLED_OFF, detail=f"HTTP {status}")
            return _out(FETCH_FAILED, detail=f"{type(exc).__name__}: {exc}"[:200])

        if written <= 0:
            return _out(FETCH_FAILED, detail="empty response body")

        rel = store_via(backend, guid, tmp_path)
        if rel is None:
            return _out(FETCH_FAILED, detail="storage backend rejected the upload")

        outcome = _out(STORED, rel_key=rel, bytes_stored=written)
        record_provenance(corpus_dir, outcome, source_url=media_url)
        return outcome
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            logger.debug("archive backfill: temp cleanup failed for %s", tmp_dir)


def plan_backfill(
    episodes: Iterable[Dict[str, Any]],
    backend: Any,
    *,
    local_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> BackfillReport:
    """Classify every episode WITHOUT fetching — the ``--dry-run`` half.

    Reports the split per feed — already in cold / harvest from a local original / download
    from the publisher — so the size and shape of the pass are known before any bytes move.
    ``local_lookup`` (when given) is what distinguishes ``harvested`` (a local original exists,
    no download) from ``stored`` (must be fetched). Download size comes from the ``?size=``
    enclosure hint; it is absent often enough that the estimate is a floor, not a total.
    """
    from .cli_handlers import _SIZE_RE

    report = BackfillReport()
    for ep in episodes:
        guid = str(ep.get("guid") or "")
        title = str(ep.get("title") or "")
        feed_title = str(ep.get("feed_title") or "unknown feed")
        media_url = str(ep.get("media_url") or "")

        existing = already_archived(backend, guid)
        if existing is not None:
            report.outcomes.append(
                EpisodeOutcome(guid, title, feed_title, ALREADY_PRESENT, rel_key=existing)
            )
            continue
        local_path = local_lookup(guid) if (local_lookup is not None and guid) else None
        if local_path:
            try:
                local_bytes = os.path.getsize(local_path)
            except OSError:
                local_bytes = 0
            report.outcomes.append(
                EpisodeOutcome(guid, title, feed_title, HARVESTED, bytes_stored=local_bytes)
            )
            continue
        if not media_url:
            report.outcomes.append(
                EpisodeOutcome(guid, title, feed_title, NO_MEDIA_URL, detail="no media_url")
            )
            continue
        # Recoverable-by-download as far as a dry run can tell. Whether it is actually still
        # served is only knowable by asking, which a dry run must not do.
        report.outcomes.append(EpisodeOutcome(guid, title, feed_title, STORED))
        if m := _SIZE_RE.search(media_url):
            report.estimated_bytes += int(m.group(1))
    return report


def format_dry_run(report: BackfillReport) -> str:
    """Operator-facing preview: per-feed table, then the totals that decide whether to run."""
    lines = ["archive backfill (dry-run) — nothing has been fetched", ""]
    feeds = report.by_feed()
    width = max(max((len(f) for f in feeds), default=4), 4)
    lines.append(f"  {'feed'.ljust(width)}  in-corpus  in-cold  move-local  download")
    for feed in sorted(feeds):
        c = feeds[feed]
        total = sum(c.values())
        in_cold = c.get(ALREADY_PRESENT, 0)
        move_local = c.get(HARVESTED, 0)
        download = c.get(STORED, 0)
        lines.append(
            f"  {feed.ljust(width)}  {total:>9}  {in_cold:>7}  {move_local:>10}  {download:>8}"
        )
    counts = report.counts()
    harvest_gb = sum(o.bytes_stored for o in report.outcomes if o.outcome == HARVESTED) / 1e9
    lines.append("")
    lines.append(
        f"  totals: {len(report.outcomes)} episode(s) — "
        f"{counts.get(ALREADY_PRESENT, 0)} already in cold, "
        f"{counts.get(HARVESTED, 0)} to move from local, "
        f"{counts.get(STORED, 0)} to download, "
        f"{counts.get(NO_MEDIA_URL, 0)} without a media_url"
    )
    if counts.get(HARVESTED, 0):
        lines.append(
            f"  move from local: {counts[HARVESTED]} original(s), ~{harvest_gb:.2f} GB "
            "(byte-identical — no download, no rolled-off risk)"
        )
    if report.estimated_bytes:
        lines.append(
            f"  estimated download: >= {report.estimated_bytes / 1e9:.2f} GB "
            "(floor — only counts enclosures that advertise a size)"
        )
    else:
        lines.append("  estimated download: unknown (no enclosure size hints in this corpus)")
    lines.append("")
    lines.append("  rolled-off episodes cannot be detected without fetching; they will be")
    lines.append("  reported as 'rolled_off' during the real run, which is a normal outcome.")
    return "\n".join(lines)


def format_result(report: BackfillReport) -> str:
    """Post-run summary. Leads with what was recovered, names what was not."""
    c = report.counts()
    lines = [
        "archive backfill: "
        f"{c.get(HARVESTED, 0)} moved from local, "
        f"{c.get(STORED, 0)} downloaded, "
        f"{c.get(ALREADY_PRESENT, 0)} already present, "
        f"{c.get(ROLLED_OFF, 0)} rolled off, "
        f"{c.get(FETCH_FAILED, 0)} failed, "
        f"{c.get(NO_MEDIA_URL, 0)} without media_url"
    ]
    harvested_bytes = sum(o.bytes_stored for o in report.outcomes if o.outcome == HARVESTED)
    downloaded_bytes = sum(o.bytes_stored for o in report.outcomes if o.outcome == STORED)
    if harvested_bytes:
        lines.append(
            f"  moved from local: {harvested_bytes / 1e9:.2f} GB (original bytes, byte-identical)"
        )
    if downloaded_bytes:
        lines.append(f"  downloaded: {downloaded_bytes / 1e9:.2f} GB (publisher re-encode)")
    rolled = [o for o in report.outcomes if o.outcome == ROLLED_OFF]
    if rolled:
        lines.append(
            f"  rolled off (unrecoverable — outside the publisher's feed window): {len(rolled)}"
        )
        for o in rolled[:10]:
            lines.append(f"    - [{o.feed_title}] {o.title[:60]}")
        if len(rolled) > 10:
            lines.append(f"    ... and {len(rolled) - 10} more")
    failed = [o for o in report.outcomes if o.outcome == FETCH_FAILED]
    if failed:
        lines.append(f"  failed (retryable — re-run to pick these up): {len(failed)}")
        for o in failed[:10]:
            lines.append(f"    - [{o.feed_title}] {o.title[:60]} — {o.detail}")
        if len(failed) > 10:
            lines.append(f"    ... and {len(failed) - 10} more")
    return "\n".join(lines)
