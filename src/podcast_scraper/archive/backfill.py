"""``archive backfill`` — recover audio for episodes ingested before the cache persisted (#1631).

The mirror of ``archive pull``: that resolves episode → guid → key and *downloads from* the
archive; this resolves the same way and *uploads into* it, fetching from the publisher's
enclosure URL for episodes the archive does not yet hold.

Why it exists: audio caching was on all along, but ``audio_cache_in_corpus`` defaulted to
``False``, so the cache resolved to ``/app/.cache/audio`` inside the container. The only
mounted volume is the corpus, and every prod job runs as ``docker compose run --rm`` — so each
run downloaded audio, cached it, and destroyed the cache on exit. Roughly 473 episodes predate
the fix and have no archived audio at all.

Two properties of the world this encodes rather than discovers at runtime:

* **Coverage is partial by nature.** Publishers truncate their feeds at wildly different
  depths, so an episode older than the window is simply not fetchable. That is ``rolled_off``
  — a reported outcome, not a failure and not something to retry.
* **Recovered audio is not the original bytes.** Dynamic-ad-insertion feeds re-encode per
  request, so what comes back is *not* the file that produced the existing transcript. Good
  enough to re-transcribe from; wrong for any WER-vs-original comparison. Recovered entries
  are stamped so a later reprocess can tell the difference.

No transcription, no enrichment, no LLM calls — this is download-and-store only and must not
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

#: HTTP statuses that mean "this episode is gone from the feed window", not "try again".
#: 410 Gone is explicit; 404 is what most CDNs actually return for an expired enclosure.
_ROLLED_OFF_STATUSES = frozenset({404, 410})

# Outcome vocabulary — closed, so a report can group by it.
STORED = "stored"
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


def backfill_episode(
    episode: Dict[str, Any],
    backend: Any,
    *,
    corpus_dir: str,
    force: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    limiter: Optional[HostRateLimiter] = None,
    opener: Any = None,
) -> EpisodeOutcome:
    """Archive one episode's audio. Never raises — every path returns a classified outcome."""
    from ..utils.audio_cache import store_via

    guid = str(episode.get("guid") or "")
    title = str(episode.get("title") or "")
    feed_title = str(episode.get("feed_title") or "unknown feed")
    media_url = str(episode.get("media_url") or "")

    def _out(outcome: str, **kw: Any) -> EpisodeOutcome:
        return EpisodeOutcome(guid=guid, title=title, feed_title=feed_title, outcome=outcome, **kw)

    if not media_url:
        return _out(NO_MEDIA_URL, detail="episode metadata carries no media_url")

    if not force:
        existing = already_archived(backend, guid)
        if existing is not None:
            return _out(ALREADY_PRESENT, rel_key=existing)

    if limiter is not None:
        limiter.wait(media_url)

    from .cli_handlers import _ext_from

    ext = _ext_from(media_url, str(episode.get("media_type") or ""))
    tmp_dir = tempfile.mkdtemp(prefix="ps-backfill-")
    tmp_path = os.path.join(tmp_dir, f"audio{ext}")
    try:
        try:
            written = _download(media_url, tmp_path, timeout_s=timeout_s, opener=opener)
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


def plan_backfill(episodes: Iterable[Dict[str, Any]], backend: Any) -> BackfillReport:
    """Classify every episode WITHOUT fetching — the ``--dry-run`` half.

    Reports what would happen per feed and an estimated download size, so the size of the pass
    is known before any bytes move. Size comes from the ``?size=`` enclosure hint publishers
    put on the URL; it is absent often enough that the estimate is a floor, not a total.
    """
    from .cli_handlers import _SIZE_RE

    report = BackfillReport()
    for ep in episodes:
        guid = str(ep.get("guid") or "")
        title = str(ep.get("title") or "")
        feed_title = str(ep.get("feed_title") or "unknown feed")
        media_url = str(ep.get("media_url") or "")

        if not media_url:
            report.outcomes.append(
                EpisodeOutcome(guid, title, feed_title, NO_MEDIA_URL, detail="no media_url")
            )
            continue
        existing = already_archived(backend, guid)
        if existing is not None:
            report.outcomes.append(
                EpisodeOutcome(guid, title, feed_title, ALREADY_PRESENT, rel_key=existing)
            )
            continue
        # Everything else is "recoverable now" as far as a dry run can tell. Whether it is
        # actually still served is only knowable by asking, which a dry run must not do.
        report.outcomes.append(EpisodeOutcome(guid, title, feed_title, STORED))
        if m := _SIZE_RE.search(media_url):
            report.estimated_bytes += int(m.group(1))
    return report


def format_dry_run(report: BackfillReport) -> str:
    """Operator-facing preview: per-feed table, then the totals that decide whether to run."""
    lines = ["archive backfill (dry-run) — nothing has been fetched", ""]
    feeds = report.by_feed()
    width = max((len(f) for f in feeds), default=4)
    lines.append(f"  {'feed'.ljust(width)}  in-corpus  archived  recoverable")
    for feed in sorted(feeds):
        c = feeds[feed]
        total = sum(c.values())
        archived = c.get(ALREADY_PRESENT, 0)
        recoverable = c.get(STORED, 0)
        lines.append(f"  {feed.ljust(width)}  {total:>9}  {archived:>8}  {recoverable:>11}")
    counts = report.counts()
    lines.append("")
    lines.append(
        f"  totals: {len(report.outcomes)} episode(s), "
        f"{counts.get(ALREADY_PRESENT, 0)} already archived, "
        f"{counts.get(STORED, 0)} to fetch, "
        f"{counts.get(NO_MEDIA_URL, 0)} without a media_url"
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
        f"{c.get(STORED, 0)} stored, "
        f"{c.get(ALREADY_PRESENT, 0)} already present, "
        f"{c.get(ROLLED_OFF, 0)} rolled off, "
        f"{c.get(FETCH_FAILED, 0)} failed, "
        f"{c.get(NO_MEDIA_URL, 0)} without media_url"
    ]
    if report.stored_bytes:
        lines.append(f"  downloaded: {report.stored_bytes / 1e9:.2f} GB")
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
