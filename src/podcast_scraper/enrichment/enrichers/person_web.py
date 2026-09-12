"""Person web-enrichment (wave-G) — the first ``EnricherTier.WEB`` enricher.

Fetches a short bio + a photo URL + attribution for each Person in the corpus from an external
web source (Wikipedia first), writing ``person_web.json``. Structured as a GENERAL web enricher with
a pluggable provider so more sources can be added: :class:`PersonWebProvider` is the seam,
:class:`WikipediaProvider` is provider #1.

**Two phases: FETCH → raw cache, then DERIVE → output.** A web source is external, costly, and can
change or vanish — unlike every corpus-internal enricher whose input is the corpus itself. So the
FULL provider payload is persisted per person under ``enrichments/person_web_raw/<slug>.json`` (the
raw cache), and the derived ``person_web.json`` is computed from it. This buys:

* **evolve without re-fetching** — add a derived field, bump the version, re-derive over cached raw
  (network only for persons not yet cached, or with ``refresh``);
* **a data-mining substrate** — the raw cache is a local, reproducible snapshot of the source to
  mine for anything later (dates, orgs, cross-links);
* **politeness + resilience** — re-runs skip already-cached persons; derivations stay reproducible
  even after the upstream drifts.

The raw cache is source data, NOT versioned by the enricher (it is what Wikipedia said, not what we
computed); only the derive step is version-gated.

**Airgap contract.** WEB-tier: it reaches a third party at FETCH time. It runs only in non-airgapped
profiles, the provider is INJECTED (tests pass a fake → no live call), and DERIVE is a pure function
(runnable in CI over fixture raw). See :class:`PersonWebProvider`.

**Photo hosting (mirrors the user avatar).** When the provider exposes ``fetch_image`` the photo
is downloaded, validated (allow-listed type + magic-byte sniff + size cap) and stored under
``enrichments/person_images/`` with an ``{ext,license,artist}`` sidecar, then served auth-gated at
``/api/app/persons/{id}/photo``. Hosting needs the per-IMAGE license (article text is CC-BY-SA; the
photo may not be), fetched from Wikipedia's ``imageinfo`` API — no license means we do NOT host.
A person that is permanently un-hostable (no license, unsupported/oversize bytes) gets a ``skip``
sidecar so we never re-fetch it; a transient network error caches nothing and retries next run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from podcast_scraper.enrichment.enrichers._loaders import (
    is_unresolved_speaker_placeholder,
    load_gi,
    nodes_of_type,
)
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.resilience import DEFAULT_POLICIES
from podcast_scraper.net.outbound_http import create_client
from podcast_scraper.rss.http_retry import RetryTransport

_logger = logging.getLogger(__name__)

#: Bound the distinct persons enriched per run (polite to the upstream; the WEB tier is opt-in).
_DEFAULT_MAX_PERSONS = 200
_USER_AGENT = "close-listening/1.0 (podcast knowledge base; contact via app)"
#: Where the raw provider payloads live, one file per person, under the corpus enrichments dir.
_RAW_SUBDIR = "person_web_raw"
#: Live Wikipedia REST summary base. Overridable via env so the e2e/mock server can stand in for
#: it (the enricher is otherwise a live external call — the mock is how the full cycle is tested).
_WIKIPEDIA_SUMMARY_ENV = "APP_WIKIPEDIA_SUMMARY_BASE"
_WIKIPEDIA_SUMMARY_DEFAULT = "https://en.wikipedia.org/api/rest_v1/page/summary/"
#: Wikipedia action API (imageinfo → per-image license/author). Env-overridable for the mock.
_WIKIPEDIA_API_ENV = "APP_WIKIPEDIA_API_BASE"
_WIKIPEDIA_API_DEFAULT = "https://en.wikipedia.org/w/api.php"
#: Where hosted person photos live under the corpus (served by /api/app/persons/{id}/photo).
_IMAGE_SUBDIR = "person_images"
_IMAGE_MAX_BYTES = 2 * 1024 * 1024  # 2 MB, matching the avatar cap
#: declared content-type → stored extension (the only image types we host).
_IMAGE_ALLOWED = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def _image_ext(content_type: str) -> str | None:
    return _IMAGE_ALLOWED.get((content_type or "").split(";", 1)[0].strip().lower())


def _image_sniff_ok(ext: str, data: bytes) -> bool:
    """Magic-byte check — the bytes must match the declared type (defence-in-depth)."""
    if ext == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == "jpg":
        return data.startswith(b"\xff\xd8\xff")
    if ext == "webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def _wiki_file_title(image_url: str) -> str:
    """The real ``File:`` name for a Wikipedia image URL.

    The REST summary's ``thumbnail.source`` is a THUMB url —
    ``…/commons/thumb/8/8d/President_Barack_Obama.jpg/330px-President_Barack_Obama.jpg?utm_source=…``
    — where the actual file is the segment BEFORE the trailing rendition, and a query string may be
    appended. Naive ``rsplit('/')`` yields ``330px-…jpg?utm_source=…``, which imageinfo reports as
    missing → we would cache a permanent skip and never host ANY photo. So: drop the query, take the
    pre-rendition segment for ``/thumb/`` urls (the last segment otherwise), and unquote so the
    caller re-encodes exactly once."""
    path = urllib.parse.urlsplit(image_url).path
    segs = [s for s in path.split("/") if s]
    if not segs:
        return ""
    name = segs[-2] if ("thumb" in segs and len(segs) >= 2) else segs[-1]
    return urllib.parse.unquote(name)


@dataclass(frozen=True)
class FetchedImage:
    """A downloaded, validated person photo + its own license/attribution (NOT the article's)."""

    data: bytes
    ext: str
    license: str | None
    artist: str | None


class _ImageSkip:
    """Sentinel returned by ``fetch_image`` for a PERMANENTLY un-hostable photo — no resolvable
    license, or bytes we reject (unsupported type / oversize / failed sniff). Distinct from ``None``
    (a transient network error): a skip is cached in a sidecar so we never re-fetch it, whereas a
    transient error caches nothing and retries on the next run."""

    __slots__ = ()


#: Singleton skip sentinel (compared by identity).
IMAGE_SKIP = _ImageSkip()


@dataclass(frozen=True)
class PersonWebInfo:
    """One person's DERIVED web enrichment — metadata only (no image bytes)."""

    person_id: str
    name: str
    bio: str | None
    #: One-line "who is this" descriptor (Wikipedia REST ``description``, e.g. "American financier
    #: and politician") — a subtitle you can read at a glance without the full bio.
    description: str | None
    image_url: str | None
    source: str  # provider label, e.g. "wikipedia"
    source_url: str | None
    license: str | None  # of the ARTICLE text (image license resolved before any hosting)


class PersonWebProvider(Protocol):
    """The pluggable web source, split into fetch (network) and derive (pure).

    ``fetch_raw`` returns the FULL upstream payload (what we persist); ``derive`` extracts the
    normalized :class:`PersonWebInfo` from a payload. Splitting them is what lets the enricher
    re-derive from cached raw without re-fetching, and lets derive run in an airgapped CI over
    fixture raw.
    """

    name: str

    def fetch_raw(self, person_id: str, display_name: str) -> dict[str, Any] | None:
        """Fetch the full upstream payload, or None on miss / failure. Never raises."""
        ...

    def derive(
        self, person_id: str, display_name: str, raw: dict[str, Any]
    ) -> PersonWebInfo | None:
        """Extract normalized info from a raw payload (pure), or None when it carries nothing."""
        ...


#: HTTP status codes worth a retry (transient upstream), mirroring the RSS downloader's forcelist.
_WEB_RETRY_STATUS = (429, 500, 502, 503, 504)
#: Only idempotent GETs are status-retried (all our calls are GETs).
_WEB_RETRY_METHODS = frozenset({"GET"})
#: Socket timeout for a single attempt (the tier policy governs retry count/backoff, not this).
_WEB_HTTP_TIMEOUT_S = 15.0


def _build_web_client() -> httpx.Client:
    """A hardened outbound client for the web enricher — the SAME resilience the RSS fetch uses.

    Routes through :func:`net.outbound_http.create_client` (admin proxy + TLS-trust registry, a
    per-subsystem o11y header) and wraps it in :class:`rss.http_retry.RetryTransport` so a transient
    upstream (429/5xx/connection error) is retried with exponential backoff + ``Retry-After``, using
    the ``EnricherTier.WEB`` policy's retry count + backoff factor. Previously this enricher used a
    bare ``urllib.urlopen`` with only a fixed timeout — no retry, no proxy/TLS routing."""
    policy = DEFAULT_POLICIES[EnricherTier.WEB]

    def _wrap(base: httpx.HTTPTransport) -> RetryTransport:
        return RetryTransport(
            base,
            total=policy.max_retries,
            backoff_factor=policy.backoff_factor,
            status_forcelist=_WEB_RETRY_STATUS,
            allowed_methods=_WEB_RETRY_METHODS,
        )

    return create_client(
        subsystem="person_web",
        follow_redirects=True,
        timeout=httpx.Timeout(_WEB_HTTP_TIMEOUT_S),
        transport_wrapper=_wrap,
    )


class WikipediaProvider:
    """Provider #1 — the Wikipedia REST summary API (bio + thumbnail + article URL)."""

    name = "wikipedia"

    def __init__(
        self,
        client: httpx.Client | None = None,
        summary_base: str | None = None,
        api_base: str | None = None,
    ) -> None:
        # Injectable client so tests drive an httpx.MockTransport (no network); the default carries
        # the shared proxy/TLS + retry resilience. Base URLs are env-overridable so the e2e mock can
        # stand in for the live host.
        self._client = client or _build_web_client()
        base = summary_base or os.environ.get(_WIKIPEDIA_SUMMARY_ENV) or _WIKIPEDIA_SUMMARY_DEFAULT
        self._summary_base = base if base.endswith("/") else base + "/"
        self._api_base = api_base or os.environ.get(_WIKIPEDIA_API_ENV) or _WIKIPEDIA_API_DEFAULT
        # Host allowlist for the image download (SSRF guard): the image URL comes from an external
        # payload (also read back from the raw cache) and the fetched bytes are re-served to users.
        # Only the configured summary host (covers the e2e mock + any override) or Wikimedia.
        self._summary_host = (urllib.parse.urlsplit(self._summary_base).hostname or "").lower()

    def _image_host_allowed(self, image_url: str) -> bool:
        host = (urllib.parse.urlsplit(image_url).hostname or "").lower()
        if not host:
            return False
        if host == self._summary_host:
            return True
        return host == "wikimedia.org" or host.endswith((".wikimedia.org", ".wikipedia.org"))

    def _get_json(self, url: str) -> dict[str, Any] | None:
        """GET + parse a JSON object. Best-effort: non-200 / network / parse error → None. The
        RetryTransport has already retried transient 429/5xx before we see a non-200 here."""
        try:
            resp = self._client.get(url, headers={"User-Agent": _USER_AGENT})
            if resp.status_code != 200:
                return None
            doc = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def fetch_raw(self, person_id: str, display_name: str) -> dict[str, Any] | None:
        """GET the REST summary. Best-effort: any network/parse error → None, never raises."""
        title = urllib.parse.quote(display_name.replace(" ", "_"), safe="")
        return self._get_json(self._summary_base + title)

    def derive(
        self, person_id: str, display_name: str, raw: dict[str, Any]
    ) -> PersonWebInfo | None:
        """Extract bio + image URL + article URL from a stored summary payload (pure)."""
        if raw.get("type") == "disambiguation":
            return None
        extract = raw.get("extract")
        if not isinstance(extract, str) or not extract.strip():
            return None
        thumb = raw.get("thumbnail")
        image_url = thumb.get("source") if isinstance(thumb, dict) else None
        page = ((raw.get("content_urls") or {}).get("desktop") or {}).get("page")
        # One-line descriptor ("American financier and politician") — a glanceable subtitle. Skip
        # the auto-generated "Wikimedia disambiguation/list page" style descriptions.
        desc = raw.get("description")
        description = desc.strip() if isinstance(desc, str) and desc.strip() else None
        return PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio=extract.strip(),
            description=description,
            image_url=image_url if isinstance(image_url, str) else None,
            source=self.name,
            source_url=page if isinstance(page, str) else None,
            license="CC-BY-SA 4.0",
        )

    def _image_attribution(self, image_url: str) -> tuple[str | None, str | None] | None:
        """(license, artist) for the image FILE via the imageinfo API — the image's OWN license,
        not the article's. Returns ``None`` on a NETWORK/parse failure (transient — the caller
        retries), or a ``(license, artist)`` tuple when the API answered (``license`` may itself be
        None → resolved-but-unlicensed, which is a PERMANENT skip)."""
        file_name = _wiki_file_title(image_url)
        query = (
            f"{self._api_base}?action=query&format=json&prop=imageinfo&iiprop=extmetadata"
            f"&titles=File:{urllib.parse.quote(file_name)}"
        )
        doc = self._get_json(query)
        # None → network/parse failure. A 200 body carrying ``error`` (MediaWiki reports rate-limits
        # etc. as 200 + {"error":…}, which RetryTransport never sees) or missing ``query`` is also a
        # transient/unexpected condition — treat all as retry, NOT a permanent skip.
        if doc is None or "error" in doc or "query" not in doc:
            return None
        pages = (doc.get("query") or {}).get("pages") or {}
        for page in pages.values() if isinstance(pages, dict) else []:
            infos = page.get("imageinfo") if isinstance(page, dict) else None
            meta = (infos[0].get("extmetadata") or {}) if isinstance(infos, list) and infos else {}
            lic = (meta.get("LicenseShortName") or {}).get("value")
            artist = (meta.get("Artist") or {}).get("value")
            return (
                lic if isinstance(lic, str) else None,
                artist if isinstance(artist, str) else None,
            )
        return (None, None)  # API answered with no imageinfo → resolved, unlicensed → permanent

    def fetch_image(self, image_url: str) -> FetchedImage | _ImageSkip | None:
        """Resolve the image's license (imageinfo) then download+validate the bytes.

        Returns a :class:`FetchedImage` on success; :data:`IMAGE_SKIP` when the photo is PERMANENTLY
        un-hostable (no resolvable license, or unsupported/oversize/mismatched bytes) so the caller
        caches a skip; and ``None`` on a transient network error so the caller retries next run."""
        attribution = self._image_attribution(image_url)
        if attribution is None:
            return None  # transient imageinfo failure → retry next run
        license_, artist = attribution
        if license_ is None:
            return IMAGE_SKIP  # no license → never host what we cannot attribute (permanent)
        if not self._image_host_allowed(image_url):
            return IMAGE_SKIP  # off-allowlist host (SSRF guard) → never fetch it (permanent)
        try:
            # Stream so an oversize image is bounded, never read whole into memory (read cap+1 then
            # reject) — the same defence the urllib version had.
            with self._client.stream("GET", image_url, headers={"User-Agent": _USER_AGENT}) as resp:
                if resp.status_code in (404, 410):
                    return IMAGE_SKIP  # the image is gone → do not retry forever (permanent)
                if resp.status_code != 200:
                    return None  # transient (5xx already retried) → retry next run
                content_type = resp.headers.get("Content-Type", "")
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > _IMAGE_MAX_BYTES:
                        return IMAGE_SKIP  # oversize → never host this URL (permanent)
                data = b"".join(chunks)
        except httpx.HTTPError:
            return None  # transient download failure → retry next run
        ext = _image_ext(content_type)
        if ext is None or not _image_sniff_ok(ext, data):
            return IMAGE_SKIP  # unsupported/mismatched bytes → never host this URL (permanent)
        return FetchedImage(data=data, ext=ext, license=license_, artist=artist)


def _distinct_persons(all_bundles: list[EpisodeArtifactBundle]) -> list[tuple[str, str]]:
    """(person_id, name) for every non-placeholder Person in the corpus GI, de-duplicated."""
    seen: dict[str, str] = {}
    for b in all_bundles:
        gi = load_gi(b)
        for node in nodes_of_type(gi, "Person"):
            pid = str(node.get("id") or "")
            if not pid or pid in seen:
                continue
            name = str((node.get("properties") or {}).get("name") or pid)
            if is_unresolved_speaker_placeholder(pid, name):
                continue
            seen[pid] = name
    return sorted(seen.items())


def _safe_name(person_id: str) -> str:
    """A filesystem-safe raw-cache stem for a person id (slug, or a hash if it sanitizes away)."""
    slug = re.sub(r"[^a-z0-9._-]", "_", person_id.split(":", 1)[-1].lower())
    return slug or hashlib.sha256(person_id.encode("utf-8")).hexdigest()[:16]


def _raw_path(corpus_root: Path, person_id: str) -> Path:
    return corpus_root / "enrichments" / _RAW_SUBDIR / f"{_safe_name(person_id)}.json"


def _read_raw_cache(corpus_root: Path, person_id: str) -> dict[str, Any] | None:
    """The persisted upstream payload for a person, or None when uncached/unreadable."""
    path = _raw_path(corpus_root, person_id)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    payload = doc.get("payload") if isinstance(doc, dict) else None
    return payload if isinstance(payload, dict) else None


def _write_raw_cache(
    corpus_root: Path, person_id: str, display_name: str, payload: dict[str, Any], now: int
) -> None:
    """Persist the raw upstream payload + fetch metadata (source data; not version-gated)."""
    path = _raw_path(corpus_root, person_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "person_id": person_id,
        "display_name": display_name,
        "fetched_at": now,
        "payload": payload,
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")


#: stored extension → served media type (the serve route needs this).
_EXT_MEDIA = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


def _image_dir(corpus_root: Path) -> Path:
    return corpus_root / "enrichments" / _IMAGE_SUBDIR


def person_image_path(corpus_root: Path, person_id: str) -> tuple[Path, str] | None:
    """The hosted photo ``(path, media_type)`` for a person, or None. Public — the serve route
    reads it. The stem is sanitized (``_safe_name``) and the filename is a fixed glob, so the path
    cannot escape the images dir."""
    from podcast_scraper.utils.path_validation import resolves_under_root

    directory = _image_dir(corpus_root)
    stem = _safe_name(person_id)
    for ext, media in _EXT_MEDIA.items():
        candidate = directory / f"{stem}.{ext}"
        # is_file() follows symlinks, so also require the resolved target stays under the corpus.
        if candidate.is_file() and resolves_under_root(candidate, corpus_root):
            return candidate, media
    return None


def _image_meta_path(corpus_root: Path, person_id: str) -> Path:
    return _image_dir(corpus_root) / f"{_safe_name(person_id)}.image.json"


def _read_image_meta(corpus_root: Path, person_id: str) -> dict[str, Any] | None:
    """The stored {ext, license, artist} sidecar for a hosted photo, or None (cache-on-skip)."""
    path = _image_meta_path(corpus_root, person_id)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _write_skip_meta(corpus_root: Path, person_id: str) -> None:
    """Record that a person is permanently un-hostable so re-runs never re-fetch the photo.

    Also removes any previously stored photo: a ``refresh`` run whose license no longer resolves
    must STOP serving the old image (``person_image_path`` globs the file, so a stale photo would
    otherwise keep being served at ``/persons/{id}/photo`` while the card reports it unhosted)."""
    directory = _image_dir(corpus_root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(person_id)
    for ext in set(_IMAGE_ALLOWED.values()):
        (directory / f"{stem}.{ext}").unlink(missing_ok=True)
    _image_meta_path(corpus_root, person_id).write_text(
        json.dumps({"skip": True}), encoding="utf-8"
    )


def _store_image(corpus_root: Path, person_id: str, image: FetchedImage) -> dict[str, Any]:
    """Write the photo (hosted like the avatar) + an {ext,license,artist} sidecar; return meta."""
    directory = _image_dir(corpus_root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(person_id)
    for ext in set(_IMAGE_ALLOWED.values()):  # drop any prior format so only one photo remains
        (directory / f"{stem}.{ext}").unlink(missing_ok=True)
    (directory / f"{stem}.{image.ext}").write_bytes(image.data)
    meta = {"ext": image.ext, "license": image.license, "artist": image.artist}
    _image_meta_path(corpus_root, person_id).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return meta


class PersonWebEnricher:
    """Corpus-scope WEB enricher: a bio/photo-URL/attribution row per Person from a web provider."""

    manifest = EnricherManifest(
        id="person_web",
        version="0.1.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.WEB,
        reads=[".gi.json"],
        writes="person_web.json",
        description="Per-person bio + photo URL + attribution from a web source (Wikipedia).",
        # ON by default in the cloud/prod profiles (not opt-in) — a free Wikipedia fetch, cheap on
        # re-run via the raw cache. The airgap is held by PROFILE membership: person_web is in the
        # cloud/prod sets only, never the airgapped/CI ones, so CI never fetches. (And the fetch is
        # best-effort — a networkless run degrades to an empty bio, never a crash.)
        requires_opt_in=False,
        expected_duration_s=120,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "max_persons": {
                    "type": "integer",
                    "minimum": 1,
                    "default": _DEFAULT_MAX_PERSONS,
                    "description": "Cap on distinct persons enriched per run (be polite).",
                },
                "refresh": {
                    "type": "boolean",
                    "default": False,
                    "description": "Re-fetch even when a raw payload is cached (bypass the cache).",
                },
            },
        },
    )

    def __init__(self, provider: PersonWebProvider | None = None) -> None:
        # Default provider does real HTTP; tests inject a fake so no live call is made. The enricher
        # is never RUN in the airgapped CI profile regardless (WEB tier is excluded there).
        self._provider: PersonWebProvider = provider or WikipediaProvider()

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Fetch-then-derive per person: cached raw is reused (no network); output is re-derived."""
        max_persons = int(config.get("max_persons", _DEFAULT_MAX_PERSONS))
        refresh = bool(config.get("refresh", False))
        persons = _distinct_persons(all_bundles or [])[:max_persons]
        now = int(time.time())

        # Image hosting is an optional provider capability (mirrors the avatar: download →
        # validate → store), gated on the provider exposing fetch_image. A cached photo (sidecar)
        # is reused without a re-download unless ``refresh``.
        fetch_image = getattr(self._provider, "fetch_image", None)

        def _host_image(pid: str, image_url: str) -> dict[str, Any] | None:
            meta = None if refresh else _read_image_meta(corpus_root, pid)
            if meta is not None:
                if meta.get("skip"):
                    return None  # known permanently un-hostable — do not re-fetch
                if person_image_path(corpus_root, pid) is not None:
                    return meta  # sidecar AND file present → reuse
                # sidecar without its image file → treat as a miss and re-fetch
            if fetch_image is None:
                return None
            fetched_image = fetch_image(image_url)
            if fetched_image is None:
                return None  # transient error — cache nothing, retry next run
            if isinstance(fetched_image, _ImageSkip):
                _write_skip_meta(corpus_root, pid)
                return None  # permanent skip cached
            return _store_image(corpus_root, pid, fetched_image)

        def _run() -> tuple[list[dict[str, Any]], int]:
            rows: list[dict[str, Any]] = []
            fetched = 0
            for pid, name in persons:
                if ctx.cancel_event.is_set():
                    break
                raw = None if refresh else _read_raw_cache(corpus_root, pid)
                if raw is None:
                    raw = self._provider.fetch_raw(pid, name)
                    if raw is not None:
                        _write_raw_cache(corpus_root, pid, name, raw, now)
                        fetched += 1
                if raw is None:
                    continue
                info = self._provider.derive(pid, name, raw)
                if info is None:
                    continue
                row = asdict(info)
                if info.image_url:
                    meta = _host_image(pid, info.image_url)
                    if meta is not None:
                        row["image_hosted"] = True
                        row["image_ext"] = meta.get("ext")
                        row["image_license"] = meta.get("license")
                        row["image_artist"] = meta.get("artist")
                rows.append(row)
            return rows, fetched

        try:
            rows, fetched = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 — enrich() never raises out of itself
            return EnricherResult(status="failed", error=str(exc), error_class=type(exc).__name__)
        _logger.info(
            "person_web derived %d/%d persons (%d freshly fetched) run_id=%s provider=%s",
            len(rows),
            len(persons),
            fetched,
            ctx.run_id,
            self._provider.name,
        )
        return EnricherResult(
            status=STATUS_OK,
            data={"provider": self._provider.name, "persons": rows},
            records_written=len(rows),
        )
