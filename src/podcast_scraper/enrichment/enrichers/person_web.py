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
import dataclasses
import hashlib
import io
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

from podcast_scraper.archive.backfill import HostRateLimiter
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

#: How long a MISS is trusted before the upstream is asked again (seconds; 30 days).
#: A person with no Wikipedia article is the common case, not an error — but without a negative
#: cache every such person is re-queried on EVERY run, which is precisely the full-corpus network
#: work ENTITY scope exists to avoid. Measured 2026-09-15: a warm run still issued one request for
#: a 404'd name. The TTL keeps it self-healing: someone who GETS an article later is picked up on
#: the next expiry rather than never, without a manual `refresh`.
_MISS_TTL_S = 30 * 24 * 3600
_USER_AGENT = "close-listening/1.0 (podcast knowledge base; contact via app)"

#: Minimum seconds between consecutive hits on one upstream host. Wikimedia's robot policy
#: (https://w.wiki/4wJS, phabricator T400119) is explicit that unthrottled clients get 403'd,
#: and on 2026-09-16 this enricher issued ~911 requests in 65 seconds (~43/s) and was blocked
#: outright — every lookup returned 403 "Please set a user-agent and respect our robot policy".
#: 1 req/s turns a 911-entity pass into ~15 minutes, which is irrelevant for a background job
#: and is the difference between being a good citizen and losing access to the source entirely.
_WEB_MIN_INTERVAL_S = 1.0


class TransientFetchError(Exception):
    """Upstream was unreachable or refused us — as opposed to authoritatively having nothing.

    This distinction is the whole point. ``fetch_raw`` returning ``None`` means "the source
    genuinely has no page for this entity", which the caller records as a negative-cache miss
    with a 30-day TTL. A 403/429/5xx/timeout means "we could not ask", which must NOT be
    recorded — otherwise a transient outage is silently converted into a month of missing data.

    On 2026-09-16 that is exactly what happened: a rate-limit block made every fetch return
    ``None``, and 911 real people and organisations (Keir Starmer, Katie Couric, Allbirds...)
    were written as permanent misses in 65 seconds. Raise this instead of returning ``None``
    whenever the answer is "we don't know", never "there is nothing".
    """


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


#: Longest stored edge, in pixels. The person card renders the photo at CSS ``size=176`` square
#: (``PersonCardContent.vue``), so 512 covers a 3x retina panel with room to spare and anything
#: larger is bytes we pay to store, back up and serve for no visible gain. We downscale rather
#: than crop: the card is square but ``object-fit`` does that in CSS, and choosing a crop box
#: here would be deciding where a face sits — a framing judgement this layer has no basis for.
_IMAGE_MAX_EDGE = 512


def _downscale_image(data: bytes, ext: str) -> bytes:
    """Shrink an oversized photo to ``_IMAGE_MAX_EDGE`` on its longest side.

    Upstream serves originals: Commons hands back a 3000px master unless asked otherwise, and
    even a thumbnailed rendition arrives larger than anything we render. ``_IMAGE_MAX_BYTES``
    only bounds the download; without this we persist the full frame forever.

    Re-encoding is skipped entirely when the image already fits, so a small photo is never
    degraded by a needless generation loss. Any Pillow failure returns the ORIGINAL bytes: a
    photo we could not resize is still a photo, and the caller has already sniffed and size-
    capped it, so falling back is safe rather than dropping it.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) <= _IMAGE_MAX_EDGE:
                return data
            frame = im.copy()  # detach from the closing file before re-encoding
            frame.thumbnail((_IMAGE_MAX_EDGE, _IMAGE_MAX_EDGE), Image.LANCZOS)
            buf = io.BytesIO()
            # Format must stay put: the sidecar records `ext` and /persons/{id}/photo serves the
            # file by that extension, so silently writing a JPEG into a .png would break both.
            if ext == "jpg":
                frame.convert("RGB").save(buf, "JPEG", quality=85, optimize=True, progressive=True)
            elif ext == "png":
                frame.save(buf, "PNG", optimize=True)
            else:
                frame.save(buf, "WEBP", quality=85, method=6)
            out = buf.getvalue()
    except Exception:  # noqa: BLE001 - decode/encode failure must not cost us the photo
        return data
    # Judged on DIMENSIONS, not bytes. An earlier version kept the original whenever the
    # re-encode came out larger, which sounds prudent and is wrong: a 1600px master that
    # happens to compress well is still a 1600px master we store, back up and serve. The only
    # reason to reject the result is that it is not a usable image of the declared type.
    return out if _image_sniff_ok(ext, out) else data


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


#: Wikidata entity API — identity resolution (search -> entity -> sitelink).
#: Env-overridable for tests.
_WIKIDATA_API_DEFAULT = "https://www.wikidata.org/w/api.php"
_WIKIDATA_API_ENV = "APP_WIKIDATA_API_BASE"
#: P31 "instance of" / Q5 "human" — the type filter that keeps a search hit
#: from being a ship or a film.
_WD_INSTANCE_OF = "P31"
_WD_HUMAN = "Q5"
#: P18 "image" — the portrait claim. Resolved through Commons Special:FilePath, the same
#: route org_web uses for logos, so the existing image host-allowlist + licence check apply.
_WD_IMAGE = "P18"
_COMMONS_FILEPATH_DEFAULT = "https://commons.wikimedia.org/wiki/Special:FilePath/"
#: ``Special:FilePath/<name>`` serves the FULL-RESOLUTION original — Katie Couric's portrait is
#: 16 MB, eight times ``_IMAGE_MAX_BYTES``, so every P18 photo would download to the cap and then
#: be cached as a PERMANENT skip. ``?width=`` asks Commons for a rendition instead (the same
#: thumbnailer the REST summary's ``thumbnail.source`` comes from): 178 KB for that same file.
#: The query string is dropped before the ``File:`` title lookup, so attribution still resolves.
_COMMONS_THUMB_WIDTH = 640
#: How many search hits to type-check. Wikidata ranks by relevance; beyond a
#: handful the tail is noise.
_WD_MAX_CANDIDATES = 5
#: Marks a payload produced by the Wikidata-resolved path. Its ABSENCE means the payload is a bare
#: Wikipedia REST summary from the original provider — 940 of those are cached
#: on prod, and they must
#: keep deriving unchanged. Never make this key required.
_RESOLVED_SCHEMA = "wikidata+wikipedia/1"


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
        # Per-host throttle shared by every request this provider makes. Constructed here (not
        # module-global) so tests get an isolated limiter and can inject a fake sleep.
        self._limiter = HostRateLimiter(_WEB_MIN_INTERVAL_S)
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
        """GET + parse a JSON object.

        Returns ``None`` ONLY for an authoritative 404 — the source looked and has nothing.
        Every other failure (403, 429, 5xx, connection error, unparsable body) raises
        :class:`TransientFetchError`, because "we could not ask" must never be recorded as
        "there is nothing there". The RetryTransport has already retried transient 429/5xx
        before we see a non-200 here, so reaching this point means retries were exhausted.

        Throttled per host: see ``_WEB_MIN_INTERVAL_S``.
        """
        self._limiter.wait(url)
        try:
            resp = self._client.get(url, headers={"User-Agent": _USER_AGENT})
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                raise TransientFetchError(f"HTTP {resp.status_code} from {url}")
            doc = resp.json()
        except TransientFetchError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise TransientFetchError(f"{type(exc).__name__} from {url}") from exc
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
        try:
            doc = self._get_json(query)
        except TransientFetchError:
            # fetch_image already encodes transient-vs-permanent as None vs IMAGE_SKIP, and that
            # distinction is correct — an unreachable imageinfo must be retried, never cached as a
            # skip. So absorb the exception here rather than propagating it: the raise exists to
            # stop the ENTITY cache recording a false miss, which is a different caller.
            return None
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
            # Same per-host throttle as _get_json: image bytes come from
            # upload./commons.wikimedia.org, which are covered by the same robot policy.
            self._limiter.wait(image_url)
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
        # Store what we render, not what upstream serves — see _downscale_image.
        return FetchedImage(
            data=_downscale_image(data, ext), ext=ext, license=license_, artist=artist
        )


def _existing_person_rows(corpus_root: Path) -> dict[str, dict[str, Any]]:
    """Rows already derived on a previous run, keyed by ``person_id`` (see org_web sibling)."""
    try:
        doc = json.loads((corpus_root / "enrichments" / "person_web.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    data = doc.get("data") if isinstance(doc, dict) else None
    rows = (data or {}).get("persons") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("person_id"):
            out[str(r["person_id"])] = r
    return out


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


def _miss_is_fresh(corpus_root: Path, person_id: str, now: int) -> bool:
    """True when a recent MISS is recorded for this person — skip the upstream call.

    Returns False once the TTL lapses so the lookup is retried, which is what makes the cache
    self-healing for someone who gains an article later.
    """
    path = _raw_path(corpus_root, person_id)
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not (isinstance(doc, dict) and doc.get("miss")):
        return False
    fetched_at = doc.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return False
    return (now - int(fetched_at)) < _MISS_TTL_S


def _write_miss(corpus_root: Path, person_id: str, display_name: str, now: int) -> None:
    """Record that the upstream had nothing for this person, with a timestamp for the TTL."""
    path = _raw_path(corpus_root, person_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "person_id": person_id,
        "display_name": display_name,
        "fetched_at": now,
        "miss": True,
    }
    try:
        path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort: a failed miss-write only costs a retry next run


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
    reads it."""
    from podcast_scraper.utils.path_validation import resolves_under_root

    directory = _image_dir(corpus_root)
    stem = _safe_name(person_id)
    # Enumerate the real directory entries and match the wanted filename against THEM, rather than
    # joining the person-derived stem onto the dir. The tainted stem only ever indexes this dict;
    # the served path comes from directory.iterdir() — a trusted enumeration — so person_id never
    # reaches a filesystem path at all (py/path-injection, PR #2049). resolves_under_root still
    # rejects a symlink entry that points outside the dir.
    try:
        entries = {p.name: p for p in directory.iterdir()}
    except OSError:
        return None
    for ext, media in _EXT_MEDIA.items():
        match = entries.get(f"{stem}.{ext}")
        if match is not None and match.is_file() and resolves_under_root(match, directory):
            return match, media
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


class WikidataResolvedProvider:
    """Provider #2 — Wikidata resolves WHO, Wikipedia supplies the PROSE.

    ``WikipediaProvider`` guesses an article title from the display name
    (``"Anish Acharya" -> /page/summary/Anish_Acharya``). That is one source and a fragile
    lookup, and it conflates three different outcomes into a single 404:

      * genuinely in neither source          -> a correct miss
      * a Wikidata item but no article       -> recoverable, invisible today
      * an article at a title we didn't guess -> recoverable, invisible today
        (middle names, initials, diacritics, ``John Smith (economist)``)

    This provider removes the guess. A Wikidata item carries ``sitelinks.enwiki``: the exact
    article title. So identity is resolved structurally, then prose is fetched at the title
    Wikidata hands us.

    **Wikidata resolves; it does not supply content.** The card contract is three fields —
    bio, description, image — and Wikidata can only ever fill the last two. A person with an
    item but no article (Adam Mastroianni: ``Q125651464``, description ``"American Rhodes
    Scholar"``, no ``enwiki``, no ``P18``) would yield a one-line stub next to fully-populated
    neighbours. That is a miss, and it is recorded as one. What Wikidata *does* contribute is
    the right title, plus gap-fill for a description or portrait the article itself lacks.

    Disambiguation is the danger, not coverage, and two prod audits narrowed the rule twice —
    both times because a stricter rule dropped people we already serve.

    First, the discriminator is the ARTICLE, not humanness. ``"Balaji Srinivasan"`` returns
    FIVE humans: the entrepreneur (``Q87684934``, has an article) and four ORCID researcher
    stubs with no article and no prose. Refusing on "more than one human" loses him.

    Second, several articles is not automatically ambiguity. Wikipedia has already
    disambiguated and records the answer in the titles — the primary topic sits at the bare
    name, everyone else carries a qualifier (``Alex Jones`` vs ``Alex Jones (actor)``).
    Refusing on "more than one article" dropped 5 of 30 audited prod rows, Henry VIII among
    them. See :meth:`_pick`.

    What remains refused is real ambiguity: two notable people, neither at the bare name. A
    confident wrong match puts someone else's biography on an episode page, which is worse
    than no match at all. Every candidate is persisted regardless, so a future re-derive can
    revisit the choice without re-fetching.
    """

    name = "wikidata+wikipedia"

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        wikidata_api: str | None = None,
        summary_base: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._wikipedia = WikipediaProvider(
            client=client, summary_base=summary_base, api_base=api_base
        )
        # Share the Wikipedia provider's client so BOTH hosts sit behind the same per-host
        # throttle (wikidata.org and en.wikipedia.org are metered separately by HostRateLimiter).
        self._client = self._wikipedia._client
        self._limiter = self._wikipedia._limiter
        self._wd_api = wikidata_api or os.environ.get(_WIKIDATA_API_ENV) or _WIKIDATA_API_DEFAULT

    # -- identity resolution -------------------------------------------------

    def _search(self, display_name: str) -> list[dict[str, Any]]:
        url = (
            f"{self._wd_api}?action=wbsearchentities&format=json&language=en&uselang=en"
            f"&type=item&limit={_WD_MAX_CANDIDATES}"
            f"&search={urllib.parse.quote(display_name)}"
        )
        doc = self._wikipedia._get_json(url)
        hits = (doc or {}).get("search")
        return [h for h in hits if isinstance(h, dict)] if isinstance(hits, list) else []

    def _entities(self, ids: list[str]) -> dict[str, Any]:
        """One batched ``wbgetentities`` for every candidate — N ids, ONE round trip."""
        if not ids:
            return {}
        url = (
            f"{self._wd_api}?action=wbgetentities&format=json&languages=en"
            f"&ids={urllib.parse.quote('|'.join(ids))}"
        )
        doc = self._wikipedia._get_json(url)
        ents = (doc or {}).get("entities")
        return ents if isinstance(ents, dict) else {}

    @staticmethod
    def _is_human(entity: dict[str, Any]) -> bool:
        for claim in (entity.get("claims") or {}).get(_WD_INSTANCE_OF) or []:
            if not isinstance(claim, dict):
                continue
            dv = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {}
            if isinstance(dv, dict) and dv.get("id") == _WD_HUMAN:
                return True
        return False

    @staticmethod
    def _image_url(entity: dict[str, Any]) -> str | None:
        """Portrait from the P18 claim, via Commons Special:FilePath.

        Gap-fill only: an article whose REST summary carries no ``thumbnail`` can still have a
        portrait on its Wikidata item. The URL goes through the SAME hosting path as any other
        (host allowlist, licence resolution, self-hosted serve), so nothing about image handling
        becomes a special case.
        """
        for claim in (entity.get("claims") or {}).get(_WD_IMAGE) or []:
            if not isinstance(claim, dict):
                continue
            fname = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
            if isinstance(fname, str) and fname.strip():
                name = urllib.parse.quote(fname.strip().replace(" ", "_"), safe="")
                return f"{_COMMONS_FILEPATH_DEFAULT}{name}?width={_COMMONS_THUMB_WIDTH}"
        return None

    @staticmethod
    def _enwiki_title(entity: dict[str, Any]) -> str | None:
        link = (entity.get("sitelinks") or {}).get("enwiki") or {}
        title = link.get("title") if isinstance(link, dict) else None
        return title if isinstance(title, str) and title.strip() else None

    def _pick(self, articled: dict[str, Any], display_name: str) -> str | None:
        """Choose among candidates that have an article, or refuse.

        One candidate is the easy case. Several is not automatically ambiguity: Wikipedia has
        ALREADY disambiguated, and it records the answer in the titles. The primary topic sits
        at the bare name; everyone else carries a parenthetical or a qualifier.

            "Alex Jones"   -> ['Alex Jones', 'Alex Jones (actor)']
            "Henry VIII"   -> ['Henry VIII of Waldeck', 'Henry VIII', 'Henry VII of Brzeg']
            "Bill Cassidy" -> ['Bill Cassidy (footballer, born 1917)', 'Bill Cassidy']

        Refusing all of those — which an earlier "more than one article means ambiguous" rule
        did — discards Wikipedia's own editorial decision and drops people we serve today. A
        prod audit measured it at 5 of 30 sampled rows.

        So: exactly one bare-title match wins. Zero means nobody is the primary topic under
        this name, and two or more cannot happen for real titles but is refused on principle.
        Only then is it genuine ambiguity — two notable people, neither at the bare name — and
        a wrong pick would put someone else's biography on an episode page.
        """
        if len(articled) == 1:
            return next(iter(articled))
        wanted = display_name.strip().casefold()
        exact = [
            qid
            for qid, entity in articled.items()
            if (self._enwiki_title(entity) or "").strip().casefold() == wanted
        ]
        return exact[0] if len(exact) == 1 else None

    def fetch_raw(self, person_id: str, display_name: str) -> dict[str, Any] | None:
        """Resolve via Wikidata, then fetch the article Wikidata points at.

        Returns None ONLY when the search found nothing at all — an authoritative "no such
        person", which the caller may cache as a miss. Transport failures raise
        :class:`TransientFetchError` from the shared ``_get_json``, so a blocked or unreachable
        upstream never becomes a 30-day absence claim.
        """
        hits = self._search(display_name)
        if not hits:
            # Wikidata knows nothing by this name. Before declaring absence, try the ORIGINAL
            # direct-title lookup: this path must never return LESS than provider #1 did.
            # Almost every article has a Wikidata item, but `wbsearchentities` can still miss
            # one that a direct title hit would have found, and losing a person we currently
            # have would be a regression dressed as an improvement.
            return self._wikipedia.fetch_raw(person_id, display_name)

        ids = [str(h["id"]) for h in hits if isinstance(h.get("id"), str)]
        entities = self._entities(ids[:_WD_MAX_CANDIDATES])
        humans = {
            qid: e for qid, e in entities.items() if isinstance(e, dict) and self._is_human(e)
        }
        # The discriminator is the ARTICLE, not humanness. Four ORCID stubs and one
        # entrepreneur are all "human"; only one of them has a biography to show.
        articled = {qid: e for qid, e in humans.items() if self._enwiki_title(e)}

        payload: dict[str, Any] = {
            "schema": _RESOLVED_SCHEMA,
            # EVERY candidate is persisted, not just the winner. org_web does the same, because
            # choosing is the hard part and a future re-derive should be able to revisit it
            # without re-fetching.
            "candidates": [
                {"id": h.get("id"), "label": h.get("label"), "description": h.get("description")}
                for h in hits
            ],
            "human_ids": sorted(humans),
            "articled_ids": sorted(articled),
        }

        winner = self._pick(articled, display_name)
        if winner is None:
            # No article at all -> fall back to the direct title lookup so this path never
            # returns less than provider #1 did, then let the caller record an honest miss.
            #
            # Several articles and no bare-title match -> genuine ambiguity. That does NOT fall
            # back: a direct title guess would pick one of them blindly, which is the exact
            # failure this provider exists to prevent.
            if not articled:
                legacy = self._wikipedia.fetch_raw(person_id, display_name)
                if legacy is not None:
                    return legacy
            return payload

        entity = articled[winner]
        qid = winner
        title = self._enwiki_title(entity)
        payload["wikidata_id"] = qid
        payload["enwiki_title"] = title
        payload["wikidata"] = {
            "description": ((entity.get("descriptions") or {}).get("en") or {}).get("value"),
            "label": ((entity.get("labels") or {}).get("en") or {}).get("value"),
            "image_url": self._image_url(entity),
        }
        # The whole point: fetch at the title Wikidata gives us, not one we invented.
        payload["wikipedia"] = self._wikipedia._get_json(
            self._wikipedia._summary_base
            + urllib.parse.quote((title or "").replace(" ", "_"), safe="")
        )
        return payload

    # -- derive --------------------------------------------------------------

    def derive(
        self, person_id: str, display_name: str, raw: dict[str, Any]
    ) -> PersonWebInfo | None:
        """Normalize a payload, old-format or new.

        BACKWARD COMPATIBILITY IS LOAD-BEARING: 940 cached payloads on prod are bare Wikipedia
        REST summaries written by provider #1. They carry no ``schema`` key and must keep
        deriving exactly as before — a re-derive pass must not silently drop 793 existing rows.
        """
        if raw.get("schema") != _RESOLVED_SCHEMA:
            return self._wikipedia.derive(person_id, display_name, raw)

        # `wikidata_id` is written ONLY when `_pick` chose a winner, so its presence is the
        # resolved/refused flag. Counting `articled_ids` here would re-implement the picking
        # rule in a second place and get it wrong the moment the two drift apart.
        if not raw.get("wikidata_id"):
            return None  # unresolved or ambiguous — never guess

        wiki = raw.get("wikipedia")
        if not isinstance(wiki, dict):
            return None
        info = self._wikipedia.derive(person_id, display_name, wiki)
        if info is None:
            return None

        # Wikipedia supplies the prose; Wikidata fills the two fields the article may omit.
        # The card contract is bio + description + image, so a row that reaches the app with
        # only one of them is the thing we are fixing, not shipping. PersonWebInfo is frozen,
        # hence replace() rather than assignment.
        wd = raw.get("wikidata") or {}
        patch: dict[str, Any] = {}
        wd_desc = wd.get("description")
        if not info.description and isinstance(wd_desc, str) and wd_desc.strip():
            patch["description"] = wd_desc.strip()
        if not info.image_url and isinstance(wd.get("image_url"), str):
            patch["image_url"] = wd["image_url"]
        return dataclasses.replace(info, **patch) if patch else info

    def fetch_image(self, image_url: str) -> "FetchedImage | _ImageSkip | None":
        """Delegate: image hosting + licensing is identical whichever source found the person."""
        return self._wikipedia.fetch_image(image_url)


class PersonWebEnricher:
    """Corpus-scope WEB enricher: a bio/photo-URL/attribution row per Person from a web provider."""

    manifest = EnricherManifest(
        id="person_web",
        version="0.1.0",
        scope=EnricherScope.ENTITY,
        tier=EnricherTier.WEB,
        reads=[".gi.json"],
        writes="person_web.json",
        description="Per-person bio + photo URL + attribution from a web source (Wikipedia).",
        # ON by default in the cloud/prod profiles (not opt-in) — a free Wikipedia fetch, cheap on
        # re-run via the raw cache. The airgap is held by PROFILE membership: person_web is in the
        # cloud/prod sets only, never the airgapped/CI ones, so CI never fetches. (And the fetch is
        # best-effort — a networkless run degrades to an empty bio, never a crash.)
        requires_opt_in=False,
        # RFC-118: the executor dispatches enrich_incremental() on this flag.
        supports_incremental=True,
        # 1800, not 120. The tier walks a RATE-LIMITED upstream (~5.5 entities/min
        # measured on prod 2026-09-15), and the per-run budget is max_persons entities — so a
        # full budget is ~36 min, not two. 120s was sized for the pre-ENTITY-scope design
        # and killed the first prod pass at 58 min with 311 payloads already fetched: the
        # raw cache survived (it writes per entity) but the merged artifact never got
        # written, so `known` stayed empty and the next run would have re-walked the same
        # entities instead of advancing. A steady-state run is milliseconds; this ceiling
        # only ever bites the initial backfill, which is exactly when it must not.
        expected_duration_s=3600,
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
        #
        # #2111: resolving through Wikidata is the DEFAULT, not an opt-in. Leaving it off would
        # leave the bug in place — the title guess is what produces the wrong-title misses. It
        # costs two extra requests per person (search + batched getentities), both throttled
        # per-host by the shared HostRateLimiter, and it falls back to the direct title lookup
        # whenever Wikidata has nothing, so it can only return more than the old path.
        self._provider: PersonWebProvider = provider or WikidataResolvedProvider()

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Full pass. Prior rows come off disk; the delta path passes them in instead."""
        refresh = bool(config.get("refresh", False))
        known = {} if refresh else _existing_person_rows(corpus_root)
        return await self._compute(
            corpus_root=corpus_root,
            all_bundles=all_bundles or [],
            config=config,
            ctx=ctx,
            known=known,
        )

    async def enrich_incremental(
        self,
        *,
        delta: Any,
        prior_output: dict[str, Any] | None,
        corpus_root: Path,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """RFC-118 delta pass — see the org_web sibling for the full rationale.

        A person's bio does not change because some OTHER episode changed, so the delta matters
        only through the people it INTRODUCES. Output-identical to :meth:`enrich` over the same
        corpus (§7 reconciliation): both funnel into ``_compute``.
        """
        refresh = bool(config.get("refresh", False)) or bool(getattr(delta, "forced", False))
        known: dict[str, dict[str, Any]] = {}
        if not refresh:
            rows = (prior_output or {}).get("persons")
            if isinstance(rows, list):
                known = {
                    str(r["person_id"]): r
                    for r in rows
                    if isinstance(r, dict) and r.get("person_id")
                }
            else:
                known = _existing_person_rows(corpus_root)
        return await self._compute(
            corpus_root=corpus_root,
            all_bundles=list(getattr(delta, "all_bundles", []) or []),
            config=config,
            ctx=ctx,
            known=known,
        )

    async def _compute(
        self,
        *,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle],
        config: dict[str, Any],
        ctx: RunContext,
        known: dict[str, dict[str, Any]],
    ) -> EnricherResult:
        """Shared body for both passes: fetch only unknown people, merge onto ``known``."""
        max_persons = int(config.get("max_persons", _DEFAULT_MAX_PERSONS))
        refresh = bool(config.get("refresh", False))
        # ENTITY scope: rows already derived are CARRIED FORWARD and the budget is spent only on
        # people never seen before. This used to be `_distinct_persons(...)[:max_persons]`, a
        # slice of an ID-SORTED corpus-wide list — not a rate limit but a permanent coverage
        # ceiling, where person N+1 was unreachable no matter how many runs happened.
        now_for_budget = int(time.time())
        all_persons = _distinct_persons(all_bundles or [])
        # Exclude BOTH already-derived people and recent misses BEFORE applying the budget.
        # Filtering misses only inside the loop would let them eat the budget: with
        # max_persons=3 and three permanently-unknown names, no new person would ever be
        # fetched — the coverage-ceiling bug this change exists to remove, in a new disguise.
        fresh = [
            (pid, name)
            for pid, name in all_persons
            if pid not in known
            and (refresh or not _miss_is_fresh(corpus_root, pid, now_for_budget))
        ]
        persons = fresh[:max_persons]
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
                    # A recorded miss inside the TTL means "upstream had nothing recently" —
                    # do not ask again. Without this, every person the source does not know is
                    # re-queried on every run forever.
                    if not refresh and _miss_is_fresh(corpus_root, pid, now):
                        continue
                    try:
                        raw = self._provider.fetch_raw(pid, name)
                    except TransientFetchError:
                        # We could not ask — upstream refused or was unreachable. Leave NO
                        # record so the next run retries. Writing a miss here is what turned a
                        # 65-second rate-limit block into 911 entities marked absent for 30 days.
                        continue
                    if raw is not None:
                        _write_raw_cache(corpus_root, pid, name, raw, now)
                        fetched += 1
                    else:
                        _write_miss(corpus_root, pid, name, now)
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
            # Merge carried-forward + this run's, deduped by person_id, id-sorted for a stable
            # artifact across runs.
            merged = dict(known)
            for r in rows:
                merged[str(r.get("person_id") or "")] = r
            merged.pop("", None)
            return [merged[k] for k in sorted(merged)], fetched

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
