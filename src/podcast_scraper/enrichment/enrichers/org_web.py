"""Organization web-enrichment (#2035) — the org analog of ``person_web``.

Mirrors :mod:`person_web` (``EnricherTier.WEB``, two-phase FETCH → raw cache → pure DERIVE) but
keyed on ``Organization`` GI nodes and sourced from **Wikidata** (structured: a logo P154,
description, founded P571, official site P856). Reuses person_web's proven image-hosting +
resilient-client primitives; only the provider (a different API) and the org id/path plumbing are
new.

**Logos are frequently non-free.** The same license gate person photos use (``no license ⇒ don't
host``) refuses most company logos, so a matched org typically gets a description without a logo —
the org card degrades to text, which is fine.

**Airgap contract.** WEB-tier, held by PROFILE membership (cloud/prod only, never airgapped/CI); the
provider is injected (tests pass a fake → no live call) and DERIVE is pure over cached raw.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from podcast_scraper.enrichment.enrichers._loaders import load_gi, nodes_of_type
from podcast_scraper.enrichment.enrichers.person_web import (
    _build_web_client,
    _EXT_MEDIA,
    _IMAGE_ALLOWED,
    _image_ext,
    _IMAGE_MAX_BYTES,
    _image_sniff_ok,
    _ImageSkip,
    _USER_AGENT,
    FetchedImage,
    IMAGE_SKIP,
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

_logger = logging.getLogger(__name__)

#: Bound the distinct orgs enriched per run (polite to the upstream).
_DEFAULT_MAX_ORGS = 200
#: Raw provider payloads, one file per org, under the corpus enrichments dir.
_RAW_SUBDIR = "org_web_raw"
#: Hosted org logos (served by /api/app/organizations/{id}/logo).
_LOGO_SUBDIR = "org_logos"

#: Wikidata endpoints — env-overridable so the e2e/mock server can stand in for the live host.
_WIKIDATA_API_ENV = "APP_WIKIDATA_API_BASE"
_WIKIDATA_API_DEFAULT = "https://www.wikidata.org/w/api.php"
#: Commons action API (imageinfo → per-file license) + file-bytes base.
_COMMONS_API_ENV = "APP_COMMONS_API_BASE"
_COMMONS_API_DEFAULT = "https://commons.wikimedia.org/w/api.php"
_COMMONS_FILEPATH_DEFAULT = "https://commons.wikimedia.org/wiki/Special:FilePath/"


def _safe_name(org_id: str) -> str:
    """A filesystem-safe stem for an org id (mirrors person_web's; strips every path separator)."""
    slug = re.sub(r"[^a-z0-9._-]", "_", org_id.split(":", 1)[-1].lower())
    return slug or re.sub(r"[^a-z0-9]", "_", org_id.lower()) or "org"


def _logo_dir(corpus_root: Path) -> Path:
    return corpus_root / "enrichments" / _LOGO_SUBDIR


def org_logo_path(corpus_root: Path, org_id: str) -> tuple[Path, str] | None:
    """The hosted logo ``(path, media_type)`` for an org, or None. Public — the serve route reads
    it. The stem is sanitized + the filename a fixed glob, so the path cannot escape the dir."""
    from podcast_scraper.utils.path_validation import resolves_under_root

    directory = _logo_dir(corpus_root)
    stem = _safe_name(org_id)
    for ext, media in _EXT_MEDIA.items():
        candidate = directory / f"{stem}.{ext}"
        # is_file() follows symlinks, so also require the resolved target stays inside the logo dir.
        if candidate.is_file() and resolves_under_root(candidate, directory):
            return candidate, media
    return None


@dataclass(frozen=True)
class OrgWebInfo:
    """One org's DERIVED web enrichment — metadata only (no image bytes)."""

    org_id: str
    name: str
    description: str | None
    summary: str | None
    source: str  # provider label, e.g. "wikidata"
    source_url: str | None
    logo_url: str | None  # external logo URL (resolved + hosted before serving)
    founded: str | None
    industry: str | None
    website: str | None


class OrgWebProvider(Protocol):
    """The pluggable web source, split into fetch (network) and derive (pure) — as person_web."""

    name: str

    def fetch_raw(self, org_id: str, display_name: str) -> dict[str, Any] | None:
        """Fetch the full upstream payload, or None on miss / failure. Never raises."""
        ...

    def derive(self, org_id: str, display_name: str, raw: dict[str, Any]) -> OrgWebInfo | None:
        """Extract normalized info from a raw payload (pure), or None when it carries nothing."""
        ...


def _distinct_orgs(all_bundles: list[EpisodeArtifactBundle]) -> list[tuple[str, str]]:
    """(org_id, name) for every Organization in the corpus GI, de-duplicated, id-sorted."""
    seen: dict[str, str] = {}
    for b in all_bundles:
        gi = load_gi(b)
        for node in nodes_of_type(gi, "Organization"):
            oid = str(node.get("id") or "")
            if not oid or oid in seen:
                continue
            name = str((node.get("properties") or {}).get("name") or oid)
            seen[oid] = name
    return sorted(seen.items())


class WikidataProvider:
    """Provider #1 — Wikidata: search the entity by name, then read its claims (logo/founded/site).

    ``fetch_raw`` persists the merged ``{search, entity}`` payload so DERIVE re-runs offline. Only
    an entity whose ``instance of`` (P31) is organization-like is accepted — the disambiguation
    guard #2031 flagged (avoids enriching "Apple" the fruit for "Apple" the company)."""

    name = "wikidata"

    #: P31 targets we accept as "an organization" (business, org, company, nonprofit, agency…).
    _ORG_INSTANCE_QIDS = frozenset(
        {"Q4830453", "Q43229", "Q783794", "Q6881511", "Q163740", "Q327333", "Q891723", "Q3918"}
    )

    def __init__(
        self,
        client: httpx.Client | None = None,
        api_base: str | None = None,
        commons_api_base: str | None = None,
        commons_filepath_base: str | None = None,
    ) -> None:
        import os

        self._client = client or _build_web_client()
        self._api_base = api_base or os.environ.get(_WIKIDATA_API_ENV) or _WIKIDATA_API_DEFAULT
        self._commons_api = (
            commons_api_base or os.environ.get(_COMMONS_API_ENV) or _COMMONS_API_DEFAULT
        )
        self._commons_filepath = commons_filepath_base or _COMMONS_FILEPATH_DEFAULT
        self._commons_host = (urllib.parse.urlsplit(self._commons_filepath).hostname or "").lower()

    def _get_json(self, url: str) -> dict[str, Any] | None:
        try:
            resp = self._client.get(url, headers={"User-Agent": _USER_AGENT})
            if resp.status_code != 200:
                return None
            doc = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def fetch_raw(self, org_id: str, display_name: str) -> dict[str, Any] | None:
        """Search Wikidata for the org name → best entity id → its full entity JSON. Best-effort."""
        search_url = (
            f"{self._api_base}?action=wbsearchentities&format=json&language=en&type=item&limit=1"
            f"&search={urllib.parse.quote(display_name)}"
        )
        search = self._get_json(search_url)
        hits = (search or {}).get("search") or []
        if not (isinstance(hits, list) and hits and isinstance(hits[0], dict)):
            return None
        qid = str(hits[0].get("id") or "")
        if not qid:
            return None
        entity = self._get_json(
            f"{self._api_base}?action=wbgetentities&format=json&languages=en&ids={qid}"
        )
        if entity is None:
            return None
        return {"search": search, "entity": entity, "qid": qid}

    @staticmethod
    def _claim_values(entity: dict[str, Any], prop: str) -> list[dict[str, Any]]:
        claims = (entity.get("claims") or {}).get(prop) or []
        out: list[dict[str, Any]] = []
        for c in claims if isinstance(claims, list) else []:
            snak = ((c or {}).get("mainsnak") or {}).get("datavalue") or {}
            val = snak.get("value")
            if isinstance(val, dict):
                out.append(val)
            elif val is not None:
                out.append({"_scalar": val})
        return out

    def derive(self, org_id: str, display_name: str, raw: dict[str, Any]) -> OrgWebInfo | None:
        """Extract description + logo + founded + site from a stored Wikidata payload (pure)."""
        qid = str(raw.get("qid") or "")
        entities = (raw.get("entity") or {}).get("entities") or {}
        entity = entities.get(qid) if isinstance(entities, dict) else None
        if not isinstance(entity, dict):
            return None
        # Disambiguation guard: accept only organization-like entities (P31 instance of).
        instance_qids = {
            str(v.get("id")) for v in self._claim_values(entity, "P31") if isinstance(v, dict)
        }
        if not (instance_qids & self._ORG_INSTANCE_QIDS):
            return None
        desc = ((entity.get("descriptions") or {}).get("en") or {}).get("value")
        description = desc.strip() if isinstance(desc, str) and desc.strip() else None
        # Logo (P154) → a Commons file name → Special:FilePath URL (bytes) resolved later.
        logo_vals = self._claim_values(entity, "P154")
        logo_file = logo_vals[0].get("_scalar") if logo_vals else None
        logo_url = (
            self._commons_filepath + urllib.parse.quote(str(logo_file))
            if isinstance(logo_file, str) and logo_file
            else None
        )
        founded_vals = self._claim_values(entity, "P571")
        founded_raw = founded_vals[0].get("time") if founded_vals else None
        founded = _wikidata_year(founded_raw) if isinstance(founded_raw, str) else None
        site_vals = self._claim_values(entity, "P856")
        website = site_vals[0].get("_scalar") if site_vals else None
        if not description and logo_url is None:
            return None  # nothing worth surfacing
        return OrgWebInfo(
            org_id=org_id,
            name=display_name,
            description=description,
            summary=None,  # Wikidata has no long summary; the description is the one-liner
            source=self.name,
            source_url=f"https://www.wikidata.org/wiki/{qid}" if qid else None,
            logo_url=logo_url,
            founded=founded,
            industry=None,  # P452 is a QID needing a second label lookup — omitted for now
            website=website if isinstance(website, str) else None,
        )

    def _logo_host_allowed(self, url: str) -> bool:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if not host:
            return False
        if host == self._commons_host:
            return True
        return host == "wikimedia.org" or host.endswith((".wikimedia.org", ".wikipedia.org"))

    def _logo_license(self, logo_file_url: str) -> str | None | _ImageSkip:
        """License of the Commons logo file. None → transient (retry); IMAGE_SKIP → no license
        (permanent, do not host); a string → resolved license."""
        # Special:FilePath/<File> → the file name is the last path segment.
        last_seg = urllib.parse.urlsplit(logo_file_url).path.rsplit("/", 1)[-1]
        file_name = urllib.parse.unquote(last_seg)
        query = (
            f"{self._commons_api}?action=query&format=json&prop=imageinfo&iiprop=extmetadata"
            f"&titles=File:{urllib.parse.quote(file_name)}"
        )
        doc = self._get_json(query)
        if doc is None or "error" in doc or "query" not in doc:
            return None  # transient
        pages = (doc.get("query") or {}).get("pages") or {}
        for page in pages.values() if isinstance(pages, dict) else []:
            infos = page.get("imageinfo") if isinstance(page, dict) else None
            meta = (infos[0].get("extmetadata") or {}) if isinstance(infos, list) and infos else {}
            lic = (meta.get("LicenseShortName") or {}).get("value")
            return lic if isinstance(lic, str) and lic.strip() else IMAGE_SKIP
        return IMAGE_SKIP  # answered with no imageinfo → unlicensed → permanent

    def fetch_image(self, image_url: str) -> FetchedImage | _ImageSkip | None:
        """Resolve the logo's license (Commons) then download+validate. Mirrors person_web."""
        license_ = self._logo_license(image_url)
        if license_ is None:
            return None  # transient license lookup → retry next run
        if isinstance(license_, _ImageSkip):
            return IMAGE_SKIP  # no license → never host what we cannot attribute (permanent)
        if not self._logo_host_allowed(image_url):
            return IMAGE_SKIP  # off-allowlist host (SSRF guard)
        try:
            with self._client.stream("GET", image_url, headers={"User-Agent": _USER_AGENT}) as resp:
                if resp.status_code in (404, 410):
                    return IMAGE_SKIP
                if resp.status_code != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > _IMAGE_MAX_BYTES:
                        return IMAGE_SKIP
                data = b"".join(chunks)
        except httpx.HTTPError:
            return None
        ext = _image_ext(content_type)
        if ext is None or not _image_sniff_ok(ext, data):
            return IMAGE_SKIP
        return FetchedImage(data=data, ext=ext, license=license_, artist=None)


def _wikidata_year(time_str: str) -> str | None:
    """Extract the year from a Wikidata time literal like ``+2015-00-00T00:00:00Z``."""
    m = re.match(r"^[+-]?(\d{1,4})-", time_str)
    return m.group(1) if m else None


def _raw_path(corpus_root: Path, org_id: str) -> Path:
    return corpus_root / "enrichments" / _RAW_SUBDIR / f"{_safe_name(org_id)}.json"


def _read_raw_cache(corpus_root: Path, org_id: str) -> dict[str, Any] | None:
    path = _raw_path(corpus_root, org_id)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc.get("payload") if isinstance(doc, dict) else None


def _write_raw_cache(
    corpus_root: Path, org_id: str, name: str, payload: dict[str, Any], ts: int
) -> None:
    path = _raw_path(corpus_root, org_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"org_id": org_id, "name": name, "fetched_at": ts, "payload": payload}),
        encoding="utf-8",
    )


def _logo_meta_path(corpus_root: Path, org_id: str) -> Path:
    return _logo_dir(corpus_root) / f"{_safe_name(org_id)}.meta.json"


def _read_logo_meta(corpus_root: Path, org_id: str) -> dict[str, Any] | None:
    path = _logo_meta_path(corpus_root, org_id)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _write_skip_meta(corpus_root: Path, org_id: str) -> None:
    directory = _logo_dir(corpus_root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(org_id)
    for ext in set(_IMAGE_ALLOWED.values()):
        (directory / f"{stem}.{ext}").unlink(missing_ok=True)
    _logo_meta_path(corpus_root, org_id).write_text(json.dumps({"skip": True}), encoding="utf-8")


def _store_logo(corpus_root: Path, org_id: str, image: FetchedImage) -> dict[str, Any]:
    directory = _logo_dir(corpus_root)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(org_id)
    for ext in set(_IMAGE_ALLOWED.values()):
        (directory / f"{stem}.{ext}").unlink(missing_ok=True)
    (directory / f"{stem}.{image.ext}").write_bytes(image.data)
    meta = {"ext": image.ext, "license": image.license}
    _logo_meta_path(corpus_root, org_id).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return meta


class OrgWebEnricher:
    """Corpus-scope WEB enricher: a description/logo/attribution row per Organization (#2035)."""

    manifest = EnricherManifest(
        id="org_web",
        version="0.1.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.WEB,
        reads=[".gi.json"],
        writes="org_web.json",
        description="Per-org description + logo + basic facts from a web source (Wikidata).",
        requires_opt_in=False,
        expected_duration_s=120,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "max_orgs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": _DEFAULT_MAX_ORGS,
                    "description": "Cap on distinct orgs enriched per run (be polite).",
                },
                "refresh": {
                    "type": "boolean",
                    "default": False,
                    "description": "Re-fetch even when a raw payload is cached.",
                },
            },
        },
    )

    def __init__(self, provider: OrgWebProvider | None = None) -> None:
        self._provider: OrgWebProvider = provider or WikidataProvider()

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Fetch-then-derive per org: cached raw is reused (no network); output is re-derived."""
        max_orgs = int(config.get("max_orgs", _DEFAULT_MAX_ORGS))
        refresh = bool(config.get("refresh", False))
        orgs = _distinct_orgs(all_bundles or [])[:max_orgs]
        now = int(time.time())
        fetch_image = getattr(self._provider, "fetch_image", None)

        def _host_logo(oid: str, logo_url: str) -> dict[str, Any] | None:
            meta = None if refresh else _read_logo_meta(corpus_root, oid)
            if meta is not None:
                if meta.get("skip"):
                    return None
                if org_logo_path(corpus_root, oid) is not None:
                    return meta
            if fetch_image is None:
                return None
            fetched = fetch_image(logo_url)
            if fetched is None:
                return None  # transient — cache nothing, retry next run
            if isinstance(fetched, _ImageSkip):
                _write_skip_meta(corpus_root, oid)
                return None
            return _store_logo(corpus_root, oid, fetched)

        def _run() -> tuple[list[dict[str, Any]], int]:
            rows: list[dict[str, Any]] = []
            fetched = 0
            for oid, name in orgs:
                if ctx.cancel_event.is_set():
                    break
                raw = None if refresh else _read_raw_cache(corpus_root, oid)
                if raw is None:
                    raw = self._provider.fetch_raw(oid, name)
                    if raw is not None:
                        _write_raw_cache(corpus_root, oid, name, raw, now)
                        fetched += 1
                if raw is None:
                    continue
                info = self._provider.derive(oid, name, raw)
                if info is None:
                    continue
                row = asdict(info)
                if info.logo_url:
                    meta = _host_logo(oid, info.logo_url)
                    if meta is not None:
                        row["logo_hosted"] = True
                        row["logo_ext"] = meta.get("ext")
                        row["logo_license"] = meta.get("license")
                rows.append(row)
            return rows, fetched

        try:
            rows, fetched = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 — enrich() never raises out of itself
            return EnricherResult(status="failed", error=str(exc), error_class=type(exc).__name__)
        _logger.info(
            "org_web derived %d/%d orgs (%d freshly fetched) run_id=%s provider=%s",
            len(rows),
            len(orgs),
            fetched,
            ctx.run_id,
            self._provider.name,
        )
        return EnricherResult(
            status=STATUS_OK,
            data={"provider": self._provider.name, "orgs": rows},
            records_written=len(rows),
        )
