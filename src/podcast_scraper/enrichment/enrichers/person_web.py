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

**Scope note — metadata only, no image bytes yet.** Records ``image_url`` + attribution but does not
download/host the photo: hosting needs the per-IMAGE license (article text is CC-BY-SA; the photo
may not be), fetched from Wikipedia's ``imageinfo`` API — a follow-up whose raw also lands in the
cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

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


@dataclass(frozen=True)
class PersonWebInfo:
    """One person's DERIVED web enrichment — metadata only (no image bytes)."""

    person_id: str
    name: str
    bio: str | None
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


#: Injectable HTTP opener (mirrors archive/backfill) so tests never touch the network.
Opener = Callable[[urllib.request.Request], Any]


def _default_opener(req: urllib.request.Request) -> Any:
    return urllib.request.urlopen(req, timeout=10)  # noqa: S310 — fixed https host, UA set


class WikipediaProvider:
    """Provider #1 — the Wikipedia REST summary API (bio + thumbnail + article URL)."""

    name = "wikipedia"

    def __init__(self, opener: Opener | None = None, summary_base: str | None = None) -> None:
        self._opener = opener or _default_opener
        base = summary_base or os.environ.get(_WIKIPEDIA_SUMMARY_ENV) or _WIKIPEDIA_SUMMARY_DEFAULT
        self._summary_base = base if base.endswith("/") else base + "/"

    def fetch_raw(self, person_id: str, display_name: str) -> dict[str, Any] | None:
        """GET the REST summary. Best-effort: any network/parse error → None, never raises."""
        title = urllib.parse.quote(display_name.replace(" ", "_"), safe="")
        req = urllib.request.Request(
            self._summary_base + title, headers={"User-Agent": _USER_AGENT}
        )
        try:
            with self._opener(req) as resp:  # type: ignore[union-attr]
                raw = resp.read()
            doc = json.loads(raw)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None
        return doc if isinstance(doc, dict) else None

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
        return PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio=extract.strip(),
            image_url=image_url if isinstance(image_url, str) else None,
            source=self.name,
            source_url=page if isinstance(page, str) else None,
            license="CC-BY-SA 4.0",
        )


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
                if info is not None:
                    rows.append(asdict(info))
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
