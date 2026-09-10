"""Person web-enrichment (wave-G) — the first ``EnricherTier.WEB`` enricher.

Fetches a short bio + a photo URL + attribution for each Person in the corpus from an external web
source (Wikipedia first), writing ``person_web.json``. Structured as a GENERAL web enricher with a
pluggable provider so more sources can be added: :class:`PersonWebProvider` is the seam,
:class:`WikipediaProvider` is provider #1.

**Airgap contract.** This is a WEB-tier enricher: it reaches a third party at enrichment time. It
runs ONLY in non-airgapped profiles (never the deterministic CI profile), and the provider is
INJECTED — tests pass a fake provider so no live call is made, and the network client is a plain
injectable opener (no new dependency). The base owns this contract so every future provider
inherits it.

**Scope of this commit — metadata only, no image bytes.** It records each person's ``image_url`` +
``attribution`` (source + license), but does NOT download or host the image yet: hosting an image
requires resolving THAT image's license (Wikipedia article text is CC-BY-SA, but an embedded photo
can be any license), which is a deliberate follow-up. Recording the URL/attribution now is safe;
storing bytes is not until the per-image license is checked.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

#: Bound the number of distinct persons enriched per run (polite to the upstream; the WEB tier is
#: opt-in and low-concurrency, but a huge corpus should still not fan out unboundedly).
_DEFAULT_MAX_PERSONS = 200
_USER_AGENT = "close-listening/1.0 (podcast knowledge base; contact via app)"


@dataclass(frozen=True)
class PersonWebInfo:
    """One person's web enrichment — metadata only (no image bytes; see module docstring)."""

    person_id: str
    name: str
    bio: str | None
    image_url: str | None
    source: str  # provider label, e.g. "wikipedia"
    source_url: str | None
    license: str | None  # of the ARTICLE text (image license resolved before any hosting)


class PersonWebProvider(Protocol):
    """The pluggable web source. A provider is pure fetch: name in, info out (or None)."""

    name: str

    def fetch(self, person_id: str, display_name: str) -> PersonWebInfo | None:
        """Return web info for a person, or None when nothing is found / the fetch fails."""
        ...


#: Injectable HTTP opener (mirrors archive/backfill) so tests never touch the network.
Opener = Callable[[urllib.request.Request], Any]


def _default_opener(req: urllib.request.Request) -> Any:
    return urllib.request.urlopen(req, timeout=10)  # noqa: S310 — fixed https host, UA set


class WikipediaProvider:
    """Provider #1 — the Wikipedia REST summary API (bio + thumbnail + article URL).

    Best-effort and total: any network / parse / not-found condition returns None rather than
    raising, so one missing person never fails the run. The image URL is recorded but NOT hosted
    here (its license is resolved in the hosting follow-up); the recorded ``license`` is the
    ARTICLE text's (CC-BY-SA), which is what we would attribute for the bio.
    """

    name = "wikipedia"
    _SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self, opener: Opener | None = None) -> None:
        self._opener = opener or _default_opener

    def fetch(self, person_id: str, display_name: str) -> PersonWebInfo | None:
        title = urllib.parse.quote(display_name.replace(" ", "_"), safe="")
        req = urllib.request.Request(self._SUMMARY + title, headers={"User-Agent": _USER_AGENT})
        try:
            with self._opener(req) as resp:  # type: ignore[union-attr]
                raw = resp.read()
            doc = json.loads(raw)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None
        if not isinstance(doc, dict) or doc.get("type") == "disambiguation":
            return None
        extract = doc.get("extract")
        if not isinstance(extract, str) or not extract.strip():
            return None
        thumb = doc.get("thumbnail")
        image_url = thumb.get("source") if isinstance(thumb, dict) else None
        page = ((doc.get("content_urls") or {}).get("desktop") or {}).get("page")
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
        requires_opt_in=True,  # external fetch — opt-in, never in the airgapped CI profile
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
        """Enumerate persons, fetch each via the provider (off the loop), write person_web.json."""
        max_persons = int(config.get("max_persons", _DEFAULT_MAX_PERSONS))
        persons = _distinct_persons(all_bundles or [])[:max_persons]

        def _run() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for pid, name in persons:
                if ctx.cancel_event.is_set():
                    break
                info = self._provider.fetch(pid, name)
                if info is not None:
                    rows.append(asdict(info))
            return rows

        try:
            rows = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 — enrich() never raises out of itself
            return EnricherResult(status="failed", error=str(exc), error_class=type(exc).__name__)
        _logger.info(
            "person_web enriched %d/%d persons run_id=%s provider=%s",
            len(rows),
            len(persons),
            ctx.run_id,
            self._provider.name,
        )
        return EnricherResult(
            status=STATUS_OK,
            data={"provider": self._provider.name, "persons": rows},
            records_written=len(rows),
        )
