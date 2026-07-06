"""Corpus ingestion primitive (#1069) — bring one feed into the corpus.

The durable write-path spine shared by both delivery phases (PRD-037): the
operator-curated corpus growth of phase 1, and the user bring-your-own-shows
self-serve of phase 2. Both call :func:`ingest_feed` through an
:class:`~podcast_scraper.ingestion.policy.IngestPolicy`; phase 1 uses the no-op
:class:`~podcast_scraper.ingestion.policy.AllowAllPolicy`, phase 2 slots its
per-user guardrails into the same seam without touching the ingest path.

See ``docs/wip/player/1069-SCRAPE-ON-DEMAND-SCOPE-ANALYSIS.md``.
"""

from __future__ import annotations

from podcast_scraper.ingestion.policy import (
    AllowAllPolicy,
    IngestNotAuthorized,
    IngestPolicy,
    IngestRequest,
)
from podcast_scraper.ingestion.primitive import ingest_feed, IngestResult

__all__ = [
    "AllowAllPolicy",
    "IngestNotAuthorized",
    "IngestPolicy",
    "IngestRequest",
    "IngestResult",
    "ingest_feed",
]
