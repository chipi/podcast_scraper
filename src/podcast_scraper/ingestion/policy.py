"""Authorization / guardrail seam for corpus ingestion (#1069).

Every ingest request passes through an :class:`IngestPolicy` **before** any
pipeline work begins. This is the seam that keeps the two PRD-037 phases on one
code path:

* **Phase 1 (curated, operator-driven)** uses :class:`AllowAllPolicy` — a no-op.
  The operator grows the corpus; there is nothing to rate-limit.
* **Phase 2 (user bring-your-own-shows)** implements the same Protocol with
  per-user rate limits, quotas, cost bounds, and abuse prevention. Because the
  guardrails live *here*, adding them later is a new policy — not a rewrite of
  the ingest path.

Rejections raise :class:`IngestNotAuthorized`; the caller maps that to a 4xx (an
endpoint) or a non-zero exit (the CLI). Designing the seam in now — while the
self-serve phase is committed but unbuilt — is the one piece of future-proofing
that is justified, precisely because phase 2 is a certainty, not a maybe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


class IngestNotAuthorized(Exception):
    """Raised by an :class:`IngestPolicy` to reject an ingest request."""


@dataclass(frozen=True)
class IngestRequest:
    """One request to bring a feed (or a specific episode) into the corpus.

    ``episode_guid`` is ``None`` for a whole-feed ingest (the phase-1 shape — the
    single-feed pipeline is incrementally deduped per episode). ``actor`` is who
    asked — ``None`` means the operator / system (phase 1); phase 2 sets a user
    id so per-user policies can scope quotas.
    """

    feed_url: str
    episode_guid: Optional[str] = None
    actor: Optional[str] = None


@runtime_checkable
class IngestPolicy(Protocol):
    """Authorizes (or rejects) an ingest request before any pipeline work."""

    def authorize(self, request: IngestRequest) -> None:
        """Return to allow the request; raise :class:`IngestNotAuthorized` to reject."""
        ...


class AllowAllPolicy:
    """Phase-1 operator policy: allow every request (no guardrails)."""

    def authorize(self, request: IngestRequest) -> None:
        """Allow unconditionally — the operator path has nothing to gate."""
        return None


__all__ = [
    "AllowAllPolicy",
    "IngestNotAuthorized",
    "IngestPolicy",
    "IngestRequest",
]
