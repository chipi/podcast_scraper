"""Decide whether an episode's enrichment output is still valid (#1649).

Enrichment has never had a skip path. ``discover_episode_bundles`` walks every bundle,
``executor`` loops over all of them unconditionally, and ``--only``/``--skip``/health gating
select **enrichers**, never **episodes**. So a 16-episode ingest triggers a full-corpus pass.
That was invisible while the pass was a 3 ms no-op (#1648) — reprocessing nothing is free —
and became real work the moment enrichment started producing output.

The trap this module exists to avoid
------------------------------------
``envelope.py`` already persists ``computed_at``, ``enricher_version`` and ``schema_version``,
so the obvious staleness key is those three. **That key would defeat the corpus repair.**
Fix speaker attribution upstream (#1646), re-run enrichment, and every episode reads as
"unchanged at the same enricher version" — all 678 skipped, the fix invisible, the run green.
It is the same shape of failure as the bug being repaired: a signal that reports on the
machinery instead of on the work.

So the key includes the **input identity**:

    stale = f(input fingerprint, enricher_version, schema_version)

An upstream change to GI/KG invalidates downstream enrichment automatically, with nobody
remembering to pass a flag. ``--force`` exists for re-running at an unchanged version, and is
an override — never the mechanism correctness depends on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Bump when the fingerprint's INPUTS change (which files, or how they are hashed) so old
# envelopes are treated as stale rather than compared against an incompatible fingerprint.
FINGERPRINT_VERSION = "1"


@dataclass(frozen=True)
class StalenessDecision:
    """Why an episode is being enriched or skipped — the reason is the point.

    "612 skipped" is a number; "612 skipped: input unchanged" versus "612 skipped: no
    fingerprint recorded" are different situations, and the second one means the incrementality
    is not actually working.
    """

    should_run: bool
    reason: str
    fingerprint: Optional[str] = None


def _hash_file(path: Path, hasher: "hashlib._Hash") -> bool:
    """Fold one file's bytes into *hasher*. Returns False when it could not be read.

    Content, not mtime: a re-run that rewrites an identical file (which this pipeline does
    routinely — see C2 in #1654) must not read as a change, or incrementality degrades to
    "always run" and we are back where we started.
    """
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return True
    except OSError:
        return False


def input_fingerprint(paths: Iterable[Optional[Path]]) -> Optional[str]:
    """Content fingerprint over an episode's enrichment INPUTS.

    Returns ``None`` when nothing could be read — an unreadable input must not produce a
    stable-looking fingerprint, because that would let a broken episode be skipped forever.
    """
    hasher = hashlib.sha256()
    hasher.update(FINGERPRINT_VERSION.encode("utf-8"))
    read_any = False
    for path in sorted((p for p in paths if p is not None), key=str):
        hasher.update(str(path.name).encode("utf-8"))
        if _hash_file(path, hasher):
            read_any = True
    return hasher.hexdigest() if read_any else None


def envelope_is_current(
    envelope: Optional[dict[str, Any]],
    *,
    fingerprint: Optional[str],
    enricher_version: str,
    schema_version: str,
) -> StalenessDecision:
    """Decide whether existing output still stands for this input and this enricher.

    Every "run it" answer carries its reason so the run stats can say *why* work happened,
    which is what makes an incrementality regression visible instead of merely slow.
    """
    if envelope is None:
        return StalenessDecision(True, "no_previous_output", fingerprint)
    if not isinstance(envelope, dict):
        return StalenessDecision(True, "unreadable_previous_output", fingerprint)

    if envelope.get("enricher_version") != enricher_version:
        return StalenessDecision(True, "enricher_version_changed", fingerprint)
    if envelope.get("schema_version") != schema_version:
        return StalenessDecision(True, "schema_version_changed", fingerprint)

    previous = envelope.get("input_fingerprint")
    if not previous:
        # Output predates fingerprinting. Re-run once so it acquires one; skipping here would
        # freeze every pre-#1649 episode as permanently "current".
        return StalenessDecision(True, "no_recorded_fingerprint", fingerprint)
    if fingerprint is None:
        return StalenessDecision(True, "inputs_unreadable", fingerprint)
    if previous != fingerprint:
        return StalenessDecision(True, "inputs_changed", fingerprint)

    return StalenessDecision(False, "inputs_unchanged", fingerprint)


def load_envelope(path: Path) -> Optional[dict[str, Any]]:
    """Read a previously written envelope; None when absent or unparsable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class EnrichmentRunStats:
    """Per-run incrementality counters, mirroring ``IndexRunStats`` (#1649).

    Indexing already reports ``episodes_skipped_unchanged=646, episodes_reindexed=16``.
    Enrichment reported nothing comparable, which is *why* nobody noticed it had no
    incrementality at all: there was no number that could have looked wrong.
    """

    episodes_total: int = 0
    episodes_enriched: int = 0
    episodes_skipped_unchanged: int = 0
    forced: bool = False
    reasons: Optional[dict[str, int]] = None

    def record(self, decision: StalenessDecision) -> None:
        """Fold one episode's staleness decision into the counters, keyed by its reason."""
        self.episodes_total += 1
        if decision.should_run:
            self.episodes_enriched += 1
        else:
            self.episodes_skipped_unchanged += 1
        if self.reasons is None:
            self.reasons = {}
        self.reasons[decision.reason] = self.reasons.get(decision.reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Serialise the counters for the run summary."""
        return {
            "episodes_total": self.episodes_total,
            "episodes_enriched": self.episodes_enriched,
            "episodes_skipped_unchanged": self.episodes_skipped_unchanged,
            "forced": self.forced,
            "reasons": dict(self.reasons or {}),
        }

    def log_summary(self) -> None:
        """Log the incrementality line — the number whose absence hid #1649."""
        logger.info(
            "enrichment incrementality: %d/%d episodes enriched, %d skipped unchanged "
            "(forced=%s, reasons=%s)",
            self.episodes_enriched,
            self.episodes_total,
            self.episodes_skipped_unchanged,
            self.forced,
            dict(self.reasons or {}),
        )
