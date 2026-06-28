"""Cross-run enricher health persistence.

``.viewer/enrichment_health.json`` records each enricher's health
state across runs:

* ``consecutive_failures`` — across runs (resets on success).
* ``last_run_id`` / ``last_status`` / ``last_run_at`` — most recent
  outcome.
* ``auto_disabled`` — true when consecutive failed runs reached
  ``policy.auto_disable_threshold``; operator-side recovery required.
* ``circuit_state`` + ``cooldown_until`` — cross-run circuit state
  (the per-run circuit is in ``resilience.EnricherCircuitState``;
  this records the persisted post-run snapshot for the viewer
  Operator-tab Enrichment panel + ``enrichment_health`` MCP tool).

Atomic write semantics: the file is written via
``write_text_atomic`` and loaded with corruption recovery (corrupt
file → empty state + WARNING; never raises into the executor).

See ``docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md``
§"Health persistence".
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.envelope import utc_iso_now
from podcast_scraper.enrichment.paths import (
    enrichment_health_path,
    ensure_directory,
    viewer_dir,
)
from podcast_scraper.enrichment.resilience import TierPolicy

logger = logging.getLogger(__name__)

# Bump when the persisted shape changes incompatibly.
HEALTH_SCHEMA_VERSION = "1"


@dataclass
class EnricherHealth:
    """One enricher's persisted health record."""

    consecutive_failures: int = 0
    last_run_at: str | None = None
    last_run_id: str | None = None
    last_status: str | None = None
    auto_disabled: bool = False
    auto_disabled_at: str | None = None
    auto_disabled_reason: str | None = None
    # Cross-run circuit state snapshot.
    circuit_state: str = "closed"
    circuit_opened_at: str | None = None
    cooldown_until: str | None = None

    def is_active(self) -> bool:
        """True when this enricher is eligible to run on the next pass.

        False when auto-disabled or when the cooldown_until window
        has not yet elapsed. The executor calls this before scheduling
        each enricher; False enrichers produce ``status: "skipped"``
        with reason ``auto_disabled`` or ``cooldown_active``.
        """
        if self.auto_disabled:
            return False
        if self.cooldown_until is None:
            return True
        return self.cooldown_until <= utc_iso_now()


class HealthRegistry:
    """In-memory registry mirroring the on-disk health file.

    Use:

        registry = HealthRegistry(corpus_root)
        registry.load()
        ...
        # after each enricher run:
        registry.update_after_run(enricher_id, run_id, "ok", policy)
        # at end of run:
        registry.save()

    Tests use a fresh ``HealthRegistry`` per fixture (no global state).
    """

    def __init__(self, corpus_root: Path) -> None:
        self._corpus_root = corpus_root
        self._states: dict[str, EnricherHealth] = {}

    @property
    def path(self) -> Path:
        return enrichment_health_path(self._corpus_root)

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Load the persisted state, idempotently and safely.

        * Missing file → empty state.
        * Corrupt JSON → log WARNING, empty state, file left untouched
          until the next save (operator can inspect).
        * Schema version mismatch → log WARNING + ignore the file.
        """
        p = self.path
        if not p.is_file():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "enrichment_health.json could not be parsed (%s); "
                "treating as empty until next save",
                exc,
            )
            return
        if not isinstance(payload, dict):
            logger.warning("enrichment_health.json is not a dict; treating as empty")
            return
        version = payload.get("schema_version")
        if version != HEALTH_SCHEMA_VERSION:
            logger.warning(
                "enrichment_health.json schema_version=%r != %r; ignoring file",
                version,
                HEALTH_SCHEMA_VERSION,
            )
            return
        records = payload.get("enrichers") or {}
        if not isinstance(records, dict):
            logger.warning("enrichment_health.json 'enrichers' is not a dict; ignoring")
            return
        for eid, rec in records.items():
            if not isinstance(rec, dict):
                continue
            try:
                self._states[str(eid)] = _from_dict(rec)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "enrichment_health.json: cannot parse record for %r (%s); " "dropping",
                    eid,
                    exc,
                )

    # ------------------------------------------------------------------ save

    def save(self) -> None:
        """Atomically write the registry to disk.

        Creates ``.viewer/`` if it doesn't exist (per chunk-1 lock
        audit §B6 — standalone runs against corpora without a
        ``.viewer/`` directory should not fail).
        """
        ensure_directory(viewer_dir(self._corpus_root))
        payload: dict[str, Any] = {
            "schema_version": HEALTH_SCHEMA_VERSION,
            "enrichers": {eid: asdict(h) for eid, h in self._states.items()},
        }
        _atomic_write_json(self.path, payload)

    # ------------------------------------------------------------------ accessors

    def get(self, enricher_id: str) -> EnricherHealth:
        """Get the health record for *enricher_id* (creating an empty one if absent)."""
        return self._states.setdefault(enricher_id, EnricherHealth())

    def all(self) -> dict[str, EnricherHealth]:
        """Snapshot of all health records (defensive copy)."""
        return dict(self._states)

    def is_active(self, enricher_id: str) -> bool:
        """Whether the named enricher is eligible to run this pass."""
        return self.get(enricher_id).is_active()

    # ------------------------------------------------------------------ mutations

    def update_after_run(
        self,
        enricher_id: str,
        *,
        run_id: str,
        status: str,
        policy: TierPolicy,
        circuit_state: str = "closed",
        circuit_opened_at: str | None = None,
        cooldown_until: str | None = None,
    ) -> EnricherHealth:
        """Record the outcome of one enricher run.

        * ``status == "ok"`` resets ``consecutive_failures`` to 0 and
          clears any prior auto-disable.
        * Failed-class outcomes (failed / timeout / quarantined)
          increment ``consecutive_failures``. When the count reaches
          ``policy.auto_disable_threshold``, ``auto_disabled`` flips
          true.
        * ``cancelled`` / ``skipped`` outcomes neither bump nor reset
          the counter (they aren't bug signals; they're operator
          actions).
        """
        h = self.get(enricher_id)
        h.last_run_id = run_id
        h.last_run_at = utc_iso_now()
        h.last_status = status
        h.circuit_state = circuit_state
        h.circuit_opened_at = circuit_opened_at
        h.cooldown_until = cooldown_until

        if status == "ok":
            h.consecutive_failures = 0
            h.auto_disabled = False
            h.auto_disabled_at = None
            h.auto_disabled_reason = None
        elif status in ("failed", "timeout", "quarantined"):
            h.consecutive_failures += 1
            if not h.auto_disabled and h.consecutive_failures >= policy.auto_disable_threshold:
                h.auto_disabled = True
                h.auto_disabled_at = utc_iso_now()
                h.auto_disabled_reason = (
                    f"{h.consecutive_failures} consecutive failed runs " f"(last status: {status})"
                )
        # cancelled / skipped: leave the counter alone.
        return h

    def re_enable(
        self, enricher_id: str, *, reason: str, clear_cooldown: bool = True
    ) -> EnricherHealth:
        """Operator-side manual recovery.

        Resets ``consecutive_failures`` to 0, clears ``auto_disabled``,
        and (by default) clears any cooldown window. Stamps
        ``auto_disabled_reason`` with the operator-supplied *reason*
        for the audit trail before clearing the flag.
        """
        h = self.get(enricher_id)
        h.auto_disabled = False
        h.auto_disabled_at = None
        h.auto_disabled_reason = f"re_enabled: {reason}"
        h.consecutive_failures = 0
        if clear_cooldown:
            h.cooldown_until = None
            h.circuit_state = "closed"
            h.circuit_opened_at = None
        return h


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_HEALTH_FIELDS: frozenset[str] = frozenset(
    {f.name for f in EnricherHealth.__dataclass_fields__.values()}
)


def _from_dict(rec: dict[str, Any]) -> EnricherHealth:
    """Construct an EnricherHealth from a persisted dict, tolerating extra keys."""
    kwargs = {k: v for k, v in rec.items() if k in _HEALTH_FIELDS}
    return EnricherHealth(**kwargs)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to *path* atomically via tmp-file + rename.

    Avoids the partial-file corruption window in case of crash mid-write.
    """
    ensure_directory(path.parent)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# Field initialization helper for ``EnricherHealth`` defaults that
# need ``field(default_factory=...)`` semantics — re-export to avoid
# linters complaining about unused imports.
_ = field
