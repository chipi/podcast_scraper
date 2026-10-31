"""The identity ledger: every cross-episode identity decision, recorded (ADR-166, RFC-131).

A deterministic step builds the ledger from corpus text and writes it next to the other derived
corpus files. The server applies it at read time. The ledger IS the decision, not a description
written beside it, so what is recorded and what is served cannot disagree.

Failure mode (ADR-166): when the ledger is missing, unreadable, or written by another schema or
rule version, there are NO cross-episode joins. Every person id stands alone, exactly as on the
episode it came from. The in-memory unguarded build is never a serving fallback: it is the very
map the guards (ADR-167) exist to correct.

A corpus newer than the ledger (an ingest, a migration or a relabel ran since) is served from the
ledger as it is. New episodes have no joins yet; a changed episode keeps its previous joins until
the next build. ``ledger_status`` reports that gap.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The ledger and its change log, at the corpus root beside the other derived corpus files.
LEDGER_NAME = "identity_ledger.json"
CHANGES_NAME = "identity_ledger_changes.jsonl"
#: The last valid ledger before the current one: the undo. ``restore_previous`` puts it back. A
#: missing ledger means no cross-episode joins at all (below), so the undo is never "delete it".
PREVIOUS_NAME = "identity_ledger.previous.json"

#: The file's shape. A reader refuses any other (ADR-166).
LEDGER_SCHEMA_VERSION = "identity-ledger/1"

#: The rules that decided the joins. Bump it whenever a rule or guard changes what joins, so a
#: ledger written by the previous rules is refused (no joins) until it is rebuilt.
RULE_VERSION = "cluster-person-no-bridge-guarded/2"

#: The person rule the ledger's joins use (RFC-131 S4): mode ``cluster_person`` (one-word names
#: join only as one spelling), no one-word bridge between full names, and the guards (ADR-167).
PERSON_MODE = "cluster_person"
PERSON_LINKAGE = "no_one_word_bridge"

#: Ledger states, for logs and the state gauge.
STATE_OK = "ok"
STATE_MISSING = "missing"
STATE_UNREADABLE = "unreadable"
STATE_VERSION_MISMATCH = "version_mismatch"

#: Changes listed per build in the change log; the counts are always complete.
_MAX_LISTED_CHANGES = 200


def corpus_token(root: Path | str) -> Dict[str, Any]:
    """The corpus state a ledger was built from: content mtime and the newest-run kg count.

    The content mtime excludes the ledger itself (``perf_cache.corpus_content_mtime``): a token
    that included the file it is written into could never match after the write.
    """
    from podcast_scraper.kg.corpus import newest_run_artifact_paths, scan_kg_artifact_paths
    from podcast_scraper.perf_cache import corpus_content_mtime

    root = Path(root)
    kg = newest_run_artifact_paths(root, scan_kg_artifact_paths(root), ".kg.json")
    return {"corpus_mtime": corpus_content_mtime(root), "newest_kg_count": len(kg)}


def build_ledger(root: Path | str) -> Dict[str, Any]:
    """Build the ledger for a corpus root, from corpus text only. Writes nothing.

    The joins are the cluster rule's for organisations. For people: rule mode ``PERSON_MODE``,
    linkage ``PERSON_LINKAGE``, and the identity guards inside the join test (ADR-167); each
    refused join is recorded in ``blocked_joins``.
    """
    import podcast_scraper
    from podcast_scraper.identity.guards import candidate_guard, collect_episode_evidence
    from podcast_scraper.kg.entity_clusters import (
        build_entity_canonical_map,
        collect_entity_candidates,
    )

    root = Path(root)
    payload, _id_map = build_entity_canonical_map(
        collect_entity_candidates(root),
        guard=candidate_guard(collect_episode_evidence(root)),
        person_mode=PERSON_MODE,
        linkage=PERSON_LINKAGE,
    )
    blocked = payload.pop("blocked_joins", [])
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "code_version": getattr(podcast_scraper, "__version__", "unknown"),
        "corpus_token": corpus_token(root),
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "clusters_payload": payload,
        "blocked_joins": blocked,
    }


def ledger_id_map(ledger: Dict[str, Any]) -> Dict[str, str]:
    """The ``variant -> canonical`` map a ledger records."""
    from podcast_scraper.kg.entity_clusters import id_map_from_clusters_payload

    return id_map_from_clusters_payload(ledger.get("clusters_payload") or {})


def read_ledger(root: Path | str) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(ledger, state)``. The ledger is None unless the state is ``ok``."""
    path = Path(root) / LEDGER_NAME
    try:
        with open(path, encoding="utf-8") as fh:
            ledger = json.load(fh)
    except FileNotFoundError:
        return None, STATE_MISSING
    except (OSError, ValueError):
        return None, STATE_UNREADABLE
    if not isinstance(ledger, dict):
        return None, STATE_UNREADABLE
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA_VERSION
        or ledger.get("rule_version") != RULE_VERSION
    ):
        return None, STATE_VERSION_MISMATCH
    if not isinstance(ledger.get("clusters_payload"), dict):
        return None, STATE_UNREADABLE
    return ledger, STATE_OK


def served_id_map(root: Path | str) -> Dict[str, str]:
    """The identity map the server applies: the ledger's, or no joins (ADR-166)."""
    ledger, state = read_ledger(root)
    _observe_state(state)
    if ledger is None:
        logger.warning(
            "identity ledger %s at %s: serving no cross-episode joins (ADR-166)",
            state,
            Path(root) / LEDGER_NAME,
        )
        return {}
    return ledger_id_map(ledger)


def ledger_status(root: Path | str) -> Dict[str, Any]:
    """State, versions and staleness of a corpus's ledger, for reports and checks."""
    ledger, state = read_ledger(root)
    out: Dict[str, Any] = {"state": state}
    if ledger is None:
        return out
    current = corpus_token(root)
    built = ledger.get("corpus_token") or {}
    out.update(
        {
            "built_at": ledger.get("built_at"),
            "code_version": ledger.get("code_version"),
            "corpus_token_at_build": built,
            "corpus_token_now": current,
            "corpus_newer_than_ledger": (current.get("corpus_mtime") or 0)
            > (built.get("corpus_mtime") or 0),
            "kg_episodes_since_build": (current.get("newest_kg_count") or 0)
            - (built.get("newest_kg_count") or 0),
        }
    )
    return out


def write_ledger(root: Path | str, ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``ledger`` atomically and append what changed to the change log. Returns the change.

    The ledger it replaces, when valid, is kept as ``PREVIOUS_NAME`` first (the undo).

    Atomic: a temp file in the corpus root, fsync, then ``os.replace``, so a reader never sees a
    half-written ledger (it would read as unreadable, which means no joins). When run as root the
    files take the corpus root's owner, as every corpus writer must.
    """
    root = Path(root)
    previous, _state = read_ledger(root)
    change = _change_entry(previous, ledger)
    if previous is not None:
        _write_atomic(root, root / PREVIOUS_NAME, previous)
    _write_atomic(root, root / LEDGER_NAME, ledger)
    _append_change(root, change)
    return change


def refresh_ledger(root: Path | str, *, after: str, log: Optional[logging.Logger] = None) -> bool:
    """Build and write the ledger as the last stage of a corpus-writing path (plan V3, D1).

    Called after a successful ``enrich`` run (the post-pipeline chain queues one after every
    ingest and reprocess batch), an ``upgrade run`` that applied a migration, and an
    ``enrich-edges`` pass that rewrote artifacts. True when the ledger was written. On failure the
    previous ledger stays as it was (joins as of the last build, never none) and the error is
    logged; the caller reports it in its exit code, so it is never silent.
    """
    log = log or logger
    try:
        change = write_ledger(root, build_ledger(root))
    except Exception:  # noqa: BLE001 - any build or write failure leaves the previous ledger
        log.exception("identity ledger refresh after %s failed; the previous ledger stays", after)
        return False
    log.info(
        "identity ledger refreshed after %s: %d joins (%d added, %d removed, %d moved)",
        after,
        change["joins"],
        change["joins_added"],
        change["joins_removed"],
        change["canonical_changed"],
    )
    return True


def restore_previous(root: Path | str) -> Dict[str, Any]:
    """Put the previous valid ledger back (the undo of the last write). Returns the change.

    Refuses (``FileNotFoundError`` / ``ValueError``) when there is no valid previous ledger, rather
    than leave the corpus with none: no ledger means no cross-episode joins.
    """
    root = Path(root)
    path = root / PREVIOUS_NAME
    with open(path, encoding="utf-8") as fh:
        previous = json.load(fh)
    if (
        not isinstance(previous, dict)
        or previous.get("schema_version") != LEDGER_SCHEMA_VERSION
        or not isinstance(previous.get("clusters_payload"), dict)
    ):
        raise ValueError(f"{path} is not a valid identity ledger; nothing restored")
    current, _state = read_ledger(root)
    change = _change_entry(current, previous)
    change["restored_from"] = PREVIOUS_NAME
    _write_atomic(root, root / LEDGER_NAME, previous)
    _append_change(root, change)
    return change


def _write_atomic(root: Path, target: Path, payload: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".identity_ledger.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        _match_corpus_owner(root, Path(tmp))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append_change(root: Path, change: Dict[str, Any]) -> None:
    changes = root / CHANGES_NAME
    with open(changes, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(change, ensure_ascii=False, sort_keys=True) + "\n")
    _match_corpus_owner(root, changes)


def _change_entry(previous: Optional[Dict[str, Any]], ledger: Dict[str, Any]) -> Dict[str, Any]:
    """What this build changes against the previous valid ledger: joins added, removed, moved."""
    before = ledger_id_map(previous) if previous else {}
    after = ledger_id_map(ledger)
    added = sorted(k for k in after if k not in before)
    removed = sorted(k for k in before if k not in after)
    moved = sorted(k for k in after if k in before and after[k] != before[k])

    def listed(keys: List[str], src: Dict[str, str], dst: Dict[str, str]) -> List[Any]:
        return [[k, src.get(k), dst.get(k)] for k in keys[:_MAX_LISTED_CHANGES]]

    return {
        "built_at": ledger.get("built_at"),
        "code_version": ledger.get("code_version"),
        "rule_version": ledger.get("rule_version"),
        "corpus_token": ledger.get("corpus_token"),
        "previous_built_at": previous.get("built_at") if previous else None,
        "joins": len(after),
        "joins_added": len(added),
        "joins_removed": len(removed),
        "canonical_changed": len(moved),
        "added": listed(added, before, after),
        "removed": listed(removed, before, after),
        "changed": listed(moved, before, after),
    }


def _match_corpus_owner(root: Path, path: Path) -> None:
    """Give ``path`` the corpus root's owner when running as root; a no-op otherwise."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    st = root.stat()
    os.chown(path, st.st_uid, st.st_gid)


_GAUGE: Dict[str, Any] = {"done": False, "gauge": None}


def _observe_state(state: str) -> None:
    """Set ``podcast_identity_ledger_state{state}`` (1 for the current state), when enabled."""
    if os.environ.get("PODCAST_METRICS_ENABLED", "").strip().lower() not in {"1", "true", "yes"}:
        return
    if not _GAUGE["done"]:
        _GAUGE["done"] = True
        try:
            from prometheus_client import Gauge
        except ImportError:
            return
        _GAUGE["gauge"] = Gauge(
            "podcast_identity_ledger_state",
            "Identity ledger state the server read (1 = current); not ok means no joins.",
            ["state"],
        )
    gauge = _GAUGE["gauge"]
    if gauge is None:
        return
    for s in (STATE_OK, STATE_MISSING, STATE_UNREADABLE, STATE_VERSION_MISMATCH):
        gauge.labels(state=s).set(1 if s == state else 0)
