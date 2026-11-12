"""Park an episode whose transcription keeps failing, so the nightly stops retrying it.

A failure is held in memory until the batch ends. ``commit_batch`` then counts it, but only when
some other episode in the same batch transcribed: a failure in a batch where nothing transcribed
is an endpoint outage, not this episode, and counting it would park the whole batch. An episode
that reaches ``Config.transcription_park_after_failures`` counted failures is parked: skipped
before download from then on, with an error log line, a GlitchTip event and a policy incident when
it is parked. An episode that transcribes clears its count. A parked episode is released only by
hand (``parked-episodes --release``); an operator can also park one by hand (``--park``).

State lives in ``parked_episodes.json`` beside the run's ``corpus_incidents.jsonl`` (the corpus
parent for a multi-feed batch).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from podcast_scraper.utils.log_redaction import redact_for_log

logger = logging.getLogger(__name__)

PARK_AFTER_FAILURES = 2
STATE_FILE = "parked_episodes.json"
SCHEMA_VERSION = 1
LOG_MARKER = "EPISODE_PARKED"

_lock = threading.Lock()
# state path -> episode id -> failure seen in this batch
_pending_failures: Dict[str, Dict[str, Dict[str, Any]]] = {}
# state path -> episode ids transcribed in this batch
_transcribed: Dict[str, set] = {}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def state_path_for(incident_log_path: Optional[str]) -> Optional[str]:
    """``parked_episodes.json`` beside the incident log; None when the run keeps no log."""
    if not incident_log_path or not str(incident_log_path).strip():
        return None
    return str(Path(incident_log_path).parent / STATE_FILE)


def _load(path: str) -> Dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": SCHEMA_VERSION, "episodes": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("episodes"), dict):
        raise ValueError(f"{path}: not a parked-episodes file")
    return doc


def _save(path: str, doc: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".parked_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, target)


def parked_entry(incident_log_path: Optional[str], episode_id: Optional[str]) -> Optional[Dict]:
    """The episode's record when it is parked, else None."""
    path = state_path_for(incident_log_path)
    if path is None or not episode_id:
        return None
    try:
        entry = _load(path)["episodes"].get(episode_id)
    except (OSError, ValueError) as exc:
        # An unreadable state file must not stop the nightly; it is logged and nothing is parked.
        logger.error("parked-episodes state unreadable, nothing parked: %s", exc)
        return None
    if isinstance(entry, dict) and entry.get("parked_at"):
        return entry
    return None


def note_failure(
    incident_log_path: Optional[str],
    episode_id: Optional[str],
    *,
    feed_url: Optional[str],
    exception_type: str,
    message: str,
) -> None:
    """Hold one transcription failure for this batch; ``commit_batch`` decides if it counts."""
    path = state_path_for(incident_log_path)
    if path is None or not episode_id:
        return
    with _lock:
        _pending_failures.setdefault(path, {})[episode_id] = {
            "at": _now(),
            "feed_url": feed_url,
            "exception_type": exception_type,
            "message": redact_for_log(message, max_len=500),
        }


def note_transcribed(incident_log_path: Optional[str], episode_id: Optional[str]) -> None:
    """Record a transcription that succeeded in this batch (clears the episode's count)."""
    path = state_path_for(incident_log_path)
    if path is None or not episode_id:
        return
    with _lock:
        _transcribed.setdefault(path, set()).add(episode_id)


def _alert_parked(episode_id: str, entry: Dict[str, Any], state_path: str) -> None:
    last = entry["failures"][-1]
    logger.error(
        "%s episode_id=%s feed_url=%s failures=%d last=%s: %s; skipped until released with "
        "`parked-episodes --output-dir %s --release %s`",
        LOG_MARKER,
        episode_id,
        entry.get("feed_url"),
        len(entry["failures"]),
        last["exception_type"],
        last["message"],
        Path(state_path).parent,
        episode_id,
    )
    try:
        from podcast_scraper.utils.sentry_init import capture_enrichment_message

        capture_enrichment_message(
            f"Episode parked after {len(entry['failures'])} transcription failures: {episode_id}",
            level="error",
            tags={"stage": "transcription", "episode_id": episode_id, "park": "parked"},
        )
    except Exception:  # pragma: no cover - telemetry must never break the pipeline
        logger.debug("park alert to GlitchTip skipped", exc_info=True)


def commit_batch(
    incident_log_path: Optional[str], park_after: int = PARK_AFTER_FAILURES
) -> Dict[str, Any]:
    """Count this batch's failures and park the episodes that reached ``park_after`` (0: off).

    Returns what it did: ``counted`` (bool), ``parked`` and ``cleared`` (episode ids).
    """
    path = state_path_for(incident_log_path)
    with _lock:
        failures = _pending_failures.pop(path, {}) if path else {}
        transcribed = _transcribed.pop(path, set()) if path else set()
    result: Dict[str, Any] = {"counted": False, "parked": [], "cleared": []}
    if path is None or park_after <= 0 or (not failures and not transcribed):
        return result
    if failures and not transcribed:
        logger.warning(
            "park: %d transcription failure(s) not counted, no episode transcribed in this batch "
            "(an endpoint outage is not the episode's fault): %s",
            len(failures),
            ", ".join(sorted(failures)),
        )
        return result
    try:
        doc = _load(path)
    except (OSError, ValueError) as exc:
        logger.error("parked-episodes state unreadable, this batch not counted: %s", exc)
        return result
    episodes = doc["episodes"]
    for eid in sorted(transcribed):
        if eid in episodes and not episodes[eid].get("parked_at"):
            del episodes[eid]
            result["cleared"].append(eid)
    for eid, failure in sorted(failures.items()):
        entry = episodes.setdefault(
            eid, {"feed_url": failure.get("feed_url"), "failures": [], "parked_at": None}
        )
        if entry.get("parked_at"):
            continue
        entry["failures"].append({k: failure[k] for k in ("at", "exception_type", "message")})
        if len(entry["failures"]) >= park_after:
            entry["parked_at"] = _now()
            result["parked"].append(eid)
    doc["schema_version"] = SCHEMA_VERSION
    doc["park_after_failures"] = park_after
    _save(path, doc)
    result["counted"] = True
    for eid in result["parked"]:
        _alert_parked(eid, episodes[eid], path)
        from podcast_scraper.utils.corpus_incidents import append_corpus_incident

        append_corpus_incident(
            incident_log_path,
            scope="episode",
            category="policy",
            message=f"parked after {len(episodes[eid]['failures'])} transcription failures",
            exception_type="EpisodeParked",
            stage="transcription",
            feed_url=episodes[eid].get("feed_url"),
            episode_id=eid,
        )
    return result


def list_episodes(corpus_dir: str) -> Dict[str, Any]:
    """The state file of a corpus parent: parked episodes and those still counting."""
    return _load(str(Path(corpus_dir) / STATE_FILE))


def park(corpus_dir: str, episode_ids: Iterable[str], *, reason: str) -> List[str]:
    """Park episodes by hand (an operator who already knows an episode hangs).

    Returns the ids newly parked; one already parked is left as it is.
    """
    path = str(Path(corpus_dir) / STATE_FILE)
    with _lock:
        doc = _load(path)
        episodes = doc["episodes"]
        newly = []
        for eid in dict.fromkeys(episode_ids):
            entry = episodes.setdefault(eid, {"feed_url": None, "failures": [], "parked_at": None})
            if entry.get("parked_at"):
                continue
            entry["failures"].append(
                {"at": _now(), "exception_type": "ParkedByHand", "message": reason}
            )
            entry["parked_at"] = _now()
            newly.append(eid)
        if newly:
            doc["schema_version"] = SCHEMA_VERSION
            _save(path, doc)
    for eid in newly:
        logger.warning("%s parked by hand episode_id=%s: %s", LOG_MARKER, eid, reason)
    return newly


def release(corpus_dir: str, episode_ids: Iterable[str], *, all_parked: bool = False) -> List[str]:
    """Drop parked episodes (and their failure history) so the next batch tries them again.

    Raises KeyError naming any id that is not parked; nothing is changed then.
    """
    path = str(Path(corpus_dir) / STATE_FILE)
    with _lock:
        doc = _load(path)
        episodes = doc["episodes"]
        parked = sorted(e for e, v in episodes.items() if v.get("parked_at"))
        wanted = parked if all_parked else list(dict.fromkeys(episode_ids))
        unknown = [e for e in wanted if e not in parked]
        if unknown:
            raise KeyError(", ".join(unknown))
        for eid in wanted:
            del episodes[eid]
        if wanted:
            _save(path, doc)
    for eid in wanted:
        logger.warning("%s released episode_id=%s", LOG_MARKER, eid)
    return wanted


def _reset_for_tests() -> None:
    with _lock:
        _pending_failures.clear()
        _transcribed.clear()
