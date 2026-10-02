"""Show-level metadata: what we concluded about a SHOW, kept beside its episodes.

Every fact about a show used to exist only as copies inside each episode's metadata, so "who hosts
this show, where did that come from, and does every episode actually put a host on a voice" could
be answered only by re-reading the whole feed by hand. This module derives it ONCE per feed run,
from artifacts already on disk, into ``feeds/<feed>/show.json``. It is our metadata, not a report:
the app does not show it; operators and tools query it.

Three sections:

* ``hosts`` — the show's hosts, and which feed fields state each one (title / description host
  statement, RSS author tag) beside the set the pipeline actually used for its episodes.
* ``transcripts`` — which transcript formats the feed offers, and how often we used the publisher's
  file versus transcribing the audio ourselves.
* ``checks`` — per-episode validation: does the episode have a host tied to a voice, and if not,
  why (no host name found / host known but on no voice / the transcript had one voice).

Derived only — never read back by the pipeline — so a wrong value here cannot change an episode.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SHOW_METADATA_FILENAME = "show.json"
SCHEMA_VERSION = 1

#: Why an episode has no host on a voice. Ordered: the first that applies is reported.
NO_HOST_ONE_VOICE = "transcript_has_one_voice"
NO_HOST_NOT_PLACED = "host_known_not_on_a_voice"
NO_HOST_NO_NAME = "no_host_name_found"
NO_RECORD = "no_placement_record"


def feed_dir_for_run(run_dir: Path) -> Optional[Path]:
    """``feeds/<feed>/run_<id>`` -> ``feeds/<feed>``; anything else -> None (no feed layout)."""
    if run_dir.name.startswith("run_") and run_dir.parent.parent.name == "feeds":
        return run_dir.parent
    return None


def _authors(raw: Any) -> List[str]:
    # Stored as a list, or (older writers) as the str() of one: "['Dan Shipper']".
    if isinstance(raw, list):
        return [str(a) for a in raw if a]
    if isinstance(raw, str) and raw.strip():
        if raw.strip().startswith("["):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed if a]
            except (ValueError, SyntaxError):
                pass
        return [raw]
    return []


def _episode_files(feed_dir: Path) -> List[Path]:
    from ..search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

    paths = sorted(feed_dir.glob("run_*/metadata/*.metadata.json"))
    corpus_root = feed_dir.parent.parent
    return [Path(p) for p in dedupe_metadata_paths_newest_run_per_episode(corpus_root, paths)]


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _pipeline_hosts(meta_path: Path, doc: Dict[str, Any]) -> List[str]:
    """The host set the roster ran with for this episode (``tried.known_hosts`` in diagnostics)."""
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    if not rel:
        return []
    diag = meta_path.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json")
    data = _read(diag) if diag.is_file() else None
    tried = (data or {}).get("tried") or {}
    return [str(h) for h in tried.get("known_hosts") or [] if h]


def _host_check(doc: Dict[str, Any]) -> Optional[str]:
    """None when a host is on a voice; otherwise the reason it is not."""
    content = doc.get("content") or {}
    speakers = [s for s in content.get("speakers") or [] if isinstance(s, dict)]
    if not any("placed" in s for s in speakers):
        return NO_RECORD
    if any(s.get("placed") is True and s.get("role") == "host" for s in speakers):
        return None
    voices = content.get("diarization_num_speakers")
    if isinstance(voices, int) and voices <= 1:
        return NO_HOST_ONE_VOICE
    if not any(s.get("placed") is True for s in speakers) and not voices:
        return NO_HOST_ONE_VOICE
    if any(s.get("placed") is False and s.get("role") == "host" for s in speakers):
        return NO_HOST_NOT_PLACED
    return NO_HOST_NO_NAME


def build_show_metadata(feed_dir: Path) -> Optional[Dict[str, Any]]:
    """Derive the show metadata from the feed's newest-run episodes. None if there are none."""
    from ..kg.speaker_coherence import same_person
    from ..speaker_detectors.hosts import (
        hosts_from_feed_statement,
        is_network_or_org_author,
        names_the_show,
        split_author_names,
    )

    episodes: List[tuple[Path, Dict[str, Any]]] = []
    for path in _episode_files(feed_dir):
        doc = _read(path)
        if doc is not None:
            episodes.append((path, doc))
    if not episodes:
        return None

    newest = max(episodes, key=lambda e: e[0].stat().st_mtime)[1]
    feed = newest.get("feed") or {}
    title, description = feed.get("title"), feed.get("description")
    authors = _authors(feed.get("authors"))

    sources: Dict[str, List[str]] = {}

    def _add(name: str, source: str) -> None:
        for known in sources:
            if same_person(known, name):
                if source not in sources[known]:
                    sources[known].append(source)
                return
        sources[name] = [source]

    for name in sorted(hosts_from_feed_statement(title, description)):
        _add(name, "feed_statement")
    for tag in authors:
        for name in split_author_names(tag):
            if name and not names_the_show(name, title) and not is_network_or_org_author(name):
                _add(name, "author_tag")
    used: Counter = Counter()
    for path, doc in episodes:
        for name in _pipeline_hosts(path, doc):
            used[name] += 1
    for name in used:
        _add(name, "used_by_pipeline")

    offered: Counter = Counter()
    used_source: Counter = Counter()
    publisher_refused = 0
    checks: Counter = Counter()
    failing: List[Dict[str, Any]] = []
    placed_per_host: Counter = Counter()
    for _path, doc in episodes:
        content = doc.get("content") or {}
        types = [str(t.get("type") or "?") for t in content.get("transcript_urls") or []]
        offered.update(set(types))
        source = content.get("transcript_source") or "none"
        used_source[source] += 1
        if types and source == "whisper_transcription":
            publisher_refused += 1
        reason = _host_check(doc)
        checks[reason or "ok"] += 1
        if reason and reason != NO_RECORD:
            ep = doc.get("episode") or {}
            failing.append(
                {"episode_id": ep.get("episode_id"), "title": ep.get("title"), "reason": reason}
            )
        for s in content.get("speakers") or []:
            if isinstance(s, dict) and s.get("placed") is True and s.get("role") == "host":
                for host in sources:
                    if same_person(host, str(s.get("name") or "")):
                        placed_per_host[host] += 1

    recorded = len(episodes) - checks[NO_RECORD]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feed": {"title": title, "url": feed.get("url"), "feed_id": feed.get("feed_id")},
        "episodes": len(episodes),
        "hosts": [
            {
                "name": name,
                "sources": srcs,
                "episodes_used_by_pipeline": sum(
                    n for u, n in used.items() if same_person(u, name)
                ),
                "episodes_on_a_voice": placed_per_host[name],
            }
            for name, srcs in sources.items()
        ],
        "transcripts": {
            "offered_types": dict(offered),
            "transcript_source": dict(used_source),
            "publisher_offered_but_transcribed": publisher_refused,
        },
        "checks": {
            "host_on_a_voice": {
                "episodes_with_record": recorded,
                "ok": checks["ok"],
                "by_reason": {k: v for k, v in checks.items() if k not in ("ok", NO_RECORD)},
                "without_record": checks[NO_RECORD],
                "failing": failing,
            }
        },
    }


def write_show_metadata(feed_dir: Path) -> Optional[Path]:
    """Build and atomically write ``show.json``. Never raises: show metadata must not fail a run."""
    from ..utils.atomic_io import write_json_atomic

    try:
        doc = build_show_metadata(feed_dir)
        if doc is None:
            return None
        out = feed_dir / SHOW_METADATA_FILENAME
        write_json_atomic(out, doc, indent=2, ensure_ascii=False)
        checks = doc["checks"]["host_on_a_voice"]
        logger.info(
            "show metadata: %s — hosts %s; host on a voice %d/%d; publisher transcript "
            "refused %d",
            out,
            [h["name"] for h in doc["hosts"]],
            checks["ok"],
            checks["episodes_with_record"],
            doc["transcripts"]["publisher_offered_but_transcribed"],
        )
        return out
    except Exception as exc:  # noqa: BLE001 — derived metadata must never fail the pipeline
        logger.warning("show metadata for %s not written: %s", feed_dir, exc)
        return None
