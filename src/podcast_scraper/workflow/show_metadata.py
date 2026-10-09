"""Show sidecar: what we concluded about a SHOW, kept beside its episodes (``show.json``).

Every fact about a show used to exist only as copies inside each episode's metadata, so "who hosts
this show, where did that come from, and does every episode actually put a host on a voice" could
be answered only by re-reading the whole feed by hand. ``feeds/<feed>/show.json`` answers it. It is
our metadata, not a report: the app does not show it; operators and tools query it, and it is the
first place to look when debugging how a show's extraction went.

Two inputs, folded at the end of every feed run (and by the backfill below):

* artifacts already on disk — per-episode metadata, speaker diagnostics, kg.json, gi.json;
* the run event files ``feeds/<feed>/show_events/<run>.jsonl`` (:mod:`show_events`) — what
  happened during a run and is written nowhere else.

Sections:

* ``hosts`` — each host with the feed fields stating it (title / description statement, RSS author
  tag) and whether the pipeline used it, with episode counts used and on a voice;
* ``transcripts`` — formats offered, which source we used, refusals in recent runs;
* ``checks.host_on_a_voice`` — per episode, whether a host is tied to a voice, and if not why;
* ``checks.artifacts`` — per episode, failed or missing KG / GI / summary / transcript;
* ``runs``, ``host_detections``, ``recent_errors`` — from the event files of the recent runs.

Derived only — never read back by the pipeline — so a wrong value here cannot change an episode.
Backfill for existing shows: ``python -m podcast_scraper.workflow.show_metadata --corpus <root>``.
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


#: Artifact problems read back from disk per episode (the backfill works from these alone).
ISSUE_KG_FAILED = "kg_extraction_failed"
ISSUE_KG_MISSING = "kg_missing"
ISSUE_GI_MISSING = "gi_missing"
ISSUE_GI_EMPTY = "gi_no_insights"
ISSUE_SUMMARY_MISSING = "summary_missing"
ISSUE_SUMMARY_INVALID = "summary_invalid"
ISSUE_TRANSCRIPT_MISSING = "transcript_missing"

#: How many run event files and error events the sidecar keeps in view.
RECENT_RUNS = 10
RECENT_ERRORS = 50
_ERROR_KINDS = frozenset({"error", "kg_extraction_failed", "stage_failed"})


def _episode_issues(meta_path: Path, doc: Dict[str, Any]) -> List[str]:
    """What is wrong with this episode's artifacts, read from disk (no run events needed)."""
    from ..kg.io import is_failed_extraction

    issues: List[str] = []
    stem = meta_path.name[: -len(".metadata.json")]
    run = meta_path.parent.parent
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    if not rel or not (run / rel).is_file():
        issues.append(ISSUE_TRANSCRIPT_MISSING)
    kg = (
        _read(meta_path.parent / f"{stem}.kg.json")
        if (meta_path.parent / f"{stem}.kg.json").is_file()
        else None
    )
    if kg is None:
        issues.append(ISSUE_KG_MISSING)
    elif is_failed_extraction(kg):
        issues.append(ISSUE_KG_FAILED)
    gi = (
        _read(meta_path.parent / f"{stem}.gi.json")
        if (meta_path.parent / f"{stem}.gi.json").is_file()
        else None
    )
    if gi is None:
        issues.append(ISSUE_GI_MISSING)
    elif not any(n.get("type") == "Insight" for n in gi.get("nodes") or [] if isinstance(n, dict)):
        issues.append(ISSUE_GI_EMPTY)
    summary = doc.get("summary")
    if not isinstance(summary, dict) or not (
        summary.get("short_summary") or summary.get("bullets")
    ):
        issues.append(ISSUE_SUMMARY_MISSING)
    elif summary.get("schema_status") not in (None, "valid"):
        issues.append(ISSUE_SUMMARY_INVALID)
    return issues


def _read_events(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _fold_events(feed_dir: Path) -> Dict[str, Any]:
    """Runs, host detections and recent errors from the newest run event files."""
    from .show_events import EVENTS_SUBDIR

    files = sorted(
        (feed_dir / EVENTS_SUBDIR).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:RECENT_RUNS]
    runs: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    refused = 0
    for path in files:
        events = _read_events(path)
        kinds = Counter(str(e.get("event_type")) for e in events)
        stamps = sorted(str(e.get("ts")) for e in events if e.get("ts"))
        runs.append(
            {
                "run": path.stem,
                "first_event": stamps[0] if stamps else None,
                "last_event": stamps[-1] if stamps else None,
                "events": dict(kinds),
            }
        )
        refused += kinds.get("transcript_refused", 0)
        for e in events:
            kind = e.get("event_type")
            if kind == "hosts_detected":
                detections.append(
                    {
                        "run": path.stem,
                        "ts": e.get("ts"),
                        "hosts": e.get("hosts") or [],
                        "source": e.get("source"),
                        "dropped_non_person": e.get("dropped_non_person") or [],
                    }
                )
            elif kind in _ERROR_KINDS:
                errors.append(
                    {
                        "run": path.stem,
                        "ts": e.get("ts"),
                        "kind": kind,
                        "episode_id": e.get("episode_id"),
                        **{
                            k: e.get(k)
                            for k in ("stage", "reason", "model", "logger_name", "message")
                            if e.get(k) is not None
                        },
                    }
                )
    errors.sort(key=lambda e: str(e.get("ts")), reverse=True)
    detections.sort(key=lambda d: str(d.get("ts")), reverse=True)
    host_sets = [tuple(d["hosts"]) for d in detections]
    return {
        "runs": runs,
        "host_detections": detections,
        "host_set_changed_between_runs": len(set(host_sets)) > 1,
        "recent_errors": errors[:RECENT_ERRORS],
        "transcripts_refused": refused,
    }


def build_show_metadata(feed_dir: Path) -> Optional[Dict[str, Any]]:
    """Derive the show metadata from the feed's newest-run episodes. None if there are none."""
    from ..kg.speaker_coherence import same_person
    from ..speaker_detectors.hosts import (
        hosts_from_feed_statement,
        is_network_or_org_author,
        names_the_show,
        refused_feed_statement_names,
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

    language = str(feed.get("language") or "")
    for name in sorted(hosts_from_feed_statement(title, description, language)):
        _add(name, "feed_statement")
    for name in refused_feed_statement_names(title, description, language):
        _add(name, "feed_statement_refused")
    for tag in authors:
        for name in split_author_names(tag, language or None):
            if (
                name
                and not names_the_show(name, title, language or None)
                and not is_network_or_org_author(name, language or None)
            ):
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
    issue_counts: Counter = Counter()
    episode_issues: List[Dict[str, Any]] = []
    for _path, doc in episodes:
        issues = _episode_issues(_path, doc)
        if issues:
            issue_counts.update(issues)
            ep = doc.get("episode") or {}
            episode_issues.append(
                {"episode_id": ep.get("episode_id"), "title": ep.get("title"), "issues": issues}
            )
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
    events = _fold_events(feed_dir)
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
            "refused_in_recent_runs": events["transcripts_refused"],
        },
        "checks": {
            "host_on_a_voice": {
                "episodes_with_record": recorded,
                "ok": checks["ok"],
                "by_reason": {k: v for k, v in checks.items() if k not in ("ok", NO_RECORD)},
                "without_record": checks[NO_RECORD],
                "failing": failing,
            },
            "artifacts": {
                "episodes_with_issues": len(episode_issues),
                "by_issue": dict(issue_counts),
                "episodes": episode_issues,
            },
        },
        "runs": events["runs"],
        "host_detections": events["host_detections"],
        "host_set_changed_between_runs": events["host_set_changed_between_runs"],
        "recent_errors": events["recent_errors"],
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


def main(argv: Optional[List[str]] = None) -> int:
    """Backfill: write show.json for every feed of a corpus (or one feed). Derived, idempotent.

    python -m podcast_scraper.workflow.show_metadata --corpus /app/output [--feed DIR] [--dry-run]
    """
    import argparse

    ap = argparse.ArgumentParser(description="Write feeds/<feed>/show.json for existing shows.")
    ap.add_argument("--corpus", type=Path, required=True, help="corpus root (contains feeds/)")
    ap.add_argument("--feed", action="append", default=[], help="feed directory name; repeatable")
    ap.add_argument(
        "--dry-run", action="store_true", help="print a one-line summary, write nothing"
    )
    args = ap.parse_args(argv)
    feeds_root = args.corpus / "feeds"
    if not feeds_root.is_dir():
        print(f"no feeds/ under {args.corpus}")
        return 2
    dirs = sorted(d for d in feeds_root.iterdir() if d.is_dir())
    if args.feed:
        dirs = [d for d in dirs if d.name in set(args.feed)]
    written = 0
    for feed_dir in dirs:
        if args.dry_run:
            doc = build_show_metadata(feed_dir)
            if doc is None:
                continue
            c = doc["checks"]["host_on_a_voice"]
            a = doc["checks"]["artifacts"]
            print(
                f"{feed_dir.name}\t{doc['feed'].get('title')}\teps={doc['episodes']}"
                f"\thosts={[h['name'] for h in doc['hosts']]}"
                f"\thost_on_voice={c['ok']}/{c['episodes_with_record']}"
                f"\tissues={a['by_issue']}"
            )
            written += 1
        elif write_show_metadata(feed_dir) is not None:
            written += 1
    print(
        f"{'would write' if args.dry_run else 'wrote'} show.json for {written} of {len(dirs)} feeds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
