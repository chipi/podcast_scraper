"""Naming COVERAGE of the served corpus: how much is attributed to a named person (#2304).

Every other naming check compares old code with new code, so an episode that both name nobody
"agrees" and stays invisible. This measures the absolute level: 296 English episodes with insights
published no named voice on 2026-10-10, and nothing reported it.

Per served episode with a GI artifact (one language by default):

- ``quotes`` / ``named_quotes``: GI quotes, and those whose ``speaker_id`` is a person;
- ``named_voices``: diarized voices published under a name in the segments;
- ``group``: ``named``, or why not: ``no_segments``, ``no_diagnostics`` (an older record that
  cannot be replayed) or ``stored_unnamed``.

The nightly batch finalize measures the corpus once and writes a snapshot
(:func:`write_snapshot`); the API's ``/metrics`` serves it (:func:`install_metrics`), so a scrape
never walks the corpus. ``scripts/measure/naming_coverage.py`` is the CLI over the same scan.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DIARIZATION_LABEL = re.compile(r"^SPEAKER(_\d+)?$")
#: Where the snapshot lives, under the corpus root.
SNAPSHOT_RELPATH = Path(".podcast_scraper") / "naming_coverage.json"
DEFAULT_LANGUAGE = "en"


def _segments(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = raw if isinstance(raw, list) else raw.get("segments", [])
    return [r for r in rows if isinstance(r, dict)]


def _language(doc: Dict[str, Any]) -> str:
    feed = doc.get("feed") or {}
    content = doc.get("content") or {}
    return str(feed.get("language") or content.get("language") or "")[:2].lower()


def scan_episode(meta: Path, lang: Optional[str]) -> Optional[Dict[str, Any]]:
    """One episode's coverage row, or None when it is out of scope (other language, no GI)."""
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if lang and _language(doc) not in (lang, ""):
        return None
    stem = meta.name[: -len(".metadata.json")]
    gi = next(iter(meta.parent.glob(stem + "*.gi.json")), None)
    if gi is None:
        return None
    try:
        nodes = json.loads(gi.read_text(encoding="utf-8")).get("nodes") or []
    except (OSError, ValueError):
        return None
    quotes = [n for n in nodes if isinstance(n, dict) and n.get("type") == "Quote"]
    named_quotes = sum(
        1
        for q in quotes
        if str((q.get("properties") or {}).get("speaker_id") or "").startswith("person:")
    )
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    run = meta.parent.parent
    row: Dict[str, Any] = {
        "meta": str(meta),
        "feed": str((doc.get("feed") or {}).get("title") or "")[:40],
        "title": str((doc.get("episode") or {}).get("title") or "")[:60],
        "quotes": len(quotes),
        "named_quotes": named_quotes,
    }
    segs = _segments(run / rel.replace(".txt", ".segments.json")) if rel else None
    if segs is None:
        row.update(named_voices=0, group="no_segments")
        return row
    labels = collections.Counter(
        str(s.get("speaker_label") or s.get("speaker") or "") for s in segs
    )
    named = {lab for lab in labels if lab and not _DIARIZATION_LABEL.match(lab)}
    row["named_voices"] = len(named)
    row["main_voice"] = labels.most_common(1)[0][0] if labels else None
    if named:
        row["group"] = "named"
    elif (run / rel.replace(".txt", ".speakers.diagnostics.json")).is_file():
        row["group"] = "stored_unnamed"
    else:
        row["group"] = "no_diagnostics"
    return row


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = collections.Counter(r["group"] for r in rows)
    unnamed = [r for r in rows if r["group"] != "named"]
    return {
        "episodes": len(rows),
        "episodes_no_named_voice": len(unnamed),
        "quotes": sum(r["quotes"] for r in rows),
        "named_quotes": sum(r["named_quotes"] for r in rows),
        "groups": dict(groups),
        "unnamed_by_feed": dict(collections.Counter(r["feed"] for r in unnamed).most_common(15)),
    }


def scan_corpus(corpus: Path, lang: Optional[str] = DEFAULT_LANGUAGE) -> List[Dict[str, Any]]:
    """Coverage rows for every SERVED episode of ``corpus`` (newest run per episode)."""
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts

    return [
        row
        for meta in select_served_artifacts(corpus, ".metadata.json")[0]
        if (row := scan_episode(meta, lang or None)) is not None
    ]


def write_snapshot(corpus: Path, summary: Dict[str, Any], *, lang: Optional[str]) -> Path:
    """Write ``summary`` (plus when and for which language) atomically; return the path."""
    path = Path(corpus) / SNAPSHOT_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(summary, language=lang or "", measured_at=time.time())
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".naming_coverage.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_snapshot(corpus: Path) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads((Path(corpus) / SNAPSHOT_RELPATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def measure_and_record(corpus: Path, lang: Optional[str] = DEFAULT_LANGUAGE) -> Optional[Dict]:
    """Scan the served corpus and write the snapshot. Never raises: a finished run must not fail
    because its coverage could not be measured; the snapshot's age shows that it is stale."""
    t0 = time.perf_counter()
    try:
        summary = summarise(scan_corpus(Path(corpus), lang))
        write_snapshot(Path(corpus), summary, lang=lang)
    except Exception as exc:  # noqa: BLE001 - telemetry must not fail the batch
        logger.warning("naming coverage: could not measure (%s)", exc)
        return None
    logger.info(
        "naming coverage: %s of %s %s episodes publish no named voice (%.1fs)",
        summary["episodes_no_named_voice"],
        summary["episodes"],
        lang or "all",
        time.perf_counter() - t0,
    )
    return summary


def install_metrics(corpus_root: Path) -> bool:
    """Publish the last snapshot on ``/metrics``. No snapshot exports nothing (not zero: zero
    would read as "everything is named" and silence an alert). Never raises (ADR-120)."""
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.core import GaugeMetricFamily
    except Exception:  # noqa: BLE001 - metrics extras absent; run without these gauges
        return False

    class _NamingCoverageCollector:
        def collect(self):  # noqa: ANN202 - prometheus_client's duck-typed collector protocol
            doc = read_snapshot(corpus_root)
            if doc is None:
                return
            lang = str(doc.get("language") or "all")
            for name, key, help_text in (
                (
                    "podcast_corpus_episodes_no_named_voice",
                    "episodes_no_named_voice",
                    "Served episodes with insights that publish no named voice (#2304).",
                ),
                ("podcast_corpus_naming_episodes", "episodes", "Served episodes with insights."),
                ("podcast_corpus_quotes", "quotes", "GI quotes on served episodes."),
                ("podcast_corpus_named_quotes", "named_quotes", "GI quotes credited to a person."),
            ):
                g = GaugeMetricFamily(name, help_text, labels=["language"])
                g.add_metric([lang], float(doc.get(key) or 0))
                yield g
            age = GaugeMetricFamily(
                "podcast_corpus_naming_coverage_age_seconds",
                "Seconds since the naming coverage was last measured.",
                labels=["language"],
            )
            age.add_metric([lang], max(0.0, time.time() - float(doc.get("measured_at") or 0)))
            yield age

    try:
        REGISTRY.register(_NamingCoverageCollector())  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - duplicate registration on app re-create is harmless
        return False
    return True
