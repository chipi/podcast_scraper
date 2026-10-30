"""Storyline summaries (#2344): one paragraph per storyline, written in the background.

A storyline (``thc:``, ``enrichments/topic_theme_clusters.json``) is a thread of topics through
several episodes. Every storyline gets a summary (operator 2026-10-13: consistency — a listener
must not wonder why one has a summary and another does not); the prompt adapts to its size: a
thread of fewer than five episodes says what it is and what each adds, a longer one also how it
developed and where the voices differ.

Written by the board-summarizer sidecar with the same use case as boards (the provider's own
``summarize``, the storyline prompts), under the same kill switch and the same zero-dollar daily
budget (``board_summaries``). Shared by every listener, so stored once in the player data dir
(``storyline_summaries.json``). Read by the storyline page and the top of its Learn path; the API
never calls a model.

A summary is redone when its storyline's episodes or topics change (``content_fingerprint``), or
its prompt or model does; a summary whose storyline no longer exists is dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from filelock import FileLock

from podcast_scraper.server import board_summaries

_LOGGER = logging.getLogger(__name__)

STAGE = "storyline_summary"
SUMMARIES_FILE = "storyline_summaries.json"
MAX_SUMMARY_CHARS = 1200


@dataclass(frozen=True)
class Limits:
    """How much one sweep may do. Every storyline is eventually summarised; the cap only spreads
    the first fill (99 on prod) over several sweeps."""

    max_per_sweep: int = 20
    max_input_episodes: int = 12

    @classmethod
    def from_env(cls) -> "Limits":
        """The limits, with ``STORYLINE_SUMMARY_*`` environment overrides."""

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except ValueError:
                return default

        return cls(
            max_per_sweep=_int("STORYLINE_SUMMARY_MAX_PER_SWEEP", cls.max_per_sweep),
            max_input_episodes=_int("STORYLINE_SUMMARY_MAX_INPUT_EPISODES", cls.max_input_episodes),
        )


@dataclass(frozen=True)
class Storyline:
    """One storyline as the artifact describes it."""

    id: str
    label: str
    topics: tuple[tuple[str, str], ...]
    episode_ids: frozenset[str]
    anchor_topic_id: str


def load_storylines(root: Path) -> list[Storyline]:
    """The corpus's storylines; ``[]`` when the artifact is missing or unreadable."""
    path = Path(root) / "enrichments" / "topic_theme_clusters.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    data = raw.get("data", raw) if isinstance(raw, dict) else {}
    from podcast_scraper.search.storylines import _anchor_topic_id

    out = []
    for c in data.get("clusters") or [] if isinstance(data, dict) else []:
        sid = str(c.get("graph_compound_parent_id") or "")
        if not sid:
            continue
        members = [m for m in c.get("members") or [] if isinstance(m, dict)]
        eps = {str(e) for m in members for e in m.get("episode_ids") or []}
        topics = tuple((str(m.get("topic_id") or ""), str(m.get("label") or "")) for m in members)
        out.append(
            Storyline(
                id=sid,
                label=str(c.get("canonical_label") or sid),
                topics=topics,
                episode_ids=frozenset(eps),
                # What the storyline page routes on (``search.storylines``).
                anchor_topic_id=str(_anchor_topic_id(members) or ""),
            )
        )
    return out


def _rows_by_episode_id(root: Path) -> dict[str, Any]:
    from podcast_scraper.server.catalog_cache import cached_catalog

    return {row.episode_id: row for row in cached_catalog(root) if row.episode_id}


def content_fingerprint(storyline: Storyline, existing: set[str]) -> str:
    """What the summary is OF: the storyline's label, topics and the episodes still in the
    corpus. An episode that left changes it; one that was never in the catalog does not."""
    payload = json.dumps(
        {
            "label": storyline.label,
            "topics": sorted(t for t, _ in storyline.topics),
            "episodes": sorted(storyline.episode_ids & existing),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _spread(items: list[Any], n: int) -> list[Any]:
    """``n`` items spread evenly across ``items`` (first and last kept)."""
    if len(items) <= n:
        return items
    if n <= 1:
        return items[:1]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def build_material(
    root: Path, storyline: Storyline, rows: Mapping[str, Any], limits: Limits
) -> Optional[dict[str, Any]]:
    """What the summary is written from, for the ``storyline_*`` templates: the storyline's
    episodes in publish order (spread across its time when there are more than the cap), each
    with its summary and top insights. None when none of its episodes is in the corpus."""
    from podcast_scraper.server.app_recap_view import build_episode_recap
    from podcast_scraper.server.slugs import slug_for_row

    clip = board_summaries._clip
    present = [rows[e] for e in storyline.episode_ids if e in rows]
    if not present:
        return None
    present.sort(key=lambda r: (str(r.publish_date or ""), str(r.episode_id)))
    episodes = []
    for row in _spread(present, limits.max_input_episodes):
        recap = build_episode_recap(root, row, slug_for_row(row), limit=2)
        episodes.append(
            {
                "title": clip(recap.title, 160),
                "show": clip(recap.podcast_title, 120),
                "date": str(row.publish_date or "")[:10],
                "summary": clip(recap.summary_text, 400),
                "insights": [
                    {
                        "speaker": clip(next((q.speaker for q in i.quotes if q.speaker), None), 60),
                        "text": clip(i.text, 200),
                    }
                    for i in recap.insights[:2]
                ],
            }
        )
    return {
        "label": clip(storyline.label, 120),
        "topics": [clip(label, 60) for _, label in storyline.topics[:6] if label],
        "episode_count": len(present),
        "episodes": episodes,
    }


# --- the store -----------------------------------------------------------------------------------


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / SUMMARIES_FILE


def read_summaries(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Every stored storyline summary, ``{storyline_id: record}``."""
    try:
        doc = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in doc.items() if isinstance(v, dict)} if isinstance(doc, dict) else {}


def _update(data_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=5.0):
        doc = read_summaries(data_dir)
        change(doc)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def summary_for(root: Path, data_dir: Path, ref: str) -> Optional[dict[str, Any]]:
    """The stored summary of the storyline ``ref`` names — its ``thc:`` id or one of its member
    topics (the storyline page routes on its anchor topic), as the storyline card accepts — with
    ``stale`` when its episodes changed since. None when there is none yet. Never calls a
    model."""
    storyline = next(
        (
            s
            for s in load_storylines(root)
            if ref in (s.id, s.anchor_topic_id) or ref in {t for t, _ in s.topics}
        ),
        None,
    )
    if storyline is None:
        return None
    rec = read_summaries(data_dir).get(storyline.id)
    if not rec or not rec.get("text"):
        return None
    existing = set(_rows_by_episode_id(root))
    return {
        "text": str(rec["text"]),
        "created_at": int(rec.get("created_at") or 0),
        "stale": rec.get("fingerprint") != content_fingerprint(storyline, existing),
    }


# --- the sweep -----------------------------------------------------------------------------------


@dataclass
class SweepResult:
    """What one sweep did, by reason."""

    counts: Counter[str] = field(default_factory=Counter)
    summarised: list[str] = field(default_factory=list)
    stopped: Optional[str] = None


#: The summary's limits (operator decision 6, 2026-10-13). The prompt asks for them; the prod
#: model exceeded them on 8 of 9 storylines in the first real run (93-143 words, 2026-10-14), so
#: they are also enforced here.
MAX_SENTENCES = 3
MAX_WORDS = 80

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'“A-Z0-9])")


def fit_summary(text: str, *, sentences: int = MAX_SENTENCES, words: int = MAX_WORDS) -> str:
    """Whole sentences, at most ``sentences`` of them and ``words`` words in all. A first sentence
    already longer than ``words`` is cut at the last word that fits, with an ellipsis — every
    storyline keeps a summary."""
    parts = [p for p in _SENTENCE_END.split(" ".join(text.split())) if p]
    kept: list[str] = []
    count = 0
    for part in parts[:sentences]:
        n = len(part.split())
        if kept and count + n > words:
            break
        kept.append(part)
        count += n
    out = " ".join(kept)
    if len(out.split()) > words:
        out = " ".join(out.split()[:words]).rstrip(",;:") + "…"
    return out


def sweep(
    root: Path,
    data_dir: Path,
    *,
    summarize: Callable[[dict[str, Any]], str],
    price: Callable[[dict[str, Any]], Optional[float]],
    model: str,
    prompt: str,
    limits: Optional[Limits] = None,
    daily_budget_usd: float = 0.0,
    now: Optional[float] = None,
    dry_run: bool = False,
) -> SweepResult:
    """Summarise every storyline that needs it, within the cap and the shared daily budget, and
    drop the summaries of storylines that no longer exist. With ``dry_run`` nothing is called or
    written; the counts say what would be."""
    lim = limits or Limits()
    t = time.time() if now is None else now
    result = SweepResult()
    if not board_summaries.summaries_enabled(data_dir):
        result.stopped = "switched_off"
        return result
    storylines = load_storylines(root)
    if not storylines:
        result.stopped = "no_storylines"
        return result
    current = {s.id for s in storylines}
    stored = read_summaries(data_dir)
    orphans = sorted(set(stored) - current)
    if orphans:
        result.counts["orphan_dropped"] += len(orphans)
        if not dry_run:

            def drop(doc: dict[str, Any]) -> None:
                for k in orphans:
                    doc.pop(k, None)

            _update(data_dir, drop)
    rows = _rows_by_episode_id(root)
    existing = set(rows)
    for storyline in sorted(storylines, key=lambda s: (-len(s.episode_ids), s.id)):
        rec = stored.get(storyline.id) or {}
        fp = content_fingerprint(storyline, existing)
        if (
            rec.get("fingerprint") == fp
            and rec.get("prompt") == prompt
            and rec.get("model") == model
        ):
            result.counts["unchanged"] += 1
            continue
        if len(result.summarised) >= lim.max_per_sweep:
            result.counts["sweep_cap"] += 1
            continue
        material = build_material(root, storyline, rows, lim)
        if material is None:
            result.counts["no_episodes"] += 1
            continue
        if dry_run:
            result.counts["would_summarise"] += 1
            result.summarised.append(storyline.id)
            continue
        cost = price(material)
        if cost is None:
            result.stopped = "price_unknown"
            return result
        if board_summaries.spent_today(data_dir, t) + cost > daily_budget_usd:
            result.stopped = "budget"
            return result
        try:
            text = fit_summary(summarize(material))
        except Exception as exc:  # noqa: BLE001 — one storyline's failure must not end the sweep
            if board_summaries._unreachable(exc):
                result.stopped = "model_unreachable"
                return result
            _LOGGER.exception("storyline summary failed for %s", storyline.id)
            result.counts["failed"] += 1
            continue
        board_summaries._record_spend(data_dir, t, cost)
        if not text or len(text) > MAX_SUMMARY_CHARS:
            result.counts["rejected"] += 1
            continue
        record = {
            "text": text,
            "fingerprint": fp,
            "model": model,
            "prompt": prompt,
            "created_at": int(t),
            "episode_count": material["episode_count"],
        }

        def put(doc: dict[str, Any], sid: str = storyline.id, r: dict[str, Any] = record) -> None:
            doc[sid] = r

        _update(data_dir, put)
        result.summarised.append(storyline.id)
        result.counts["summarised"] += 1
    return result
