"""Board summaries (RFC-127): one paragraph on what a whole board is about, written by an LLM.

The first LLM call outside the pipeline, so everything here is about WHEN it may run and WHAT it
may read:

- It runs only in the ``board-summarizer`` sidecar
  (``podcast_scraper.server.board_summarizer_runner``), never in a request. This module takes the
  model call as a plain function and imports no SDK, so the API process that reads summaries for
  the Brief cannot call a model even by mistake.
- A board is summarised when its content changed, has been quiet for a while, was not summarised
  in the last day, and has two episodes or more; per user and per sweep there is a cap.
- The budget is zero (operator 2026-10-13: the DGX profile costs nothing). Every call is priced
  first with the pipeline's own pricing table; a price that is unknown, or above what is left of
  the day's budget, stops the sweep instead of spending.
- What goes in is the episodes' Brief content and the words of saved moments: text the pipeline
  wrote or the listener chose from a transcript. Nothing the user typed (the board's name, link
  titles, saved searches) reaches the prompt.
- A kill switch in the appdata root, read at the start of every sweep, stops new summaries without
  a deploy. On by default.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from filelock import FileLock

from podcast_scraper.server import app_collections_store
from podcast_scraper.server.app_board_view import Board, board_episode_slugs, load_board
from podcast_scraper.server.app_user_store import _is_safe_user_id

_LOGGER = logging.getLogger(__name__)

#: The ``llm_cost`` stage, so a summary is never counted as pipeline work.
STAGE = "board_summary"
#: Per user, beside ``collections.json``: ``{board_id: {text, fingerprint, model, ...}}``.
SUMMARIES_FILE = "board_summaries.json"
#: In the appdata root: the kill switch, ``{"enabled": false}`` to stop.
SETTINGS_FILE = "board_summaries_settings.json"
#: In the appdata root: each day's summary spend, ``{"2026-10-13": {"usd": 0.0, "calls": 3}}``.
LEDGER_FILE = "board_summaries_ledger.json"
#: A summary longer than this is not the paragraph the prompt asks for.
MAX_SUMMARY_CHARS = 1200


@dataclass(frozen=True)
class Limits:
    """Every number that decides what runs (RFC-127 §2). Env overrides, for the sidecar."""

    # Fresh within minutes of the last edit (operator 2026-10-13, for the beta): the model is ours
    # and free, so these bound DGX load and runaway loops, not spend. The budget does that.
    quiet_minutes: float = 10.0
    min_hours_between: float = 0.25
    min_episodes: int = 2
    max_per_user_per_day: int = 30
    max_per_sweep: int = 50
    daily_budget_usd: float = 0.0
    max_input_episodes: int = 20

    @classmethod
    def from_env(cls) -> "Limits":
        """The defaults, each overridable by ``BOARD_SUMMARY_<NAME>`` (the sidecar's env); an
        unparsable value keeps the default."""

        def num(name: str, default: float) -> float:
            raw = os.environ.get(f"BOARD_SUMMARY_{name}", "").strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        d = cls()
        return cls(
            quiet_minutes=num("QUIET_MINUTES", d.quiet_minutes),
            min_hours_between=num("MIN_HOURS_BETWEEN", d.min_hours_between),
            min_episodes=int(num("MIN_EPISODES", d.min_episodes)),
            max_per_user_per_day=int(num("MAX_PER_USER_PER_DAY", d.max_per_user_per_day)),
            max_per_sweep=int(num("MAX_PER_SWEEP", d.max_per_sweep)),
            daily_budget_usd=num("DAILY_BUDGET_USD", d.daily_budget_usd),
            max_input_episodes=int(num("MAX_INPUT_EPISODES", d.max_input_episodes)),
        )


# --- the kill switch and the ledger ---------------------------------------------------------


def summaries_enabled(data_dir: Path) -> bool:
    """On unless the settings file says ``"enabled": false``. Unreadable counts as on: the switch
    is for stopping a job that misbehaves, and a damaged file must not be what stops it."""
    try:
        doc = json.loads((data_dir / SETTINGS_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return True
    return not (isinstance(doc, dict) and doc.get("enabled") is False)


def _today(now: float) -> str:
    return _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc).date().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), "utf-8")
    os.replace(tmp, path)


def spent_today(data_dir: Path, now: float) -> float:
    """Today's recorded spend in USD, from the ledger in the appdata root."""
    day = _read_json(data_dir / LEDGER_FILE).get(_today(now)) or {}
    return float(day.get("usd") or 0.0)


def _record_spend(data_dir: Path, now: float, usd: float) -> None:
    path = data_dir / LEDGER_FILE
    with FileLock(str(path) + ".lock", timeout=5.0):
        doc = _read_json(path)
        day = doc.get(_today(now)) or {}
        doc[_today(now)] = {
            "usd": float(day.get("usd") or 0.0) + usd,
            "calls": int(day.get("calls") or 0) + 1,
        }
        _write_json(path, doc)


# --- the store --------------------------------------------------------------------------------


def _summaries_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / SUMMARIES_FILE


def read_summaries(data_dir: Path, user_id: str) -> dict[str, dict[str, Any]]:
    """A user's stored board summaries, ``{board_id: record}``; empty for an unsafe user id."""
    if not _is_safe_user_id(user_id):
        return {}
    doc = _read_json(_summaries_path(data_dir, user_id))
    return {k: v for k, v in doc.items() if isinstance(v, dict)}


def write_summary(data_dir: Path, user_id: str, board_id: str, record: dict[str, Any]) -> None:
    """Store one board's summary record under the user's file lock, as the board store writes."""
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    path = _summaries_path(data_dir, user_id)
    with FileLock(str(path) + ".lock", timeout=5.0):
        doc = _read_json(path)
        doc[board_id] = record
        _write_json(path, doc)


def drop_summaries(data_dir: Path, user_id: str, board_ids: list[str]) -> None:
    """Remove these boards' summaries under the user's file lock."""
    if not _is_safe_user_id(user_id):
        return
    path = _summaries_path(data_dir, user_id)
    with FileLock(str(path) + ".lock", timeout=5.0):
        doc = _read_json(path)
        for board_id in board_ids:
            doc.pop(board_id, None)
        _write_json(path, doc)


def content_fingerprint(board: Board, root: Optional[Path] = None) -> str:
    """What the summary is OF: the board's episodes and its saved moments. A show, a topic, a link
    or a search added to the board changes nothing the summary reads, so it calls nothing.

    With ``root``, only episodes still in the corpus count (#2344): one that left the corpus
    changes what the summary can be written from, so it is redone; without it, every slug."""
    from podcast_scraper.server.app_slugs import resolve_slug

    moments = sorted(str(m.get("ref")) for m in board.items if m.get("kind") == "highlight")
    slugs = board_episode_slugs(board)
    if root is not None:
        slugs = [s for s in slugs if resolve_slug(root, s) is not None]
    payload = json.dumps({"episodes": sorted(slugs), "moments": moments})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def summary_for_brief(
    data_dir: Path, user_id: str, board: Board, root: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    """The stored summary for the Brief, with ``stale`` when the board changed since. None when
    there is none yet. Never calls a model."""
    rec = read_summaries(data_dir, user_id).get(str(board.meta["id"]))
    if not rec or not rec.get("text"):
        return None
    return {
        "text": str(rec["text"]),
        "created_at": int(rec.get("created_at") or 0),
        "stale": rec.get("fingerprint") != content_fingerprint(board, root),
    }


# --- the prompt -------------------------------------------------------------------------------


def _clip(text: Optional[str], n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def build_material(root: Path, board: Board, limits: Limits) -> Optional[dict[str, Any]]:
    """What the summary is written from, for the templates in ``prompts/shared/board_summary``:
    each episode's Brief content and the saved moments' words, every part clipped so the prompt has
    a ceiling whatever the board holds. None when fewer than ``limits.min_episodes`` of the board's
    episodes are in the corpus. Nothing the user typed is in it."""
    from podcast_scraper.server.app_recap_view import build_episode_recap
    from podcast_scraper.server.app_slugs import resolve_slug

    episodes: list[dict[str, Any]] = []
    titles: dict[str, str] = {}
    n = 0
    for slug in board_episode_slugs(board):
        row = resolve_slug(root, slug)
        if row is None:
            continue
        n += 1
        if n > limits.max_input_episodes:
            continue
        recap = build_episode_recap(root, row, slug, limit=3)
        titles[slug] = recap.title or slug
        episodes.append(
            {
                "title": _clip(recap.title, 160),
                "show": _clip(recap.podcast_title, 120),
                "summary": _clip(recap.summary_text, 600),
                "key_points": [_clip(pt, 200) for pt in recap.key_points[:5]],
                "insights": [
                    {
                        "speaker": _clip(
                            next((q.speaker for q in i.quotes if q.speaker), None), 60
                        ),
                        "text": _clip(i.text, 220),
                    }
                    for i in recap.insights[:3]
                ],
            }
        )
    if n < limits.min_episodes:
        return None
    moments = [
        board.highlights[str(m.get("ref"))]
        for m in board.items
        if m.get("kind") == "highlight" and str(m.get("ref")) in board.highlights
    ]
    return {
        "episodes": episodes,
        "moments": [
            {
                "text": _clip(h.get("quote_text"), 280),
                "speaker": _clip(h.get("speaker"), 60),
                "episode": _clip(titles.get(str(h.get("episode_slug"))), 120),
            }
            for h in moments[:10]
            if h.get("quote_text")
        ],
    }


# --- the sweep --------------------------------------------------------------------------------


@dataclass
class SweepResult:
    """What one sweep did, by reason: the line the sidecar logs and the dashboard counts."""

    counts: Counter[str] = field(default_factory=Counter)
    summarised: list[tuple[str, str]] = field(default_factory=list)
    stopped: Optional[str] = None


def _user_ids(data_dir: Path) -> list[str]:
    users = data_dir / "users"
    if not users.is_dir():
        return []
    return sorted(
        p.name
        for p in users.iterdir()
        if p.is_dir() and _is_safe_user_id(p.name) and (p / "collections.json").is_file()
    )


def _unreachable(exc: BaseException) -> bool:
    """The model endpoint is down (the DGX in another GPU mode, the tailnet out): stop the sweep,
    the next one tries again. Judged by type name so this module imports no SDK."""
    seen: Optional[BaseException] = exc
    while seen is not None:
        name = type(seen).__name__
        if isinstance(seen, (ConnectionError, TimeoutError)) or any(
            k in name for k in ("Connection", "Timeout", "Unavailable")
        ):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def sweep(
    root: Path,
    data_dir: Path,
    *,
    summarize: Callable[[dict[str, Any]], str],
    price: Callable[[dict[str, Any]], Optional[float]],
    model: str,
    prompt: str,
    limits: Optional[Limits] = None,
    now: Optional[float] = None,
) -> SweepResult:
    """Summarise every board that should be, within every limit.

    ``summarize(material)`` is the provider's ``summarize`` with the board use case;
    ``price(material)`` its estimated cost in USD, None when unknown. ``prompt`` is the identity of
    the templates in use (names and content hash): a changed template re-summarises every board
    once, as a new model does."""
    lim = limits or Limits()
    t = time.time() if now is None else now
    result = SweepResult()
    if not summaries_enabled(data_dir):
        result.stopped = "switched_off"
        return result
    day_start = t - (t % 86400)
    for user_id in _user_ids(data_dir):
        stored = read_summaries(data_dir, user_id)
        boards = app_collections_store.list_collections(data_dir, user_id)
        # A deleted board's summary is dropped (#2344): nothing reads it, and it would outlive
        # the board in the user's data.
        orphans = sorted(set(stored) - {str(b["id"]) for b in boards})
        if orphans:
            drop_summaries(data_dir, user_id, orphans)
            result.counts["orphan_dropped"] += len(orphans)
            stored = {k: v for k, v in stored.items() if k not in orphans}
        made_today = sum(1 for r in stored.values() if float(r.get("created_at") or 0) >= day_start)
        for row in boards:
            board = load_board(data_dir, user_id, str(row["id"]))
            if board is None:
                continue
            rec = stored.get(str(board.meta["id"])) or {}
            fp = content_fingerprint(board, root)
            if (
                rec.get("fingerprint") == fp
                and rec.get("prompt") == prompt
                and rec.get("model") == model
            ):
                result.counts["unchanged"] += 1
                continue
            if len(board_episode_slugs(board)) < lim.min_episodes:
                result.counts["too_few"] += 1
                continue
            changed = float(board.meta.get("updated_at") or board.meta.get("created_at") or 0)
            if t - changed < lim.quiet_minutes * 60:
                result.counts["quiet"] += 1
                continue
            if rec and t - float(rec.get("created_at") or 0) < lim.min_hours_between * 3600:
                result.counts["too_soon"] += 1
                continue
            if made_today >= lim.max_per_user_per_day:
                result.counts["user_cap"] += 1
                continue
            if len(result.summarised) >= lim.max_per_sweep:
                result.counts["sweep_cap"] += 1
                continue
            material = build_material(root, board, lim)
            if material is None:
                result.counts["too_few"] += 1
                continue
            cost = price(material)
            if cost is None:
                result.stopped = "price_unknown"
                return result
            if spent_today(data_dir, t) + cost > lim.daily_budget_usd:
                result.stopped = "budget"
                return result
            try:
                text = " ".join(summarize(material).split())
            except Exception as exc:  # noqa: BLE001 — one board's failure must not end the sweep
                if _unreachable(exc):
                    result.stopped = "model_unreachable"
                    return result
                # At ERROR, as the pipeline logs a failed stage: Sentry's logging integration
                # reports it where Sentry is on (the sidecar), and it is a log line everywhere.
                _LOGGER.exception("board summary failed for board %s", board.meta.get("id"))
                result.counts["failed"] += 1
                continue
            _record_spend(data_dir, t, cost)
            if not text or len(text) > MAX_SUMMARY_CHARS:
                result.counts["rejected"] += 1
                continue
            write_summary(
                data_dir,
                user_id,
                str(board.meta["id"]),
                {
                    "text": text,
                    "fingerprint": fp,
                    "model": model,
                    "prompt": prompt,
                    "created_at": int(t),
                },
            )
            made_today += 1
            result.summarised.append((user_id, str(board.meta["id"])))
            result.counts["summarised"] += 1
    return result
