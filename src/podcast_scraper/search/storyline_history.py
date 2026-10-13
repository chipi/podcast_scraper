"""Seed the storyline registry (RFC-128, #2349) from the corpus's history, keeping every storyline.

Without this, the first run with the registry seeds it from that day's storylines only: every
storyline "created" and "last active" on deploy day, every topic at the same strength, no quiet
storyline for 90 days. This replays the registry month by month over the corpus (episodes
published up to each month's end, the production enricher, no LLM), then merges the result with
the storylines published today:

- a replayed storyline that shares half of its topics, or of a current storyline's, with a current
  storyline folds into it: the current storyline takes the replayed history (created, last active,
  topic strengths, aliases) and keeps its own id, name and anchor — its follows, saves and page do
  not move; a different replayed name becomes an alias;
- a current storyline with no replayed match is kept as it is, dated by its episodes;
- a replayed storyline with no current match is added (it cleared the bar in the replay).

Only the artifact's ``registry`` is written; the next enricher run publishes from it. Dry run by
default; ``--apply`` keeps a backup next to the artifact (undo: copy it back).

    python -m podcast_scraper.search.storyline_history --corpus <corpus root> [--apply]
"""

from __future__ import annotations

import argparse
import calendar
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from podcast_scraper.search.storylines import _anchor_topic_id

_ARTIFACT = Path("enrichments") / "topic_theme_clusters.json"


def _published(bundle: Any) -> str:
    try:
        meta = json.loads(Path(bundle.metadata_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str((meta.get("episode") or {}).get("published_date") or "")


def _month_end(month: str) -> float:
    y, m = (int(x) for x in month.split("-"))
    day = calendar.monthrange(y, m)[1]
    return datetime(y, m, day, 23, 59, tzinfo=timezone.utc).timestamp()


def _date_ts(value: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def run_month(bundles: list[Any], state_dir: Path, now: float) -> dict[str, Any]:
    """One enricher run over ``bundles`` at clock ``now``, reconciled against ``state_dir``'s
    previous artifact; the result is written there for the next month."""
    from podcast_scraper.enrichment.enrichers import topic_theme_clusters as ttc

    clock = ttc._now
    ttc._now = lambda: now  # the enricher's own clock hook
    try:
        ctx = type("Ctx", (), {"run_id": "history", "enricher_id": "topic_theme_clusters"})()
        data: dict[str, Any] = ttc._compute(None, state_dir, bundles, {}, ctx)
    finally:
        ttc._now = clock
    art = state_dir / _ARTIFACT
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({"data": data}), encoding="utf-8")
    return data


def replay_registry(bundles: list[Any], min_episodes: int = 50) -> dict[str, Any]:
    """The registry as if it had run at every month's end of the corpus's history."""
    dated = [(_published(b)[:7], b) for b in bundles]
    months = sorted({m for m, _ in dated if m})
    data: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as state:
        for month in months:
            sub = [b for m, b in dated if m and m <= month]
            if len(sub) >= min_episodes:
                data = run_month(sub, Path(state), _month_end(month))
    registry = data.get("registry")
    return dict(registry) if isinstance(registry, Mapping) else {}


def _topics(cluster: Mapping[str, Any]) -> set[str]:
    return {str(m.get("topic_id")) for m in cluster.get("members") or [] if m.get("topic_id")}


def _episodes(cluster: Mapping[str, Any]) -> set[str]:
    return {str(e) for m in cluster.get("members") or [] for e in m.get("episode_ids") or []}


def merge(
    history: Mapping[str, Any],
    current_clusters: Sequence[Mapping[str, Any]],
    *,
    episode_dates: Mapping[str, float],
    now: float,
    attach_share: float = 0.5,
    held_strength: float = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(registry, report)``: the replayed registry merged with today's storylines (see module)."""
    replayed: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in (history.get("entries") or {}).items()
    }
    held = {
        k: {t for t, w in (v.get("topics") or {}).items() if w >= held_strength}
        for k, v in replayed.items()
    }
    best: dict[str, tuple[float, str]] = {}  # replayed id -> (share, current id)
    for c in current_clusters:
        topics, cid = _topics(c), str(c["graph_compound_parent_id"])
        for rid, h in held.items():
            common = len(topics & h)
            share = max(common / len(topics) if topics else 0.0, common / len(h) if h else 0.0)
            if share >= attach_share and (rid not in best or (share, cid) > best[rid]):
                best[rid] = (share, cid)
    folded: dict[str, list[str]] = {}  # current id -> replayed ids, best first
    for rid, (share, cid) in sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0])):
        folded.setdefault(cid, []).append(rid)

    entries: dict[str, dict[str, Any]] = {}
    for c in current_clusters:
        cid, name = str(c["graph_compound_parent_id"]), str(c.get("canonical_label") or "")
        topics, eps = _topics(c), _episodes(c)
        dates = sorted(episode_dates[e] for e in eps if e in episode_dates)
        rids = folded.get(cid) or []
        if not rids:
            entries[cid] = {
                "name": name,
                "aliases": [],
                "topics": {t: 1.0 for t in topics},
                "anchor": _anchor_topic_id(list(c.get("members") or [])),
                "episodes": sorted(eps),
                "created_at": dates[0] if dates else now,
                "last_active_at": dates[-1] if dates else now,
                "rename": None,
            }
            continue
        group = [replayed[r] for r in rids]
        strengths: dict[str, float] = {}
        aliases: list[str] = []
        for r in group:
            for t, w in (r.get("topics") or {}).items():
                strengths[t] = max(strengths.get(t, 0.0), float(w))
            for a in [*(r.get("aliases") or []), r.get("name")]:
                if a and a != name and a not in aliases:
                    aliases.append(a)
        for t in topics:  # published today: held, so the anchor among them stays put
            strengths[t] = max(strengths.get(t, 0.0), 1.0)
        entries[cid] = {
            "name": name,
            "aliases": aliases,
            "topics": strengths,
            "anchor": _anchor_topic_id(list(c.get("members") or [])),
            "episodes": sorted(set().union(eps, *(r.get("episodes") or [] for r in group))),
            "created_at": min(float(r.get("created_at", now)) for r in group),
            "last_active_at": max(float(r.get("last_active_at", now)) for r in group),
            "rename": None,
        }

    added = []
    for rid in sorted(set(replayed) - set(best)):
        sid, k = rid, 1
        while sid in entries:
            sid, k = f"{rid}-{k}", k + 1
        entries[sid] = replayed[rid]
        added.append(sid)

    registry = {
        "as_of": history.get("as_of", now),
        "entries": entries,
        "candidates": list(history.get("candidates") or []),
    }
    report = {
        "current": len(current_clusters),
        "replayed": len(replayed),
        "with_history": len(folded),
        "folded": sum(len(v) for v in folded.values()),
        "current_only": len(current_clusters) - len(folded),
        "added_from_history": added,
        "candidates": len(registry["candidates"]),
    }
    return registry, report


def seed(corpus: Path, *, apply: bool, now: Optional[float] = None) -> dict[str, Any]:
    """Replay, merge, and (with ``apply``) write the registry into the corpus's artifact."""
    from podcast_scraper.enrichment.paths import discover_episode_bundles

    now = time.time() if now is None else now
    art = corpus / _ARTIFACT
    raw = json.loads(art.read_text(encoding="utf-8"))
    data = raw.get("data", raw)
    bundles = discover_episode_bundles(corpus)
    episode_dates = {
        str(b.episode_id): ts
        for b in bundles
        if getattr(b, "episode_id", None) and (ts := _date_ts(_published(b))) is not None
    }
    registry, report = merge(
        replay_registry(bundles),
        list(data.get("clusters") or []),
        episode_dates=episode_dates,
        now=now,
    )
    if apply:
        backup = art.with_name(f"{art.name}.pre-history-{int(now)}")
        shutil.copy2(art, backup)
        data["registry"] = registry
        tmp = art.with_name(art.name + ".tmp")
        tmp.write_text(json.dumps(raw if "data" in raw else data), encoding="utf-8")
        shutil.copymode(art, tmp)
        tmp.replace(art)
        report["backup"] = str(backup)
    report["applied"] = apply
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """The command: replay and merge; print the report; write only with ``--apply``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="write the registry (default: dry run)")
    args = ap.parse_args(argv)
    print(json.dumps(seed(args.corpus, apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
