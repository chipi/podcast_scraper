"""Durable storylines (RFC-128, #2349): the clustering proposes, the registry decides.

A storyline used to be whatever one run's co-occurrence clustering produced: named after the
member topic that ranked highest that day, gone when its topics stopped co-occurring enough. A
replay over the production history (RFC-128) showed 24 of 99 storylines renamed as the corpus grew,
some back and forth four times, each rename breaking every follow, save and Learn path on it.

Here a storyline is a record that outlives any run:

- **Identity separate from membership.** Each run's clustering yields *proposals*. A proposal
  attaches to the registry entry that already holds at least half its topics; only what fits
  none becomes a *candidate*.
- **A bar to create.** A candidate becomes a storyline at 2 topics and 5 episodes, proposed across
  at least 28 days (the replay's 2/5/2).
- **Membership drifts, identity stays.** Topic strengths fade with time (×0.85 per 30 days) and
  grow when proposed again, so a thread can change every topic it began with and keep its id.
- **A name that sticks.** Another label must lead the storyline's proposals for 60 days before a
  rename; the old name becomes an alias.
- **One storyline per topic**, as the clustering's own output: a topic shared by several entries is
  published in the one it anchors, else the one where it is strongest.
- **Never deleted.** A storyline with no new episode for 90 days is *quiet*; a new episode makes it
  active again. One whose topics have all faded into other storylines is published as their
  successor (``id_history.successors``); one whose every topic left the corpus is ``gone``. The
  reference pass moves what listeners kept on either.

Storyline pages are routed by their anchor topic, so the registry keeps it: the anchor a storyline
had before the registry, or its most-episode topic when created. It moves only when that topic
leaves the storyline, to the strongest remaining one. The published member carries
``"anchor": true`` (read by ``search.storylines``); ``lift_to_cluster`` is the topic's strength.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from podcast_scraper.search.storylines import _anchor_topic_id

_DAY_S = 86400.0


@dataclass(frozen=True)
class Params:
    """The registry's rules; the defaults are RFC-128's, from the replay."""

    attach_share: float = 0.5
    decay_per_30_days: float = 0.85
    held_strength: float = 0.2
    rename_days: float = 60.0
    bar_topics: int = 2
    bar_episodes: int = 5
    bar_days: float = 28.0
    quiet_days: float = 90.0
    candidate_forget_days: float = 90.0


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "storyline"


def _members(cluster: Mapping[str, Any]) -> set[str]:
    return {str(m.get("topic_id")) for m in cluster.get("members") or [] if m.get("topic_id")}


def _episodes(cluster: Mapping[str, Any]) -> set[str]:
    return {str(e) for m in cluster.get("members") or [] for e in m.get("episode_ids") or []}


def _seed(prior: Mapping[str, Any], now: float) -> dict[str, Any]:
    """A registry from a storyline artifact written before there was one (the first run): every
    storyline kept, with its id and name."""
    entries = {}
    for c in prior.get("clusters") or []:
        sid = str(c.get("graph_compound_parent_id") or "")
        if not sid:
            continue
        entries[sid] = {
            "name": str(c.get("canonical_label") or sid),
            "aliases": [],
            "topics": {t: 1.0 for t in _members(c)},
            "anchor": _anchor_topic_id(list(c.get("members") or [])),
            "episodes": sorted(_episodes(c)),
            "created_at": now,
            "last_active_at": now,
            "rename": None,
        }
    return {"as_of": now, "entries": entries, "candidates": []}


def reconcile(
    proposals: Sequence[Mapping[str, Any]],
    prior: Optional[Mapping[str, Any]],
    *,
    topic_label: Mapping[str, str],
    topic_eps: Mapping[str, list[str]],
    now: float,
    params: Params = Params(),
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """``(storylines, registry, id_history)`` for this run.

    ``proposals`` are this run's clusters (``canonical_label`` and ``members`` with ``topic_id``);
    ``prior`` is the previous artifact's ``data`` (its ``registry``, or its clusters to seed one);
    ``topic_label`` / ``topic_eps`` describe the corpus now. Deterministic for a given input.
    """
    p = params
    prior = prior or {}
    registry = prior.get("registry")
    if not isinstance(registry, Mapping):
        # No registry yet. From an artifact written before there was one, keep its storylines;
        # on a corpus's very first run, this run's own clusters — otherwise a new corpus would
        # have no storyline until a candidate cleared the 28-day bar.
        registry = _seed(prior if prior.get("clusters") else {"clusters": proposals}, now)
    entries: dict[str, dict[str, Any]] = {
        k: {**v, "topics": dict(v.get("topics") or {}), "aliases": list(v.get("aliases") or [])}
        for k, v in (registry.get("entries") or {}).items()
    }
    candidates: list[dict[str, Any]] = [dict(c) for c in registry.get("candidates") or []]

    as_of = registry.get("as_of")
    elapsed = max(0.0, now - float(now if as_of is None else as_of))
    factor = p.decay_per_30_days ** (elapsed / (30 * _DAY_S))
    for e in entries.values():
        e["topics"] = {t: w * factor for t, w in e["topics"].items()}

    # The largest proposal attached to an entry decides which label it is proposed under.
    attached_label: dict[str, tuple[int, str]] = {}
    for prop in sorted(
        proposals, key=lambda c: (-len(_episodes(c)), str(c.get("canonical_label")))
    ):
        topics, eps = _members(prop), _episodes(prop)
        if not topics:
            continue
        label = str(prop.get("canonical_label") or "")
        best_id, best = None, 0.0
        for sid in sorted(entries):
            held = {t for t, w in entries[sid]["topics"].items() if w >= p.held_strength}
            share = len(topics & held) / len(topics)
            if share > best:
                best_id, best = sid, share
        if best_id is not None and best >= p.attach_share:
            e = entries[best_id]
            for t in topics:
                e["topics"][t] = e["topics"].get(t, 0.0) + 1.0
            known = set(e.get("episodes") or [])
            if eps - known:
                e["last_active_at"] = now
            e["episodes"] = sorted(known | eps)
            if best_id not in attached_label or len(eps) > attached_label[best_id][0]:
                attached_label[best_id] = (len(eps), label)
            continue
        _candidate(candidates, topics, eps, label, now)

    for sid, (_n, label) in attached_label.items():
        e = entries[sid]
        if label == e["name"] or not label:
            e["rename"] = None
        elif (e.get("rename") or {}).get("label") == label:
            if now - float(e["rename"]["since"]) >= p.rename_days * _DAY_S:
                e["aliases"] = [*e["aliases"], e["name"]]
                e["name"], e["rename"] = label, None
        else:
            e["rename"] = {"label": label, "since": now}

    taken = set(entries) | {f"thc:{_slug(a)}" for e in entries.values() for a in e["aliases"]}
    kept_candidates = []
    for c in candidates:
        ready = (
            len(c["topics"]) >= p.bar_topics
            and len(c["episodes"]) >= p.bar_episodes
            and now - float(c["first_seen"]) >= p.bar_days * _DAY_S
            and float(c["last_seen"]) == now
        )
        if ready:
            base, sid, k = f"thc:{_slug(c['label'])}", f"thc:{_slug(c['label'])}", 1
            while sid in taken:
                sid, k = f"{base}-{k}", k + 1
            taken.add(sid)
            entries[sid] = {
                "name": c["label"],
                "aliases": [],
                "topics": {t: 1.0 for t in c["topics"]},
                "anchor": min(c["topics"], key=lambda t: (-len(topic_eps.get(t) or []), t)),
                "episodes": sorted(c["episodes"]),
                "created_at": now,
                "last_active_at": now,
                "rename": None,
            }
        elif now - float(c["last_seen"]) <= p.candidate_forget_days * _DAY_S:
            kept_candidates.append(c)

    storylines, gone, absorbed = _publish(entries, p, topic_label, topic_eps, now)

    history = dict(prior.get("id_history") or {})
    successors = {**dict(history.get("successors") or {}), **absorbed}
    published = {s["graph_compound_parent_id"] for s in storylines}
    id_history = {
        "successors": {k: v for k, v in sorted(successors.items()) if v in published},
        "gone": sorted((set(history.get("gone") or []) | set(gone)) - published),
    }
    state = {"as_of": now, "entries": entries, "candidates": kept_candidates}
    return storylines, state, id_history


def _publish(
    entries: dict[str, dict[str, Any]],
    p: Params,
    topic_label: Mapping[str, str],
    topic_eps: Mapping[str, list[str]],
    now: float,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """``(storylines, gone ids, {absorbed id: successor})``: each held topic published in one
    storyline only — the one it anchors, else the one where it is strongest."""
    in_corpus = {
        sid: {t: w for t, w in e["topics"].items() if t in topic_label and topic_eps.get(t)}
        for sid, e in entries.items()
    }
    owner: dict[str, tuple[bool, float, str]] = {}  # topic -> best (is anchor, strength, -id)
    for sid in sorted(entries, reverse=True):
        e = entries[sid]
        kept = {t: w for t, w in in_corpus[sid].items() if w >= p.held_strength}
        if kept and e.get("anchor") not in kept:
            e["anchor"] = min(kept, key=lambda t: (-kept[t], t))
        for t, w in kept.items():
            key = (t == e.get("anchor"), w, sid)
            if t not in owner or key[:2] >= owner[t][:2]:
                owner[t] = key

    storylines: list[dict[str, Any]] = []
    gone: list[str] = []
    absorbed: dict[str, str] = {}
    for sid in sorted(entries):
        e = entries[sid]
        strengths = {
            t: w for t, w in in_corpus[sid].items() if w >= p.held_strength and owner[t][2] == sid
        }
        if not strengths:
            rest = sorted(in_corpus[sid].items(), key=lambda kv: (-kv[1], kv[0]))
            took = next((owner[t][2] for t, _w in rest if t in owner), None)
            if took is None:
                gone.append(sid)
            else:
                absorbed[sid] = took
            continue
        if e.get("anchor") not in strengths:
            e["anchor"] = min(strengths, key=lambda t: (-strengths[t], t))
        last = e.get("last_active_at")
        quiet = now - float(now if last is None else last) >= p.quiet_days * _DAY_S
        members = [
            {
                "topic_id": t,
                "label": topic_label.get(t, t),
                "episode_ids": sorted(set(topic_eps.get(t, []))),
                "lift_to_cluster": round(w, 4),
                **({"anchor": True} if t == e["anchor"] else {}),
            }
            for t, w in sorted(strengths.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        storylines.append(
            {
                "cluster_type": "theme",
                "canonical_label": e["name"],
                "graph_compound_parent_id": sid,
                "member_count": len(members),
                "members": members,
                "state": "quiet" if quiet else "active",
                "aliases": list(e["aliases"]),
                "created_at": int(float(e["created_at"])),
                "last_active_at": int(float(e["last_active_at"])),
            }
        )
    return storylines, gone, absorbed


def _candidate(
    candidates: list[dict[str, Any]], topics: set[str], eps: set[str], label: str, now: float
) -> None:
    for c in candidates:
        ct = set(c["topics"])
        if len(ct & topics) / len(ct | topics) >= 0.5:
            c["topics"] = sorted(ct | topics)
            c["episodes"] = sorted(set(c["episodes"]) | eps)
            c["last_seen"] = now
            c["label"] = label or c["label"]
            return
    candidates.append(
        {
            "topics": sorted(topics),
            "episodes": sorted(eps),
            "label": label,
            "first_seen": now,
            "last_seen": now,
        }
    )
