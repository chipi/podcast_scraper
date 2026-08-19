"""Measure what the recommender actually does on a REAL corpus (#1682, #1683).

Everything we claim about ranking and personalisation has only been measured against
``tests/fixtures/app-validation-corpus/v3`` — 36 episodes, 9 feeds, 2 topic clusters. That corpus
is controllable, which is why it exists. It is also small enough that several signals are
*unobservable* in it, and one code path has never executed in any test we own:

* ``build_discover_pool`` takes the newest ``4 * limit`` episodes (48 at the default) unioned with
  the newest 48 that MATCH an interest. At 36 episodes ``48 > 36``, so the pool IS the whole
  corpus — the union, the relevance leg and the fallback have never run.
* The picker offers 2 clusters on v3 and both cover 36/36 episodes, so every option yields one
  identical feed. Whether that is a picker defect or an artifact of having two clusters is not
  decidable at that size.

This module is the measurement, not the fix. It lives in ``src/`` rather than ``scripts/`` for one
concrete reason: the prod corpus lives at ``/app/output`` inside the ``pipeline-llm`` container,
and that image ships ``src/``. A measurement that cannot be imported where the data is cannot be
run against production at all.

READ-ONLY. Nothing here writes to the corpus. The operator plane is not touched, no artifact is
created, no index is built. That is a contract, not an accident: it is what lets this run against
production behind ``inspect-prod-corpus.yml`` without a backup first.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map
from podcast_scraper.search.topic_clusters import (
    _load_topic_clusters_payload,
    consumer_topic_cluster_map,
    top_clusters_by_member_count,
)
from podcast_scraper.server.app_discover_view import (
    _episode_features,
    build_discover_pool,
    DISCOVER_POOL_MULTIPLE,
    rank_discover,
)
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

#: The picker's default page size, mirroring ``GET /api/app/clusters?limit=``.
DEFAULT_PICKER_LIMIT = 12
#: The discover feed length these measurements compare.
DEFAULT_FEED_LIMIT = 12

#: A token is a candidate option when it covers at least this many episodes (below it, following
#: it is indistinguishable from following one episode) and at most this share of them (above it,
#: it stops separating the corpus at all).
BAND_MIN_EPISODES = 2
BAND_MAX_SHARE = 0.6


@dataclass
class TokenCoverage:
    """How much of the corpus a single followable token reaches."""

    token: str
    episodes: int
    share: float


@dataclass
class AuditReport:
    """Everything one corpus walk can say. Areas are independent; a failure in one is not fatal."""

    corpus_root: str
    episodes: int
    feeds: int
    sections: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


def _coverage(root: Path, rows: Sequence[Any]) -> Tuple[Counter, Dict[str, set]]:
    """(token -> episode count, token -> the feeds carrying it).

    One KG load per episode — the expensive walk — so both are collected in the same pass. The
    per-feed map is what distinguishes a cluster that merges two names for the same idea inside
    ONE show from one that genuinely spans shows.
    """
    cluster_map = consumer_topic_cluster_map(root)
    theme_map = consumer_theme_cluster_map(root)
    counts: Counter = Counter()
    feeds: Dict[str, set] = {}
    for row in rows:
        clusters, topics, persons = _episode_features(root, row, cluster_map, theme_map)
        feed_id = str(getattr(row, "feed_id", "") or "?")
        for token in (*clusters, *topics, *persons):
            counts[token] += 1
            feeds.setdefault(token, set()).add(feed_id)
    return counts, feeds


def _ranked(counts: Counter) -> List[Tuple[str, int]]:
    """(token, episodes) ordered by coverage desc, then token asc.

    `Counter.most_common()` breaks ties in INSERTION order, and insertion here comes from
    iterating a set per episode — so with eleven tokens tied at 4 episodes the "top" ones changed
    between runs with the hash seed. A measurement that reports different tokens each time it is
    run against the same corpus is not a measurement. Ties break alphabetically instead.
    """
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _feed_slugs(root: Path, rows: Sequence[Any], tokens: Sequence[str], limit: int) -> Tuple:
    return tuple(s.slug for s in rank_discover(root, list(tokens), rows, limit=limit))


def measure_cluster_structure(root: Path, rows: Sequence[Any], counts: Counter) -> Dict[str, Any]:
    """How many clusters exist and how much of the corpus each reaches (#1669 item 3).

    This is what makes the picker measurement interpretable. Two universal clusters cannot
    distinguish "the picker ranks options by the wrong thing" from "this corpus cannot be clustered
    meaningfully" — so any picker fix decided without this number is a guess about which.
    """
    total = len(rows)
    offered = top_clusters_by_member_count(root, DEFAULT_PICKER_LIMIT)
    all_clusters = top_clusters_by_member_count(root, 10_000)
    sizes = [int(c.get("size") or 0) for c in all_clusters]
    coverage = [
        TokenCoverage(
            str(c["id"]),
            counts.get(str(c["id"]), 0),
            (counts.get(str(c["id"]), 0) / total) if total else 0.0,
        )
        for c in all_clusters
    ]
    universal = [c for c in coverage if total and c.episodes >= total]
    return {
        "clusters_total": len(all_clusters),
        "clusters_offered": len(offered),
        "cluster_size_min": min(sizes) if sizes else 0,
        "cluster_size_median": statistics.median(sizes) if sizes else 0,
        "cluster_size_max": max(sizes) if sizes else 0,
        "clusters_covering_every_episode": len(universal),
        "coverage": [
            c.__dict__ for c in sorted(coverage, key=lambda c: (-c.episodes, c.token))[:25]
        ],
    }


def measure_cluster_reach(
    root: Path, rows: Sequence[Any], token_feeds: Dict[str, set]
) -> Dict[str, Any]:
    """Do clusters SPAN SHOWS, or merge synonyms inside one? (#1682)

    A cluster is meant to be a theme that crosses shows — "AI safety" pulling from three different
    podcasts — which is the whole reason the picker offers clusters rather than raw topics: one
    pick, a coherent cross-show feed.

    Production 2026-08-19 measured 278 clusters at median size 2, from 870 candidate tokens. Median
    2 means most clusters merge exactly two topics, which is consistent with either reading:
    near-synonyms inside one show (`ai-safety` + `ai-alignment`), or a genuine small theme across
    two. Size cannot tell those apart. **Feed span can**, and it is the number that decides whether
    `topic_cluster_threshold` (0.75, tuned on v2 fixtures in June and never re-measured on real
    data) is doing the job it was tuned for.
    """
    # Read the PAYLOAD, not `top_clusters_by_member_count` — that returns {id,label,size} and
    # DROPS `members`, so asking it for member topic ids silently yields nothing. My first
    # version did exactly that and reported "0 topics across 0 feeds" for every cluster,
    # which read like a finding and was a bug in the measurement.
    payload = _load_topic_clusters_payload(root) or {}
    raw = payload.get("clusters")
    clusters = [c for c in raw if isinstance(c, Mapping)] if isinstance(raw, list) else []
    spans: List[Dict[str, Any]] = []
    for c in clusters:
        members = c.get("members")
        member_ids = (
            [
                str(m.get("topic_id"))
                for m in members
                if isinstance(m, Mapping) and m.get("topic_id")
            ]
            if isinstance(members, list)
            else []
        )
        feeds: set = set()
        for tid in member_ids:
            feeds |= token_feeds.get(tid, set())
        spans.append(
            {
                "cluster": str(
                    c.get("graph_compound_parent_id") or c.get("canonical_label") or "?"
                ),
                "topics": len(member_ids),
                "feeds": len(feeds),
            }
        )

    multi = [x for x in spans if x["feeds"] > 1]
    single = [x for x in spans if x["feeds"] == 1]
    unknown = [x for x in spans if x["feeds"] == 0]
    return {
        "clusters": len(spans),
        "cross_feed": len(multi),
        "single_feed": len(single),
        "no_feed_data": len(unknown),
        "cross_feed_share": (len(multi) / len(spans)) if spans else 0.0,
        "widest": sorted(spans, key=lambda x: (-x["feeds"], -x["topics"], x["cluster"]))[:10],
    }


def measure_picker_discrimination(
    root: Path, rows: Sequence[Any], counts: Counter, *, limit: int = DEFAULT_FEED_LIMIT
) -> Dict[str, Any]:
    """Do the picker's options produce DIFFERENT feeds, or one identical feed? (#1669 item 2)

    Following a token adds affinity to every episode carrying it. A token on every episode adds the
    same constant everywhere, leaves the relative order untouched, and yields the feed you would
    have got by picking anything else — so the choice is decorative.
    """
    total = len(rows)
    offered = [str(c["id"]) for c in top_clusters_by_member_count(root, DEFAULT_PICKER_LIMIT)]
    offered_feeds = {t: _feed_slugs(root, rows, [t], limit) for t in offered}

    band = [
        TokenCoverage(t, n, n / total if total else 0.0)
        for t, n in _ranked(counts)
        if n >= BAND_MIN_EPISODES and total and n <= total * BAND_MAX_SHARE
    ]
    band_top = band[:DEFAULT_PICKER_LIMIT]
    band_feeds = {c.token: _feed_slugs(root, rows, [c.token], limit) for c in band_top}

    return {
        "offered": [
            {
                "token": t,
                "episodes": counts.get(t, 0),
                "share": (counts.get(t, 0) / total) if total else 0.0,
            }
            for t in offered
        ],
        "offered_distinct_feeds": len(set(offered_feeds.values())),
        "offered_covering_every_episode": sum(
            1 for t in offered if total and counts.get(t, 0) >= total
        ),
        "band_candidates": len(band),
        "band_top": [c.__dict__ for c in band_top],
        "band_distinct_feeds": len(set(band_feeds.values())),
        "decorative": len(offered) > 1 and len(set(offered_feeds.values())) == 1,
        # Every corpus-wide token, not just the cluster ones. Measured on v3 2026-08-19: five
        # tokens cover 36/36 and only two are `tc:` clusters — `topic:lifelong-learning`,
        # `topic:expert-interviews` and `thc:managing-risk` are equally decorative. So "offer
        # topics instead of clusters" is not a fix, and the storylines rail (`thc:`) shares the
        # defect. Filtering has to key on COVERAGE, not on the token's kind.
        "universal_tokens_any_kind": [t for t, n in _ranked(counts) if total and n >= total],
    }


def measure_pool_reachability(
    root: Path, rows: Sequence[Any], counts: Counter, *, limit: int = DEFAULT_FEED_LIMIT
) -> Dict[str, Any]:
    """What fraction of the corpus can reach the ranker at all (#1669 item 1).

    The pool is the newest ``4 * limit`` UNION the newest ``4 * limit`` that match an interest. The
    recency leg is a fixed window regardless of corpus size, so its reach shrinks as the corpus
    grows; everything older depends entirely on the relevance leg finding a match.

    Reported per candidate interest: how many episodes the relevance leg RESCUES — matching
    episodes that the recency window alone would never have shown the ranker. An interest that
    rescues nothing is decorative for a different reason than a corpus-wide one.
    """
    total = len(rows)
    window = limit * DISCOVER_POOL_MULTIPLE
    recency_only = build_discover_pool(rows, limit=limit, interests=(), root=root)
    recency_slugs = {getattr(r, "metadata_relative_path", None) or id(r) for r in recency_only}

    rescued: List[Dict[str, Any]] = []
    for token, n in _ranked(counts)[: DEFAULT_PICKER_LIMIT * 2]:
        if n < BAND_MIN_EPISODES:
            continue
        pool = build_discover_pool(rows, limit=limit, interests=[token], root=root)
        slugs = {getattr(r, "metadata_relative_path", None) or id(r) for r in pool}
        extra = len(slugs - recency_slugs)
        rescued.append({"token": token, "episodes": n, "rescued_by_relevance_leg": extra})

    return {
        "episodes": total,
        "recency_window": window,
        "recency_reach": len(recency_only),
        "recency_share": (len(recency_only) / total) if total else 0.0,
        "unreachable_without_a_match": max(total - len(recency_only), 0),
        "pool_is_whole_corpus": window >= total,
        "rescued": sorted(rescued, key=lambda r: (-r["rescued_by_relevance_leg"], r["token"]))[:20],
        "interests_rescuing_nothing": sum(1 for r in rescued if r["rescued_by_relevance_leg"] == 0),
    }


def measure_corpus_shape(rows: Sequence[Any]) -> Dict[str, Any]:
    """Feed balance and publish spread — the distributions #1684 tunes against.

    ``significance / feed_mean`` and a 365-day recency half-life are both meaningless in isolation:
    a mean over four episodes is noise, and a half-life only says something relative to how far the
    corpus actually spreads.
    """
    per_feed = Counter(getattr(r, "feed_id", "") or "?" for r in rows)
    sizes = sorted(per_feed.values())
    dates = sorted(d for d in (getattr(r, "publish_date", None) for r in rows) if d)
    return {
        "feeds": len(per_feed),
        "episodes_per_feed_min": sizes[0] if sizes else 0,
        "episodes_per_feed_median": statistics.median(sizes) if sizes else 0,
        "episodes_per_feed_max": sizes[-1] if sizes else 0,
        "feeds_with_fewer_than_5": sum(1 for n in sizes if n < 5),
        "publish_date_earliest": dates[0] if dates else None,
        "publish_date_latest": dates[-1] if dates else None,
        "episodes_without_publish_date": sum(
            1 for r in rows if not getattr(r, "publish_date", None)
        ),
    }


def measure(root: Path, *, limit: int = DEFAULT_FEED_LIMIT) -> AuditReport:
    """One corpus walk, every number. An area that raises is recorded and does not stop the rest."""
    rows = list(build_catalog_rows_cumulative(root))
    rows.sort(key=lambda r: (getattr(r, "publish_date", "") or ""), reverse=True)
    report = AuditReport(
        corpus_root=str(root),
        episodes=len(rows),
        feeds=len({getattr(r, "feed_id", "") for r in rows}),
    )
    if not rows:
        report.errors["corpus"] = "no episodes found"
        return report

    counts, token_feeds = _coverage(root, rows)
    for name, fn in (
        ("cluster_structure", lambda: measure_cluster_structure(root, rows, counts)),
        ("cluster_reach", lambda: measure_cluster_reach(root, rows, token_feeds)),
        (
            "picker_discrimination",
            lambda: measure_picker_discrimination(root, rows, counts, limit=limit),
        ),
        ("pool_reachability", lambda: measure_pool_reachability(root, rows, counts, limit=limit)),
        ("corpus_shape", lambda: measure_corpus_shape(rows)),
    ):
        try:
            report.sections[name] = fn()
        except Exception as exc:  # noqa: BLE001 — one broken area must not lose the others
            report.errors[name] = f"{type(exc).__name__}: {exc}"
    return report


def format_report(report: AuditReport) -> str:
    """Markdown for ``$GITHUB_STEP_SUMMARY`` — a baseline attached to a run, not scrollback."""
    out: List[str] = []
    out.append(
        f"**Corpus:** `{report.corpus_root}` — {report.episodes} episodes, {report.feeds} feeds"
    )
    out.append("")

    pool = report.sections.get("pool_reachability")
    if pool:
        out.append("### Discover pool reachability")
        out.append(
            f"- recency leg reaches **{pool['recency_reach']}/{pool['episodes']}** "
            f"({pool['recency_share']:.1%}); window is {pool['recency_window']}"
        )
        out.append(
            f"- **{pool['unreachable_without_a_match']}** episodes cannot reach the ranker without "
            "matching a followed interest"
        )
        if pool["pool_is_whole_corpus"]:
            out.append(
                "- ⚠ the pool IS the whole corpus here, so the union / relevance leg / fallback "
                "are NOT exercised (this is the v3 blind spot)"
            )
        out.append(
            "- interests whose relevance leg rescues nothing: "
            f"**{pool['interests_rescuing_nothing']}**"
        )
        out.append("")

    picker = report.sections.get("picker_discrimination")
    if picker:
        out.append("### Picker discrimination")
        out.append(
            f"- offered **{len(picker['offered'])}** options → "
            f"**{picker['offered_distinct_feeds']}** "
            f"distinct feed(s); {picker['offered_covering_every_episode']} cover every episode"
        )
        out.append(
            f"- discriminating band ({BAND_MIN_EPISODES} <= episodes <= {BAND_MAX_SHARE:.0%} of "
            f"corpus) holds **{picker['band_candidates']}** tokens; its top "
            f"{len(picker['band_top'])} produce "
            f"**{picker['band_distinct_feeds']}** distinct feed(s)"
        )
        # Discriminating power is necessary but NOT sufficient: on v3 this band contains
        # `person:a-correspondent` ("A. correspondent") and first-name-only entities, which
        # separate the corpus perfectly and are not things anyone would offer as an interest.
        # Print the band so that is visible rather than inferred from a count.
        for entry in picker["band_top"][:8]:
            out.append(f"    - `{entry['token']}` — {entry['episodes']} ep, {entry['share']:.0%}")
        if picker["decorative"]:
            out.append("- ⚠ **DECORATIVE**: every offered option yields the same feed")
        universal = picker.get("universal_tokens_any_kind") or []
        if universal:
            out.append(
                f"- corpus-wide tokens of ANY kind: **{len(universal)}** "
                f"({', '.join('`' + t + '`' for t in universal[:6])})"
                " — a fix that only filters `tc:` clusters would leave these offerable"
            )
        out.append("")

    clusters = report.sections.get("cluster_structure")
    if clusters:
        out.append("### Cluster structure")
        out.append(
            f"- **{clusters['clusters_total']}** clusters (size min/median/max "
            f"{clusters['cluster_size_min']}/{clusters['cluster_size_median']}"
            f"/{clusters['cluster_size_max']})"
        )
        out.append(f"- covering EVERY episode: **{clusters['clusters_covering_every_episode']}**")
        out.append("")

    reach = report.sections.get("cluster_reach")
    if reach and reach["clusters"]:
        out.append("### Cluster reach (do clusters span shows?)")
        out.append(
            f"- **{reach['cross_feed']}/{reach['clusters']}** clusters span more than one feed "
            f"({reach['cross_feed_share']:.0%}); **{reach['single_feed']}** are inside a single "
            "show"
        )
        if reach["cross_feed_share"] < 0.5:
            out.append(
                "- ⚠ most clusters merge topics WITHIN one show — that is a synonym merge, not a "
                "cross-show theme, which is what the picker offers them as"
            )
        for w in reach["widest"][:5]:
            out.append(f"    - `{w['cluster']}` — {w['topics']} topics across {w['feeds']} feeds")
        out.append("")

    shape = report.sections.get("corpus_shape")
    if shape:
        out.append("### Corpus shape (ranking calibration inputs)")
        out.append(
            f"- episodes per feed min/median/max: "
            f"{shape['episodes_per_feed_min']}/{shape['episodes_per_feed_median']}"
            f"/{shape['episodes_per_feed_max']}"
            f" — {shape['feeds_with_fewer_than_5']} feed(s) under 5 episodes"
        )
        out.append(
            f"- publish dates {shape['publish_date_earliest']} → {shape['publish_date_latest']}"
            f" ({shape['episodes_without_publish_date']} without a date)"
        )
        out.append("")

    if report.errors:
        out.append("### Areas that did not complete")
        for name, err in sorted(report.errors.items()):
            out.append(f"- `{name}`: {err}")
        out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and print the report.

    Exists so this file can be MOUNTED into the already-deployed image and executed there, rather
    than having to be baked into a new one. The module imports ``podcast_scraper`` but does not
    need to be part of it: the image supplies the dependencies, this file supplies the
    measurement. That is the difference between "answer the question now" and "wait for a
    main -> stack-test -> publish cycle first", which for a read-only question is the wrong price.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Measure ranking + personalisation on a corpus.")
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=DEFAULT_FEED_LIMIT)
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = args.corpus_root.expanduser()
    if not root.is_dir():
        print(f"corpus root is not a directory: {root}")
        return 2
    report = measure(root, limit=args.limit)
    print(format_report(report))
    # An empty corpus is an ERROR, not a finding. Run under compose against a mis-named volume and
    # you get a fresh empty one — the audit would then print "0 episodes" and exit clean, which
    # reads like a measurement rather than like a mount that did not land. Exit non-zero so the
    # difference between "measured nothing" and "measured a corpus with nothing in it" is loud.
    if report.episodes == 0:
        print("\nERROR: no episodes found — check that the corpus is actually mounted.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
