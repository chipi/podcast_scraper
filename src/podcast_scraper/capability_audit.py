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

import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from podcast_scraper.identity.bare_name_scope import SCOPED_PREFIX
from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map
from podcast_scraper.search.topic_clusters import (
    _load_topic_clusters_payload,
    consumer_topic_cluster_map,
    top_clusters_by_member_count,
)
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_discover_view import (
    _episode_features,
    _pool_window,
    build_discover_pool,
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


def _slug(token: str) -> str:
    """`person:elon-musk` -> `elon-musk`. The bare id, without its kind prefix.

    Module level so `measure_entity_identity` and `classify_bare_name` share ONE definition —
    two copies of "what counts as the slug" is how the two measures would drift into disagreeing
    about the same token.
    """
    return token.split(":", 1)[1] if ":" in token else token


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


def _feed_label(row: Any) -> str:
    """A feed's name for a human, falling back to its id.

    Production `feed_id` is a sha256 — the audit's first run printed
    `sha256:0c54c0cf2a4f95...` as the "worst feed", which is the actionable half of a coverage or
    defect report and was unreadable. The fixture hid this by using `p01`-style ids.
    """
    title = str(getattr(row, "feed_title", "") or "").strip()
    if title:
        return title
    feed_id = str(getattr(row, "feed_id", "") or "?")
    return feed_id[:16] + "…" if len(feed_id) > 17 else feed_id


def measure_graph_coverage(rows: Sequence[Any]) -> Dict[str, Any]:
    """How often the knowledge graph the product PROMISES is actually there (#1685).

    Your Week, the share card and the next-arc export all distribute graph nodes rather than flat
    clips, via `refs_for_slug`. When an episode has no KG those return `[]` cleanly — the code is
    honest — and the surface quietly shows less. Nobody has measured how often that happens.

    Free on this walk: `has_gi` / `has_kg` are already on the catalog row, so this costs no extra
    artifact loads. Broken down by feed because a single badly-ingested show can account for most
    of a gap, and "82% coverage" reads very differently from "one show is at 4%".
    """
    total = len(rows)
    by_feed: Dict[str, Dict[str, int]] = {}
    kg = gi = both = 0
    for row in rows:
        feed = _feed_label(row)
        slot = by_feed.setdefault(feed, {"episodes": 0, "kg": 0, "gi": 0})
        slot["episodes"] += 1
        has_kg = bool(getattr(row, "has_kg", False))
        has_gi = bool(getattr(row, "has_gi", False))
        kg += has_kg
        gi += has_gi
        both += has_kg and has_gi
        slot["kg"] += has_kg
        slot["gi"] += has_gi

    # Build the (share, share, feed) tuples first so the sort key is typed. Sorting dicts whose
    # values are str | int | float leaves mypy unable to prove the key is comparable, and silencing
    # that with an ignore would hide a real ordering bug just as effectively.
    ranked: List[Tuple[float, float, str, int]] = sorted(
        (
            (
                v["kg"] / v["episodes"] if v["episodes"] else 0.0,
                v["gi"] / v["episodes"] if v["episodes"] else 0.0,
                f,
                v["episodes"],
            )
            for f, v in by_feed.items()
        )
    )
    worst = [
        {"feed": f, "episodes": eps, "kg_share": kg_s, "gi_share": gi_s}
        for kg_s, gi_s, f, eps in ranked
    ]
    return {
        "episodes": total,
        "with_kg": kg,
        "with_gi": gi,
        "with_both": both,
        "kg_share": kg / total if total else 0.0,
        "gi_share": gi / total if total else 0.0,
        "feeds_below_half_kg": sum(1 for kg_s, _gi, _f, _e in ranked if kg_s < 0.5),
        "worst_feeds": worst[:8],
    }


def measure_entity_identity(
    root: Path, token_feeds: Dict[str, set], counts: Counter
) -> Dict[str, Any]:
    """Near-duplicate PERSON entities, which split the affinity signal (#1685).

    Affinity is keyed on `person:<slug>`, so one human appearing under two ids means a follow
    matches only half their episodes — and every downstream count understates. The fixture showed
    the shape (`Skanda Amarnath` / `Skanda Amarnauth`, plus `A. correspondent` and first-name-only
    entities); this measures how much of it production carries.

    Deliberately mechanical: normalised-slug collisions and shared surnames, NOT fuzzy similarity.
    A cheap check that flags candidates for a human to judge is worth more here than a clever one
    that merges two different people — that failure is worse than the duplicate.
    """
    persons = [t for t in counts if t.startswith("person:")]

    # first-name-only: a single word, so it cannot be disambiguated from anyone else with that name
    # Feed span is what separates "harmless" from "harmful" here, and it is the number that
    # decides whether this needs a canonicalisation pass at all:
    #
    #   `person:sam` on ONE show is very likely a recurring first-name reference to a single
    #   person the extractor never got a surname for. Untidy; costs nothing.
    #   `person:alex` across SIX shows is almost certainly six different Alexes pooled under one
    #   followable token — and that is a precision failure the user cannot undo, because there is
    #   no way for them to say WHICH Alex they meant.
    #
    # Counting ids without this cannot tell the two apart, which is why the first version's "155
    # single-word names" was a number without a verdict attached.
    single_word = sorted(p for p in persons if "-" not in _slug(p))
    single_word_rows = sorted(
        (
            {
                "token": p,
                "episodes": counts[p],
                "feeds": len(token_feeds.get(p, set())),
            }
            for p in single_word
        ),
        key=lambda r: (-int(r["feeds"]), -int(r["episodes"]), str(r["token"])),
    )
    spanning = [r for r in single_word_rows if int(r["feeds"]) > 1]

    # same surname, different id — the shape `Amarnath` / `Amarnauth` does NOT match, so this
    # catches the other direction: two people who may be one, sharing a last name.
    by_surname: Dict[str, List[str]] = {}
    for p in persons:
        parts = _slug(p).split("-")
        if len(parts) >= 2:
            by_surname.setdefault(parts[-1], []).append(p)
    shared_surname = {k: sorted(v) for k, v in by_surname.items() if len(v) > 1}

    # a prefix of another id: `sam` vs `sam-altman`, the merge-risk direction
    prefixes = []
    slugs = {_slug(p): p for p in persons}
    for slug, token in sorted(slugs.items()):
        for other_slug, other in slugs.items():
            if other_slug != slug and other_slug.startswith(slug + "-"):
                prefixes.append({"short": token, "long": other, "short_episodes": counts[token]})

    return {
        "person_entities": len(persons),
        "single_word_names": len(single_word),
        "single_word_examples": single_word[:10],
        # The harmful subset: one token, several shows.
        "single_word_spanning_feeds": len(spanning),
        "single_word_worst": single_word_rows[:8],
        "shared_surname_groups": len(shared_surname),
        "shared_surname_examples": [
            {"surname": k, "ids": v} for k, v in sorted(shared_surname.items())[:8]
        ],
        "prefix_pairs": len(prefixes),
        "prefix_examples": prefixes[:8],
    }


#: How much of the transcript's opening must appear in the summary to call it an echo. Matches
#: the fixture-build audit's `_ECHO_PREFIX_CHARS` so the two agree on what "echo" means.
ECHO_PREFIX_CHARS = 60

#: Read only this much of a transcript artifact. Production transcripts are ~200 KB of JSON for an
#: hour-long episode (measured on the acceptance corpus; the 36-episode fixture's are 8 KB, so the
#: fixture would have hidden this by a factor of 25). The echo check needs the FIRST segment only,
#: so parsing 135 MB to read 678 openings would be pure waste.
_TRANSCRIPT_HEAD_BYTES = 8192


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _summary_text(meta: Dict[str, Any]) -> str:
    summary = meta.get("summary")
    if not isinstance(summary, Mapping):
        return ""
    raw = summary.get("raw_text") or summary.get("text") or ""
    if not raw:
        bullets = summary.get("bullets")
        if isinstance(bullets, list):
            raw = " ".join(str(b) for b in bullets if b)
    return str(raw or "")


def _transcript_opening(root: Path, metadata_rel: str) -> str:
    """The first segment's text, read WITHOUT parsing the whole transcript.

    Production transcripts are ~200 KB of JSON; only the opening is needed. Reading a bounded head
    and regex-ing the first `"text"` avoids ~135 MB of parsing across 678 episodes to extract 678
    short strings.

    Returns "" whenever the artifact is absent, truncated mid-token, or shaped differently — this
    is a measurement, and a missing opening means "cannot judge this episode", never "defect".
    """
    rel = PurePosixPath(metadata_rel)
    stem = rel.name[: -len(".metadata.json")] if rel.name.endswith(".metadata.json") else rel.stem
    # Transcripts live in a SIBLING `transcripts/` directory, not beside the metadata:
    #   feeds/<feed>/<run>/metadata/<stem>.metadata.json
    #   feeds/<feed>/<run>/transcripts/<stem>.segments.json
    # My first version appended the suffix to the metadata path, resolved 0/36 openings, and the
    # report printed "defect rate 0.0%". A check that finds nothing and a corpus with nothing
    # wrong are indistinguishable in the output — the exact failure this epic exists to catch.
    run_dir = rel.parent.parent if rel.parent.name == "metadata" else rel.parent
    for suffix in (".segments.json", ".adfree.segments.json", ".transcript.json"):
        candidate = (root / run_dir / "transcripts" / (stem + suffix)).resolve()
        try:
            if not candidate.is_file():
                continue
            with candidate.open("r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(_TRANSCRIPT_HEAD_BYTES)
        except OSError:
            continue
        match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.){10,})"', head)
        if match:
            return match.group(1)
    return ""


def classify_bare_name(bare: str, episode_persons: Iterable[str]) -> Tuple[str, List[str]]:
    """``(verdict, candidates)`` for one single-token person id WITHIN one episode (#1685).

    The rule is the one production already trusts for insight mentions
    (``gi/relational_edges.py::_resolve_span_to_entities``, #1076 chunk 4-A): a span resolves to
    an entity whose token set is a SUPERSET of it, and refuses when more than one qualifies.

        resolvable  exactly one candidate  — `alex` + `alex-mayassi` in the same episode
        ambiguous   two or more            — `trump` + `donald-trump` + `eric-trump`
        orphan      none                   — `jensen` alone, no full name anywhere in the episode

    OUR OWN PLACEHOLDERS ARE NOT CANDIDATES. `unresolved-dario-ep-42` tokenises to
    ``{unresolved, dario, ep, 42}``, a superset of `dario`, so without the exclusion it counts as
    a "full name" and inflates `resolvable` with self-matches — a bare name "resolving" to a
    person we already failed to identify. This measure is what #1685 was decided on, so it has to
    count what it claims to count. Mirrors the same exclusion in
    ``identity/bare_name_scope.resolve_candidates``, which is the rule the pipeline applies.

    Token-subset, not prefix, so a SURNAME-only reference resolves too: `musk` is a token of
    `elon-musk`. Prefix matching would catch the first-name case and miss that one, which is
    half the population.

    Scoped to the episode on purpose. `person:alex` spans two feeds in production, and in the
    other feed "Alex" is somebody else — so a corpus-wide merge would pick one and be wrong for
    the other. Per-episode is what makes the answer safe.
    """
    tokens = _slug(bare).split("-")
    if len(tokens) != 1:
        return "not_bare", []
    bare_slug = tokens[0]
    candidates = sorted(
        p
        for p in {_slug(x) for x in episode_persons}
        if p != bare_slug and not p.startswith(SCOPED_PREFIX) and bare_slug in p.split("-")
    )
    if len(candidates) == 1:
        return "resolvable", candidates
    if len(candidates) > 1:
        return "ambiguous", candidates
    return "orphan", []


def _episode_person_ids(root: Path, row: Any, kg_persons: Iterable[str]) -> set:
    """Every `person:` id this episode carries, across BOTH graph layers (#1685).

    The first version of this measure read only the KG entity set, and reported `person:alex`
    as an ORPHAN — while the corpus itself says `person:alex` and `person:alex-mayassi` are
    co-speakers in the very same episode (mutually, from both dossiers). The co-speaker relation
    lives in the GI layer; the KG entity set is a different population. So the measure was
    asking "is the full name among this episode's KG entities?" while reporting the answer to
    "is the full name in this episode?" — adjacent, not identical, and it undercounted.

    Union of both, because the question is about the EPISODE, not about one artifact.
    """
    ids = {str(x) for x in kg_persons}
    if not getattr(row, "has_gi", False):
        return ids
    artifact = load_json_artifact(root, getattr(row, "gi_relative_path", "") or "")
    if not isinstance(artifact, dict):
        return ids
    nodes = artifact.get("nodes")
    if not isinstance(nodes, list):
        return ids
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id.startswith("person:"):
            ids.add(node_id)
    return ids


def _raw_person_ids(root: Path, row: Any) -> set:
    """Every `person:` node id in this episode's RAW artifacts, placeholders included.

    Deliberately NOT :func:`_episode_person_ids`, which routes the KG layer through
    ``entities_from_kg`` — and that filters `is_unresolved_speaker_placeholder` on purpose, so a
    placeholder can never surface as an entity in the product. Correct there; fatal here.

    A measure of placeholders built on the view that hides placeholders reports zero, forever,
    against any corpus however damaged — and reads as good news. The unit tests for
    :func:`measure_placeholder_health` exist because the first version of it did exactly that.
    """
    ids: set = set()
    for attr, present in (("kg_relative_path", "has_kg"), ("gi_relative_path", "has_gi")):
        if not getattr(row, present, False):
            continue
        artifact = load_json_artifact(root, getattr(row, attr, "") or "")
        if not isinstance(artifact, dict):
            continue
        nodes = artifact.get("nodes")
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if isinstance(node_id, str) and node_id.startswith("person:"):
                ids.add(node_id)
    return ids


def measure_bare_name_resolvability(root: Path, rows: Sequence[Any]) -> Dict[str, Any]:
    """Could a mint-time rule FIX the single-word person ids, or only quarantine them? (#1685)

    The production run counted 155 single-word person ids and left the verdict open. A count
    cannot say whether the episode that produced `person:alex` also contained `Alex Mayassi` —
    and that is the whole decision: resolvable occurrences HEAL the graph (the mention joins the
    real person and strengthens the signal), while ambiguous and orphan ones can only be
    quarantined behind an episode-scoped id.

    Reported per OCCURRENCE (token x episode), not per token, because the same token can be
    resolvable in one episode and orphan in another — `person:alex` demonstrably is. Per-token
    rollups say how consistent each one is.
    """
    from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map as _themes
    from podcast_scraper.search.topic_clusters import consumer_topic_cluster_map as _topics

    cluster_map = _topics(root)
    theme_map = _themes(root)

    verdicts: Counter = Counter()
    per_token: Dict[str, Counter] = {}
    examples: Dict[str, List[Dict[str, str]]] = {"resolvable": [], "ambiguous": [], "orphan": []}

    for row in rows:
        _clusters, _topics_set, kg_persons = _episode_features(root, row, cluster_map, theme_map)
        persons = _episode_person_ids(root, row, kg_persons)
        bare = [p for p in persons if len(_slug(p).split("-")) == 1]
        for token in sorted(bare):
            verdict, candidates = classify_bare_name(token, persons)
            if verdict == "not_bare":
                continue
            verdicts[verdict] += 1
            per_token.setdefault(_slug(token), Counter())[verdict] += 1
            if len(examples[verdict]) < 6:
                examples[verdict].append(
                    {
                        "token": _slug(token),
                        "feed": _feed_label(row),
                        "candidates": ", ".join(candidates) or "-",
                    }
                )

    total = sum(verdicts.values())
    consistent = sum(1 for c in per_token.values() if len(c) == 1)
    return {
        "occurrences": total,
        "distinct_tokens": len(per_token),
        "resolvable": verdicts.get("resolvable", 0),
        "ambiguous": verdicts.get("ambiguous", 0),
        "orphan": verdicts.get("orphan", 0),
        "resolvable_share": verdicts.get("resolvable", 0) / total if total else 0.0,
        "tokens_with_one_verdict": consistent,
        "tokens_mixed": len(per_token) - consistent,
        "examples": examples,
    }


def measure_placeholder_health(root: Path, rows: Sequence[Any]) -> Dict[str, Any]:
    """What the buggy heal already wrote, and whether an enricher is worth building (#1685/#1801).

    Three questions the existing measures cannot answer, all computed from the person sets this
    audit already walks.

    1. CONTAMINATED PLACEHOLDERS — a placeholder id is unique to one episode BY CONSTRUCTION;
       the episode is in the id. So the same `unresolved-…` id appearing in two episodes proves
       at least one of them imported another episode's scope. That is the cross-episode failure
       of the un-fixed `resolve_candidates`, which accepted a placeholder as a heal TARGET. This
       needs no episode-id derivation, which is why it is framed as a uniqueness check rather
       than a comparison — the invariant is self-evidencing.

    2. BLOCKED HEALS — a placeholder sitting in an episode that ALSO contains a real full name
       carrying its token. Under the old rule the placeholder was itself a candidate, so two
       candidates existed, the rule declined to guess, and the bare name was scoped instead of
       joining the real person. Post-fix these are exactly the references that should have
       healed, so this is the repair work-list.

    3. RECURRENCE — how many episodes each single-token name appears in. This is the number that
       decides whether #1801's enricher is worth building: a name appearing once is an incidental
       mention worth nothing, while a name recurring across episodes is a real person whose
       mentions are being lost. A count of bare names does not distinguish those, and the
       decision hangs entirely on the split.

    Reported alongside, never instead of, `measure_bare_name_resolvability` — that one asks
    "could this be fixed?", this one asks "what did we already break, and is fixing it worth it?".
    """
    placeholder_episodes: Dict[str, set] = {}
    blocked: List[Dict[str, str]] = []
    token_episodes: Dict[str, set] = {}

    for row in rows:
        key = str(getattr(row, "metadata_relative_path", "") or _feed_label(row))
        slugs = {_slug(p) for p in _raw_person_ids(root, row)}
        real = {s for s in slugs if not s.startswith(SCOPED_PREFIX)}

        for slug in slugs:
            if not slug.startswith(SCOPED_PREFIX):
                if len(slug.split("-")) == 1:
                    token_episodes.setdefault(slug, set()).add(key)
                continue
            placeholder_episodes.setdefault(slug, set()).add(key)
            # The name a placeholder stands for is always ONE token — only single-token ids are
            # ever scoped — so it is the token immediately after the prefix.
            name = slug[len(SCOPED_PREFIX) :].split("-")[0]
            token_episodes.setdefault(name, set()).add(key)
            healable = sorted(r for r in real if name in r.split("-"))
            if len(healable) == 1:
                blocked.append(
                    {
                        "placeholder": slug,
                        "should_be": healable[0],
                        "feed": _feed_label(row),
                    }
                )

    contaminated = {pid: sorted(eps) for pid, eps in placeholder_episodes.items() if len(eps) > 1}
    recurring = {t: len(eps) for t, eps in token_episodes.items() if len(eps) > 1}

    return {
        "placeholders_total": len(placeholder_episodes),
        "contaminated_ids": len(contaminated),
        "contaminated_examples": [
            {"placeholder": pid, "episodes": len(eps)}
            for pid, eps in sorted(contaminated.items())[:6]
        ],
        "blocked_heals": len(blocked),
        "blocked_examples": blocked[:6],
        "names_total": len(token_episodes),
        "names_recurring": len(recurring),
        "names_once_only": len(token_episodes) - len(recurring),
        "recurring_examples": [
            {"name": t, "episodes": n}
            for t, n in sorted(recurring.items(), key=lambda kv: -kv[1])[:8]
        ],
    }


def measure_content_quality(root: Path, rows: Sequence[Any]) -> Dict[str, Any]:
    """A DEFECT RATE for the most-read text in the product (#1686).

    The summary is what a user sees before deciding to listen, and what /discover is effectively
    selling. A junk summary is not cosmetic — it is the product misdescribing an episode.

    The checks exist already: they were written while fixing the validation corpus, where they
    found real defects (a summary that was the transcript's opening greeting, summaries quoting
    the prompt's own few-shot examples). None has ever been pointed at production.

    Distinct from the runtime guard: `_reject_if_prompt_examples_leaked` rejects at GENERATION
    time, so what it catches never lands. This measures what is ALREADY in the corpus — written
    before the guard existed, or by paths it does not cover. Those episodes do not heal
    themselves; they sit there being served.

    Reported as a rate with the worst feeds named, not pass/fail: the decision this feeds is
    whether the count justifies a re-summarisation pass, which is an LLM bill.
    """
    # The guard's own matcher, not a re-implementation: 'leaked' must mean the same thing at
    # generation time and in the audit, or the two would disagree about the same summary.
    from podcast_scraper.workflow.metadata_generation import (
        _looks_copied_from_example,
        _PROMPT_EXAMPLE_FRAGMENTS,
    )

    total = len(rows)
    missing = absent = blank = echo = short = leak = 0
    by_feed: Counter = Counter()
    examples: List[Dict[str, str]] = []
    leak_examples: List[Dict[str, str]] = []

    for row in rows:
        rel = getattr(row, "metadata_relative_path", None)
        if not rel:
            continue
        feed = _feed_label(row)
        try:
            meta = load_json_artifact(root, str(rel)) or {}
        except Exception:  # noqa: BLE001 — one unreadable artifact must not lose the rest
            missing += 1
            continue

        meta_dict = meta if isinstance(meta, dict) else {}
        text = _summary_text(meta_dict)
        if not text.strip():
            # `summary: null` is the DESIGNED #1496 degradation — summarisation failed, the
            # episode was kept, and a Sentry warning was filed at the time. A summary OBJECT
            # holding no readable text is the other thing entirely: the pipeline produced
            # something, recorded success, and said nothing. Counting them together was why
            # "8 empty summaries" could not tell us which we had.
            if isinstance(meta_dict.get("summary"), Mapping):
                blank += 1
            else:
                absent += 1
            by_feed[feed] += 1
            continue
        if len(text.split()) < 10:
            short += 1
            by_feed[feed] += 1
            continue

        opening = _transcript_opening(root, str(rel))
        if opening:
            head = _norm(opening)[:ECHO_PREFIX_CHARS]
            if len(head) >= ECHO_PREFIX_CHARS and head in _norm(text):
                echo += 1
                by_feed[feed] += 1
                if len(examples) < 5:
                    examples.append({"episode": str(rel).split("/")[-1], "opening": head[:60]})

        # Prompt-example leakage (#1686): a summary about the PROMPT's few-shot subject, not the
        # episode. Same shape as the runtime guard: cheap vocabulary pre-filter, then only a line
        # that reproduces an example SENTENCE counts as a copy.
        summary_obj = meta_dict.get("summary")
        summary_map = summary_obj if isinstance(summary_obj, Mapping) else {}
        bullets = summary_map.get("bullets")
        lines = [str(summary_map.get("title") or "")] + [
            str(b) for b in (bullets if isinstance(bullets, list) else [])
        ]
        haystack = " ".join(lines).lower()
        if any(fragment in haystack for fragment in _PROMPT_EXAMPLE_FRAGMENTS):
            copied = next(
                (
                    (line, hit)
                    for line in lines
                    if line.strip() and (hit := _looks_copied_from_example(line)) is not None
                ),
                None,
            )
            if copied is not None:
                leak += 1
                by_feed[feed] += 1
                if len(leak_examples) < 5:
                    leak_examples.append(
                        {"episode": str(rel).split("/")[-1], "line": copied[0][:80]}
                    )

    defects = absent + blank + short + echo + leak
    return {
        "episodes": total,
        "empty_summary": absent + blank,
        "absent_summary": absent,
        "blank_summary": blank,
        "very_short_summary": short,
        "transcript_echo": echo,
        "prompt_example_leak": leak,
        "unreadable_metadata": missing,
        "defects": defects,
        "defect_rate": defects / total if total else 0.0,
        "worst_feeds": [{"feed": f, "defects": n} for f, n in by_feed.most_common(8)],
        "echo_examples": examples,
        "leak_examples": leak_examples,
    }


#: The gate `TrendingTopics.vue` applies (`RISING`, `MIN_TOTAL`). Mirrored here so the audit
#: reports what the RAIL would actually show, not what the enrichment merely computed. Chosen
#: without data to tune against — which is half of what #1668 asks.
TRENDING_RISING_GATE = 1.5
TRENDING_MIN_TOTAL = 3


def measure_topic_momentum(root: Path) -> Dict[str, Any]:
    """Does the `temporal_velocity` rail ever have anything to show? (#1668)

    Home carries TWO measures of "what's hot" and on the validation corpus they contradict each
    other on the same topic in the same week: the momentum rail called `systems thinking` 1.78x
    and "heating up", while `TrendingTopics` computed 0.86x and showed nothing.

    The reason nobody noticed is the shape this measures. `TrendingTopics` gates on
    `velocity_last_over_6mo >= 1.5` with `total >= 3`, and the fixture's MAXIMUM reading is 0.857.
    So the rail is fully built, mounted and fetching, and has always concluded "nothing
    qualifies" — a component that has never once rendered its own content.

    This reports the distribution and how many topics clear the gate, which decides all four
    questions in that issue: is the rail dead weight, is the disagreement a sparsity artefact, is
    1.5 the right threshold, and does Home need both.
    """
    path = root / "enrichments" / "temporal_velocity.json"
    if not path.is_file():
        return {"available": False, "reason": "no temporal_velocity.json in enrichments/"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"unreadable: {type(exc).__name__}"}

    # The enricher wraps its output in an envelope; `topics` is under `data`, and reading the top
    # level instead silently yields zero topics — which would look exactly like "nothing is
    # trending" rather than "the audit read the wrong key".
    data = payload.get("data")
    topics = (data or {}).get("topics") if isinstance(data, Mapping) else None
    if not isinstance(topics, list) or not topics:
        return {
            "available": False,
            "reason": "envelope carries no topics",
            "enricher_error": payload.get("error"),
            "circuit_state": payload.get("circuit_state"),
        }

    scored = [
        (
            float(t.get("velocity_last_over_6mo") or 0.0),
            int(t.get("total") or 0),
            str(t.get("topic_id") or "?"),
        )
        for t in topics
        if isinstance(t, Mapping)
    ]
    eligible = [x for x in scored if x[1] >= TRENDING_MIN_TOTAL]
    qualifying = [x for x in eligible if x[0] >= TRENDING_RISING_GATE]
    velocities = sorted((v for v, _t, _i in scored), reverse=True)

    return {
        "available": True,
        "topics": len(scored),
        "window_months": (data or {}).get("window_months"),
        "eligible_by_total": len(eligible),
        "qualifying": len(qualifying),
        "gate": TRENDING_RISING_GATE,
        "max_velocity": velocities[0] if velocities else 0.0,
        "median_velocity": statistics.median(velocities) if velocities else 0.0,
        "rail_is_always_empty": len(qualifying) == 0,
        "headroom_to_gate": (TRENDING_RISING_GATE - velocities[0]) if velocities else None,
        # WHAT THE RAIL WOULD RENDER, not what the enrichment computed. The first version listed
        # the highest-velocity topics from `scored` — unfiltered — so the report showed
        # `ai-alignment 6.0x over 2 episodes` beside a gate of `total >= 3`, and I read that as
        # "MIN_TOTAL is not being applied". It is: `TrendingTopics.vue` filters on both. The
        # component was right and the report was showing rows it would never display.
        "would_render": [
            {"topic": i, "velocity": round(v, 4), "total": t}
            for v, t, i in sorted(qualifying, key=lambda x: (-x[0], -x[1], x[2]))[:8]
        ],
        # Kept separately, and labelled, because "high ratio but too few episodes to trust" is a
        # real category worth seeing when tuning the gate — just not one to confuse with output.
        "high_ratio_below_min_total": [
            {"topic": i, "velocity": round(v, 4), "total": t}
            for v, t, i in sorted(scored, key=lambda x: (-x[0], -x[1], x[2]))
            if v >= TRENDING_RISING_GATE and t < TRENDING_MIN_TOTAL
        ][:8],
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
    # ASK the pool for its window; do not recompute it. This line used to be
    # `limit * DISCOVER_POOL_MULTIPLE`, a second implementation of the policy — so when the real
    # one started scaling with corpus size, the report kept printing the old fixed number. The
    # production run on 2026-08-19 said "recency leg reaches 101/678; window is 48", which is
    # self-contradictory: the measurement was right and the number beside it was stale. A
    # measurement tool that restates a policy instead of reading it will always drift from it.
    window = _pool_window(limit, total)
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


def measure_ranking_calibration(rows: Sequence[Any]) -> Dict[str, Any]:
    """#1684: the two ``app_ranking_config`` numbers whose tuning is unverifiable at 36 episodes.

    Reads the SHIPPED config and the SAME scoring kernels ``rank_discover`` uses
    (``_significance`` / ``_feed_significance_means`` / ``_recency_boost``) — a re-implementation
    here would drift and the audit would measure a ranking that does not exist.

    1. Significance normalisation: ``base / feed_mean`` lets a sparse show compete with a
       prolific one. The question production answers: does dividing by a small, noisy mean
       OVER-reward sparse feeds? Read ``sparse_top_share`` against ``sparse_corpus_share`` —
       sparse feeds holding a much larger share of the top normalized scores than of the corpus
       is the over-reward shape.
    2. Recency: a half-life only means something relative to the corpus's spread. Reported as
       the decay multiplier's actual range plus the share of episodes inside one and two
       half-lives — near-1.0 shares mean the signal is flat in practice and the config
       overstates what it does.
    """
    from podcast_scraper.server.app_discover_view import (
        _feed_significance_means,
        _newest_publish_date,
        _recency_boost,
        _significance,
    )
    from podcast_scraper.server.app_ranking_config import (
        DEFAULT_RANKING_CONFIG,
        SIGNAL_RECENCY,
        SIGNAL_SIGNIFICANCE,
    )

    sig_params = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_SIGNIFICANCE)
    feed_means = _feed_significance_means(rows, sig_params)
    per_feed: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        per_feed[getattr(r, "feed_id", "") or "?"].append(float(_significance(r, sig_params)))

    normalized: List[Tuple[float, str]] = []
    for feed, values in per_feed.items():
        mean = feed_means.get(feed, 0.0)
        for value in values:
            normalized.append((value / mean if mean > 0 else value, feed))
    normalized.sort(reverse=True)

    sparse = {feed for feed, values in per_feed.items() if len(values) < 5}
    top_n = min(20, len(normalized))
    top = normalized[:top_n]
    sparse_norms = [v for v, feed in normalized if feed in sparse]
    means_sorted = sorted(feed_means.values())
    significance = {
        "feeds": len(per_feed),
        "sparse_feeds": len(sparse),
        "feed_mean_min": round(means_sorted[0], 3) if means_sorted else 0.0,
        "feed_mean_median": round(statistics.median(means_sorted), 3) if means_sorted else 0.0,
        "feed_mean_max": round(means_sorted[-1], 3) if means_sorted else 0.0,
        "top_n": top_n,
        "sparse_top_share": (
            round(sum(1 for _v, feed in top if feed in sparse) / top_n, 3) if top_n else 0.0
        ),
        "sparse_corpus_share": (
            round(sum(len(per_feed[feed]) for feed in sparse) / len(normalized), 3)
            if normalized
            else 0.0
        ),
        "sparse_median_normalized": (
            round(statistics.median(sparse_norms), 3) if sparse_norms else 0.0
        ),
        "global_median_normalized": (
            round(statistics.median(v for v, _feed in normalized), 3) if normalized else 0.0
        ),
    }

    rec_params = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_RECENCY)
    half_life = float(rec_params.get("half_life_days", 0.0))
    newest = _newest_publish_date(rows)
    boosts = sorted(
        _recency_boost(getattr(r, "publish_date", None), newest, half_life)
        for r in rows
        if getattr(r, "publish_date", None)
    )
    dates = sorted(d for d in (getattr(r, "publish_date", None) for r in rows) if d)
    span_days = 0
    if len(dates) >= 2:
        from datetime import date as _date

        try:
            span_days = (
                _date.fromisoformat(str(dates[-1])[:10]) - _date.fromisoformat(str(dates[0])[:10])
            ).days
        except ValueError:
            span_days = 0
    recency = {
        "half_life_days": half_life,
        "newest": str(dates[-1])[:10] if dates else None,
        "span_days": span_days,
        "multiplier_min": round(boosts[0], 3) if boosts else 0.0,
        "multiplier_median": round(statistics.median(boosts), 3) if boosts else 0.0,
        "multiplier_max": round(boosts[-1], 3) if boosts else 0.0,
        # boost >= 0.5 IS "inside one half-life of the newest" — same function, no date math.
        "share_within_one_half_life": (
            round(sum(1 for b in boosts if b >= 0.5) / len(boosts), 3) if boosts else 0.0
        ),
        "share_within_two_half_lives": (
            round(sum(1 for b in boosts if b >= 0.25) / len(boosts), 3) if boosts else 0.0
        ),
    }
    return {"significance": significance, "recency": recency}


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
        ("graph_coverage", lambda: measure_graph_coverage(rows)),
        ("entity_identity", lambda: measure_entity_identity(root, token_feeds, counts)),
        ("content_quality", lambda: measure_content_quality(root, rows)),
        ("bare_name_resolvability", lambda: measure_bare_name_resolvability(root, rows)),
        ("placeholder_health", lambda: measure_placeholder_health(root, rows)),
        ("topic_momentum", lambda: measure_topic_momentum(root)),
        (
            "picker_discrimination",
            lambda: measure_picker_discrimination(root, rows, counts, limit=limit),
        ),
        ("pool_reachability", lambda: measure_pool_reachability(root, rows, counts, limit=limit)),
        ("corpus_shape", lambda: measure_corpus_shape(rows)),
        ("ranking_calibration", lambda: measure_ranking_calibration(rows)),
    ):
        try:
            report.sections[name] = fn()
        except Exception as exc:  # noqa: BLE001 — one broken area must not lose the others
            report.errors[name] = f"{type(exc).__name__}: {exc}"
    return report


def _render_pool_reachability(report: AuditReport, out: List[str]) -> None:
    """Render the `pool_reachability` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
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


def _render_picker_discrimination(report: AuditReport, out: List[str]) -> None:
    """Render the `picker_discrimination` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
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


def _render_cluster_structure(report: AuditReport, out: List[str]) -> None:
    """Render the `cluster_structure` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
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


def _render_cluster_reach(report: AuditReport, out: List[str]) -> None:
    """Render the `cluster_reach` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
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


def _render_graph_coverage(report: AuditReport, out: List[str]) -> None:
    """Render the `graph_coverage` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
    cov = report.sections.get("graph_coverage")
    if cov and cov["episodes"]:
        out.append("### Graph coverage (what Your Week can actually distribute)")
        out.append(
            f"- KG on **{cov['with_kg']}/{cov['episodes']}** ({cov['kg_share']:.0%}), "
            f"GI on **{cov['with_gi']}** ({cov['gi_share']:.0%}), both on **{cov['with_both']}**"
        )
        if cov["feeds_below_half_kg"]:
            out.append(
                f"- ⚠ **{cov['feeds_below_half_kg']}** feed(s) below 50% KG coverage — a gap "
                "concentrated in one show is a different problem from one spread evenly"
            )
        for f in cov["worst_feeds"][:5]:
            out.append(
                f"    - `{f['feed']}` — {f['episodes']} ep, KG {f['kg_share']:.0%}, "
                f"GI {f['gi_share']:.0%}"
            )
        out.append("")


def _render_entity_identity(report: AuditReport, out: List[str]) -> None:
    """Render the `entity_identity` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
    ident = report.sections.get("entity_identity")
    if ident and ident["person_entities"]:
        out.append("### Entity identity (duplicates split the affinity signal)")
        out.append(
            f"- **{ident['person_entities']}** person entities; "
            f"**{ident['single_word_names']}** are a single word (cannot be disambiguated), "
            f"**{ident['shared_surname_groups']}** surnames appear on more than one id, "
            f"**{ident['prefix_pairs']}** ids are a prefix of another"
        )
        spanning = ident.get("single_word_spanning_feeds") or 0
        if spanning:
            out.append(
                f"- ⚠ **{spanning}** single-word id(s) appear across MORE THAN ONE show — very "
                "likely several people pooled under one followable token, which costs precision "
                "the user cannot undo"
            )
        for r in (ident.get("single_word_worst") or [])[:4]:
            out.append(f"    - `{r['token']}` — {r['episodes']} ep across {r['feeds']} feed(s)")
        for ex in ident["prefix_examples"][:4]:
            out.append(f"    - `{ex['short']}` ({ex['short_episodes']} ep) may be `{ex['long']}`")
        for ex in ident["shared_surname_examples"][:3]:
            out.append(
                f"    - surname `{ex['surname']}`: {', '.join('`' + i + '`' for i in ex['ids'])}"
            )
        out.append("")


def _render_placeholder_health(report: AuditReport, out: List[str]) -> None:
    """Render `placeholder_health` — damage already written, and whether #1801 is worth it."""
    r = report.sections.get("placeholder_health")
    if not r or not r["placeholders_total"]:
        return
    out.append("### Placeholder health — damage already written, and is an enricher worth it?")
    out.append(f"- **{r['placeholders_total']}** episode-scoped placeholder id(s) in the corpus")

    if r["contaminated_ids"]:
        out.append(
            f"- ⚠ **{r['contaminated_ids']}** placeholder id(s) appear in MORE THAN ONE episode. "
            "A placeholder carries its own episode, so it cannot legitimately be shared — each "
            "of these is one episode that imported another episode's scope, written by the "
            "un-fixed heal. These need repairing, and the migration will NOT do it: the bare id "
            "was merged away, so there is nothing left for m0007 to match."
        )
        for ex in r["contaminated_examples"]:
            out.append(f"    - `{ex['placeholder']}` in {ex['episodes']} episodes")
    else:
        out.append("- 0 placeholder ids are shared across episodes — no cross-episode damage")

    if r["blocked_heals"]:
        out.append(
            f"- ⚠ **{r['blocked_heals']}** placeholder(s) sit in an episode that DOES contain "
            "their person — the old rule counted the placeholder as a rival candidate, refused "
            "to guess, and scoped instead of healing. This is the forward-repair work-list."
        )
        for ex in r["blocked_examples"]:
            out.append(f"    - `{ex['placeholder']}` should be `{ex['should_be']}` [{ex['feed']}]")
    else:
        out.append("- 0 blocked heals — no placeholder has a real person available in its episode")

    out.append(
        f"- enricher value (#1801): of **{r['names_total']}** single-token name(s), "
        f"**{r['names_recurring']}** recur across 2+ episodes and **{r['names_once_only']}** "
        "appear exactly once. Only the recurring ones represent a person whose mentions are "
        "being lost; a one-off is an incidental reference worth nothing to resolve."
    )
    for ex in r["recurring_examples"]:
        out.append(f"    - `{ex['name']}` — {ex['episodes']} episodes")
    out.append("")


def _render_bare_name_resolvability(report: AuditReport, out: List[str]) -> None:
    """Render `bare_name_resolvability` — can a mint-time rule FIX the bare ids,
    or only hide them?"""
    r = report.sections.get("bare_name_resolvability")
    if not r or not r["occurrences"]:
        return
    out.append("### Bare person names — would within-episode resolution fix them? (#1685)")
    out.append(
        f"- **{r['occurrences']}** occurrence(s) of **{r['distinct_tokens']}** single-word "
        f"person id(s), judged against the other people in the SAME episode:"
    )
    out.append(
        f"    - **{r['resolvable']}** resolvable ({r['resolvable_share']:.1%}) — exactly one "
        "full name in that episode contains the bare token, so minting the full id would HEAL "
        "the reference instead of splitting it off"
    )
    out.append(
        f"    - **{r['ambiguous']}** ambiguous — two or more candidates (the Donald/Eric shape); "
        "the rule refuses to guess and the id must be episode-scoped"
    )
    out.append(
        f"    - **{r['orphan']}** orphan — no full name anywhere in the episode; nothing to "
        "resolve to, so episode-scoping is the only option"
    )
    out.append(
        f"    - {r['tokens_with_one_verdict']} token(s) always get the same verdict, "
        f"{r['tokens_mixed']} are MIXED across episodes — a mixed token is why this is judged "
        "per episode and never corpus-wide"
    )
    for kind in ("resolvable", "ambiguous", "orphan"):
        for ex in r["examples"].get(kind, [])[:3]:
            out.append(f"    - `{kind}` `{ex['token']}` -> {ex['candidates']} [{ex['feed']}]")
    out.append("")


def _render_content_quality(report: AuditReport, out: List[str]) -> None:
    """Render the `content_quality` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
    cq = report.sections.get("content_quality")
    if cq and cq["episodes"]:
        out.append("### Content quality (the most-read text in the product)")
        out.append(
            f"- defect rate **{cq['defect_rate']:.1%}** — {cq['defects']}/{cq['episodes']}: "
            f"{cq['absent_summary']} absent (no summary written), "
            f"{cq['blank_summary']} blank (summary written, no text), "
            f"{cq['very_short_summary']} very short, "
            f"{cq['transcript_echo']} echoing the transcript's opening, "
            f"{cq.get('prompt_example_leak', 0)} quoting the prompt's style examples"
        )
        if cq.get("prompt_example_leak"):
            out.append(
                f"    - ⚠ {cq['prompt_example_leak']} summary(ies) are about the PROMPT's "
                "few-shot subject, not the episode — fabricated content being served"
            )
        if cq["blank_summary"]:
            out.append(
                f"    - ⚠ {cq['blank_summary']} summary object(s) hold no readable "
                "text — the pipeline recorded success and produced nothing. An absent "
                "summary is the designed #1496 degradation; a blank one is not."
            )
        if cq["unreadable_metadata"]:
            out.append(f"- ⚠ {cq['unreadable_metadata']} metadata artifact(s) could not be read")
        for f in cq["worst_feeds"][:5]:
            out.append(f"    - `{f['feed']}` — {f['defects']} defect(s)")
        for ex in cq["echo_examples"][:3]:
            out.append(f"    - echo in `{ex['episode']}`: {ex['opening']!r}")
        for ex in cq.get("leak_examples", [])[:3]:
            out.append(f"    - leak in `{ex['episode']}`: {ex['line']!r}")
        out.append("")


def _render_topic_momentum(report: AuditReport, out: List[str]) -> None:
    """Render the `topic_momentum` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
    mom = report.sections.get("topic_momentum")
    if mom:
        out.append("### Topic momentum (does the Trending rail ever fire?)")
        if not mom.get("available"):
            out.append(f"- not measurable: {mom.get('reason')}")
        else:
            out.append(
                f"- **{mom['qualifying']}/{mom['eligible_by_total']}** eligible topics clear the "
                f"{mom['gate']}x gate; max velocity **{mom['max_velocity']:.2f}**, "
                f"median {mom['median_velocity']:.2f}"
            )
            if mom["rail_is_always_empty"]:
                out.append(
                    "- ⚠ **the rail can never render**: nothing reaches the gate, so it is fully "
                    f"built and always concludes 'nothing qualifies' (short by "
                    f"{mom['headroom_to_gate']:.2f})"
                )
            for t in mom["would_render"][:5]:
                out.append(f"    - `{t['topic']}` — {t['velocity']}x over {t['total']} episodes")
            below = mom.get("high_ratio_below_min_total") or []
            if below:
                out.append(
                    f"- {len(below)}+ topic(s) clear the ratio but fall under "
                    f"`MIN_TOTAL={TRENDING_MIN_TOTAL}` — correctly hidden, listed because they "
                    "are what a lower floor would admit:"
                )
                for t in below[:3]:
                    out.append(
                        f"    - `{t['topic']}` — {t['velocity']}x over only {t['total']} episode(s)"
                    )
        out.append("")


def _render_corpus_shape(report: AuditReport, out: List[str]) -> None:
    """Render the `corpus_shape` section. Split out of `format_report`, which grew past the
    complexity gate as areas were added — one renderer per area keeps each readable and
    makes a missing section a missing CALL rather than a branch buried in 200 lines."""
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


def _render_ranking_calibration(report: AuditReport, out: List[str]) -> None:
    """Render the `ranking_calibration` section (#1684) — numbers plus the verdicts they carry."""
    section = report.sections.get("ranking_calibration")
    if not section:
        return
    sig = section["significance"]
    rec = section["recency"]
    out.append("### Ranking calibration (#1684 — the shipped config vs this corpus)")
    out.append(
        f"- significance/feed_mean: {sig['feeds']} feeds ({sig['sparse_feeds']} sparse <5 eps), "
        f"feed means min/median/max {sig['feed_mean_min']}/{sig['feed_mean_median']}"
        f"/{sig['feed_mean_max']}"
    )
    out.append(
        f"- sparse feeds hold **{sig['sparse_top_share']:.0%}** of the top {sig['top_n']} "
        f"normalized scores vs **{sig['sparse_corpus_share']:.0%}** of the corpus; "
        f"sparse median normalized {sig['sparse_median_normalized']} vs global "
        f"{sig['global_median_normalized']}"
    )
    if sig["sparse_corpus_share"] > 0 and sig["sparse_top_share"] > 2 * sig["sparse_corpus_share"]:
        out.append(
            "    - ⚠ sparse feeds are over-represented at the top — the noisy-denominator "
            "over-reward shape #1684 predicted"
        )
    out.append(
        f"- recency: half-life {rec['half_life_days']:.0f}d over a {rec['span_days']}-day span; "
        f"multiplier min/median/max {rec['multiplier_min']}/{rec['multiplier_median']}"
        f"/{rec['multiplier_max']}"
    )
    out.append(
        f"- **{rec['share_within_one_half_life']:.0%}** of dated episodes are inside one "
        f"half-life, {rec['share_within_two_half_lives']:.0%} inside two"
    )
    if rec["multiplier_min"] >= 0.5:
        out.append(
            "    - ⚠ the whole corpus sits inside one half-life — recency is nearly flat here "
            "and the config overstates what it does today (it is set for where the corpus is "
            "going; see app_ranking_config)"
        )
    out.append("")


def format_report(report: AuditReport) -> str:
    """Markdown for ``$GITHUB_STEP_SUMMARY`` — a baseline attached to a run, not scrollback."""
    out: List[str] = []
    out.append(
        f"**Corpus:** `{report.corpus_root}` — {report.episodes} episodes, " f"{report.feeds} feeds"
    )
    out.append("")
    for render in (
        _render_pool_reachability,
        _render_picker_discrimination,
        _render_cluster_structure,
        _render_cluster_reach,
        _render_graph_coverage,
        _render_entity_identity,
        _render_content_quality,
        _render_bare_name_resolvability,
        _render_placeholder_health,
        _render_topic_momentum,
        _render_corpus_shape,
        _render_ranking_calibration,
    ):
        render(report, out)
    if report.errors:
        out.append("### Areas that did not complete")
        for name, err in sorted(report.errors.items()):
            out.append(f"- `{name}`: {err}")
        out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and print the report.

    Exists so this file can be MOUNTED into the already-deployed image and executed there, rather
    than baked into a new one. The module imports ``podcast_scraper`` but need not be part of it:
    the image supplies the dependencies, this file supplies the measurement. That is the
    difference between answering a read-only question now and waiting for a
    main -> stack-test -> publish cycle first.
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
    # reads like a measurement rather than like a mount that did not land.
    if report.episodes == 0:
        print("\nERROR: no episodes found — check that the corpus is actually mounted.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
