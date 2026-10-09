"""Consumer relational projections — person/topic cards from KG co-occurrence (PRD-043 FR2/FR3).

Read-only aggregation over each episode's ``*.kg.json`` (the same view the entities endpoint
uses via :func:`entities_from_kg`). *KG-grounded*: an entity *appears in* — and a topic *is
about* — an episode iff that episode's KG asserts the node; relatedness is co-occurrence within
those episodes. No ``CorpusGraph`` build, no LLM, no operator-route coupling.

One corpus scan per request (consistent with the existing ``/related`` and ``/search`` consumer
endpoints). The scan is the cost; cache later if the corpus grows large enough to matter.
"""

from __future__ import annotations

import os
import urllib.parse
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from podcast_scraper.search.storylines import (
    storyline_map_by_topic,
    storyline_member_lift,
    storyline_siblings_by_topic,
    STORYLINES_REL,
    top_storylines_by_member_count,
)
from podcast_scraper.search.topic_clusters import (
    theme_map_by_topic,
    theme_siblings_by_topic,
    top_themes_by_member_count,
    TOPIC_CLUSTERS_FILENAME,
)
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_index import (
    get_kg_index,
    iter_kg_entities,
    normalize_label,
)
from podcast_scraper.server.cil_queries import topic_perspectives, topics_perspectives
from podcast_scraper.server.corpus_catalog import (
    CatalogEpisodeRow,
)
from podcast_scraper.server.schemas import (
    AppClusterCard,
    AppClusterMember,
    AppClusterPair,
    AppEntity,
    AppEntityRef,
    AppEpisodeSummary,
    AppInsight,
    AppInterestHit,
    AppOrgCard,
    AppOrgWeb,
    AppPersonCard,
    AppPersonShow,
    AppPersonWeb,
    AppTopic,
    AppTopicCard,
    AppTopicPerspective,
    AppTopicPerspectivesResponse,
)

_DEFAULT_TOP_K = 12

# topic_id -> {cluster_id, cluster_label, cluster_size}; from search/topic_clusters.json.
ClusterMap = dict[str, dict[str, object]]


def _photo_route(person_id: str) -> str:
    """The auth-gated served-photo route for a person, percent-encoding the id so the ``:`` in a
    GI person id (``person:jane-doe``) stays one path segment instead of splitting the path."""
    return f"/api/app/persons/{urllib.parse.quote(person_id, safe='')}/photo"


def _person_web_payload(root: Path) -> dict[str, Any] | None:
    """The person_web payload ``{provider, persons:[…]}``, unwrapping the enrichment ENVELOPE.

    The executor writes every enrichment artifact as an envelope
    (``{derived, status, data:{provider, persons}, …}``), so the payload the card reads lives under
    ``data`` — the same convention every other enrichment reader uses (routes/app_enrichment.py,
    routes/corpus_storylines.py, cil_queries.py). Reading the top level instead found nothing
    on a real corpus (only the hand-written flat test fixtures matched), so bios/photos never
    surfaced. Tolerates an already-flat dict too. Uncached: the corpus-mtime token keys on
    corpus_run_summary.json, which an enrichment run does not bump, so a cached miss would hide
    freshly-enriched rows until the next ingest; person_web.json is small, read it live."""
    doc = load_json_artifact(root, "enrichments/person_web.json")
    if not isinstance(doc, dict):
        return None
    inner = doc.get("data")
    return inner if isinstance(inner, dict) else doc


def hosted_photo_urls(root: Path) -> dict[str, str]:
    """``{person_id: served photo route}`` for every person the web enricher HOSTS a photo for.

    Read once, reused by every people-listing surface (person card, key voices, topic Top voices)
    so a small circular avatar hydrates consistently wherever a name appears. We expose only the
    served (our-domain) route, never the raw external URL — that would leak the viewer's IP to the
    source. Best-effort: an absent/malformed artifact → ``{}``."""
    doc = _person_web_payload(root)
    if doc is None:
        return {}
    out: dict[str, str] = {}
    for row in doc.get("persons") or []:
        if not isinstance(row, dict) or not row.get("image_hosted"):
            continue
        pid = row.get("person_id")
        if isinstance(pid, str) and pid:
            out[pid] = _photo_route(pid)
    return out


def with_photos(people: list[AppEntity], photos: dict[str, str]) -> list[AppEntity]:
    """Hydrate ``image_url`` on each person that has a hosted photo (leaves the data available to
    every people-list surface; whether the UI renders the avatar is a per-surface choice)."""
    if not photos:
        return people
    return [
        p.model_copy(update={"image_url": photos[p.id]}) if p.id in photos else p for p in people
    ]


def _person_web(root: Path, person_id: str) -> AppPersonWeb | None:
    """The person's external bio + attribution from ``enrichments/person_web.json``, if present.

    Read-time projection: absent artifact / no matching row / missing bio → None (the card stays
    lean, exactly as before the enricher ran). Best-effort — a malformed artifact never breaks the
    card."""
    doc = _person_web_payload(root)
    if doc is None:
        return None
    source = str(doc.get("provider") or "")
    for row in doc.get("persons") or []:
        if not isinstance(row, dict) or row.get("person_id") != person_id:
            continue
        bio = row.get("bio")
        if not isinstance(bio, str) or not bio.strip():
            return None
        # Only expose a photo we HOST (served from our domain) — never the raw external URL, which
        # would leak the viewer's IP to the source. image_hosted is set by the enricher image step.
        hosted = bool(row.get("image_hosted"))
        image_url = _photo_route(person_id) if hosted else None
        return AppPersonWeb(
            bio=bio.strip(),
            description=(
                row.get("description") if isinstance(row.get("description"), str) else None
            ),
            source=str(row.get("source") or source or "web"),
            source_url=row.get("source_url") if isinstance(row.get("source_url"), str) else None,
            image_url=image_url,
            license=row.get("license") if isinstance(row.get("license"), str) else None,
            image_license=(
                row.get("image_license") if isinstance(row.get("image_license"), str) else None
            ),
            image_artist=(
                row.get("image_artist") if isinstance(row.get("image_artist"), str) else None
            ),
        )
    return None


# (row, persons, topics) for the episodes a card actually aggregates over.
_MatchedEpisodes = list[tuple[CatalogEpisodeRow, list[AppEntity], list[AppTopic]]]


def _person_episodes(
    root: Path, person_id: str, rows: Sequence[CatalogEpisodeRow] | None
) -> _MatchedEpisodes:
    """Episodes where ``person_id`` appears, in catalog order.

    Default (route) path: O(matches) via the corpus-mtime-cached inverted KG index — no KG re-parse.
    ``rows`` override (tests): an uncached scan of that subset, preserving the pre-index behaviour.
    """
    if rows is None:
        return [(e.row, e.persons, e.topics) for e in get_kg_index(root).person_episodes(person_id)]
    return [
        (row, persons, topics)
        for row, persons, _orgs, topics in iter_kg_entities(root, rows)
        if any(p.id == person_id for p in persons)
    ]


def _topic_episodes(
    root: Path, topic_id: str, rows: Sequence[CatalogEpisodeRow] | None
) -> _MatchedEpisodes:
    """Episodes about ``topic_id``, in catalog order (index fast-path, or a ``rows`` scan)."""
    if rows is None:
        return [(e.row, e.persons, e.topics) for e in get_kg_index(root).topic_episodes(topic_id)]
    return [
        (row, persons, topics)
        for row, persons, _orgs, topics in iter_kg_entities(root, rows)
        if any(t.id == topic_id for t in topics)
    ]


def _org_episodes(
    root: Path, org_id: str, rows: Sequence[CatalogEpisodeRow] | None
) -> list[tuple[CatalogEpisodeRow, list[AppEntity], list[AppEntity], list[AppTopic]]]:
    """Episodes mentioning ``org_id`` — ``(row, persons, orgs, topics)`` — in catalog order (#2031).

    Unlike the person/topic helpers, this carries ORGS through too, because the org card's
    co-occurrence lists both the people and the other orgs mentioned alongside it.
    """
    if rows is None:
        return [
            (e.row, e.persons, e.orgs, e.topics) for e in get_kg_index(root).org_episodes(org_id)
        ]
    return [
        (row, persons, orgs, topics)
        for row, persons, orgs, topics in iter_kg_entities(root, rows)
        if any(o.id == org_id for o in orgs)
    ]


# How many storylines the entity resolver indexes. The artifact is small (a corpus has tens of
# themes, not thousands) and this is a NAME lookup, so the cap only guards against a pathological
# artifact rather than shaping results.
_STORYLINE_INDEX_CAP = 500


def resolve_entity(
    root: Path,
    query: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
) -> AppEntityRef | None:
    """Resolve an exact/near-exact person/topic/org/storyline match for ``query``, else ``None``.

    Precedence person > topic > org > storyline on a tie (all four have cards). Default path is an
    O(1) lookup in the cached KG index; a ``rows`` override scans that subset.

    Storylines are resolved here rather than matched in the client (operator 2026-09-17). They live
    in a different artifact from the KG — ``enrichments/topic_theme_clusters.json``, not the
    per-episode graphs — so they are read from the theme-cluster summaries and normalised the same
    way, which keeps ONE definition of "does this query name an entity". A label match in the client
    could not rank, could not see past the endpoint's 50-item cap, and left every other consumer of
    this resolver blind to storylines.
    """
    norm = normalize_label(query)
    if not norm:
        return None
    if rows is None:
        index = get_kg_index(root)
        return (
            index.person_ref_by_norm.get(norm)
            or index.topic_ref_by_norm.get(norm)
            or index.org_ref_by_norm.get(norm)  # #2031 — orgs now have cards, so search finds them
            or _storyline_ref_by_norm(root).get(norm)
            or _theme_ref_by_norm(root).get(norm)
        )
    persons_idx: dict[str, AppEntityRef] = {}
    topics_idx: dict[str, AppEntityRef] = {}
    orgs_idx: dict[str, AppEntityRef] = {}
    for _row, persons, orgs, topics in iter_kg_entities(root, rows):
        for p in persons:
            persons_idx.setdefault(
                normalize_label(p.name), AppEntityRef(id=p.id, kind="person", label=p.name)
            )
        for t in topics:
            topics_idx.setdefault(
                normalize_label(t.label), AppEntityRef(id=t.id, kind="topic", label=t.label)
            )
        for o in orgs:
            orgs_idx.setdefault(
                normalize_label(o.name),
                AppEntityRef(id=o.id, kind="organization", label=o.name),
            )
    return (
        persons_idx.get(norm)
        or topics_idx.get(norm)
        or orgs_idx.get(norm)
        # Storylines and themes are corpus-level, not per-episode, so the `rows` subset does not
        # bound them — the same maps serve both paths.
        or _storyline_ref_by_norm(root).get(norm)
        or _theme_ref_by_norm(root).get(norm)
    )


def _artifact_mtime(root: Path, rel: str) -> float:
    """The artifact's own mtime (0.0 when absent) — the token the two label maps below cache on.

    They used to be ``lru_cache``d on the corpus root ALONE, so a re-enrichment that rewrote the
    storyline or theme artifact left entity search resolving against the old clusters until the
    process restarted (2026-10-09). The payload loaders already token on these same mtimes.
    """
    try:
        # callers pass a validated corpus root; `rel` is a module constant.
        # codeql[py/path-injection] -- validated corpus root + constant relative path (Type 1).
        return os.path.getmtime(root / rel)
    except OSError:
        return 0.0


def _theme_ref_by_norm(root: Path) -> Mapping[str, AppEntityRef]:
    """Normalised theme label → ref; cached until the topic-cluster artifact changes."""
    return _theme_ref_by_norm_at(
        root, _artifact_mtime(root, os.path.join("search", TOPIC_CLUSTERS_FILENAME))
    )


@lru_cache(maxsize=8)
def _theme_ref_by_norm_at(root: Path, _token: float) -> Mapping[str, AppEntityRef]:
    """Normalised theme label → ref, from the topic-cluster artifact.

    Themes are resolved here for the same reason storylines are (operator 2026-09-17): a label match
    in the client cannot rank, cannot see past the endpoint's item cap, and leaves every OTHER
    consumer of this resolver blind. Until now that is exactly what themes were — a theme label
    typed verbatim returned nothing.

    LAST in the precedence chain, deliberately. A label can name both a topic and a theme
    (`Lifelong Learning` is `topic:lifelong-learning` AND `tc:lifelong-learning` in the v3 fixture),
    and putting themes ahead of topics would silently change where every such query lands today.
    Going last makes this purely additive: a theme resolves only when nothing narrower claims the
    label. The cost is that an ambiguous label still opens the topic — the narrower thing wins —
    which is a ranking decision worth revisiting with real queries rather than guessing now.

    Unlike the storyline map this carries the cluster's OWN `tc:` id, because `/theme/:id` routes by
    it. The storyline equivalent passes an anchor topic id because `/storyline/:id` routes that way.
    """
    out: dict[str, AppEntityRef] = {}
    for th in top_themes_by_member_count(root, _STORYLINE_INDEX_CAP):
        label = str(th.get("label") or "").strip()
        tid = str(th.get("id") or "").strip()
        if not label or not tid:
            continue
        norm = normalize_label(label)
        if norm:
            out.setdefault(norm, AppEntityRef(id=tid, kind="theme", label=label))
    return out


def _storyline_ref_by_norm(root: Path) -> Mapping[str, AppEntityRef]:
    """Normalised storyline label → ref; cached until the storyline artifact changes."""
    return _storyline_ref_by_norm_at(root, _artifact_mtime(root, STORYLINES_REL))


@lru_cache(maxsize=8)
def _storyline_ref_by_norm_at(root: Path, _token: float) -> Mapping[str, AppEntityRef]:
    """Normalised storyline label → ref, from the theme-cluster artifact.

    Cached on the artifact's own mtime (``_storyline_ref_by_norm``), because this is read on every
    entity search and the artifact only changes when the corpus is re-enriched.
    ``min_members=1`` deliberately: the /4
    floor on the Home rail is a SURFACING decision about where a listener is sent, and refusing to
    resolve a storyline the user typed the exact name of would be a different, worse thing —
    searching for something by name and being told it does not exist.
    """
    out: dict[str, AppEntityRef] = {}
    for s in top_storylines_by_member_count(root, _STORYLINE_INDEX_CAP, min_members=1):
        label = str(s.get("label") or "").strip()
        # The ANCHOR TOPIC id, not the `thc:` id, because `/storyline/:id` ROUTES by anchor topic.
        # The original reason — "there is no storyline endpoint" — stopped being true when
        # `/api/app/storylines/{id}` was added; the route still takes an anchor, so this still does.
        # Every other producer of a storyline destination passes the anchor too: FollowedInterests,
        # PodcastSignalsBand, TopicBrowseView.
        anchor = str(s.get("anchor_topic_id") or "").strip()
        if not label or not anchor:
            continue
        norm = normalize_label(label)
        if norm:
            out.setdefault(norm, AppEntityRef(id=anchor, kind="storyline", label=label))
    return out


InterestKind = Literal["topic", "person", "theme", "storyline"]


def _match_rank(norm_label: str, norm_query: str) -> int | None:
    """0 = label starts with the query, 1 = a later word does, 2 = mid-word; else None."""
    if norm_label.startswith(norm_query):
        return 0
    if f" {norm_query}" in norm_label:
        return 1
    if norm_query in norm_label:
        return 2
    return None


def search_interests(
    root: Path, kind: InterestKind, query: str, limit: int = 20
) -> list[AppInterestHit]:
    """Followables of ``kind`` whose label contains ``query`` — the Interests search box.

    ``resolve_entity`` answers "does this name ONE entity"; this answers "what could I follow", so
    it substring-matches and returns many. Ranked by where the query lands in the label, then by
    how much of the corpus the entity covers (episodes for topics and people, members for themes and
    storylines), so a common name beats an obscure one that merely starts the same way.

    Ids are interest TOKENS, which is why a storyline returns its ``thc:`` id here while
    ``_storyline_ref_by_norm`` returns its anchor topic: this result is followed, that one opened.
    """
    norm_query = normalize_label(query)
    if not norm_query:
        return []
    scored: list[tuple[int, int, str, AppInterestHit]] = []
    if kind in ("topic", "person"):
        index = get_kg_index(root)
        refs = index.topic_ref_by_norm if kind == "topic" else index.person_ref_by_norm
        eps = index.topic_to_eps if kind == "topic" else index.person_to_eps
        # Two spellings of one canonical person share an id; keep whichever matches better.
        best: dict[str, tuple[int, int, str, AppInterestHit]] = {}
        for norm, ref in refs.items():
            rank = _match_rank(norm, norm_query)
            if rank is None:
                continue
            weight = -len(eps.get(ref.id, ()))
            if ref.id not in best or (rank, weight, norm) < best[ref.id][:3]:
                hit = AppInterestHit(id=ref.id, kind=kind, label=ref.label)
                best[ref.id] = (rank, weight, norm, hit)
        scored = list(best.values())
    else:
        rows = (
            top_themes_by_member_count(root, _STORYLINE_INDEX_CAP)
            if kind == "theme"
            else top_storylines_by_member_count(root, _STORYLINE_INDEX_CAP, min_members=1)
        )
        for row in rows:
            label = str(row.get("label") or "").strip()
            token = str(row.get("id") or "").strip()
            norm = normalize_label(label)
            rank = _match_rank(norm, norm_query) if norm else None
            if rank is None or not token:
                continue
            anchor = str(row.get("anchor_topic_id") or "").strip() or None
            hit = AppInterestHit(id=token, kind=kind, label=label, anchor_topic_id=anchor)
            scored.append((rank, -int(row.get("size") or 0), norm, hit))
    scored.sort(key=lambda s: (s[0], s[1], s[2]))
    return [s[3] for s in scored[: max(limit, 0)]]


def _enrich_topic(
    topic: AppTopic, cluster_map: ClusterMap, storyline_map: ClusterMap | None = None
) -> AppTopic:
    """Attach semantic + theme cluster identity to a topic (no-op when unclustered)."""
    update: dict[str, object] = {}
    info = cluster_map.get(topic.id)
    if info:
        update.update(info)
    if storyline_map:
        tinfo = storyline_map.get(topic.id)
        if tinfo:
            update.update(tinfo)
    return topic.model_copy(update=update) if update else topic


def _sorted_episode_cards(root: Path, rows: list[CatalogEpisodeRow]) -> list[AppEpisodeSummary]:
    """Project rows to episode cards, newest-first (undated episodes sort last)."""
    cards = [row_to_summary(root, r) for r in rows]
    cards.sort(key=lambda c: (c.publish_date is not None, c.publish_date or ""), reverse=True)
    return cards


# Role precedence for the aggregate card badge: a person who hosts anywhere is a "host",
# a guest anywhere (but never a host) is a "guest", otherwise the weakest role seen.
_ROLE_RANK = {"host": 3, "guest": 2, "mentioned": 1}


def _aggregate_role(roles: Sequence[str | None]) -> str | None:
    """The strongest speaker role across a person's episode nodes (host > guest > mentioned)."""
    ranked = [r for r in roles if r]
    if not ranked:
        return None
    return max(ranked, key=lambda r: _ROLE_RANK.get(r, 0))


def _person_shows(pairs: Sequence[tuple[CatalogEpisodeRow, str | None]]) -> list[AppPersonShow]:
    """Group a person's appearances by show, aggregating their role within each show.

    A person can host one show and guest on another, so role is resolved per-feed. Hosted shows
    sort first, then by episode footprint, then title — so the card leads with "their" shows.
    """
    by_feed: dict[str, dict[str, Any]] = {}
    for row, role in pairs:
        fid = row.feed_id or ""
        if not fid:
            continue
        slot = by_feed.setdefault(fid, {"title": row.feed_title or fid, "roles": [], "count": 0})
        if row.feed_title:
            slot["title"] = row.feed_title
        slot["roles"].append(role)
        slot["count"] += 1
    shows = [
        AppPersonShow(
            feed_id=fid,
            title=slot["title"],
            role=_aggregate_role(slot["roles"]),
            episode_count=slot["count"],
        )
        for fid, slot in by_feed.items()
    ]
    shows.sort(key=lambda s: (-_ROLE_RANK.get(s.role or "", 0), -s.episode_count, s.title))
    return shows


def build_person_card(
    root: Path,
    person_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppPersonCard | None:
    """Project the person's corpus footprint to a card, or ``None`` if they appear nowhere."""
    cluster_map: ClusterMap = theme_map_by_topic(root)
    storyline_map: ClusterMap = storyline_map_by_topic(root)

    label = ""
    roles: list[str | None] = []
    appears_in: list[CatalogEpisodeRow] = []
    people_by_id: dict[str, AppEntity] = {}
    related_roles: dict[str, list[str]] = {}
    topics_by_id: dict[str, AppTopic] = {}
    person_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()

    for row, persons, topics in _person_episodes(root, person_id, rows):
        match = next((p for p in persons if p.id == person_id), None)
        if match is None:
            continue
        if not label:
            label = match.name
        roles.append(match.role)
        appears_in.append(row)
        for p in persons:
            if p.id == person_id:
                continue
            people_by_id[p.id] = p
            person_counts[p.id] += 1
            # Keep EVERY shared episode's role, not just the last one. The line above is
            # last-write-wins, so `p.role` by itself is whichever episode happened to be processed
            # last — a co-host would read "mentioned" whenever their final shared episode merely
            # mentioned them. Aggregated below to host > guest > mentioned, the same precedence the
            # subject's own role and the show page already use.
            if p.role:
                related_roles.setdefault(p.id, []).append(p.role)
        for t in topics:
            topics_by_id[t.id] = t
            topic_counts[t.id] += 1

    if not appears_in:
        return None

    # Role aggregated across the SHARED episodes before any chip can claim it — see the note at the
    # collection site. `model_copy` rather than mutation: these entities come from the cached KG
    # index and are shared with every other card built in this process, so writing through them
    # would leak one card's aggregate into the next.
    related_people = with_photos(
        [
            people_by_id[i].model_copy(update={"role": _aggregate_role(related_roles.get(i, []))})
            for i, _ in person_counts.most_common(top_k)
        ],
        hosted_photo_urls(root),
    )
    related_topics = [
        _enrich_topic(topics_by_id[i], cluster_map, storyline_map)
        for i, _ in topic_counts.most_common(top_k)
    ]
    return AppPersonCard(
        id=person_id,
        label=label or person_id.split(":", 1)[-1],
        role=_aggregate_role(roles),
        shows=_person_shows(list(zip(appears_in, roles))),
        episode_count=len(appears_in),
        episodes=_sorted_episode_cards(root, appears_in),
        related_people=related_people,
        related_topics=related_topics,
        web=_person_web(root, person_id),
    )


def _build_cluster_card(
    root: Path,
    cluster_id: str,
    *,
    group_map: ClusterMap,
    id_key: str,
    label_key: str,
    lift: Mapping[str, float] | None = None,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> (
    tuple[
        str,
        list[AppClusterMember],
        list[CatalogEpisodeRow],
        list[AppEntity],
        AppClusterPair | None,
    ]
    | None
):
    """Members, merged episodes and people for a GROUPING of topics — themes and storylines alike.

    Both are groupings rather than entities: neither is ever a node on an episode, so neither can be
    built the way a topic card is (``build_topic_card`` matches a topic node by id, and a `tc:` or
    `thc:` id matches nothing). The only difference between them is which map they come from and
    what the keys are called — `tc:` groups topics that MEAN the same thing, `thc:` groups topics
    that keep coming up TOGETHER — so the projection is shared and the callers supply the map.

    The episode list is the UNION across every member, de-duplicated. That merge is the point:
    a grouping exists because looking at one member misses the others, so a list showing a single
    member's episodes would not answer the question the grouping poses. ``related_people`` is
    counted across the whole union too, so a grouping's top voices are the people who recur across
    it rather than inside one member.
    """
    # The maps are topic -> cluster; invert rather than re-reading the artifact, so one parser owns
    # membership and the two views cannot disagree about it.
    label = ""
    member_ids: list[str] = []
    for tid, info in group_map.items():
        if info.get(id_key) != cluster_id:
            continue
        member_ids.append(tid)
        if not label:
            raw = info.get(label_key)
            if isinstance(raw, str) and raw.strip():
                label = raw.strip()
    if not member_ids:
        return None

    # De-duplicate by episode id: a member topic's episodes overlap heavily with its siblings' —
    # that overlap is WHY they cluster — so without this the list repeats the same episode once per
    # member it mentions.
    seen_eps: set[str] = set()
    # Per-member episode keys, so the strongest co-occurring PAIR falls out of the same walk. The
    # grouping's evidence — "these two keep turning up together" — is otherwise asserted and
    # never shown.
    eps_by_member: dict[str, set[str]] = {}
    # Publish dates per member, so "what changed" comes out of the same walk. A grouping is a thing
    # that MOVES — members join it, carry it for a while, drop out — and a flat list of names cannot
    # say any of that.
    dates_by_member: dict[str, list[str]] = {}
    about: list[CatalogEpisodeRow] = []
    people_by_id: dict[str, AppEntity] = {}
    person_counts: Counter[str] = Counter()
    episodes_per_member: Counter[str] = Counter()
    labels: dict[str, str] = {}

    for tid in member_ids:
        for row, persons, topics in _topic_episodes(root, tid, rows):
            match = next((t for t in topics if t.id == tid), None)
            if match is None:
                continue
            episodes_per_member[tid] += 1
            labels.setdefault(tid, match.label)
            # `metadata_relative_path` — the row's non-optional natural key. `episode_id` is
            # `Optional[str]` on CatalogEpisodeRow, so keying on it needs a fallback, and an
            # `id(row)` fallback would silently de-duplicate nothing (every row object is distinct).
            # This avoids the question entirely. NOT a bug that was observed: the fixture's rows all
            # carry an episode_id, so both keys de-duplicate correctly here.
            key = row.metadata_relative_path
            eps_by_member.setdefault(tid, set()).add(key)
            if row.publish_date:
                dates_by_member.setdefault(tid, []).append(str(row.publish_date)[:10])
            if key in seen_eps:
                continue
            seen_eps.add(key)
            about.append(row)
            for p in persons:
                people_by_id[p.id] = p
                person_counts[p.id] += 1

    if not about:
        return None

    related_people = with_photos(
        [people_by_id[i] for i, _ in person_counts.most_common(top_k)], hosted_photo_urls(root)
    )
    # Members ordered by how much of the corpus each carries — the first row is the member a reader
    # is most likely to recognise, which is what makes the grouping legible at a glance.
    # LIFT first when the caller has it. Episode count answers "which member is biggest"; lift
    # answers "which member makes this a grouping" — only the second explains why the set exists.
    # It also breaks ties size cannot: two members on 4 episodes each can carry different lift.
    if lift:
        ordered = sorted(member_ids, key=lambda t: (-lift.get(t, 0.0), -episodes_per_member[t], t))
    else:
        ordered = sorted(member_ids, key=lambda t: (-episodes_per_member[t], t))
    # The split is the grouping's OWN median episode date, not a fixed window: a six-month corpus
    # and a six-year one should both read sensibly, and "last 12 months" would brand every member of
    # a young corpus `new`.
    all_dates = sorted(d for ds in dates_by_member.values() for d in ds)
    midpoint = all_dates[len(all_dates) // 2] if all_dates else ""

    def _trend(tid: str) -> tuple[str | None, str | None, str]:
        ds = sorted(dates_by_member.get(tid, []))
        if not ds or not midpoint:
            return None, None, "steady"
        early = sum(1 for d in ds if d < midpoint)
        late = len(ds) - early
        if early == 0:
            state = "new"  # was not here in the first half at all
        elif late == 0:
            state = "gone"  # has not appeared since
        elif late >= early * 2:
            state = "growing"
        elif early >= late * 2:
            state = "fading"
        else:
            state = "steady"
        return ds[0], ds[-1], state

    members = []
    for i, tid in enumerate(ordered):
        first, last, state = _trend(tid)
        members.append(
            AppClusterMember(
                id=tid,
                label=labels.get(tid) or tid.split(":", 1)[-1],
                episode_count=episodes_per_member[tid],
                # Only a lift-ordered grouping has an anchor. A theme is symmetric — "means the same
                # thing" has no centre — so flagging one of its members would invent a hierarchy.
                anchor=bool(lift) and i == 0,
                first_seen=first,
                last_seen=last,
                trend=state,  # type: ignore[arg-type]  # _trend returns the Literal's values
            )
        )

    pair: AppClusterPair | None = None
    if lift and len(ordered) > 1:
        best = max(
            (
                (len(eps_by_member.get(a, set()) & eps_by_member.get(b, set())), a, b)
                for i_a, a in enumerate(ordered)
                for b in ordered[i_a + 1 :]
            ),
            default=(0, "", ""),
        )
        if best[0] > 0:
            pair = AppClusterPair(
                a_label=labels.get(best[1]) or best[1],
                b_label=labels.get(best[2]) or best[2],
                shared_episode_count=best[0],
            )

    resolved = label or cluster_id.split(":", 1)[-1].replace("-", " ")
    return resolved, members, about, related_people, pair


def build_theme_card(
    root: Path,
    theme_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppClusterCard | None:
    """A THEME (`tc:`) — topics that MEAN the same thing. See :func:`_build_cluster_card`."""
    built = _build_cluster_card(
        root,
        theme_id,
        group_map=theme_map_by_topic(root),
        id_key="cluster_id",
        label_key="cluster_label",
        # No lift, deliberately: a theme groups topics that MEAN the same thing, which is symmetric.
        # There is no "most central" member to anchor on and no co-occurrence claim to evidence.
        rows=rows,
        top_k=top_k,
    )
    if built is None:
        return None
    label, members, about, people, pair = built
    return AppClusterCard(
        id=theme_id,
        label=label,
        member_topics=members,
        episode_count=len(about),
        episodes=_sorted_episode_cards(root, about),
        related_people=people,
        strongest_pair=pair,
    )


def build_storyline_card(
    root: Path,
    storyline_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppClusterCard | None:
    """A STORYLINE (`thc:`) — topics that keep coming up TOGETHER.

    Accepts EITHER the storyline's own `thc:` id or one of its member topics' ids. The second form
    exists because ``/storyline/:id`` routes by ANCHOR TOPIC — there was no storyline endpoint when
    that page was built, so it derives everything from the anchor's card — and that route must keep
    working while now getting merged episodes.

    Same shape as a theme card: both are groupings, and a reader should not meet two different
    objects for what is, to them, the same kind of thing.
    """
    smap: ClusterMap = storyline_map_by_topic(root)
    resolved_id = storyline_id
    if not any(info.get("storyline_id") == storyline_id for info in smap.values()):
        # Not a cluster id — try it as a member/anchor topic id.
        info = smap.get(storyline_id) or {}
        candidate = info.get("storyline_id")
        if not isinstance(candidate, str) or not candidate:
            return None
        resolved_id = candidate

    built = _build_cluster_card(
        root,
        resolved_id,
        group_map=smap,
        id_key="storyline_id",
        label_key="storyline_label",
        lift=storyline_member_lift(root).get(resolved_id),
        rows=rows,
        top_k=top_k,
    )
    if built is None:
        return None
    label, members, about, people, pair = built
    return AppClusterCard(
        id=resolved_id,
        label=label,
        member_topics=members,
        episode_count=len(about),
        episodes=_sorted_episode_cards(root, about),
        related_people=people,
        strongest_pair=pair,
    )


def build_topic_card(
    root: Path,
    topic_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppTopicCard | None:
    """Project the topic's corpus footprint + cluster siblings to a card, or ``None`` if absent."""
    cluster_map: ClusterMap = theme_map_by_topic(root)
    storyline_map: ClusterMap = storyline_map_by_topic(root)

    label = ""
    about: list[CatalogEpisodeRow] = []
    people_by_id: dict[str, AppEntity] = {}
    person_counts: Counter[str] = Counter()

    for row, persons, topics in _topic_episodes(root, topic_id, rows):
        match = next((t for t in topics if t.id == topic_id), None)
        if match is None:
            continue
        if not label:
            label = match.label
        about.append(row)
        for p in persons:
            people_by_id[p.id] = p
            person_counts[p.id] += 1

    if not about:
        return None

    related_people = with_photos(
        [people_by_id[i] for i, _ in person_counts.most_common(top_k)], hosted_photo_urls(root)
    )
    info = cluster_map.get(topic_id) or {}
    cid, clabel, csize = info.get("cluster_id"), info.get("cluster_label"), info.get("cluster_size")
    tinfo = storyline_map.get(topic_id) or {}
    tcid, tclabel, tcsize = (
        tinfo.get("storyline_id"),
        tinfo.get("storyline_label"),
        tinfo.get("storyline_size"),
    )
    siblings = [
        _enrich_topic(AppTopic(id=s["id"], label=s["label"]), cluster_map, storyline_map)
        for s in theme_siblings_by_topic(root, topic_id)[:top_k]
    ]
    storyline_siblings = [
        _enrich_topic(AppTopic(id=s["id"], label=s["label"]), cluster_map, storyline_map)
        for s in storyline_siblings_by_topic(root, topic_id)[:top_k]
    ]
    return AppTopicCard(
        id=topic_id,
        label=label or topic_id.split(":", 1)[-1],
        cluster_id=cid if isinstance(cid, str) else None,
        cluster_label=clabel if isinstance(clabel, str) else None,
        cluster_size=csize if isinstance(csize, int) else 0,
        sibling_topics=siblings,
        storyline_id=tcid if isinstance(tcid, str) else None,
        storyline_label=tclabel if isinstance(tclabel, str) else None,
        storyline_size=tcsize if isinstance(tcsize, int) else 0,
        storyline_sibling_topics=storyline_siblings,
        episode_count=len(about),
        episodes=_sorted_episode_cards(root, about),
        related_people=related_people,
    )


def _cluster_member_topic_ids(
    root: Path, cluster_id: str, kind: Literal["theme", "storyline"]
) -> tuple[str, list[str]]:
    """``(label, member topic ids)`` for a grouping. ``("", [])`` when unknown.

    ``kind`` is REQUIRED, not sniffed from the id prefix, because the ids are not sufficient to
    disambiguate: ``/storyline/:id`` routes by ANCHOR TOPIC, so the storyline route legitimately
    passes a bare ``topic:`` id — and that same topic is usually a member of a theme as well.
    Trying the theme map first resolved ``topic:risk-management`` to the THEME "Show Themes" and
    served a storyline page the wrong grouping's speakers.

    The member-topic second form is kept for storylines for the same reason
    :func:`build_storyline_card` keeps it: that route has no other id to offer.
    """
    group_map, id_key, label_key = (
        (theme_map_by_topic(root), "cluster_id", "cluster_label")
        if kind == "theme"
        else (storyline_map_by_topic(root), "storyline_id", "storyline_label")
    )
    resolved = cluster_id
    if not any(info.get(id_key) == cluster_id for info in group_map.values()):
        candidate = (group_map.get(cluster_id) or {}).get(id_key)
        if not isinstance(candidate, str) or not candidate:
            return "", []
        resolved = candidate
    label = ""
    members: list[str] = []
    for tid, info in group_map.items():
        if info.get(id_key) != resolved:
            continue
        members.append(tid)
        if not label:
            raw = info.get(label_key)
            if isinstance(raw, str) and raw.strip():
                label = raw.strip()
    return label, sorted(members)


def build_cluster_perspectives(
    root: Path,
    cluster_id: str,
    kind: Literal["theme", "storyline"],
    *,
    mine_slugs: set[str] | None = None,
) -> AppTopicPerspectivesResponse | None:
    """What is SAID across a grouping — its members' insights, grouped by speaker.

    The grouping analogue of :func:`build_topic_perspectives`, and the answer to the question the
    pages could not previously answer: a theme and a storyline listed their member topics and their
    episodes, but nothing on either page was a sentence anybody actually said.

    Scoped to the UNION of the grouping's member topics, which is what the grouping IS. A speaker
    is counted once across the whole grouping, so someone who argues the same line under three
    members is one perspective holding three takes — not three perspectives.

    Returns ``None`` when the grouping has no speaker-attributable insight, which the route turns
    into a 404 and the client renders as absence. That is the honest outcome for a grouping whose
    members are abstract labels nobody says aloud, and it is common enough to be the normal case
    rather than an error.
    """
    label, members = _cluster_member_topic_ids(root, cluster_id, kind)
    if not members:
        return None
    keep: set[str] | None = None
    if mine_slugs is not None:
        keep = {
            r.episode_id
            for r in cached_catalog(root)
            if r.episode_id and row_to_summary(root, r).slug in mine_slugs
        }
    groups = topics_perspectives(str(root), str(root), members, keep_episode_ids=keep)
    if not groups:
        return None
    photos = hosted_photo_urls(root)
    slug_by_episode = {
        r.episode_id: row_to_summary(root, r).slug for r in cached_catalog(root) if r.episode_id
    }
    perspectives = [
        AppTopicPerspective(
            person_id=str(g["person_id"]),
            person_name=str(g["person_name"]),
            image_url=photos.get(str(g["person_id"])),
            insight_count=int(g["insight_count"]),
            episode_count=int(g["episode_count"]),
            insights=_rank_for_display(
                [_node_to_app_insight(n, slug_by_episode) for n in g["insights"]]
            ),
        )
        for g in groups
    ]
    # `topic_id` / `topic_label` carry the CLUSTER here. The response shape is shared with the topic
    # page on purpose — the client renders one component for all three surfaces, and inventing a
    # parallel schema would buy a second set of types for the same payload.
    return AppTopicPerspectivesResponse(
        topic_id=cluster_id,
        topic_label=label or cluster_id.split(":", 1)[-1],
        perspective_count=len(perspectives),
        perspectives=perspectives,
    )


def _org_web_payload(root: Path) -> dict[str, Any] | None:
    """The org_web payload ``{provider, orgs:[…]}``, unwrapping the enrichment envelope (#2035).

    Same envelope convention as person_web (payload under ``data``); tolerates an already-flat
    dict. Read live (small artifact; an enrichment run does not bump the corpus-mtime token)."""
    doc = load_json_artifact(root, "enrichments/org_web.json")
    if not isinstance(doc, dict):
        return None
    inner = doc.get("data")
    return inner if isinstance(inner, dict) else doc


def _org_web(root: Path, org_id: str) -> AppOrgWeb | None:
    """The org's external description + logo + attribution from ``enrichments/org_web.json``.

    Read-time projection: absent artifact / no matching row / nothing worth showing → None (the card
    stays lean, exactly as before the enricher ran). Best-effort — a malformed artifact never breaks
    the card. Only a HOSTED logo (served from our domain) is exposed, never the raw external URL."""
    doc = _org_web_payload(root)
    if doc is None:
        return None
    provider = str(doc.get("provider") or "")
    for row in doc.get("orgs") or []:
        if not isinstance(row, dict) or row.get("org_id") != org_id:
            continue
        description = row.get("description") if isinstance(row.get("description"), str) else None
        summary = row.get("summary") if isinstance(row.get("summary"), str) else None
        # A description/summary is the point; a row with neither carries nothing to show.
        if not (description and description.strip()) and not (summary and summary.strip()):
            return None
        logo_url = f"/api/app/organizations/{org_id}/logo" if row.get("logo_hosted") else None

        def _s(key: str) -> str | None:
            v = row.get(key)
            return v if isinstance(v, str) and v.strip() else None

        return AppOrgWeb(
            description=description.strip() if description else None,
            summary=summary.strip() if summary else None,
            source=str(row.get("source") or provider or "web"),
            source_url=_s("source_url"),
            logo_url=logo_url,
            logo_license=_s("logo_license"),
            founded=_s("founded"),
            industry=_s("industry"),
            website=_s("website"),
        )
    return None


def build_org_card(
    root: Path,
    org_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppOrgCard | None:
    """Project an org's corpus footprint to a (bio-less) card, or ``None`` if it appears nowhere.

    Mirrors ``build_person_card`` but keyed on MENTIONS_ORG, and carries a co-occurring-orgs list
    the person card has no analog for. No web enrichment — orgs have no bio/photo (#2031).
    """
    cluster_map: ClusterMap = theme_map_by_topic(root)
    storyline_map: ClusterMap = storyline_map_by_topic(root)

    label = ""
    appears_in: list[CatalogEpisodeRow] = []
    people_by_id: dict[str, AppEntity] = {}
    orgs_by_id: dict[str, AppEntity] = {}
    topics_by_id: dict[str, AppTopic] = {}
    person_counts: Counter[str] = Counter()
    org_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()

    for row, persons, orgs, topics in _org_episodes(root, org_id, rows):
        match = next((o for o in orgs if o.id == org_id), None)
        if match is None:
            continue
        if not label:
            label = match.name
        appears_in.append(row)
        for p in persons:
            people_by_id[p.id] = p
            person_counts[p.id] += 1
        for o in orgs:
            if o.id == org_id:
                continue
            orgs_by_id[o.id] = o
            org_counts[o.id] += 1
        for t in topics:
            topics_by_id[t.id] = t
            topic_counts[t.id] += 1

    if not appears_in:
        return None

    related_people = with_photos(
        [people_by_id[i] for i, _ in person_counts.most_common(top_k)], hosted_photo_urls(root)
    )
    related_orgs = [orgs_by_id[i] for i, _ in org_counts.most_common(top_k)]
    related_topics = [
        _enrich_topic(topics_by_id[i], cluster_map, storyline_map)
        for i, _ in topic_counts.most_common(top_k)
    ]
    return AppOrgCard(
        id=org_id,
        label=label or org_id.split(":", 1)[-1],
        episode_count=len(appears_in),
        episodes=_sorted_episode_cards(root, appears_in),
        related_people=related_people,
        related_orgs=related_orgs,
        related_topics=related_topics,
        web=_org_web(root, org_id),
    )


def _node_to_app_insight(
    node: dict[str, Any], slug_by_episode: dict[str, str] | None = None
) -> AppInsight:
    """Project a GI Insight node to AppInsight (grounded; quotes omitted here).

    ``slug_by_episode`` maps episode id -> slug so a perspective insight carries its source
    episode + the supporting quote's start moment (#2032); both are annotated onto the node by
    ``topic_perspectives`` (``_episode_id`` / ``_quote_start_ms``) and are absent elsewhere.
    """
    props = node.get("properties") or {}
    text = props.get("text") or props.get("title") or ""
    conf = props.get("confidence")
    itype = props.get("insight_type") or props.get("type")
    phint = props.get("position_hint")
    sal = props.get("salience")
    rnk = props.get("rank")
    rtag = props.get("routing_tag")
    tier = props.get("tier")
    ep_id = node.get("_episode_id")
    start_ms = node.get("_quote_start_ms")
    episode_slug = (slug_by_episode or {}).get(str(ep_id)) if ep_id else None
    return AppInsight(
        id=str(node.get("id") or ""),
        text=str(text),
        grounded=True,
        insight_type=str(itype) if isinstance(itype, str) and itype.strip() else None,
        confidence=float(conf) if isinstance(conf, (int, float)) else None,
        position_hint=str(phint) if phint is not None else None,
        salience=float(sal) if isinstance(sal, (int, float)) else None,
        rank=int(rnk) if isinstance(rnk, int) else None,
        routing_tag=str(rtag) if isinstance(rtag, str) and rtag.strip() else None,
        tier=int(tier) if isinstance(tier, int) else None,
        episode_slug=episode_slug,
        start_ms=start_ms if isinstance(start_ms, int) else None,
        quotes=[],
    )


def _rank_for_display(insights: list[AppInsight]) -> list[AppInsight]:
    """ADR-135/#1191: drop `drop`-tagged, sort by salience desc (stable for ties / pre-3.1)."""
    kept = [i for i in insights if i.routing_tag != "drop"]
    kept.sort(key=lambda i: i.salience if i.salience is not None else 0.0, reverse=True)
    return kept


def build_topic_perspectives(
    root: Path, topic_id: str, *, mine_slugs: set[str] | None = None
) -> AppTopicPerspectivesResponse | None:
    """Group a topic's grounded insights by speaker — one take per speaker (#1146).

    ``mine_slugs`` (scope=mine, #1149) restricts to episodes in the user's heard∪captured
    set; an empty set yields no perspectives (honest-empty). Returns ``None`` when the topic
    has no speaker-attributable insight in the (scoped) GI.
    """
    keep: set[str] | None = None
    if mine_slugs is not None:
        rows = cached_catalog(root)
        keep = {
            r.episode_id
            for r in rows
            if r.episode_id and row_to_summary(root, r).slug in mine_slugs
        }
    groups = topic_perspectives(str(root), str(root), topic_id, keep_episode_ids=keep)
    if not groups:
        return None
    photos = hosted_photo_urls(root)
    # episode id -> slug, so each perspective insight can carry a jump-to-moment link (#2032).
    slug_by_episode = {
        r.episode_id: row_to_summary(root, r).slug for r in cached_catalog(root) if r.episode_id
    }
    perspectives = [
        AppTopicPerspective(
            person_id=str(g["person_id"]),
            person_name=str(g["person_name"]),
            image_url=photos.get(str(g["person_id"])),
            insight_count=int(g["insight_count"]),
            episode_count=int(g["episode_count"]),
            insights=_rank_for_display(
                [_node_to_app_insight(n, slug_by_episode) for n in g["insights"]]
            ),
        )
        for g in groups
    ]
    return AppTopicPerspectivesResponse(
        topic_id=topic_id,
        topic_label=topic_id.split(":", 1)[-1],
        perspective_count=len(perspectives),
        perspectives=perspectives,
    )
