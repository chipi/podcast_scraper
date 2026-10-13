"""Umbrella subjects: on so much of the corpus they say nothing about any one thing (#1669).

A topic on nearly every episode — the "lifelong learning" a corpus generator once stamped on all
of them, or a real corpus's equivalent — says nothing about what any episode, show or week is
about, yet it wins every ranking by count: Home's week, a show's topics, "Also about", the
interests picker. The same holds for a theme or storyline whose member topics together reach
nearly every episode. These are listed once here, per corpus version, and every list of what
something is about leaves them out (operator 2026-10-12).

One threshold for the whole app: ``UMBRELLA_TOPIC_SHARE``, the one "Mine" and short listens
already use, so the three cannot disagree about what an umbrella is. Real data has none (prod,
read 2026-10-12: the most common topic is on 1.0% of 3,528 episodes, the widest theme 2.6%), so
this is a guard, not a filter that changes what listeners see today.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from podcast_scraper import perf_cache
from podcast_scraper.server.app_catalog_cache import cached_catalog

#: A followed interest, or a listed subject, reaching at least this share of episodes is an
#: umbrella. Re-exported by ``app_user_corpus`` for "Mine" (#1669).
UMBRELLA_TOPIC_SHARE = 0.9
#: Below this many episodes a share says too little: in a 5-episode corpus a topic on all 5 is
#: what the corpus is about, not an umbrella.
UMBRELLA_MIN_EPISODES = 20

_CORPUS_EPISODE_TOPICS_NS = "corpus_episode_topics"
_UMBRELLA_NS = "umbrella_tokens"


def corpus_episode_topics(root: Path) -> list[frozenset[str]]:
    """Every corpus episode's topic ids, one set per episode with a KG.

    The reach of a THEME or STORYLINE — episodes touching ANY of its topics — is a union, so it
    cannot be summed from per-topic counts. One pass over the episode KGs, cached per corpus
    version like the catalog. Shared by the show page's distinctiveness and the umbrella guard.
    """
    from podcast_scraper.enrichment.enrichers._loaders import nodes_of_type
    from podcast_scraper.server.feed_signals import _read_kg_artifact

    def build() -> list[frozenset[str]]:
        out: list[frozenset[str]] = []
        for r in cached_catalog(root):
            if not r.has_kg or not r.kg_relative_path:
                continue
            art = _read_kg_artifact(str(root), r.kg_relative_path)
            if art is None:
                continue
            out.append(
                frozenset(str(n.get("id")) for n in nodes_of_type(art, "Topic") if n.get("id"))
            )
        return out

    result: list[frozenset[str]] = perf_cache.get_or_compute(
        _CORPUS_EPISODE_TOPICS_NS, str(root), perf_cache.corpus_mtime(root), build
    )
    return result


def umbrella_tokens(root: Path) -> frozenset[str]:
    """The topics, themes (``tc:``) and storylines (``thc:``) on at least ``UMBRELLA_TOPIC_SHARE``
    of the corpus's episodes; empty below ``UMBRELLA_MIN_EPISODES``. A group's reach is the union
    of its members' episodes. Never raises: a guard that fails must not take a page down."""

    def build() -> frozenset[str]:
        from podcast_scraper.search.storylines import storyline_map_by_topic
        from podcast_scraper.search.topic_clusters import theme_map_by_topic

        episodes = corpus_episode_topics(root)
        if len(episodes) < UMBRELLA_MIN_EPISODES:
            return frozenset()
        ceiling = UMBRELLA_TOPIC_SHARE * len(episodes)
        df: Counter[str] = Counter()
        for topics in episodes:
            df.update(topics)
        out = {t for t, n in df.items() if n >= ceiling}
        members: dict[str, set[str]] = {}
        for tid, info in theme_map_by_topic(root).items():
            members.setdefault(str(info.get("cluster_id") or ""), set()).add(tid)
        for tid, info in storyline_map_by_topic(root).items():
            members.setdefault(str(info.get("storyline_id") or ""), set()).add(tid)
        for group, member_ids in members.items():
            if group and sum(1 for ep in episodes if ep & member_ids) >= ceiling:
                out.add(group)
        return frozenset(out)

    try:
        result: frozenset[str] = perf_cache.get_or_compute(
            _UMBRELLA_NS, str(root), perf_cache.corpus_mtime(root), build
        )
        return result
    except Exception:  # noqa: BLE001 — a guard must not take a page down
        return frozenset()
