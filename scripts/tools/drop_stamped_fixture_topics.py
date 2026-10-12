#!/usr/bin/env python3
"""Remove the STAMPED topics from the committed app-validation corpus (operator 2026-10-12).

The corpus builder gave every episode two "shared umbrella" topics (``lifelong learning``,
``expert interviews``) and every show two "cross-cutting" ones (``risk management`` on 12 of 14
shows). Nothing in an episode said them: they existed so test themes had overlap. They made the
app's count-ranked surfaces — Home's week, stats, "Also about", the show page, the interests
picker — lead with topics on 100% of episodes, which real data never has (prod, read-only
2026-10-12: the most common topic is on 1.0% of 3,528 episodes).

The NATURAL corpus keeps what the episode itself carries: its ground-truth topics (authored per
episode) and the topics a real pipeline run extracted from its transcript. Measured over the 55
episodes: 57 topics still shared by two or more shows, the most common (risk management) on 56%.

A topic is STAMPED for an episode when it came only from the show's cross-cutting list or the
shared umbrellas — neither the episode's ground truth nor its extraction has it. For each episode
this removes those topics from the KG (node + MENTIONS), the GI (node + every edge to it) and the
bridge identities, then re-authors the four corpus enrichments the builder writes from the
topics left on disk. The framework-written ones are re-run afterwards (see the fixture README).

Usage::

    .venv/bin/python scripts/tools/drop_stamped_fixture_topics.py          # dry run, reports
    .venv/bin/python scripts/tools/drop_stamped_fixture_topics.py --apply  # rewrites the files

Idempotent: a second ``--apply`` finds nothing to remove.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import build_app_validation_corpus as builder  # noqa: E402

#: What the builder used to stamp; kept here because the builder no longer defines them.
SHARED_UMBRELLAS = ["lifelong learning", "expert interviews"]
CROSS_CUTTING_TOPICS: dict[str, list[str]] = {
    "p05": ["personal finance", "risk management"],
    "p02": ["systems thinking", "risk management"],
    "p03": ["safety practices", "risk management"],
    "p04": ["visual craft", "systems thinking"],
    "p01": ["endurance sport", "risk management"],
    "p07": ["systems thinking", "risk management"],
    "p06": ["long-form", "systems thinking"],
    "p08": ["public radio", "risk management"],
    "p09": ["systems thinking", "risk management"],
    "p10": ["endurance sport", "risk management"],
    "p11": ["endurance sport", "risk management"],
    "p12": ["endurance sport", "risk management"],
    "p13": ["endurance sport", "risk management"],
    "p14": ["endurance sport", "risk management"],
}


def _tid(label: str) -> str:
    return label if label.startswith("topic:") else f"topic:{builder.slug(label)}"


def stamped_for(show: str, ep_label: str, gt_dir: Path, version: str) -> set[str]:
    """Topic ids the builder stamped on this episode that the episode itself does not carry."""
    own = {_tid(t) for t in builder.episode_topics_for(ep_label, gt_dir)}
    own |= {_tid(t) for t in builder._captured_topics_for(ep_label, version)}
    stamped = {_tid(t) for t in CROSS_CUTTING_TOPICS.get(show, []) + SHARED_UMBRELLAS}
    return stamped - own


def _drop_from_graph(doc: dict, drop: set[str]) -> int:
    nodes = doc.get("nodes") or []
    edges = doc.get("edges") or []
    kept_nodes = [n for n in nodes if n.get("id") not in drop]
    kept_edges = [e for e in edges if e.get("from") not in drop and e.get("to") not in drop]
    removed = (len(nodes) - len(kept_nodes)) + (len(edges) - len(kept_edges))
    doc["nodes"], doc["edges"] = kept_nodes, kept_edges
    return removed


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=REPO / "tests/fixtures/app-validation-corpus/v3")
    p.add_argument("--apply", action="store_true", help="rewrite the files (default: report)")
    args = p.parse_args()

    version = (REPO / "tests/fixtures/FIXTURES_VERSION").read_text("utf-8").strip()
    gt_dir = REPO / "tests/fixtures/ground-truth" / version / "ground_truth"
    per_topic: collections.Counter[str] = collections.Counter()
    episodes = 0
    for kg_path in sorted(args.corpus.glob("feeds/*/run_*/metadata/*.kg.json")):
        show = kg_path.relative_to(args.corpus).parts[1]
        ep_label = kg_path.name[: -len(".kg.json")]
        drop = stamped_for(show, ep_label, gt_dir, version)
        if not drop:
            continue
        changed = False
        for suffix in ("kg", "gi"):
            path = kg_path.with_name(f"{ep_label}.{suffix}.json")
            doc = json.loads(path.read_text("utf-8"))
            if _drop_from_graph(doc, drop):
                changed = True
                if args.apply:
                    _write(path, doc)
        bridge_path = kg_path.with_name(f"{ep_label}.bridge.json")
        bridge = json.loads(bridge_path.read_text("utf-8"))
        ids = bridge.get("identities") or []
        kept = [i for i in ids if i.get("id") not in drop]
        if len(kept) != len(ids):
            changed = True
            bridge["identities"] = kept
            if args.apply:
                _write(bridge_path, bridge)
        if changed:
            episodes += 1
            per_topic.update(drop)
    verb = "removed" if args.apply else "would remove"
    print(f"{verb} stamped topics from {episodes} episodes:")
    for tid, n in per_topic.most_common():
        print(f"  {n:3d}  {tid}")
    _regenerate_authored_enrichments(args.corpus, args.apply)
    return 0


def _regenerate_authored_enrichments(corpus: Path, apply: bool) -> None:
    """Re-author the four corpus enrichments the BUILDER writes, from the topics on disk.

    ``temporal_velocity``, ``topic_theme_clusters``, ``topic_similarity`` and ``topic_consensus``
    are authored by ``build_app_validation_corpus.py`` (deterministic, ``computed_at`` pinned),
    not by the enrichment framework — the framework's ``temporal_velocity`` embeds a wall-clock
    ``now``. This replays the builder's own accumulators and authoring functions over every
    episode's KG topics, in the builder's show and episode order, so the files are what a rebuild
    of the natural corpus writes. On a dry run it reports which files would change.
    """
    months: dict[str, list[str]] = {}
    episodes: dict[str, list[str]] = {}
    weeks: dict[str, list[str]] = {}
    person_weeks: dict[str, list[str]] = {}
    persons: dict[str, list[tuple[str, str]]] = {}
    counts: dict[str, int] = {}
    for _rss, show in builder.APP_SHOWS:
        for kg_path in sorted(corpus.glob(f"feeds/{show}/run_*/metadata/*.kg.json")):
            kg = json.loads(kg_path.read_text("utf-8"))
            meta_path = kg_path.with_name(kg_path.name.replace(".kg.json", ".metadata.json"))
            episode = json.loads(meta_path.read_text("utf-8"))["episode"]
            publish = str(episode["published_date"])[:10]
            tids = [str(n["id"]) for n in kg.get("nodes", []) if n.get("type") == "Topic"]
            builder._tally_episode_content(
                kg,
                tids,
                builder._iso_week(publish),
                publish[:7],
                str(episode["episode_id"]),
                counts=counts,
                months=months,
                episodes=episodes,
                weeks=weeks,
                person_weeks=person_weeks,
            )
            builder._accumulate_topic_persons(
                builder._episode_persons(kg), tids, topic_persons=persons
            )
    velocity = builder._temporal_velocity_data(months)
    velocity["content_series"] = builder._content_series_data(weeks, person_weeks)
    authored = {
        "temporal_velocity": (velocity, "1.0"),
        "topic_theme_clusters": (builder._theme_clusters_data(episodes), "1.1.0"),
        "topic_consensus": (builder._topic_consensus_data(persons), "1.0"),
        "topic_similarity": (builder._topic_similarity_data(episodes), "1.0"),
    }
    for enricher_id, (data, version) in authored.items():
        path = corpus / "enrichments" / f"{enricher_id}.json"
        env = builder._enrichment_envelope(enricher_id, data, version=version)
        text = json.dumps(env, indent=2, sort_keys=True) + "\n"
        changed = path.read_text("utf-8") != text
        state = "unchanged" if not changed else "rewritten" if apply else "would change"
        print(f"{enricher_id}: {state}")
        if apply and changed:
            path.write_text(text, "utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
