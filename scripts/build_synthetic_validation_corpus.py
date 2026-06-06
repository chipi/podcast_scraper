#!/usr/bin/env python3
"""Build a synthetic Tier-3 validation corpus from the text fixtures.

GH #774 (RFC-086 Tier-3 CI automation). Generates a self-contained,
schema-valid corpus from ``tests/fixtures/rss/*.xml`` + matching
``tests/fixtures/transcripts/*.txt``. Output is checked into the repo
so CI / scheduled GHA can run Tier-3 validation against a stable,
version-pinned corpus without external dependencies or operator setup.

This corpus is **synthetic, low-fidelity**. The text fixtures have low
cross-episode topic intermingling, so this catches:
  - Schema regressions (endpoints return wrong shape)
  - Empty-state bugs (digest/dashboard with sparse data)
  - Per-episode handoff contract (cold-start Library, Episode panel)
It does NOT catch:
  - Cross-episode topic-cluster supersession bugs (need richer mock data)
  - KG-second-wave timing at production scale
  - Real-vector-index search

Usage:
    python scripts/build_synthetic_validation_corpus.py \\
        [--rss-dir tests/fixtures/rss] \\
        [--transcripts-dir tests/fixtures/transcripts] \\
        [--output tests/fixtures/viewer-validation-corpus] \\
        [--max-feeds 5] [--max-episodes-per-feed 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Use defusedxml to satisfy bandit B314 (untrusted-XML risk). Our input is
# the checked-in RSS test fixtures (low real-world risk), but defusedxml
# is the standards-compliant safe parser and is already a project dep.
import defusedxml.ElementTree as ET


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len] or "x"


# Cross-cutting umbrella topics injected into each podcast's episodes so
# the synthetic corpus has enough topic overlap for V2 (Digest pill) +
# V4 (Dashboard cluster) validation. Maps ``p01 → [topic, ...]``.
# Designed so each umbrella spans 2+ podcasts:
#   technology     → p02, p04, p08, p09   (4 podcasts)
#   outdoor        → p01, p03             (2 podcasts)
#   gear           → p01, p03, p04        (3 podcasts)
#   environment    → p07, p08             (2 podcasts)
#   health         → p07, p09             (2 podcasts)
CROSS_CUTTING_TOPICS: dict[str, list[str]] = {
    "p01": ["outdoor activities", "gear"],
    "p02": ["technology"],
    "p03": ["outdoor activities", "gear"],
    "p04": ["technology", "gear"],
    "p05": [],  # standalone — "investing" is its own niche
    "p06": [],  # edge cases — no umbrella
    "p07": ["environment", "health"],
    "p08": ["environment", "technology"],
    "p09": ["technology", "health"],
}

# Digest endpoint defaults (``src/podcast_scraper/server/corpus_digest.py``
# DEFAULT_DIGEST_TOPICS). When a user clicks a digest topic pill the viewer
# builds a graph handoff envelope targeting ``topic:<slug>`` (e.g.
# ``topic:science-research``). For the FSM apply step to succeed, a kg_topic
# node with that exact id MUST exist somewhere in the loaded graph. Without
# these, V2 (digest pill) fails with "no cy node found for envelope target".
# Each label is added to one episode of every feed so the topics span the
# corpus (matching real digest hit distribution).
DIGEST_HEADLINE_TOPICS: list[str] = [
    "Science & research",
    "Technology",
    "Business & markets",
]


def stable_feed_id(rss_basename: str) -> str:
    return "sha256:" + hashlib.sha256(rss_basename.encode("utf-8")).hexdigest()


def parse_rss_feed_metadata(rss_path: Path) -> dict[str, Any]:
    """Extract feed-level metadata from an RSS XML fixture."""
    try:
        tree = ET.parse(rss_path)
        root = tree.getroot()
        channel = root.find("channel") or root
        title = (channel.findtext("title") or rss_path.stem).strip()
        description = (channel.findtext("description") or "").strip()
        # RSS link can be in <link> or be the file URL itself.
        link = (channel.findtext("link") or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: failed to parse {rss_path.name}: {exc}")
        title = rss_path.stem
        description = ""
        link = ""
    return {
        "feed_id": stable_feed_id(rss_path.name),
        "display_title": title,
        "description": description,
        "rss_url": link or f"https://example.invalid/{rss_path.stem}",
    }


def read_transcript_excerpts(
    transcript_path: Path, n_topics: int = 3, n_insights: int = 3, n_quotes: int = 3
) -> dict[str, list[str]]:
    """Pull rough excerpts from a transcript to seed synthetic GI/KG content.

    Returns ``{topics, insights, quotes}`` lists. Topics are short phrases
    (3-5 words); insights are full sentences; quotes are full sentences
    with speaker prefix stripped.
    """
    try:
        text = transcript_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  warn: failed to read {transcript_path.name}: {exc}")
        return {"topics": [], "insights": [], "quotes": []}

    # Strip stage directions and speaker labels for cleaner content.
    cleaned = re.sub(r"\*[^*]*\*", "", text)
    cleaned = re.sub(r"^\s*\[?\d{1,2}:\d{2}\]?\s*", "", cleaned, flags=re.MULTILINE)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if 30 <= len(s) <= 200]
    # Strip speaker prefixes like "Maya:" from each sentence start.
    sentences = [re.sub(r"^[A-Z][a-z]+:\s*", "", s) for s in sentences]

    # Topic phrases — pick capitalised noun-ish phrases.
    cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[a-z]+){1,3})\b", cleaned)
    topic_phrases: list[str] = []
    seen: set[str] = set()
    for p in cap_phrases:
        p = p.strip().lower()
        if len(p) < 10 or p in seen:
            continue
        seen.add(p)
        topic_phrases.append(p)
        if len(topic_phrases) >= n_topics:
            break
    # Fallback: derive from first few sentences if not enough capitalised phrases.
    while len(topic_phrases) < n_topics and sentences:
        s = sentences[len(topic_phrases) % len(sentences)]
        words = s.split()[:4]
        candidate = " ".join(words).lower().rstrip(".!?:,")
        if candidate and candidate not in seen:
            seen.add(candidate)
            topic_phrases.append(candidate)
        else:
            break

    insights = sentences[:n_insights]
    quotes = sentences[n_insights : n_insights + n_quotes]
    return {
        "topics": topic_phrases[:n_topics],
        "insights": insights,
        "quotes": quotes,
    }


def short_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def build_gi(
    episode_id: str,
    podcast_id: str,
    title: str,
    publish_date: str,
    excerpts: dict[str, list[str]],
    metadata_relative_path: str | None = None,
) -> dict[str, Any]:
    """Build a minimal GI artifact matching the pipeline schema."""
    ep_node_id = f"episode:{episode_id}"
    ep_props: dict[str, Any] = {
        "podcast_id": podcast_id,
        "title": title,
        "publish_date": publish_date,
        "duration_ms": 1800000,
        # Critical for episode resolution: the viewer's
        # ``findEpisodeGraphNodeIdForMetadataPath`` matches Episode
        # nodes by these fields. Without them, no Library row click
        # can resolve to a cy node (handoffFailed fires + "Could not
        # open episode" error strip).
        "episode_id": episode_id,
    }
    if metadata_relative_path:
        ep_props["metadata_relative_path"] = metadata_relative_path
    nodes: list[dict[str, Any]] = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": ep_props,
        }
    ]
    edges: list[dict[str, Any]] = []
    for label in excerpts["topics"]:
        tid = f"topic:{slug(label)}"
        nodes.append(
            {
                "id": tid,
                "type": "Topic",
                "properties": {"label": label, "score": 0.8},
            }
        )
        edges.append({"type": "MENTIONS", "from": ep_node_id, "to": tid})
    for txt in excerpts["insights"]:
        iid = f"insight:{short_hash(txt)}"
        nodes.append(
            {
                "id": iid,
                "type": "Insight",
                "properties": {"text": txt, "confidence": 0.7, "grounded": True},
            }
        )
        edges.append({"type": "HAS_INSIGHT", "from": ep_node_id, "to": iid})
    for txt in excerpts["quotes"]:
        qid = f"quote:{short_hash(txt)}"
        nodes.append(
            {
                "id": qid,
                "type": "Quote",
                "properties": {"text": txt, "speaker": "Speaker 1"},
            }
        )
        edges.append({"type": "HAS_QUOTE", "from": ep_node_id, "to": qid})
    return {
        "schema_version": "2",
        "model_version": "synthetic-validation-corpus-v1",
        "prompt_version": "n/a",
        "episode_id": episode_id,
        "nodes": nodes,
        "edges": edges,
    }


def build_kg(
    episode_id: str,
    title: str,
    excerpts: dict[str, list[str]],
    metadata_relative_path: str | None = None,
) -> dict[str, Any]:
    """Build a minimal KG artifact."""
    ep_node_id = f"episode:{episode_id}"
    ep_props: dict[str, Any] = {"title": title, "episode_id": episode_id}
    if metadata_relative_path:
        ep_props["metadata_relative_path"] = metadata_relative_path
    nodes: list[dict[str, Any]] = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": ep_props,
        }
    ]
    edges: list[dict[str, Any]] = []
    # Synthesize a couple of Entity nodes from topic phrases.
    for i, label in enumerate(excerpts["topics"][:2]):
        eid = f"entity:{slug(label)}"
        nodes.append(
            {
                "id": eid,
                "type": "Entity",
                "properties": {"label": label.title(), "category": "Concept"},
            }
        )
        edges.append({"type": "MENTIONS", "from": ep_node_id, "to": eid})
    # Carry every topic over as a KG Topic node for cross-link parity with
    # GI. Single-topic-only KGs were insufficient: V2 (digest pill click)
    # requires the kg_topic node matching ``topic:<digest-headline-slug>``
    # to exist on at least one loaded episode for the FSM apply to find
    # it. Emitting all topics also makes V4 (topic-cluster) richer and is
    # what production KGs do.
    seen_tids: set[str] = set()
    for label in excerpts["topics"]:
        tid = f"topic:{slug(label)}"
        if tid in seen_tids:
            continue
        seen_tids.add(tid)
        nodes.append({"id": tid, "type": "Topic", "properties": {"label": label}})
        edges.append({"type": "RELATES_TO", "from": ep_node_id, "to": tid})
    return {
        "schema_version": "2",
        "episode_id": episode_id,
        "extraction": {"provider": "synthetic", "model": "n/a"},
        "nodes": nodes,
        "edges": edges,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rss-dir", type=Path, default=Path("tests/fixtures/rss"))
    p.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("tests/fixtures/transcripts")
        / Path("tests/fixtures/FIXTURES_VERSION").read_text(encoding="utf-8").strip(),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/viewer-validation-corpus"),
    )
    p.add_argument("--max-feeds", type=int, default=9)
    p.add_argument("--max-episodes-per-feed", type=int, default=3)
    args = p.parse_args()

    if not args.rss_dir.is_dir():
        sys.exit(f"--rss-dir does not exist: {args.rss_dir}")
    if not args.transcripts_dir.is_dir():
        sys.exit(f"--transcripts-dir does not exist: {args.transcripts_dir}")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    # Pick the canonical RSS per podcast (p01, p02, ..., p09), skipping
    # test-only variants (``_fast``, ``_multi``, ``_selection``,
    # ``_with_transcript``, ``_episode_selection``, ``_fast_with_transcript``).
    # The canonical file is the one with a topic suffix:
    #   p01_mtb.xml, p02_software.xml, p03_scuba.xml, p04_photo.xml,
    #   p05_investing.xml, p06_edge_cases.xml, p07_sustainability.xml,
    #   p08_solar.xml, p09_biohacking.xml
    # All others are variants for pipeline-edge testing, not for
    # representing distinct podcasts here.
    variant_markers = {
        "fast",
        "multi",
        "selection",
        "with_transcript",
        "episode_selection",
        "fast_with_transcript",
    }
    all_rss = sorted(args.rss_dir.glob("p*.xml"))
    by_podcast: dict[str, Path] = {}
    for f in all_rss:
        # Parse "pNN_<suffix>.xml"
        m = re.match(r"^(p\d+)_(.+)\.xml$", f.name)
        if not m:
            continue
        podcast_key = m.group(1)
        suffix = m.group(2)
        if suffix in variant_markers or any(suffix.startswith(v + "_") for v in variant_markers):
            continue
        # First canonical file wins; deterministic via sorted iteration.
        if podcast_key not in by_podcast:
            by_podcast[podcast_key] = f
    rss_files = [by_podcast[k] for k in sorted(by_podcast)][: args.max_feeds]
    print(f"feeds: {len(rss_files)}/{len(by_podcast)} picked (max={args.max_feeds})")

    feeds: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    episode_details: dict[str, dict[str, Any]] = {}
    artifact_index: list[dict[str, Any]] = []
    # All episodes within the default 7-day graph lens so every episode
    # shows in the default Library / Graph view (no manual lens widening
    # needed for Tier-3 validation). Spread across last 7 days.
    base_date = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)

    for fi, rss_path in enumerate(rss_files):
        feed_meta = parse_rss_feed_metadata(rss_path)
        feed_prefix = rss_path.stem.split("_")[0]  # e.g. p01

        # Find transcripts named pNN_eM.txt for this feed.
        transcripts = sorted(args.transcripts_dir.glob(f"{feed_prefix}_e[0-9]*.txt"))
        transcripts = [t for t in transcripts if "_multi_" not in t.stem and "_fast" not in t.stem]
        transcripts = transcripts[: args.max_episodes_per_feed]

        feed_episode_count = 0
        for ei, transcript_path in enumerate(transcripts):
            ep_label = transcript_path.stem  # e.g. p01_e01
            ep_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, ep_label))
            # Spread within last 6 days so default 7-day lens captures
            # everything. Cycle: feed 0 ep 0 = today, feed 0 ep 1 = -1
            # day, ..., feed 8 ep 2 = -6 days mod 7. Determinism via
            # modulo + feed/ep indices.
            days_back = (fi + ei * 2) % 6
            publish = (base_date - timedelta(days=days_back)).strftime("%Y-%m-%d")
            title = f"{feed_meta['display_title']} — Episode {ei + 1}"
            podcast_id = feed_meta["feed_id"]
            metadata_rel = f"feeds/{feed_prefix}/metadata/{ep_label}.metadata.json"
            gi_rel = f"feeds/{feed_prefix}/metadata/{ep_label}.gi.json"
            kg_rel = f"feeds/{feed_prefix}/metadata/{ep_label}.kg.json"

            excerpts = read_transcript_excerpts(transcript_path)
            # #774a — inject cross-cutting umbrella topics so multiple
            # podcasts share topic ids. Without these, topic-bands and
            # topic-clusters are all singletons (no cross-episode hits)
            # and V2 / V4 validation rows fail. Umbrella topics go FIRST
            # in the list so they appear in summary_bullet_graph_topic_ids
            # (the digest topic-band hits are derived from these).
            umbrellas = CROSS_CUTTING_TOPICS.get(feed_prefix, [])
            # Inject one rotating digest-headline topic per episode so that
            # at least one episode in each feed carries each headline label.
            # The cycle (ei % len) gives every episode index a different
            # headline; across 23 episodes every headline appears 7-8x.
            headline_topic = DIGEST_HEADLINE_TOPICS[ei % len(DIGEST_HEADLINE_TOPICS)]
            excerpts["topics"] = (
                umbrellas + [headline_topic] + excerpts["topics"][: max(0, 3 - len(umbrellas) - 1)]
            )
            gi = build_gi(
                ep_uuid,
                podcast_id,
                title,
                publish + "T12:00:00",
                excerpts,
                metadata_relative_path=metadata_rel,
            )
            kg = build_kg(ep_uuid, title, excerpts, metadata_relative_path=metadata_rel)

            # Lay artifacts out at the paths referenced by ``episodes.json``
            # so the real backend's ``/api/artifacts/<path>`` handler can
            # serve them from disk. Also write a minimal metadata.json so
            # the corpus-detail endpoint can resolve the episode.
            gi_full = out / gi_rel
            kg_full = out / kg_rel
            metadata_full = out / metadata_rel
            bridge_rel = f"feeds/{feed_prefix}/metadata/{ep_label}.bridge.json"
            bridge_full = out / bridge_rel
            gi_full.parent.mkdir(parents=True, exist_ok=True)
            gi_full.write_text(json.dumps(gi, indent=2, sort_keys=True) + "\n")
            kg_full.write_text(json.dumps(kg, indent=2, sort_keys=True) + "\n")
            # bridge.json — required by
            # ``build_cil_digest_topics_for_row`` in the API server. Without
            # one, ``cil_digest_topics`` is empty in the digest response,
            # so the per-row CIL topic pills don't render and V2 has no
            # clickable handoff target. Identities mirror the episode's
            # topic list (umbrellas + headline + transcript-derived).
            bridge_full.write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "episode_id": ep_uuid,
                        "identities": [
                            {
                                "id": f"topic:{slug(t)}",
                                "display_name": t,
                            }
                            for t in excerpts["topics"]
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            # #774a — rich metadata so the API's dynamic digest /
            # corpus-library computations have ``cil_digest_topics``,
            # ``summary_*`` etc. to work with. Minimal versions caused
            # the digest endpoint to return empty bands.
            ep_topics_for_metadata = excerpts["topics"][:3]
            ep_summary_for_metadata = (
                excerpts["insights"][0]
                if excerpts["insights"]
                else f"Synthesised summary for {ep_label}."
            )
            ep_bullets_for_metadata = excerpts["insights"][:3] or [
                f"Bullet {n + 1}" for n in range(3)
            ]
            # Production-shaped nested schema. ``corpus_catalog`` +
            # ``indexer`` both read ``doc["episode"]["episode_id"]`` /
            # ``doc["feed"]["feed_id"]`` / ``doc["grounded_insights"]
            # ["artifact_path"]`` / ``doc["knowledge_graph"]["artifact_path"]``.
            # Flat top-level fields (``summary_*``, ``cil_digest_topics``)
            # remain for any consumer that reads them directly.
            metadata_full.write_text(
                json.dumps(
                    {
                        "schema_version": "1",
                        "episode": {
                            "episode_id": ep_uuid,
                            "title": title,
                            "published_date": publish,
                        },
                        "feed": {
                            "feed_id": podcast_id,
                            "title": feed_meta["display_title"],
                            "url": feed_meta["rss_url"],
                            "description": feed_meta["description"],
                        },
                        "grounded_insights": {
                            "artifact_path": gi_rel,
                            "schema_version": "1.0",
                        },
                        "knowledge_graph": {
                            "artifact_path": kg_rel,
                            "schema_version": "1.1",
                        },
                        "summary": {
                            "title": f"{title} — synthetic",
                            "bullets": ep_bullets_for_metadata,
                            "text": "\n".join(ep_bullets_for_metadata),
                            "preview": (
                                ep_summary_for_metadata[:140] + "…"
                                if len(ep_summary_for_metadata) > 140
                                else ep_summary_for_metadata
                            ),
                        },
                        # Flat compatibility fields (read directly by older
                        # callers and digest topic-band derivation):
                        "episode_id": ep_uuid,
                        "episode_title": title,
                        "publish_date": publish,
                        "feed_id": podcast_id,
                        "feed_display_title": feed_meta["display_title"],
                        "summary_title": f"{title} — synthetic",
                        "summary_bullets": ep_bullets_for_metadata,
                        "summary_text": "\n".join(ep_bullets_for_metadata),
                        "summary_preview": (
                            ep_summary_for_metadata[:140] + "…"
                            if len(ep_summary_for_metadata) > 140
                            else ep_summary_for_metadata
                        ),
                        "topics": ep_topics_for_metadata,
                        "summary_bullet_graph_topic_ids": [
                            f"topic:{slug(t)}" for t in ep_topics_for_metadata
                        ],
                        "cil_digest_topics": [
                            {
                                "topic_id": f"topic:{slug(t)}",
                                "label": t,
                                "in_topic_cluster": t in umbrellas,
                                "topic_cluster_compound_id": (
                                    f"tc:{slug(t)}-cluster" if t in umbrellas else None
                                ),
                            }
                            for t in ep_topics_for_metadata[:2]
                        ],
                        "gi_relative_path": gi_rel,
                        "kg_relative_path": kg_rel,
                        "has_gi": True,
                        "has_kg": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            artifact_index.append(
                {
                    "name": f"{ep_label}.gi.json",
                    "relative_path": gi_rel,
                    "kind": "gi",
                    "size_bytes": gi_full.stat().st_size,
                    "mtime_utc": "2026-05-01T00:00:00Z",
                    "publish_date": publish,
                }
            )
            artifact_index.append(
                {
                    "name": f"{ep_label}.kg.json",
                    "relative_path": kg_rel,
                    "kind": "kg",
                    "size_bytes": kg_full.stat().st_size,
                    "mtime_utc": "2026-05-01T00:00:00Z",
                    "publish_date": publish,
                }
            )

            ep_summary = (
                excerpts["insights"][0]
                if excerpts["insights"]
                else f"Synthesised summary for {ep_label}."
            )
            ep_bullets = excerpts["insights"][:3] or [f"Bullet {n + 1}" for n in range(3)]
            episodes.append(
                {
                    "metadata_relative_path": metadata_rel,
                    "feed_id": podcast_id,
                    "feed_display_title": feed_meta["display_title"],
                    "feed_rss_url": feed_meta["rss_url"],
                    "feed_description": feed_meta["description"],
                    "topics": excerpts["topics"][:3],
                    "summary_title": f"{title} — synthetic",
                    "summary_bullets_preview": ep_bullets,
                    "summary_bullet_graph_topic_ids": [
                        f"topic:{slug(t)}" for t in excerpts["topics"][:3]
                    ],
                    "summary_preview": (
                        (ep_summary[:140] + "…") if len(ep_summary) > 140 else ep_summary
                    ),
                    "episode_id": ep_uuid,
                    "episode_title": title,
                    "publish_date": publish,
                    "feed_image_url": None,
                    "episode_image_url": None,
                    "duration_seconds": 1800,
                    "gi_relative_path": gi_rel,
                    "kg_relative_path": kg_rel,
                    "has_gi": True,
                    "has_kg": True,
                    "cil_digest_topics": [
                        {
                            "topic_id": f"topic:{slug(t)}",
                            "label": t,
                            "in_topic_cluster": False,
                            "topic_cluster_compound_id": None,
                        }
                        for t in excerpts["topics"][:2]
                    ],
                }
            )

            episode_details[metadata_rel] = {
                "path": str(out),
                "metadata_relative_path": metadata_rel,
                "feed_id": podcast_id,
                "episode_id": ep_uuid,
                "episode_title": title,
                "publish_date": publish,
                "summary_title": f"{title} — synthetic",
                "summary_bullets": ep_bullets,
                "summary_text": "\n".join(ep_bullets),
                "gi_relative_path": gi_rel,
                "kg_relative_path": kg_rel,
                "has_gi": True,
                "has_kg": True,
                "cil_digest_topics": [],
            }
            feed_episode_count += 1

        feeds.append(
            {
                **feed_meta,
                "episode_count": feed_episode_count,
                "image_url": None,
                "image_local_relpath": None,
            }
        )
        print(f"  {rss_path.name}: {feed_episode_count} episodes")

    # Top-level corpus files.
    def write(rel: str, payload: Any) -> None:
        (out / rel).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    write("corpus/feeds.json", {"path": str(out), "feeds": feeds})
    write(
        "corpus/episodes.json",
        {"path": str(out), "feed_id": None, "items": episodes, "next_cursor": None},
    )
    write("corpus/episode-details.json", episode_details)

    # #774a — topic clusters + digest topic-bands derived from the
    # umbrella topics. For each umbrella (technology, outdoor, gear,
    # environment, health), gather every episode whose topic_ids include
    # the umbrella's slug. Multi-member clusters and multi-hit topic-bands
    # emerge naturally from the cross-podcast injection.
    umbrella_labels: set[str] = set()
    for labels in CROSS_CUTTING_TOPICS.values():
        umbrella_labels.update(labels)
    by_umbrella: dict[str, list[dict[str, Any]]] = {label: [] for label in umbrella_labels}
    for ep in episodes:
        for t_label in ep["topics"]:
            if t_label in umbrella_labels:
                by_umbrella[t_label].append(ep)

    clusters: list[dict[str, Any]] = []
    topic_bands: list[dict[str, Any]] = []
    for label, eps_in_umbrella in by_umbrella.items():
        if not eps_in_umbrella:
            continue
        topic_id = f"topic:{slug(label)}"
        compound_id = f"tc:{slug(label)}-cluster"
        clusters.append(
            {
                "graph_compound_parent_id": compound_id,
                "cil_alias_target_topic_id": topic_id,
                "canonical_label": label.title(),
                "canonical_topic_id": topic_id,
                "member_count": 1,
                "members": [{"topic_id": topic_id}],
            }
        )
        # Mark these episodes as participating in the cluster.
        for ep in eps_in_umbrella:
            for tid in ep["cil_digest_topics"]:
                if tid["topic_id"] == topic_id:
                    tid["in_topic_cluster"] = True
                    tid["topic_cluster_compound_id"] = compound_id
        # Digest topic band: list of hits per umbrella.
        topic_bands.append(
            {
                "topic_id": f"t-{slug(label)}",
                "label": label.title(),
                "query": label,
                "graph_topic_id": topic_id,
                "hits": [
                    {
                        "metadata_relative_path": ep["metadata_relative_path"],
                        "episode_title": ep["episode_title"],
                        "feed_id": ep["feed_id"],
                        "feed_display_title": ep["feed_display_title"],
                        "publish_date": ep["publish_date"],
                        "score": 0.85,
                        "summary_preview": ep["summary_preview"],
                        "episode_id": ep["episode_id"],
                        "gi_relative_path": ep["gi_relative_path"],
                        "kg_relative_path": ep["kg_relative_path"],
                        "has_gi": True,
                        "has_kg": True,
                    }
                    for ep in eps_in_umbrella
                ],
            }
        )

    # All singletons (non-umbrella topics that only appear once).
    all_topic_ids: set[str] = set()
    for ep in episodes:
        for tid_info in ep["cil_digest_topics"]:
            all_topic_ids.add(tid_info["topic_id"])
    umbrella_ids = {f"topic:{slug(label)}" for label in umbrella_labels}
    singletons = all_topic_ids - umbrella_ids

    write(
        "corpus/topic-clusters.json",
        {
            "schema_version": "2",
            "clusters": clusters,
            "topic_count": len(all_topic_ids),
            "cluster_count": len(clusters),
            "singletons": len(singletons),
        },
    )

    # Digest: recent rows from episodes + topic bands derived above.
    digest_recent = episodes[:5]
    write(
        "corpus/digest.json",
        {
            "path": str(out),
            "window": "all",
            "window_start_utc": "2025-01-01T00:00:00Z",
            "window_end_utc": "2027-01-01T00:00:00Z",
            "compact": False,
            "rows": digest_recent,
            "topic_bands": topic_bands,
            "topics_unavailable_reason": (
                None if topic_bands else "synthetic-validation-corpus has no topic bands"
            ),
        },
    )

    write(
        "corpus/stats.json",
        {
            "path": str(out),
            "publish_month_histogram": {"2026-05": len(episodes)},
            "catalog_episode_count": len(episodes),
            "catalog_feed_count": len(feeds),
            "digest_topics_configured": 0,
        },
    )
    write("corpus/coverage.json", {"path": str(out), "items": []})
    write("corpus/persons-top.json", {"path": str(out), "persons": [], "total_persons": 0})
    write("corpus/runs-summary.json", {"path": str(out), "items": []})
    write(
        "corpus/index-stats.json",
        {
            "available": False,
            "reason": "synthetic-validation-corpus has no vector index",
            "stats": None,
            "reindex_recommended": False,
        },
    )
    write("corpus/artifacts.json", {"path": str(out), "artifacts": artifact_index})

    # Manifest.
    total_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    write(
        "manifest.json",
        {
            "schema_version": "1",
            "kind": "synthetic-validation-corpus",
            "generated_from": str(args.rss_dir.resolve()),
            "feed_count": len(feeds),
            "episode_count": len(episodes),
            "total_size_bytes": total_size,
        },
    )
    print(f"\nfixture written to {out}")
    print(f"  feeds: {len(feeds)}")
    print(f"  episodes: {len(episodes)}")
    print(f"  total size: {total_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
