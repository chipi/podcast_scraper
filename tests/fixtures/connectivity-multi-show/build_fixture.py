#!/usr/bin/env python3
"""#1058 chunk 4 — Generator for the multi-show connectivity fixture.

This script is the SPEC for what the fixture under
``tests/fixtures/connectivity-multi-show/feeds/`` exercises. Re-running
it produces byte-equal output (deterministic JSON ordering + fixed
ids) so the checked-in fixture stays in sync with this contract.

Why a generator instead of hand-rolling 9 JSON files: the connectivity
surfaces we want to exercise (cross-show Person, cross-show
Organization, intra-episode co-speakers, cross-show concept-Topic via
RELATED_TO, person→insight→topic neighborhood) require careful
referential consistency — every edge target must resolve to a real
node, every Person id must appear in every show where they speak.
Hand-editing two-dozen referentially-linked files invites silent
drift. The generator encodes the connectivity contract as Python,
emits the JSON, and is the single source of truth.

Usage:

    .venv/bin/python tests/fixtures/connectivity-multi-show/build_fixture.py

Idempotent. The output is checked in so most callers don't need to
run the script; the script exists so future contributors can
regenerate after editing the connectivity contract here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_FEEDS = _HERE / "feeds"


# ─────────────────────────────────────────────────────────────────
# The connectivity contract — what this fixture must exercise
# ─────────────────────────────────────────────────────────────────


# Persons that recur across shows (the cross-show connectivity backbone).
_PERSONS: Dict[str, Dict[str, str]] = {
    "person:alice-hayes": {"name": "Dr. Alice Hayes", "role": "host"},
    "person:bob-chen": {"name": "Bob Chen", "role": "host"},
    "person:maya-okonkwo": {"name": "Maya Okonkwo", "role": "guest"},
    "person:dan-tran": {"name": "Dan Tran", "role": "guest"},
}

# Organizations that recur — drives MENTIONS_ORG + entity_neighborhood.
_ORGS: Dict[str, Dict[str, str]] = {
    "org:acme-labs": {"name": "Acme Labs"},
    "org:greentech-ventures": {"name": "GreenTech Ventures"},
}

# Topic clusters — surface-different labels per show, same concept.
# Drives the cross-show concept-Topic + RELATED_TO contract.
_CONCEPT_CLUSTERS: List[Tuple[str, str, List[Tuple[str, str]]]] = [
    # (concept_id, canonical_label, [(podcast_short, surface_label), ...])
    (
        "concept:topic-ai-safety",
        "AI safety",
        [
            ("show-a", "AI safety"),
            ("show-b", "AI alignment"),
            ("show-c", "alignment problem"),
        ],
    ),
    (
        "concept:topic-clean-energy",
        "clean energy",
        [
            ("show-a", "clean energy"),
            ("show-c", "renewable energy transition"),
        ],
    ),
]

# (podcast_short, episode_short, title, host_ids, guest_ids,
#  topic_labels, mentioned_person_ids, mentioned_org_ids)
_EPISODES: List[Tuple[str, str, str, List[str], List[str], List[str], List[str], List[str]]] = [
    # show-a — Dr. Alice Hayes hosts; Maya guest in ep1, Dan in ep2.
    (
        "show-a",
        "ep1",
        "Building a clean energy grid",
        ["person:alice-hayes"],
        ["person:maya-okonkwo"],
        ["AI safety", "clean energy"],
        ["person:bob-chen"],  # mentioned in passing
        ["org:greentech-ventures"],
    ),
    (
        "show-a",
        "ep2",
        "AI safety roundtable",
        ["person:alice-hayes"],
        ["person:dan-tran"],
        ["AI safety"],
        ["person:maya-okonkwo"],
        ["org:acme-labs"],
    ),
    # show-b — Bob Chen hosts; Alice guest in ep1.
    (
        "show-b",
        "ep1",
        "AI alignment in practice",
        ["person:bob-chen"],
        ["person:alice-hayes"],
        ["AI alignment"],
        [],
        ["org:acme-labs"],
    ),
    (
        "show-b",
        "ep2",
        "The future of compute",
        ["person:bob-chen"],
        ["person:maya-okonkwo"],
        ["AI alignment"],
        [],
        ["org:acme-labs"],
    ),
    # show-c — Alice Hayes hosts; Bob Chen guest in ep1; Dan in ep2.
    (
        "show-c",
        "ep1",
        "Alignment problem and policy",
        ["person:alice-hayes"],
        ["person:bob-chen"],
        ["alignment problem", "renewable energy transition"],
        [],
        ["org:greentech-ventures"],
    ),
    (
        "show-c",
        "ep2",
        "Renewable energy economics",
        ["person:alice-hayes"],
        ["person:dan-tran"],
        ["renewable energy transition"],
        ["person:maya-okonkwo"],
        ["org:greentech-ventures"],
    ),
]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _episode_id(podcast_short: str, episode_short: str) -> str:
    return f"episode:{podcast_short}-{episode_short}"


def _podcast_id(podcast_short: str) -> str:
    return f"podcast:{podcast_short}"


def _topic_id(podcast_short: str, label: str) -> str:
    slug = label.lower().replace(" ", "-").replace("/", "-")
    return f"topic:{podcast_short}-{slug}"


def _insight_id(podcast_short: str, episode_short: str, ix: int) -> str:
    return f"insight:{podcast_short}-{episode_short}-i{ix}"


def _quote_id(podcast_short: str, episode_short: str, ix: int) -> str:
    return f"quote:{podcast_short}-{episode_short}-q{ix}"


def _publish_date(podcast_short: str, episode_short: str) -> str:
    """Stable per-episode timestamp so re-runs are byte-equal."""
    order = {
        ("show-a", "ep1"): 1,
        ("show-a", "ep2"): 2,
        ("show-b", "ep1"): 3,
        ("show-b", "ep2"): 4,
        ("show-c", "ep1"): 5,
        ("show-c", "ep2"): 6,
    }
    n = order.get((podcast_short, episode_short), 0)
    return f"2024-01-{n:02d}T00:00:00Z"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _person_name(person_id: str) -> str:
    return _PERSONS[person_id]["name"]


def _org_name(org_id: str) -> str:
    return _ORGS[org_id]["name"]


def _concept_id_for_label(podcast_short: str, label: str) -> str:
    """Return the concept-Topic id that should link to this per-show
    Topic, if it's part of a known cluster. Returns empty if standalone."""
    for concept_id, _canonical, members in _CONCEPT_CLUSTERS:
        for ps, surface in members:
            if ps == podcast_short and surface == label:
                return concept_id
    return ""


def _all_concepts_for_show(podcast_short: str) -> List[Tuple[str, str]]:
    """All (concept_id, canonical_label) clusters that this show
    participates in. Used to add the concept-Topic nodes + RELATED_TO."""
    out: List[Tuple[str, str]] = []
    for concept_id, canonical, members in _CONCEPT_CLUSTERS:
        if any(ps == podcast_short for ps, _ in members):
            out.append((concept_id, canonical))
    return out


# ─────────────────────────────────────────────────────────────────
# Per-episode artifact builders
# ─────────────────────────────────────────────────────────────────


def _build_metadata_json(
    podcast_short: str,
    episode_short: str,
    title: str,
    hosts: List[str],
    guests: List[str],
) -> Dict[str, Any]:
    return {
        "feed": {"title": podcast_short.replace("-", " ").title()},
        "episode": {
            "episode_id": _episode_id(podcast_short, episode_short),
            "title": title,
            "publish_date": _publish_date(podcast_short, episode_short),
        },
        "content": {
            "transcript_file_path": f"{podcast_short}_{episode_short}.txt",
            "detected_hosts": [_person_name(p) for p in hosts],
            "detected_guests": [_person_name(p) for p in guests],
        },
        "grounded_insights": {"artifact_path": f"{podcast_short}_{episode_short}.gi.json"},
    }


def _build_bridge_json(podcast_short: str, episode_short: str) -> Dict[str, Any]:
    eid = _episode_id(podcast_short, episode_short)
    pid = _podcast_id(podcast_short)
    return {
        "schema_version": "1.0",
        "episode_id": eid,
        "podcast_id": pid,
        "topic_to_canonical": {},
        "entity_to_canonical": {},
    }


def _build_kg_json(
    podcast_short: str,
    episode_short: str,
    hosts: List[str],
    guests: List[str],
    topic_labels: List[str],
    mentioned_org_ids: List[str],
) -> Dict[str, Any]:
    eid = _episode_id(podcast_short, episode_short)
    pid = _podcast_id(podcast_short)
    nodes: List[Dict[str, Any]] = [
        {
            "id": eid,
            "type": "Episode",
            "properties": {
                "podcast_id": pid,
                "title": _episode_title(podcast_short, episode_short),
                "publish_date": _publish_date(podcast_short, episode_short),
            },
        },
        {
            "id": pid,
            "type": "Podcast",
            "properties": {"title": podcast_short.replace("-", " ").title()},
        },
    ]
    edges: List[Dict[str, Any]] = [
        {"type": "HAS_EPISODE", "from": pid, "to": eid},
    ]
    for person_id in hosts + guests:
        nodes.append(
            {
                "id": person_id,
                "type": "Person",
                "properties": {
                    "name": _person_name(person_id),
                    "role": _PERSONS[person_id]["role"],
                },
            }
        )
    for label in topic_labels:
        nodes.append(
            {
                "id": _topic_id(podcast_short, label),
                "type": "Topic",
                "properties": {"label": label},
            }
        )
    for org_id in mentioned_org_ids:
        nodes.append(
            {
                "id": org_id,
                "type": "Organization",
                "properties": {
                    "name": _org_name(org_id),
                    "role": "mentioned",
                },
            }
        )
    # Add concept-Topic nodes + RELATED_TO from each member Topic.
    for concept_id, canonical in _all_concepts_for_show(podcast_short):
        # Only add the concept-Topic if at least one of this episode's
        # topic labels is a member of the cluster.
        member_labels_in_ep = [
            label
            for label in topic_labels
            if _concept_id_for_label(podcast_short, label) == concept_id
        ]
        if not member_labels_in_ep:
            continue
        nodes.append(
            {
                "id": concept_id,
                "type": "Topic",
                "properties": {"label": canonical, "is_concept": True},
            }
        )
        for label in member_labels_in_ep:
            edges.append(
                {
                    "type": "RELATED_TO",
                    "from": _topic_id(podcast_short, label),
                    "to": concept_id,
                }
            )

    return {
        "schema_version": "2.0",
        "episode_id": eid,
        "extraction": {
            "model_version": "fixture",
            "extracted_at": _publish_date(podcast_short, episode_short),
            "transcript_ref": f"{podcast_short}_{episode_short}.txt",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _episode_title(podcast_short: str, episode_short: str) -> str:
    for ps, es, title, *_ in _EPISODES:
        if ps == podcast_short and es == episode_short:
            return title
    return episode_short


def _build_gi_json(
    podcast_short: str,
    episode_short: str,
    hosts: List[str],
    guests: List[str],
    topic_labels: List[str],
    mentioned_person_ids: List[str],
    mentioned_org_ids: List[str],
) -> Dict[str, Any]:
    eid = _episode_id(podcast_short, episode_short)
    nodes: List[Dict[str, Any]] = [
        {
            "id": eid,
            "type": "Episode",
            "properties": {
                "podcast_id": _podcast_id(podcast_short),
                "title": _episode_title(podcast_short, episode_short),
                "publish_date": _publish_date(podcast_short, episode_short),
            },
        }
    ]
    edges: List[Dict[str, Any]] = []
    # Generate one Insight per topic, mentioning host + each mentioned
    # person + organization so MENTIONS_PERSON / MENTIONS_ORG resolve.
    for ix, label in enumerate(topic_labels):
        iid = _insight_id(podcast_short, episode_short, ix)
        host_name = _person_name(hosts[0]) if hosts else "the host"
        # Stitch every mentioned name into the text so the regex path
        # of add_insight_entity_edges resolves them.
        mentioned_names = [_person_name(p) for p in mentioned_person_ids]
        mentioned_org_names = [_org_name(o) for o in mentioned_org_ids]
        text_parts = [
            f"{host_name} discussed {label}.",
            *(f"They referenced {n}." for n in mentioned_names),
            *(f"They noted {n}." for n in mentioned_org_names),
        ]
        nodes.append(
            {
                "id": iid,
                "type": "Insight",
                "properties": {
                    "text": " ".join(text_parts),
                    "episode_id": eid,
                    "grounded": True,
                    "insight_type": "claim",
                    "position_hint": round(0.2 + 0.2 * ix, 2),
                },
            }
        )
        edges.append({"type": "HAS_INSIGHT", "from": eid, "to": iid})
        # ABOUT edge to per-show Topic (which already exists in KG).
        edges.append({"type": "ABOUT", "from": iid, "to": _topic_id(podcast_short, label)})
        # #1058 chunk 3 contract — when this per-show Topic is a
        # member of a cross-show cluster, also emit ABOUT to the
        # concept-Topic so the relational layer's `related_topics`
        # surface can fold across shows via shared insights on the
        # concept-Topic.
        concept_id_for_topic = _concept_id_for_label(podcast_short, label)
        if concept_id_for_topic:
            edges.append({"type": "ABOUT", "from": iid, "to": concept_id_for_topic})
            if not any(n.get("id") == concept_id_for_topic for n in nodes):
                nodes.append(
                    {
                        "id": concept_id_for_topic,
                        "type": "Topic",
                        "properties": {
                            "label": next(
                                canonical
                                for (cid, canonical, _) in _CONCEPT_CLUSTERS
                                if cid == concept_id_for_topic
                            ),
                            "is_concept": True,
                        },
                    }
                )
        # Quote + SUPPORTED_BY for at least one insight per episode so
        # grounded=True is honest under the v3 contract.
        qid = _quote_id(podcast_short, episode_short, ix)
        nodes.append(
            {
                "id": qid,
                "type": "Quote",
                "properties": {
                    "text": f"This is what we said about {label}.",
                    "episode_id": eid,
                    "speaker_id": hosts[0] if hosts else None,
                    "char_start": 0,
                    "char_end": 40,
                    "timestamp_start_ms": ix * 1000,
                    "timestamp_end_ms": ix * 1000 + 500,
                    "transcript_ref": f"{podcast_short}_{episode_short}.txt",
                },
            }
        )
        edges.append({"type": "SUPPORTED_BY", "from": iid, "to": qid})
        # MENTIONS_PERSON edges per mentioned person.
        for person_id in mentioned_person_ids:
            edges.append({"type": "MENTIONS_PERSON", "from": iid, "to": person_id})
        # MENTIONS_ORG edges per mentioned organization.
        for org_id in mentioned_org_ids:
            edges.append({"type": "MENTIONS_ORG", "from": iid, "to": org_id})
        # SPOKEN_BY for the quote — alternate host / guest so both
        # have STATES → Insight → ABOUT → Topic chains and
        # co_speakers can resolve within the episode.
        speaker = (
            hosts[0]
            if (ix % 2 == 0 and hosts)
            else (guests[0] if guests else (hosts[0] if hosts else None))
        )
        if speaker:
            edges.append({"type": "SPOKEN_BY", "from": qid, "to": speaker})
    # Mention nodes (so the targets resolve locally without a KG join).
    for person_id in mentioned_person_ids:
        nodes.append(
            {
                "id": person_id,
                "type": "Person",
                "properties": {"name": _person_name(person_id)},
            }
        )
    for org_id in mentioned_org_ids:
        nodes.append(
            {
                "id": org_id,
                "type": "Organization",
                "properties": {
                    "name": _org_name(org_id),
                    "role": "mentioned",
                },
            }
        )
    # Speaker nodes for hosts (so SPOKEN_BY resolves).
    for person_id in hosts + guests:
        nodes.append(
            {
                "id": person_id,
                "type": "Person",
                "properties": {"name": _person_name(person_id)},
            }
        )
    return {
        "schema_version": "3.0",
        "model_version": "fixture",
        "prompt_version": "v1",
        "episode_id": eid,
        "nodes": nodes,
        "edges": edges,
    }


def _build_transcript(title: str, hosts: List[str], guests: List[str]) -> str:
    """Tiny canned transcript — enough to make the file present without
    real audio. Tier-3 tests treat transcript_file_path as opaque."""
    lines = [f"# {title}"]
    if hosts:
        lines.append(f"{_person_name(hosts[0])}: Welcome to the show.")
    if guests:
        lines.append(f"{_person_name(guests[0])}: Thanks for having me.")
    return "\n".join(lines) + "\n"


def main() -> int:
    for (
        podcast_short,
        episode_short,
        title,
        hosts,
        guests,
        topic_labels,
        mentioned_persons,
        mentioned_orgs,
    ) in _EPISODES:
        feed_dir = _FEEDS / podcast_short
        meta_dir = feed_dir / "metadata"
        tx_dir = feed_dir / "transcripts"

        base = f"{podcast_short}_{episode_short}"
        _write_json(
            meta_dir / f"{base}.metadata.json",
            _build_metadata_json(podcast_short, episode_short, title, hosts, guests),
        )
        _write_json(
            meta_dir / f"{base}.bridge.json",
            _build_bridge_json(podcast_short, episode_short),
        )
        _write_json(
            meta_dir / f"{base}.kg.json",
            _build_kg_json(
                podcast_short,
                episode_short,
                hosts,
                guests,
                topic_labels,
                mentioned_orgs,
            ),
        )
        _write_json(
            meta_dir / f"{base}.gi.json",
            _build_gi_json(
                podcast_short,
                episode_short,
                hosts,
                guests,
                topic_labels,
                mentioned_persons,
                mentioned_orgs,
            ),
        )
        transcript = _build_transcript(title, hosts, guests)
        (tx_dir).mkdir(parents=True, exist_ok=True)
        (tx_dir / f"{base}.txt").write_text(transcript, encoding="utf-8")
    print(f"built {len(_EPISODES)} episodes under {_FEEDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
