"""Identity guards inside the join (ADR-167, RFC-131 S4).

A name joins a cluster only if it matches some member AND no guard fires against any member. The
guards read evidence already on disk, per episode, and only over the episodes the two ids share:

* (a) ``co_voice`` — both are placed on voices in one episode, on different voices: two people.
* (b) ``publisher_list`` — the publisher's text names both in one episode: two people.
* (c) ``mention_cooccurrence`` — both are Person nodes in one episode and NEITHER is placed on a
  voice there: unknown, so no join. (Decided at G0, 2026-10-30: (c) does not apply when either is
  placed on a voice; the in-episode mention-to-speaker collapse owns that shape.)

The evidence is built once per episode, before clustering, so the order candidates are visited in
cannot change what a guard sees (D-b).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

GUARD_CO_VOICE = "co_voice"
GUARD_PUBLISHER_LIST = "publisher_list"
GUARD_MENTION_COOCCURRENCE = "mention_cooccurrence"

#: Every guard, in the order they are tried; the first that fires is recorded.
GUARDS: Tuple[str, ...] = (GUARD_CO_VOICE, GUARD_PUBLISHER_LIST, GUARD_MENTION_COOCCURRENCE)

#: What a guard's firing says about the two names.
VERDICT = {
    GUARD_CO_VOICE: "different",
    GUARD_PUBLISHER_LIST: "different",
    GUARD_MENTION_COOCCURRENCE: "unknown",
}

#: The speakers diagnostics' publisher-text names (feed and show notes); m0025 reads these keys.
PUBLISHER_KEYS: Tuple[str, ...] = ("known_hosts", "metadata_named", "detected_guests")


@dataclass(frozen=True)
class EpisodeEvidence:
    """What one episode says about its Person ids."""

    present: FrozenSet[str]
    voice_of: Mapping[str, FrozenSet[str]]
    publisher_named: FrozenSet[str]

    @property
    def mention_only(self) -> FrozenSet[str]:
        return frozenset(i for i in self.present if not self.voice_of.get(i))


def _keys(name: str) -> Tuple[str, str]:
    from podcast_scraper.identity.slugify import person_id

    name = " ".join(str(name or "").split())
    return name.casefold(), person_id(name) if name else ""


def episode_evidence(
    people: Iterable[Tuple[str, str]],
    roster: Iterable[Tuple[str, Sequence[str]]],
    publisher_names: Iterable[str],
) -> EpisodeEvidence:
    """Evidence from one episode's KG Person nodes ``(id, name)``, roster ``(name, voices)`` and
    publisher-text names. A roster entry or a publisher name belongs to a Person node when the
    names are equal ignoring case and spacing, or when the name mints the node's id."""
    people = [(str(i), str(n or "")) for i, n in people if i]
    by_key: Dict[str, List[str]] = {}
    for pid, name in people:
        folded, minted = _keys(name)
        for k in (folded, pid, minted):
            if k:
                by_key.setdefault(k, [])
                if pid not in by_key[k]:
                    by_key[k].append(pid)

    def ids_for(name: str) -> List[str]:
        folded, minted = _keys(name)
        return sorted(set(by_key.get(folded, [])) | set(by_key.get(minted, [])))

    voices: Dict[str, set] = {}
    for name, vs in roster:
        for pid in ids_for(name):
            voices.setdefault(pid, set()).update(str(v) for v in vs or [] if v)
    named = {pid for name in publisher_names for pid in ids_for(name)}
    return EpisodeEvidence(
        present=frozenset(pid for pid, _ in people),
        voice_of={k: frozenset(v) for k, v in voices.items() if v},
        publisher_named=frozenset(named),
    )


def guard_fires(
    a: str,
    b: str,
    shared_episodes: Iterable[str],
    evidence: Mapping[str, EpisodeEvidence],
    *,
    enabled: Iterable[str] = GUARDS,
) -> Optional[Tuple[str, str]]:
    """``(guard, episode_id)`` for the first guard that fires against joining ids ``a`` and ``b``,
    over the episodes they share (sorted, so the record is deterministic); None when none does."""
    on = set(enabled)
    for eid in sorted(shared_episodes):
        ev = evidence.get(eid)
        if ev is None or a == b or a not in ev.present or b not in ev.present:
            continue
        va, vb = ev.voice_of.get(a, frozenset()), ev.voice_of.get(b, frozenset())
        if GUARD_CO_VOICE in on and va and vb and not (va & vb):
            return GUARD_CO_VOICE, eid
        if GUARD_PUBLISHER_LIST in on and a in ev.publisher_named and b in ev.publisher_named:
            return GUARD_PUBLISHER_LIST, eid
        if GUARD_MENTION_COOCCURRENCE in on and not va and not vb:
            return GUARD_MENTION_COOCCURRENCE, eid
    return None


def collect_episode_evidence(root: Path | str) -> Dict[str, EpisodeEvidence]:
    """Evidence for every newest-run KG artifact of a corpus, keyed by the KG's ``episode_id`` (the
    key the candidates' episodes use). The roster's voices follow m0025's placement rule."""
    from podcast_scraper.builders.bridge_builder import strip_layer_prefixes
    from podcast_scraper.kg.corpus import (
        load_kg_artifacts,
        newest_run_artifact_paths,
        scan_kg_artifact_paths,
    )
    from podcast_scraper.quality import name_report as NR

    root = Path(root)
    out: Dict[str, EpisodeEvidence] = {}
    paths = newest_run_artifact_paths(root, scan_kg_artifact_paths(root), ".kg.json")
    for path, data in load_kg_artifacts(paths):
        eid = str(data.get("episode_id") or "")
        if not eid:
            continue
        people = [
            (
                strip_layer_prefixes(str(n.get("id") or "")),
                str((n.get("properties") or {}).get("name") or "").strip(),
            )
            for n in data.get("nodes") or []
            if isinstance(n, dict) and n.get("type") == "Person"
        ]
        meta_path = Path(str(path)[: -len(".kg.json")] + ".metadata.json")
        meta = NR._load_json(meta_path)
        roster: List[Tuple[str, Sequence[str]]] = []
        publisher: List[str] = []
        if isinstance(meta, dict):
            ep = NR.Episode("", eid, meta_path.parent.parent.name, meta_path, meta)
            label_voices = NR.segment_label_voices(ep.segments() or [])
            roster = [(s["name"].strip(), NR.entry_voices(s, label_voices)) for s in ep.roster()]
            publisher = _publisher_names(meta, meta_path.parent.parent)
        out[eid] = episode_evidence(people, roster, publisher)
    return out


def _publisher_names(meta: Mapping, run: Path) -> List[str]:
    """The publisher-text names the episode's speakers diagnostics record."""
    from podcast_scraper.quality import name_report as NR

    rel = str((meta.get("content") or {}).get("transcript_file_path") or "")
    path = Path(rel)
    if path.is_absolute():
        rel = (
            str(Path(*path.parts[path.parts.index(run.name) + 1 :]))
            if run.name in path.parts
            else ""
        )
    if not rel.endswith(".txt"):
        return []
    diag = NR._load_json(run / rel.replace(".txt", ".speakers.diagnostics.json"))
    tried = (diag or {}).get("tried") if isinstance(diag, dict) else None
    tried = tried if isinstance(tried, dict) else {}
    return [str(n) for k in PUBLISHER_KEYS for n in tried.get(k) or [] if n]


def candidate_guard(evidence: Mapping[str, EpisodeEvidence], *, enabled: Iterable[str] = GUARDS):
    """The ``guard(candidate, member)`` the cluster build takes: :func:`guard_fires` over the
    episodes both candidates appear in."""
    on = tuple(enabled)

    def guard(cand, member) -> Optional[Tuple[str, str]]:
        return guard_fires(
            cand.id, member.id, cand.episodes & member.episodes, evidence, enabled=on
        )

    return guard
