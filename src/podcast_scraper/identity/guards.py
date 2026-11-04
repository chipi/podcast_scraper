"""Identity guards inside the join (ADR-167, RFC-131 S4).

A name joins a cluster only if it matches some member AND no guard fires against any member. The
guards read evidence already on disk, per episode, and only over the episodes the two ids share:

* (a) ``co_voice`` — both are placed on voices in one episode, on different voices: two people.
* (b) ``publisher_list`` — the publisher's text names both in one episode: two people.
* (c) ``mention_cooccurrence`` — both are Person nodes in one episode and NEITHER is placed on a
  voice there: unknown, so no join. (Decided at G0, 2026-10-30: (c) does not apply when either is
  placed on a voice; the in-episode mention-to-speaker collapse owns that shape.)

Two ids with one name (equal ignoring case and spacing) carry no evidence of two people: (b) and
(c) do not fire on them. Found in the S4 replay: a publisher listing one guest's name once held
two ids carrying that same name apart.

The evidence is built once per episode, before clustering, so the order candidates are visited in
cannot change what a guard sees (D-b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

GUARD_CO_VOICE = "co_voice"
GUARD_PUBLISHER_LIST = "publisher_list"
GUARD_MENTION_COOCCURRENCE = "mention_cooccurrence"
#: A pair a human read judged two people (record s5d): refused in any episode, at any count.
GUARD_REVIEWED_APART = "reviewed_apart"

#: Every guard, in the order they are tried; the first that fires is recorded.
GUARDS: Tuple[str, ...] = (GUARD_CO_VOICE, GUARD_PUBLISHER_LIST, GUARD_MENTION_COOCCURRENCE)

#: What a guard's firing says about the two names.
VERDICT = {
    GUARD_CO_VOICE: "different",
    GUARD_PUBLISHER_LIST: "different",
    GUARD_MENTION_COOCCURRENCE: "unknown",
    GUARD_REVIEWED_APART: "different",
}

#: The reviewed near-spelling list (record s5d): ``groups`` a human read judged one person, and
#: ``apart`` pairs it judged two.
REVIEWED_LIST = (
    Path(__file__).resolve().parents[1] / "upgrade" / "lists" / "s5d_reviewed_fuzzy_joins.json"
)

#: The speakers diagnostics' publisher-text names (feed and show notes); m0025 reads these keys.
PUBLISHER_KEYS: Tuple[str, ...] = ("known_hosts", "metadata_named", "detected_guests")


@dataclass(frozen=True)
class EpisodeEvidence:
    """What one episode says about its Person ids."""

    present: FrozenSet[str]
    voice_of: Mapping[str, FrozenSet[str]]
    publisher_named: FrozenSet[str]
    #: The node's name, normalised (case and spacing): two ids with one name carry no evidence of
    #: two people, so the publisher-list and mention guards do not fire on them.
    name_of: Mapping[str, str] = field(default_factory=dict)
    #: The publisher-text names as written, in order (the display rule's reference, s6e), the
    #: Person nodes' names as written, and the names only ``detected_guests`` states (NER over the
    #: metadata, not the publisher's own list).
    publisher_names: Tuple[str, ...] = ()
    person_names: Mapping[str, str] = field(default_factory=dict)
    ner_only: FrozenSet[str] = frozenset()
    #: Evidence mode ``publisher_verified`` only (record l_publisher_verified_mode): the
    #: ``(publisher spelling, name)`` pairs (folded) V2 found in the text, the Person ids seated
    #: as host here, the show's key, and the show's verified names this episode supports.
    publisher_pairs: FrozenSet[Tuple[str, str]] = frozenset()
    host_seated: FrozenSet[str] = frozenset()
    show: str = ""
    show_names: Tuple[str, ...] = ()

    @property
    def mention_only(self) -> FrozenSet[str]:
        return frozenset(i for i in self.present if not self.voice_of.get(i))


def _keys(name: str) -> Tuple[str, str]:
    from podcast_scraper.identity.slugify import person_id

    name = " ".join(str(name or "").split())
    try:
        minted = person_id(name) if name else ""
    except ValueError:  # no letters the slug keeps (e.g. a name in Chinese script): name only
        minted = ""
    return name.casefold(), minted


def episode_evidence(
    people: Iterable[Tuple[str, str]],
    roster: Iterable[Tuple[str, Sequence[str]]],
    publisher_names: Iterable[str],
    *,
    ner_only: Iterable[str] = (),
    publisher_pairs: Iterable[Tuple[str, str]] = (),
    host_seated: Iterable[str] = (),
    show: str = "",
    show_names: Iterable[str] = (),
) -> EpisodeEvidence:
    """Evidence from one episode's KG Person nodes ``(id, name)``, roster ``(name, voices)`` and
    publisher-text names. A roster entry or a publisher name belongs to a Person node when the
    names are equal ignoring case and spacing, or when the name mints the node's id."""
    people = [(str(i), str(n or "")) for i, n in people if i]
    publisher_names = list(dict.fromkeys(str(n) for n in publisher_names if n))
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
        name_of={pid: _keys(name)[0] for pid, name in people},
        voice_of={k: frozenset(v) for k, v in voices.items() if v},
        publisher_named=frozenset(named),
        publisher_names=tuple(publisher_names),
        person_names={pid: name for pid, name in people if name},
        ner_only=frozenset(ner_only),
        publisher_pairs=frozenset(publisher_pairs),
        host_seated=frozenset(host_seated),
        show=show,
        show_names=tuple(show_names),
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
        one_name = bool(ev.name_of.get(a)) and ev.name_of.get(a) == ev.name_of.get(b)
        if GUARD_CO_VOICE in on and va and vb and not (va & vb):
            return GUARD_CO_VOICE, eid
        if one_name:
            continue
        if GUARD_PUBLISHER_LIST in on and a in ev.publisher_named and b in ev.publisher_named:
            return GUARD_PUBLISHER_LIST, eid
        if GUARD_MENTION_COOCCURRENCE in on and not va and not vb:
            return GUARD_MENTION_COOCCURRENCE, eid
    return None


def collect_episode_evidence(
    root: Path | str, mode: str = "extracted"
) -> Dict[str, EpisodeEvidence]:
    """Evidence for every newest-run KG artifact of a corpus, keyed by the KG's ``episode_id`` (the
    key the candidates' episodes use). The roster's voices follow m0025's placement rule.

    ``mode`` (``identity.publisher_text.EVIDENCE_MODES``): ``extracted`` takes the diagnostics'
    extracted names as the publisher's, as every build before record l_publisher_verified_mode;
    ``publisher_verified`` keeps only those the publisher's own text contains, adds the text's own
    spellings (V1, V2), and records each episode's host seats and show for V3."""
    from podcast_scraper.builders.bridge_builder import strip_layer_prefixes
    from podcast_scraper.identity import publisher_text as PT

    if mode not in PT.EVIDENCE_MODES:
        raise ValueError(f"unknown evidence mode {mode!r}; one of {PT.EVIDENCE_MODES}")
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
                str(
                    (n.get("properties") or {}).get("respelled_from")
                    or (n.get("properties") or {}).get("name")
                    or ""
                ).strip(),
            )
            for n in data.get("nodes") or []
            if isinstance(n, dict) and n.get("type") == "Person"
        ]
        meta_path = Path(str(path)[: -len(".kg.json")] + ".metadata.json")
        meta = NR._load_json(meta_path)
        roster: List[Tuple[str, Sequence[str]]] = []
        publisher: List[str] = []
        ner: List[str] = []
        verified: Dict[str, object] = {}
        if isinstance(meta, dict):
            ep = NR.Episode("", eid, meta_path.parent.parent.name, meta_path, meta)
            label_voices = NR.segment_label_voices(ep.segments() or [])
            # The entry's own spelling (record 0041, 4g); its voices from the entry as written.
            roster = [
                (
                    str(s.get("respelled_from") or s["name"]).strip(),
                    NR.entry_voices(s, label_voices),
                )
                for s in ep.roster()
            ]
            by_key = _publisher_names_by_key(meta, meta_path.parent.parent)
            if mode == PT.PUBLISHER_VERIFIED:
                speakers = (meta.get("content") or {}).get("speakers") or []
                entries = [
                    str(s.get("respelled_from") or s.get("name") or "")
                    for s in speakers
                    if isinstance(s, dict)
                ]
                by_key, pairs = PT.verify_and_find(meta, by_key, entries)
                verified = {
                    "publisher_pairs": pairs,
                    "host_seated": _host_seated(data, strip_layer_prefixes),
                    "show": str((meta.get("feed") or {}).get("feed_id") or "") or _kg_show(data),
                    "show_names": [
                        n
                        for k in ("known_hosts", "metadata_named")
                        for n in by_key.get(k, [])
                        if len(PT.fold(n).split()) >= 2
                    ],
                }
            publisher = [n for k in PUBLISHER_KEYS for n in by_key.get(k, [])]
            stated = {
                n for k in PUBLISHER_KEYS if k != "detected_guests" for n in by_key.get(k, [])
            }
            ner = [n for n in by_key.get("detected_guests", []) if n not in stated]
        out[eid] = episode_evidence(
            people, roster, publisher, ner_only=ner, **verified  # type: ignore[arg-type]
        )
    return out


def _host_seated(kg: Mapping, strip) -> List[str]:
    """The Person ids the episode's KG seats as host."""
    return [
        strip(str(n.get("id") or ""))
        for n in kg.get("nodes") or []
        if isinstance(n, dict)
        and n.get("type") == "Person"
        and (n.get("properties") or {}).get("role") == "host"
    ]


def _kg_show(kg: Mapping) -> str:
    """The show key the KG's Episode node carries."""
    for n in kg.get("nodes") or []:
        if isinstance(n, dict) and n.get("type") == "Episode":
            props = n.get("properties") or {}
            return str(props.get("feed_id") or props.get("podcast_id") or "")
    return ""


def host_join_pair_rule(evidence: Mapping[str, EpisodeEvidence], base):
    """V3 over ``base`` (the build's pair rule): ``"join"`` for a person candidate seated as host
    in a show whose verified names hold one that :func:`publisher_text.host_route` matches, and
    for that name's own id against it; otherwise ``base``'s answer (record
    l_publisher_verified_mode)."""
    from collections import defaultdict

    from podcast_scraper.identity import publisher_text as PT
    from podcast_scraper.identity.slugify import person_id

    names: Dict[str, Dict[str, None]] = defaultdict(dict)
    seats: Dict[str, set] = defaultdict(set)
    own_name: Dict[str, str] = {}
    for ev in evidence.values():
        if not ev.show:
            continue
        for n in ev.show_names:
            names[ev.show][n] = None
        for pid in ev.host_seated:
            seats[pid].add(ev.show)
            own_name[pid] = ev.person_names.get(pid) or own_name.get(pid, "")
    joins: set = set()
    for show, hosts in names.items():
        by_first: Dict[str, List[str]] = defaultdict(list)
        for h in hosts:
            by_first[PT.split_name(h)[0]].append(h)
        for pid, shows in seats.items():
            if show not in shows:
                continue
            own = own_name.get(pid, "")
            for h in hosts:
                if own and PT.host_route(h, own, by_first):
                    try:
                        joins.add(frozenset((pid, person_id(h))))
                    except ValueError:
                        continue

    def rule(cand, member):
        if (
            cand.kind == "person"
            and member.kind == "person"
            and frozenset((cand.id, member.id)) in joins
        ):
            return "join"
        return base(cand, member) if base is not None else None

    rule.joins = joins  # type: ignore[attr-defined]
    return rule


def _publisher_names(meta: Mapping, run: Path) -> List[str]:
    """The publisher-text names the episode's speakers diagnostics record."""
    by_key = _publisher_names_by_key(meta, run)
    return [n for k in PUBLISHER_KEYS for n in by_key.get(k, [])]


def _publisher_names_by_key(meta: Mapping, run: Path) -> Dict[str, List[str]]:
    """``{publisher key: names}`` from the episode's speakers diagnostics (``tried``)."""
    from podcast_scraper.quality import name_report as NR
    from podcast_scraper.utils.run_relative import run_relative

    rel = run_relative((meta.get("content") or {}).get("transcript_file_path"), run)
    if not rel.endswith(".txt"):
        return {}
    diag = NR._load_json(run / rel.replace(".txt", ".speakers.diagnostics.json"))
    tried = (diag or {}).get("tried") if isinstance(diag, dict) else None
    tried = tried if isinstance(tried, dict) else {}
    return {k: [str(n) for n in tried.get(k) or [] if n] for k in PUBLISHER_KEYS}


def candidate_guard(evidence: Mapping[str, EpisodeEvidence], *, enabled: Iterable[str] = GUARDS):
    """The ``guard(candidate, member)`` the cluster build takes: :func:`guard_fires` over the
    episodes both candidates appear in."""
    on = tuple(enabled)

    def guard(cand, member) -> Optional[Tuple[str, str]]:
        return guard_fires(
            cand.id, member.id, cand.episodes & member.episodes, evidence, enabled=on
        )

    return guard


def load_reviewed(path: Optional[Path] = None) -> Dict[str, list]:
    """The reviewed list: ``{"groups": [[id, ...]], "apart": [[id, id, basis]]}``."""
    import json

    data = json.loads(Path(path or REVIEWED_LIST).read_text(encoding="utf-8"))
    return {"groups": list(data.get("groups") or []), "apart": list(data.get("apart") or [])}


def reviewed_pair_rule(evidence: Mapping[str, EpisodeEvidence], reviewed: Mapping[str, list]):
    """The cluster build's ``pair_rule(candidate, member)`` (record s5d): ``"apart"`` for a pair
    the reviewed list judged two people; ``"ignore"`` for a near spelling between two ids of at
    most one episode each, neither a voice nor named by publisher text, not in one reviewed group;
    None otherwise. Only people; an episode-scoped id is s5c's."""
    from podcast_scraper.identity.bare_name_scope import is_scoped_person_id
    from podcast_scraper.identity.person_match import near_spelling_only
    from podcast_scraper.kg.entity_clusters import _pair_language

    group_of = {str(i): n for n, g in enumerate(reviewed.get("groups") or []) for i in g}
    apart = {frozenset((str(p[0]), str(p[1]))) for p in reviewed.get("apart") or []}
    voiced = {pid for ev in evidence.values() for pid, vs in ev.voice_of.items() if vs}
    named = {pid for ev in evidence.values() for pid in ev.publisher_named}

    def rule(cand, member) -> Optional[str]:
        if cand.kind != "person" or member.kind != "person":
            return None
        if frozenset((cand.id, member.id)) in apart:
            return "apart"
        ids = (cand.id, member.id)
        if any(is_scoped_person_id(i) for i in ids) or cand.freq > 1 or member.freq > 1:
            return None
        if any(i in voiced or i in named for i in ids):
            return None
        g = group_of.get(cand.id)
        if g is not None and g == group_of.get(member.id):
            return None
        if near_spelling_only(cand.name, member.name, _pair_language(cand, member)):
            return "ignore"
        return None

    return rule
