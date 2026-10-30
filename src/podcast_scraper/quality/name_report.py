"""What the listener sees of people's names, measured over a served corpus (read-only).

The name-consolidation arc's report. It walks the served episodes (newest run per feed and
episode, the rule the catalog uses) and runs the app's own builders on each one, so the numbers
describe what the shipped code serves, not a re-implementation of it.

Two modes:

* ``roster`` counts the names in ``content.speakers`` as listed, the population of the 2026-10-30
  prod scan. Its job is faithfulness: it must reproduce that scan's pairs exactly.
* ``voice`` counts only names bound to a voice: a roster entry placed on one or more voices
  (``voices`` / ``placed``), or, on an older record without those fields, an entry whose name is
  a segment ``speaker_label`` (the placement rule of migration m0025's ``_Episode._voices``).

Measures produced here:

* **R1** raw labels shown, per surface: the insights panel (default and fallback paths), the
  transcript segments, and the episode people chips.
* **R2** candidate groups of names that may be one person, per feed, with spellings and
  insights, plus how the identity map (``cached_entity_id_map``'s builder) groups the same ids.
* **R4** (voice mode) one person across shows: identical names on two or more feeds, and
  rule-b variants across feeds, each with whether today's identity map already joins them.
* **R5** names listed but not heard, and names heard (insight speakers) but not listed.

R3 (rates against the labelled truth set), R6 and R7 come in later steps.

Nothing is written. Output is one JSON object on stdout; person names are in it, so it belongs
in the private evaluation data, never in this repository.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from podcast_scraper.graph_id_utils import is_bare_speaker_label, is_scoped_placeholder_person_id

#: The run directory's timestamp suffix, ``run_<id>_<YYYYMMDD-HHMMSS>``.
_RUN_TS = re.compile(r"_(\d{8}-\d{6})$")

#: A raw diarization label or the speaker detector's failure default, as a shown name. Wider
#: than ``is_bare_speaker_label`` on purpose: ``unknown_guest_1`` passes that predicate today, and
#: the report must see what the predicate misses.
_RAW_LABEL = re.compile(r"^\s*(speaker[\s_\-]*\d+|unknown[_\s]guest[_\s]*\d+)\s*$", re.IGNORECASE)

#: Labels the product shows on purpose for a voice that is not a named person.
DELIBERATE_LABELS = frozenset({"Unidentified speaker", "Brief speaker", "Advertisement", "Host"})

#: Pairing rules of the 2026-10-30 scan: rule a = equal after normalisation; rule b = edit
#: distance 1-2 when the shorter normalised name has at least this many characters.
_RULE_B_MIN_LEN = 5
_RULE_B_MAX_EDITS = 2


def label_kind(name: Optional[str]) -> Optional[str]:
    """How a shown speaker name reads: ``raw``, ``role``, ``deliberate`` or None (a name).

    ``raw`` is a numbered diarization label or the detector's default (``unknown_guest_1``);
    ``role`` is a bare role word shown as a name ("Guest", "Interviewer"); ``deliberate`` is a
    label the product shows on purpose ("Unidentified speaker", "Host" in its own episode).
    """
    if not name or not name.strip():
        return None
    if _RAW_LABEL.match(name):
        return "raw"
    if name.strip().casefold() in {label.casefold() for label in DELIBERATE_LABELS}:
        return "deliberate"
    if is_bare_speaker_label(name):
        return "role"
    return None


def is_raw_label(name: Optional[str]) -> bool:
    """True for a shown name that is a raw diarization label or the detector's default."""
    return label_kind(name) == "raw"


def normalise(name: str) -> str:
    """NFKD, accents removed, lowercased, spaces / hyphens / dots / apostrophes dropped."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[\s\-.'’‘`]", "", s)


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, stopping at ``cap`` (the scan's helper, so pairs match exactly)."""
    if abs(len(a) - len(b)) >= cap:
        return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) >= cap:
            return cap
        prev = cur
    return prev[-1]


def pair_rule(a: str, b: str) -> Optional[str]:
    """``"a"``, ``"b"`` or None for two names of one feed, by the scan's rules."""
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return None
    if na == nb:
        return "a"
    if min(len(na), len(nb)) >= _RULE_B_MIN_LEN and (edit_distance(na, nb) <= _RULE_B_MAX_EDITS):
        return "b"
    return None


# ---- the served corpus ------------------------------------------------------------------------


@dataclass
class Episode:
    """One served episode and the artifacts the report reads."""

    feed: str
    episode_id: str
    run_ts: str
    metadata_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return str(self.metadata_path)[: -len(".metadata.json")]

    @property
    def run_dir(self) -> Path:
        return self.metadata_path.parent.parent

    def artifact(self, suffix: str) -> Optional[Dict[str, Any]]:
        """The episode's sibling artifact with ``suffix`` (``.gi.json``, ``.kg.json``), parsed."""
        return _load_json(Path(self.stem + suffix))

    def segments(self) -> Optional[List[Dict[str, Any]]]:
        """The raw segments the transcript route serves (first existing candidate)."""
        from podcast_scraper.server.app_content_source import transcript_relpath
        from podcast_scraper.server.segments_view import segments_relpaths_for_transcript

        content = self.metadata.get("content") or {}
        rel = transcript_relpath(content) if isinstance(content, dict) else None
        if not rel:
            return None
        for candidate in segments_relpaths_for_transcript(rel):
            payload = _load_json(self.run_dir / candidate)
            if payload is None:
                continue
            rows = payload if isinstance(payload, list) else payload.get("segments")
            return [r for r in rows or [] if isinstance(r, dict)]
        return None

    def roster(self) -> List[Dict[str, Any]]:
        """The ``content.speakers`` entries that carry a name."""
        speakers = (self.metadata.get("content") or {}).get("speakers") or []
        return [s for s in speakers if isinstance(s, dict) and (s.get("name") or "").strip()]


def _load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def served_episodes(root: Path) -> List[Episode]:
    """Newest run per ``(feed, episode_id)``, by the run directory's timestamp suffix."""
    newest: Dict[Tuple[str, str], Episode] = {}
    for md in sorted(root.glob("feeds/*/run_*/metadata/*.metadata.json")):
        feed = md.relative_to(root).parts[1]
        run = md.parent.parent.name
        m = _RUN_TS.search(run)
        ts = m.group(1) if m else ""
        meta = _load_json(md)
        if not isinstance(meta, dict):
            continue
        eid = (meta.get("episode") or {}).get("episode_id")
        if not eid:
            continue
        key = (feed, str(eid))
        if key not in newest or ts > newest[key].run_ts:
            newest[key] = Episode(feed, str(eid), ts, md, meta)
    return [newest[k] for k in sorted(newest)]


# ---- voice placement --------------------------------------------------------------------------


def segment_label_voices(segments: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    """``{speaker_label: [voice, ...]}`` from segment rows (m0025's ``_load_seg_voices``)."""
    out: Dict[str, List[str]] = {}
    for row in segments:
        voice, label = row.get("speaker"), row.get("speaker_label")
        if voice and label:
            have = out.setdefault(str(label), [])
            if str(voice) not in have:
                have.append(str(voice))
    return out


def entry_voices(entry: Dict[str, Any], label_voices: Dict[str, List[str]]) -> List[str]:
    """The voices a roster entry is placed on (m0025's ``_Episode._voices`` rule)."""
    if "voices" in entry:
        return [str(v) for v in entry.get("voices") or []]
    if entry.get("placed") is False:
        return []
    return list(label_voices.get(str(entry.get("name")), []))


# ---- per-episode measurement ------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """What the app's builders show for one episode, as the report counts it."""

    feed: str
    episode_id: str
    record: str  # "new" (placed/voices fields) or "old"
    roster_names: Set[str]
    voice_names: Set[str]
    insight_names_all: Counter
    insight_names_served: Counter
    raw_rows: List[Dict[str, Any]]
    kg_ids_by_name: Dict[str, Set[str]] = field(default_factory=dict)


def measure_episode(ep: Episode) -> EpisodeResult:
    """Run the app's builders on one episode and record what they show."""
    from podcast_scraper.server.app_gi_view import insights_from_gi
    from podcast_scraper.server.app_kg_view import entities_from_kg
    from podcast_scraper.server.segments_view import to_contract_segments

    roster = ep.roster()
    segments = ep.segments() or []
    label_voices = segment_label_voices(segments)
    record = "new" if any("voices" in s or "placed" in s for s in roster) else "old"
    roster_names = {s["name"].strip() for s in roster}
    voice_names = {s["name"].strip() for s in roster if entry_voices(s, label_voices)}

    raw_rows: List[Dict[str, Any]] = []
    gi = ep.artifact(".gi.json")
    all_c, served_c = _insight_names(gi) if isinstance(gi, dict) else (Counter(), Counter())
    if isinstance(gi, dict):
        for path, kwargs in (("default", {}), ("fallback", {"unattributed_fallback": True})):
            for ins in insights_from_gi(gi, **kwargs):
                for name in {q.speaker for q in ins.quotes or [] if q.speaker}:
                    kind = label_kind(name)
                    if kind in ("raw", "role"):
                        raw_rows.append(_row(ep, f"insights_{path}", name, kind, insight_id=ins.id))

    for seg in to_contract_segments(segments):
        kind = label_kind(seg.speaker)
        if kind in ("raw", "role"):
            raw_rows.append(_row(ep, "transcript_segments", seg.speaker or "", kind))

    kg = ep.artifact(".kg.json")
    kg_ids_by_name: Dict[str, Set[str]] = defaultdict(set)
    if isinstance(kg, dict):
        persons, _orgs, _topics = entities_from_kg(kg)
        for person in persons:
            if person.name:
                kg_ids_by_name[person.name.strip()].add(str(person.id))
            kind = label_kind(person.name)
            if kind in ("raw", "role"):
                raw_rows.append(
                    _row(ep, "episode_people_chips", person.name, kind, person_id=person.id)
                )

    return EpisodeResult(
        ep.feed,
        ep.episode_id,
        record,
        roster_names,
        voice_names,
        all_c,
        served_c,
        raw_rows,
        dict(kg_ids_by_name),
    )


def _row(ep: Episode, surface: str, name: str, kind: str, **extra: Any) -> Dict[str, Any]:
    return {
        "surface": surface,
        "kind": kind,
        "feed": ep.feed,
        "episode_id": ep.episode_id,
        "run_ts": ep.run_ts,
        "name": name,
        **extra,
    }


def _insight_names(gi: Dict[str, Any]) -> Tuple[Counter, Counter]:
    """Insights per resolved quote-speaker name: all, and served (the app's gates).

    The scan's own resolution (speaker_name, then the Person via speaker_id, then via SPOKEN_BY),
    kept so roster mode reproduces the scan; the served path is measured by ``insights_from_gi``.
    """
    nodes = gi.get("nodes") or []
    persons: Dict[Any, str] = {}
    for n in nodes:
        if isinstance(n, dict) and n.get("type") == "Person":
            p = n.get("properties") or {}
            name = (p.get("name") or p.get("display_name") or "").strip()
            if name:
                persons[n.get("id")] = name
    quotes = {
        n.get("id"): (n.get("properties") or {})
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "Quote"
    }
    supported: Dict[Any, List[Any]] = defaultdict(list)
    spoken: Dict[Any, Any] = {}
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "SUPPORTED_BY":
            supported[e.get("from")].append(e.get("to"))
        elif e.get("type") == "SPOKEN_BY":
            spoken[e.get("from")] = e.get("to")
    all_c: Counter = Counter()
    served_c: Counter = Counter()
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "Insight":
            continue
        p = n.get("properties") or {}
        names: Set[str] = set()
        for qid in supported.get(n.get("id"), []):
            qp = quotes.get(qid)
            if qp is None:
                continue
            sp = (
                (qp.get("speaker_name") or "").strip()
                or persons.get(qp.get("speaker_id"))
                or persons.get(spoken.get(qid))
            )
            if sp:
                names.add(sp)
        served = p.get("surfaceable") is not False and p.get("routing_tag") != "drop"
        for name in names:
            all_c[name] += 1
            if served:
                served_c[name] += 1
    return all_c, served_c


# ---- corpus-level measures --------------------------------------------------------------------


def candidate_pairs(
    results: List[EpisodeResult], mode: str
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, int]]]:
    """Spelling-variant pairs per feed for ``mode`` (``roster`` or ``voice``)."""
    stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "insights": 0, "insights_served": 0}
    )
    for r in results:
        names = r.roster_names if mode == "roster" else r.voice_names
        for name in names:
            st = stats[(r.feed, name)]
            st["episodes"] += 1
            st["insights"] += r.insight_names_all.get(name, 0)
            st["insights_served"] += r.insight_names_served.get(name, 0)
    by_feed: Dict[str, List[str]] = defaultdict(list)
    for feed, name in stats:
        by_feed[feed].append(name)
    pairs: List[Dict[str, Any]] = []
    for feed, feed_names in sorted(by_feed.items()):
        ordered = sorted(feed_names)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                rule = pair_rule(a, b)
                if rule is None:
                    continue
                sa, sb = stats[(feed, a)], stats[(feed, b)]
                pairs.append(
                    {
                        "feed": feed,
                        "rule": rule,
                        "name_1": a,
                        "name_2": b,
                        "episodes_1": sa["episodes"],
                        "episodes_2": sb["episodes"],
                        "insights_served_1": sa["insights_served"],
                        "insights_served_2": sb["insights_served"],
                    }
                )
    return pairs, stats


def groups_of(pairs: List[Dict[str, Any]], stats: Dict[Tuple[str, str], Dict[str, int]]):
    """Union pairs into same-feed groups; the non-winning spellings' served insights per group."""
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(x: Tuple[str, str]) -> Tuple[str, str]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        a, b = (p["feed"], p["name_1"]), (p["feed"], p["name_2"])
        parent[find(a)] = find(b)
    members: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for key in list(parent):
        members[find(key)].append(key)
    out = []
    for group in members.values():
        served = {name: stats[(feed, name)]["insights_served"] for feed, name in group}
        largest = max(served.values())
        out.append(
            {
                "feed": group[0][0],
                "names": sorted(served),
                "insights_served": served,
                "upper_bound_non_winning": sum(served.values()) - largest,
            }
        )
    return sorted(out, key=lambda g: -g["upper_bound_non_winning"])


def raw_label_totals(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Per surface: episodes and items showing a raw label (insights counted once per insight)."""
    out: Dict[str, Dict[str, int]] = {}
    by_surface: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_surface[f"{row['surface']}:{row['kind']}"].append(row)
    for surface, items in sorted(by_surface.items()):
        episodes = {(r["feed"], r["episode_id"]) for r in items}
        if surface.startswith("insights_"):
            n = len({(r["feed"], r["episode_id"], r.get("insight_id")) for r in items})
        else:
            n = len(items)
        out[surface] = {"episodes": len(episodes), "items": n}
    return out


def cross_show(
    stats: Dict[Tuple[str, str], Dict[str, int]],
    ids_by_feed_name: Dict[Tuple[str, str], Set[str]],
    id_map: Dict[str, str],
) -> Dict[str, Any]:
    """R4: one person across shows. Identical names on 2+ feeds, and rule-b variants across feeds.

    ``map_joined`` says whether today's identity map already puts every member under one
    canonical id. Either way these are candidates, not verdicts: identical names on two shows
    can be two people.
    """
    by_norm: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for feed, name in stats:
        norm = normalise(name)
        if norm:
            by_norm[norm].append((feed, name))

    def member(key: Tuple[str, str]) -> Dict[str, Any]:
        feed, name = key
        return {
            "feed": feed,
            "name": name,
            "episodes": stats[key]["episodes"],
            "insights_served": stats[key]["insights_served"],
            "canonicals": sorted({id_map.get(p, p) for p in ids_by_feed_name.get(key, ())}),
            "one_word": len(name.split()) == 1,
        }

    def joined(members: List[Dict[str, Any]]) -> bool:
        canon = [set(m["canonicals"]) for m in members]
        return all(canon) and len(set.union(*canon)) == 1

    identical = []
    for norm, keys in sorted(by_norm.items()):
        if len({feed for feed, _ in keys}) < 2:
            continue
        members = [member(k) for k in sorted(keys)]
        identical.append({"norm": norm, "members": members, "map_joined": joined(members)})

    norms = sorted((n for n in by_norm if len(n) >= _RULE_B_MIN_LEN), key=lambda n: (len(n), n))
    variants = []
    for i, a in enumerate(norms):
        for b in norms[i + 1 :]:
            if len(b) - len(a) > _RULE_B_MAX_EDITS:
                break
            if edit_distance(a, b) > _RULE_B_MAX_EDITS:
                continue
            feeds_a = {f for f, _ in by_norm[a]}
            feeds_b = {f for f, _ in by_norm[b]}
            if feeds_a == feeds_b and len(feeds_a) == 1:
                continue  # same single show: that is R2's within-show pair
            members = [member(k) for k in sorted(by_norm[a] + by_norm[b])]
            variants.append({"norms": [a, b], "members": members, "map_joined": joined(members)})
    return {
        "identical_groups": len(identical),
        "identical_map_joined": sum(g["map_joined"] for g in identical),
        "variant_pairs": len(variants),
        "variant_map_joined": sum(v["map_joined"] for v in variants),
        "identical": identical,
        "variants": variants,
    }


def run(root: Path) -> Dict[str, Any]:
    """The full report for a corpus root."""
    import podcast_scraper

    started = time.monotonic()
    episodes = served_episodes(root)
    results = [measure_episode(ep) for ep in episodes]
    raw_rows = [row for r in results for row in r.raw_rows]

    id_map = _identity_map(root)
    ids_by_feed_name: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for r in results:
        for name, ids in r.kg_ids_by_name.items():
            ids_by_feed_name[(r.feed, name)].update(ids)

    modes: Dict[str, Any] = {}
    r4: Dict[str, Any] = {}
    for mode in ("roster", "voice"):
        pairs, stats = candidate_pairs(results, mode)
        if mode == "voice":
            r4 = cross_show(stats, ids_by_feed_name, id_map)
        groups = groups_of(pairs, stats)
        for g in groups:
            g["map_canonicals"] = sorted(
                {
                    id_map.get(pid, pid)
                    for name in g["names"]
                    for pid in ids_by_feed_name.get((g["feed"], name), ())
                }
            )
        modes[mode] = {
            "distinct_names_feed_scoped": len(stats),
            "pairs_rule_a": sum(p["rule"] == "a" for p in pairs),
            "pairs_rule_b": sum(p["rule"] == "b" for p in pairs),
            "groups": len(groups),
            "upper_bound_non_winning_served": sum(g["upper_bound_non_winning"] for g in groups),
            "pairs": pairs,
            "group_rows": groups,
        }

    listed_not_heard = sum(len(r.roster_names - r.voice_names) for r in results)
    heard_not_listed = Counter(
        name for r in results for name in set(r.insight_names_all) - r.roster_names
    )
    return {
        "marker": {
            "naming_arc_marker": getattr(podcast_scraper, "NAMING_ARC_MARKER", "absent"),
            "package_file": podcast_scraper.__file__,
        },
        "corpus_root": str(root),
        "corpus_token": _corpus_token(root),
        "episodes": len(episodes),
        "served_runs": [[e.feed, e.episode_id, e.run_ts] for e in episodes],
        "feeds": len({e.feed for e in episodes}),
        "records": dict(Counter(r.record for r in results)),
        "identity_map": {"merged_variants": len(id_map)},
        "r1_raw_labels": raw_label_totals(raw_rows),
        "r1_rows": raw_rows,
        "r2_modes": modes,
        "r4_voice": r4,
        "r5": {
            "listed_not_heard_entries": listed_not_heard,
            "heard_not_listed_names": sum(heard_not_listed.values()),
            "heard_not_listed_top": heard_not_listed.most_common(30),
            "scoped_or_placeholder_chip_ids": sum(
                1
                for row in raw_rows
                if row["surface"] == "episode_people_chips"
                and is_scoped_placeholder_person_id(row.get("person_id"))
            ),
        },
        "seconds": round(time.monotonic() - started, 1),
    }


def _corpus_token(root: Path) -> Dict[str, Any]:
    """The corpus state the report read: ``corpus_mtime`` plus the newest-run kg count."""
    from podcast_scraper.kg.corpus import newest_run_artifact_paths, scan_kg_artifact_paths
    from podcast_scraper.perf_cache import corpus_mtime

    kg = newest_run_artifact_paths(root, scan_kg_artifact_paths(root), ".kg.json")
    return {"corpus_mtime": corpus_mtime(root), "newest_kg_count": len(kg)}


def _identity_map(root: Path) -> Dict[str, str]:
    """The variant -> canonical map the app serves today (the builder, not its cache)."""
    from podcast_scraper.kg.entity_clusters import build_entity_id_map

    return build_entity_id_map(root)


def _summary_line(report: Dict[str, Any]) -> str:
    r1 = " ".join(f"{s}={v['items']}/{v['episodes']}ep" for s, v in report["r1_raw_labels"].items())
    modes = " ".join(
        f"{m}:pairs={v['pairs_rule_a']}a+{v['pairs_rule_b']}b,groups={v['groups']},"
        f"ub={v['upper_bound_non_winning_served']}"
        for m, v in report["r2_modes"].items()
    )
    r4 = report.get("r4_voice") or {}
    modes += (
        f" R4 identical={r4.get('identical_groups')}(joined {r4.get('identical_map_joined')})"
        f" variants={r4.get('variant_pairs')}(joined {r4.get('variant_map_joined')})"
    )
    return (
        f"marker={report['marker']['naming_arc_marker']} "
        f"file={report['marker']['package_file']} episodes={report['episodes']} "
        f"records={report['records']} R1 {r1} R2 {modes} seconds={report['seconds']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m podcast_scraper.quality.name_report [corpus_root]`` (default /app/output)."""
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0] if args else "/app/output")
    if not root.is_dir():
        sys.stderr.write(f"corpus root not found: {root}\n")
        return 2
    report = run(root)
    json.dump(report, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stderr.write(_summary_line(report) + "\n")
    return 0


def iter_raw_rows(report: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """The R1 rows of a report, for callers that post-process."""
    yield from report.get("r1_rows") or []


if __name__ == "__main__":
    raise SystemExit(main())
