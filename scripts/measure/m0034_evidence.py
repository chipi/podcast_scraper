#!/usr/bin/env python3
"""Read-only evidence for corpus migration 0034 (the talk floor): acceptance criteria 2, 7, 12, 13.

    PYTHONPATH=src python scripts/measure/m0034_evidence.py \\
        --before <corpus before 0034> --after <copy of it with 0034 applied> --out evidence.json \\
        [--labels-rows rowsA.gz,rowsB.gz --labels-cases segment.txt] ...

Nothing is written into either corpus. The output holds person names: keep it out of the repo.

Definitions (docs/wip/changes/0034_talk_floor_names_removed.acceptance*.md):

* Voice key: (episode id, raw diarizer voice id). Talk share and the served name come from the
  episode's ``segments.json`` exactly as the migration reads them (``m0034.floor_changes``).
* L is the frozen list; the REMOVALS are the entries ``_reviewed()`` applies (verdict
  ``WRONG_PERSON``), read from the verdicts file the working tree holds.

Criterion 2 (counts on the BEFORE corpus):
  (a) named voices at or above the floor; under-floor voices already unnamed.
  (b) under-floor voices whose label is a placeholder ("SPEAKER_01", "Speaker 2", ...), and those
      with no label at all.
  (c) the episodes ingested after the snapshot are not in this corpus: printed as such.
  (d) HEURISTIC, since the corpus has no record of whom a name "belongs to": a removed voice
      counts when (1) no other voice of the episode that keeps its label carries the same
      normalised name (so the name is on no voice after 0034) and (2) the episode has a voice at or
      above the floor that is served unnamed. It over-counts when the episode simply has an
      unnamed minor voice and the removed name is a wrong one; read the list, not just the count.
  (e) episodes of L where one listed name sits on two or more under-floor voices, and episodes
      where a listed name is also held by a voice at or above the floor.

Criterion 12: every GI insight is compared before and after by id (properties ``surfaceable``,
``speaker_voice_type``, ``speaker``). An insight belongs to the voice of the raw segment holding its
first SUPPORTED_BY quote's ``char_start`` (the scoreboard's join, never the ``speaker`` field).
Insights on a removed voice are classed by their before-state and what became of them; any change
on an insight outside a removed voice (or with no placeable quote) is listed and must be 0. The
corpus-wide S1/S2 movement is reconciled with the gate report.

Criterion 13: a person "loses its last voice" when a removed voice's name is carried by no
remaining voice of the episode. Such persons are matched by normalised name or id slug on the GI
Person nodes, KG Person nodes and bridge person identities; each is listed with kept / removed
and the reason (edges that still reference it after). Persons that vanished without being
such a candidate are listed too.

Criterion 7: the 10-17 labelled measurement (eval repo ``autoresearch/naming_rethink/talk_floor.py
--served-corpus ... --show-removed``) is re-run here from its own inputs: each ``--labels-rows``
(comma-separated row files) is paired by position with a ``--labels-cases`` segment file. The
under-floor labelled names are joined to 0034's verdicts and compared.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.quality.corpus_scoreboard import _load, _Texts, _voice_of_quote  # noqa: E402
from podcast_scraper.upgrade.corpus_selection import select_served_artifacts  # noqa: E402
from podcast_scraper.upgrade.migrations import m0034_talk_floor_names_removed as m34  # noqa: E402
from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import (  # noqa: E402
    _run_relative,
)

GATE_DEFAULT = REPO / "docs" / "wip" / "changes" / "0034_talk_floor_names_removed.gate.json"
EVAL_REPO_DEFAULT = Path("/Users/claude/projects/podcast-scraper-eval-data-naming")
LABEL_CASES_SUBDIR = "data/eval/references/silver/naming_speakers_v1"
NAMED_OUTCOMES = ("correct_name", "role_error", "wrong_name", "non_participant", "spurious_name")
LABEL_KEEP = {"correct_name", "role_error"}
LABEL_REMOVE = {"wrong_name", "spurious_name", "non_participant"}
Key = Tuple[str, str]


def norm(text: Any) -> str:
    """Casefolded, accent-free, alphanumerics only: the comparison form of a name or id slug."""
    s = unicodedata.normalize("NFKD", str(text or "")).casefold()
    return re.sub(r"[^a-z0-9]+", "", s)


def is_placeholder(label: str) -> bool:
    """True when a non-empty label is one of the migration's placeholder shapes."""
    return bool(label) and bool(m34._LABEL.match(label))


class Episode:
    """One served episode of a corpus: where its files are and what its voices look like."""

    def __init__(self, meta_path: Path) -> None:
        self.meta_path = meta_path
        self.run = meta_path.parent.parent
        self.stem = meta_path.name[: -len(".metadata.json")]
        meta = _load(meta_path) or {}
        self.eid = str((meta.get("episode") or {}).get("episode_id") or "")
        rel = _run_relative((meta.get("content") or {}).get("transcript_file_path"), self.run)
        self.transcript_rel = rel
        self._voices: Optional[Dict[str, Dict[str, Any]]] = None

    def artifact(self, suffix: str) -> Any:
        return _load(self.run / "metadata" / f"{self.stem}.{suffix}.json")

    def voices(self) -> Dict[str, Dict[str, Any]]:
        """``{voice: {name, label, role, talk_s, share}}`` from the segments, as 0034 reads them."""
        if self._voices is None:
            out: Dict[str, Dict[str, Any]] = {}
            if self.transcript_rel.endswith(".txt"):
                rows = m34._segments(self.meta_path, self.transcript_rel)
                talk = m34.talk_by_voice(rows)
                served = m34.served_by_voice(rows)
                labels: Dict[str, str] = {}
                for r in rows:
                    v = r.get("speaker")
                    if v and str(v) not in labels:
                        labels[str(v)] = str(r.get("speaker_label") or "")
                total = sum(talk.values()) or 1.0
                for v, (name, role) in served.items():
                    out[v] = {
                        "name": name,
                        "label": labels.get(v, ""),
                        "role": role,
                        "talk_s": talk.get(v, 0.0),
                        "share": talk.get(v, 0.0) / total,
                    }
            self._voices = out
        return self._voices


def index_corpus(root: Path) -> Dict[str, Episode]:
    """``{episode id: Episode}`` over the served artifacts of ``root`` (read only)."""
    out: Dict[str, Episode] = {}
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        ep = Episode(meta_path)
        if ep.eid:
            out[ep.eid] = ep
    return out


def keys_str(keys: Iterable[Key]) -> List[str]:
    return [f"{e}|{v}" for e, v in sorted(keys)]


# --------------------------------------------------------------------------- criterion 2


def criterion2(
    before: Dict[str, Episode], listed: Dict[Key, Any], removed: Dict[Key, Any]
) -> Dict[str, Any]:
    floor = m34.FLOOR
    named_ge = under_unnamed = under_named = placeholder = no_label = 0
    ep_named_ge: Set[str] = set()
    under_by_ep: Dict[str, List[str]] = collections.defaultdict(list)
    for eid, ep in before.items():
        for v, d in ep.voices().items():
            if d["share"] >= floor:
                if d["name"]:
                    named_ge += 1
                    ep_named_ge.add(eid)
            else:
                under_by_ep[eid].append(v)
                if d["name"]:
                    under_named += 1
                else:
                    under_unnamed += 1
                    if is_placeholder(d["label"]):
                        placeholder += 1
                    elif not d["label"]:
                        no_label += 1

    def heuristic_d(pool: Dict[Key, Any]) -> Dict[str, Any]:
        hits: List[Key] = []
        eps: Set[str] = set()
        for (eid, voice), entry in pool.items():
            ep = before.get(eid)
            if ep is None:
                continue
            n = norm(entry[0])
            vs = ep.voices()
            on_other = any(
                u != voice and (eid, u) not in removed and norm(d["name"]) == n
                for u, d in vs.items()
            )
            unnamed_ge = any(d["share"] >= floor and not d["name"] for d in vs.values())
            if not on_other and unnamed_ge:
                hits.append((eid, voice))
                eps.add(eid)
        return {
            "voices": len(hits),
            "episodes": len(eps),
            "keys": keys_str(hits),
            "of": len(pool),
        }

    def heuristic_e(pool: Dict[Key, Any]) -> Dict[str, Any]:
        by_ep: Dict[str, List[Key]] = collections.defaultdict(list)
        for k in pool:
            by_ep[k[0]].append(k)
        dup: List[Key] = []
        dup_eps: Set[str] = set()
        held: List[Key] = []
        held_eps: Set[str] = set()
        for eid, ks in by_ep.items():
            ep = before.get(eid)
            if ep is None:
                continue
            counts = collections.Counter(norm(pool[k][0]) for k in ks)
            for k in ks:
                if counts[norm(pool[k][0])] > 1:
                    dup.append(k)
                    dup_eps.add(eid)
                n = norm(pool[k][0])
                if any(
                    u != k[1] and d["share"] >= floor and norm(d["name"]) == n
                    for u, d in ep.voices().items()
                ):
                    held.append(k)
                    held_eps.add(eid)
        return {
            "same_name_on_two_under_floor_voices": {
                "voices": len(dup),
                "episodes": len(dup_eps),
                "keys": keys_str(dup),
            },
            "name_also_held_at_or_above_floor": {
                "voices": len(held),
                "episodes": len(held_eps),
                "keys": keys_str(held),
            },
        }

    return {
        "floor": floor,
        "episodes_in_before_corpus": len(before),
        "a_named_voices_at_or_above_floor": named_ge,
        "a_named_voices_at_or_above_floor_episodes": len(ep_named_ge),
        "a_under_floor_voices_named": under_named,
        "a_under_floor_voices_already_unnamed": under_unnamed,
        "b_placeholder_labelled_under_floor_voices": placeholder,
        "b_under_floor_voices_with_no_label": no_label,
        "c_post_snapshot_episodes": "not in snapshot",
        "d_removed_name_real_voice_unnamed": heuristic_d(removed),
        "d_all_listed_name_real_voice_unnamed": heuristic_d(listed),
        "e_listed": heuristic_e(listed),
        "e_removed": heuristic_e(removed),
    }


# --------------------------------------------------------------------------- criterion 12


def insight_state(props: Dict[str, Any]) -> Tuple[str, str]:
    """``(visibility state, speaker)``: shown, or hidden_<voice type>."""
    vt = props.get("speaker_voice_type") or "none"
    state = "shown" if props.get("surfaceable") is not False else f"hidden_{vt}"
    return state, str(props.get("speaker") or "")


def insights_of(gi: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in (gi or {}).get("nodes") or []:
        if isinstance(n, dict) and n.get("type") == "Insight":
            out[str(n.get("id"))] = n.get("properties") or {}
    return out


def place_quote(quote: Dict[str, Any], texts: _Texts) -> Tuple[Optional[str], str]:
    """``(voice, how)``: the scoreboard's join on ``char_start``; failing that, on the quote's last
    character. A quote that starts inside a turn's ``Name: `` prefix has its ``char_start`` in the
    gap between two segments, so the strict join finds no voice for it."""
    voice = _voice_of_quote(quote, texts)
    if voice:
        return voice, "strict"
    props = quote.get("properties") or {}
    end, start = props.get("char_end"), props.get("char_start")
    if isinstance(end, int) and isinstance(start, int) and end > start:
        voice = _voice_of_quote({"properties": {**props, "char_start": end - 1}}, texts)
        if voice:
            return voice, "char_end fallback"
    return None, "unplaced"


def insight_voices(ep: Episode, gi: Any) -> Dict[str, Tuple[Optional[str], str]]:
    """``{insight id: (raw voice of its first supported-by quote, how it was placed)}``."""
    nodes = {str(n.get("id")): n for n in (gi or {}).get("nodes") or [] if isinstance(n, dict)}
    first: Dict[str, str] = {}
    for e in (gi or {}).get("edges") or []:
        if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
            first.setdefault(str(e.get("from")), str(e.get("to")))
    texts = _Texts(ep.run / "transcripts")
    out: Dict[str, Tuple[Optional[str], str]] = {}
    for iid in insights_of(gi):
        q = nodes.get(first.get(iid, ""))
        out[iid] = place_quote(q, texts) if q else (None, "no quote")
    return out


CLASSES = {
    ("shown", "hidden_unknown"): "Iw_vis",
    ("hidden_unidentified", "hidden_unknown"): "Iw_unid",
    ("hidden_unknown", "hidden_unknown"): "Iw_unk",
    ("hidden_commercial", "hidden_commercial"): "Iw_com",
    ("shown", "shown"): "shown_stays_shown",
}


def criterion12(
    before: Dict[str, Episode],
    after: Dict[str, Episode],
    removed: Dict[Key, Any],
    gate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    removed_eps = {k[0] for k in removed}
    per_voice: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    totals: collections.Counter = collections.Counter()
    transitions: collections.Counter = collections.Counter()
    speaker_changed_on_removed = 0
    outside: List[Dict[str, str]] = []
    unplaced_changed: List[Dict[str, str]] = []
    missing_after: List[str] = []
    added_after: List[str] = []
    episodes_without_after = 0
    corpus = {"before": collections.Counter(), "after": collections.Counter()}
    removed_insights_total = 0
    placed_by_fallback: List[str] = []
    stays_shown: List[Dict[str, Any]] = []

    for eid, ep in before.items():
        ep_after = after.get(eid)
        gi_b = ep.artifact("gi")
        gi_a = ep_after.artifact("gi") if ep_after else None
        if ep_after is None:
            episodes_without_after += 1
        ins_b, ins_a = insights_of(gi_b), insights_of(gi_a)
        for side, ins in (("before", ins_b), ("after", ins_a)):
            for props in ins.values():
                state, _ = insight_state(props)
                corpus[side]["insights"] += 1
                if state != "shown":
                    corpus[side]["S2_hidden"] += 1
                    corpus[side][f"S2_hidden_{state[len('hidden_'):]}"] += 1
                    if state == "hidden_unknown":
                        corpus[side]["S1_hidden_unknown"] += 1
        missing_after += [f"{eid}|{i}" for i in ins_b if i not in ins_a]
        added_after += [f"{eid}|{i}" for i in ins_a if i not in ins_b]
        voices = insight_voices(ep, gi_b) if eid in removed_eps else {}
        for iid, pb in ins_b.items():
            if iid not in ins_a:
                continue
            sb, spk_b = insight_state(pb)
            sa, spk_a = insight_state(ins_a[iid])
            changed = (sb, spk_b) != (sa, spk_a)
            voice, how = voices.get(iid, (None, "episode has no removed voice"))
            is_removed_voice = voice is not None and (eid, voice) in removed
            if is_removed_voice:
                removed_insights_total += 1
                cls = CLASSES.get((sb, sa), f"other:{sb}->{sa}")
                transitions[f"{sb}->{sa}"] += 1
                per_voice[f"{eid}|{voice}"][cls].append(iid)
                totals[cls] += 1
                if how != "strict":
                    placed_by_fallback.append(iid)
                if cls == "shown_stays_shown":
                    kept_by = [
                        u
                        for u, d in ep.voices().items()
                        if u != voice and (eid, u) not in removed and norm(d["name"]) == norm(spk_b)
                    ]
                    stays_shown.append(
                        {"episode": eid, "insight": iid, "voice": voice, "name_kept_by": kept_by}
                    )
                if spk_b != spk_a:
                    speaker_changed_on_removed += 1
            elif changed:
                row = {"episode": eid, "insight": iid, "transition": f"{sb}->{sa}"}
                row["voice"] = voice or ""
                row["placed"] = how
                (unplaced_changed if eid in removed_eps and voice is None else outside).append(row)

    deltas = {
        k: corpus["after"][k] - corpus["before"][k]
        for k in sorted(set(corpus["before"]) | set(corpus["after"]))
    }
    predicted = {
        "S1_hidden_unknown": totals["Iw_vis"] + totals["Iw_unid"],
        "S2_hidden": totals["Iw_vis"],
        "S2_hidden_unknown": totals["Iw_vis"] + totals["Iw_unid"],
        "S2_hidden_unidentified": -totals["Iw_unid"],
        "S2_hidden_commercial": 0,
    }
    reconciliation: Dict[str, Any] = {
        "predicted_from_removed_voices": predicted,
        "measured_corpus_wide_delta": {k: deltas.get(k, 0) for k in predicted},
        "predicted_equals_measured": all(deltas.get(k, 0) == v for k, v in predicted.items()),
    }
    if gate:
        sb, sa = gate.get("scoreboard_before") or {}, gate.get("scoreboard_after") or {}
        gate_delta = {k: int(sa.get(k, 0)) - int(sb.get(k, 0)) for k in predicted}
        reconciliation["gate_report_delta"] = gate_delta
        reconciliation["gate_gating_moves"] = gate.get("gating_moves")
        reconciliation["gate_equals_measured"] = all(
            deltas.get(k, 0) == v for k, v in gate_delta.items()
        )
        reconciliation["gate_equals_predicted"] = gate_delta == predicted

    return {
        "removed_voices": len(removed),
        "removed_voices_with_insights": len(per_voice),
        "insights_on_removed_voices": removed_insights_total,
        "class_totals": dict(sorted(totals.items())),
        "transitions_on_removed_voices": dict(sorted(transitions.items())),
        "speaker_property_changed_on_removed_voices": speaker_changed_on_removed,
        "insights_placed_by_char_end_fallback": {
            "count": len(placed_by_fallback),
            "ids": placed_by_fallback,
        },
        "shown_insights_that_stay_shown": {"count": len(stays_shown), "rows": stays_shown},
        "reconciliation": reconciliation,
        "insights_changed_outside_removed_voices": {"count": len(outside), "rows": outside},
        "insights_changed_unplaced_in_removed_episodes": {
            "count": len(unplaced_changed),
            "rows": unplaced_changed,
        },
        "insights_missing_after": {"count": len(missing_after), "ids": missing_after},
        "insights_added_after": {"count": len(added_after), "ids": added_after},
        "episodes_without_after": episodes_without_after,
        "per_voice": {k: {c: ids for c, ids in v.items()} for k, v in sorted(per_voice.items())},
    }


# --------------------------------------------------------------------------- criterion 13


def person_nodes(doc: Any, kind: str) -> Dict[str, Dict[str, Any]]:
    """``{id: {names}}`` of the person entries of a GI / KG / bridge document."""
    out: Dict[str, Dict[str, Any]] = {}
    if kind == "bridge":
        for ident in (doc or {}).get("identities") or []:
            if isinstance(ident, dict) and ident.get("type") == "person":
                names = [ident.get("display_name")] + list(ident.get("aliases") or [])
                out[str(ident.get("id"))] = {"names": [n for n in names if n]}
        return out
    for n in (doc or {}).get("nodes") or []:
        if isinstance(n, dict) and str(n.get("type")).lower() == "person":
            props = n.get("properties") or {}
            out[str(n.get("id"))] = {
                "names": [x for x in (props.get("name"), props.get("label")) if x]
            }
    return out


def refs_after(doc: Any, pid: str) -> Dict[str, int]:
    """Edge types of ``doc`` that still reference ``pid`` (either end)."""
    c: collections.Counter = collections.Counter()
    for e in (doc or {}).get("edges") or []:
        if isinstance(e, dict) and pid in (str(e.get("from")), str(e.get("to"))):
            c[str(e.get("type"))] += 1
    return dict(c)


def matches(pid: str, info: Dict[str, Any], n: str) -> bool:
    slug = norm(pid.split(":", 1)[-1])
    return n in {norm(x) for x in info["names"]} or slug == n


def criterion13(
    before: Dict[str, Episode], after: Dict[str, Episode], removed: Dict[Key, Any]
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    no_node: List[Key] = []
    held_elsewhere: List[Key] = []
    candidate_ids: Set[Tuple[str, str, str]] = set()
    by_ep: Dict[str, List[Key]] = collections.defaultdict(list)
    for k in removed:
        by_ep[k[0]].append(k)

    for eid, keys in sorted(by_ep.items()):
        ep, ep_a = before.get(eid), after.get(eid)
        if ep is None:
            continue
        docs_b = {s: ep.artifact(s) for s in ("gi", "kg", "bridge")}
        docs_a = {s: (ep_a.artifact(s) if ep_a else None) for s in ("gi", "kg", "bridge")}
        for key in keys:
            n = norm(removed[key][0])
            holders = [
                u
                for u, d in ep.voices().items()
                if u != key[1] and (eid, u) not in removed and norm(d["name"]) == n
            ]
            if holders:
                held_elsewhere.append(key)
                continue
            found = False
            for surface in ("gi", "kg", "bridge"):
                pb = person_nodes(docs_b[surface], surface)
                pa = person_nodes(docs_a[surface], surface)
                for pid, info in pb.items():
                    if not matches(pid, info, n):
                        continue
                    found = True
                    candidate_ids.add((eid, surface, pid))
                    still = pid in pa
                    refs = refs_after(docs_a[surface], pid) if still else {}
                    if not still:
                        reason = "removed"
                    elif surface == "bridge":
                        src = next(
                            (
                                i.get("sources")
                                for i in (docs_a[surface] or {}).get("identities") or []
                                if str(i.get("id")) == pid
                            ),
                            None,
                        )
                        reason = f"kept: bridge sources {src}"
                    elif refs.get("MENTIONS_PERSON"):
                        reason = (
                            f"kept: still mentioned (MENTIONS_PERSON x{refs['MENTIONS_PERSON']})"
                        )
                    elif surface == "gi" and refs.get("SPOKEN_BY"):
                        reason = f"kept: held by another voice (SPOKEN_BY x{refs['SPOKEN_BY']})"
                    elif refs:
                        reason = f"kept: referenced by {refs}"
                    else:
                        reason = "kept: no edge references it (no stated rule applies)"
                    rows.append(
                        {
                            "episode_id": eid,
                            "person_id": pid,
                            "surface": surface,
                            "removed_voice": key[1],
                            "name": removed[key][0],
                            "outcome": "removed" if not still else "kept",
                            "reason": reason,
                            "edges_after": refs,
                        }
                    )
            if not found:
                no_node.append(key)

    unexpected: List[Dict[str, Any]] = []
    for eid, ep in before.items():
        ep_a = after.get(eid)
        for surface in ("gi", "kg", "bridge"):
            pb = person_nodes(ep.artifact(surface), surface)
            pa = person_nodes(ep_a.artifact(surface), surface) if ep_a else {}
            for pid in pb:
                if pid not in pa and (eid, surface, pid) not in candidate_ids:
                    names = {norm(x) for x in pb[pid]["names"]}
                    elsewhere = sorted(
                        other
                        for other in ("gi", "kg", "bridge")
                        if other != surface
                        and any(
                            matches(q, info, n)
                            for q, info in (
                                person_nodes(ep_a.artifact(other), other) if ep_a else {}
                            ).items()
                            for n in names
                        )
                    )
                    unexpected.append(
                        {
                            "episode_id": eid,
                            "surface": surface,
                            "person_id": pid,
                            "names": pb[pid]["names"],
                            "episode_has_removed_voice": eid in by_ep,
                            "same_person_still_on_surfaces_after": elsewhere,
                        }
                    )

    per_surface: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        per_surface[r["surface"]][r["outcome"]] += 1
    kept_reasons = collections.Counter(
        re.sub(r"x\d+|\{.*\}", "", r["reason"]) for r in rows if r["outcome"] == "kept"
    )
    return {
        "removed_voices": len(removed),
        "removed_voices_whose_name_is_held_by_another_voice": {
            "count": len(held_elsewhere),
            "keys": keys_str(held_elsewhere),
        },
        "removed_voices_with_no_person_node_on_any_surface": {
            "count": len(no_node),
            "keys": keys_str(no_node),
        },
        "persons_losing_last_voice": len(rows),
        "by_surface": {s: dict(c) for s, c in sorted(per_surface.items())},
        "kept_reasons": dict(kept_reasons),
        "persons_vanished_that_were_not_candidates": {
            "count": len(unexpected),
            "rows": unexpected,
        },
        "rows": rows,
    }


# --------------------------------------------------------------------------- criterion 7


def read_rows(paths: str) -> Dict[str, List[Dict[str, Any]]]:
    """Label rows by case id; a later file replaces an earlier one's rows for the same case."""
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths.split(","):
        cur: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        with gzip.open(p, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                cur[r["case"]].append(r)
        rows.update(cur)
    return rows


def case_files(eval_repo: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for f in sorted((eval_repo / LABEL_CASES_SUBDIR).glob("*/cases/*.json")):
        c = json.loads(f.read_text())
        out[str(c["case_id"])].append(c)
    return out


def same_person(a: str, b: str) -> bool:
    ta, tb = set(re.findall(r"[a-z0-9]+", norm_words(a))), set(
        re.findall(r"[a-z0-9]+", norm_words(b))
    )
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def norm_words(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).casefold()


def criterion7(
    root: Path,
    recomputed: Dict[Key, Any],
    before: Dict[str, Episode],
    listed: Dict[Key, Any],
    verdicts: Any,
    segments: Sequence[Tuple[str, str]],
    eval_repo: Path,
) -> Dict[str, Any]:
    floor = m34.FLOOR
    cases_by_id = case_files(eval_repo)
    joined: List[Dict[str, Any]] = []
    unjoinable: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    labelled_under_floor = 0
    seen: Set[Tuple[str, str]] = set()
    for rows_arg, cases_arg in segments:
        case_ids = {
            ln.split()[1] for ln in open(cases_arg) if ln.strip() and not ln.startswith("#")
        }
        rows = read_rows(rows_arg)
        for cid in sorted(case_ids):
            if cid not in rows or cid not in cases_by_id:
                continue
            for case in cases_by_id[cid]:
                meta_path = root / case["meta_relpath"]
                seg_path = root / case["meta_relpath"].replace(
                    "/metadata/", "/transcripts/"
                ).replace(".metadata.json", ".segments.json")
                if not seg_path.is_file():
                    continue
                segs = json.load(open(seg_path))
                talk: Dict[str, float] = collections.defaultdict(float)
                for s in segs:
                    if s.get("speaker"):
                        talk[s["speaker"]] += max(
                            0.0, float(s.get("end") or 0) - float(s.get("start") or 0)
                        )
                total = sum(talk.values()) or 1.0
                hosts = sorted(
                    {
                        r.get("label_name")
                        for r in rows[cid]
                        if r.get("label_role") == "host" and r.get("label_name")
                    }
                )
                meta = _load(meta_path) or {}
                eid = str((meta.get("episode") or {}).get("episode_id") or "")
                for r in rows[cid]:
                    out = r["before"]
                    voice = r["voice"]
                    if out not in NAMED_OUTCOMES or talk.get(voice, 0.0) / total >= floor:
                        continue
                    if (cid, voice) in seen:
                        continue
                    seen.add((cid, voice))
                    labelled_under_floor += 1
                    served = r.get("served_before") or ""
                    host_read = out == "non_participant" and any(
                        same_person(served, h) for h in hosts
                    )
                    label_action = (
                        "keep"
                        if out in LABEL_KEEP or host_read
                        else "remove" if out in LABEL_REMOVE else "?"
                    )
                    base = {
                        "case": cid,
                        "episode_id": eid,
                        "voice": voice,
                        "label_outcome": out,
                        "host_read_ad": host_read,
                        "label_action": label_action,
                        "served_before_name": served,
                        "label_name": r.get("label_name"),
                    }
                    key = (eid, voice)
                    cause = None
                    if not eid or eid not in before:
                        cause = "episode not in the before corpus"
                    elif key not in listed:
                        d = before[eid].voices().get(voice)
                        if d is None:
                            cause = "voice absent from the episode's segments on 10-25"
                        elif not d["name"]:
                            cause = "voice unnamed on 10-25 (name already gone)"
                        elif d["share"] >= floor:
                            cause = f"share {d['share']:.1%} at or above the floor on 10-25"
                        elif key in recomputed:
                            cause = "under the floor and named, but dropped from the frozen list"
                        else:
                            cause = "under floor and named but not in the list"
                        # name-based fallback: the same name on another listed voice of the episode
                        alt = [
                            k
                            for k, e in listed.items()
                            if k[0] == eid and norm(e[0]) == norm(served) and norm(served)
                        ]
                        if alt:
                            key = alt[0]
                            cause = None
                            base["joined_by"] = "episode id + name"
                    v = verdicts.get(*key) if cause is None else None
                    if cause is None and v is None:
                        cause = "listed voice without a verdict"
                    if cause is not None:
                        unjoinable.append({**base, "cause": cause})
                        continue
                    base["key"] = f"{key[0]}|{key[1]}"
                    base["listed_name"] = listed[key][0]
                    base["verdict"] = v.verdict
                    base["basis"] = v.basis
                    base["action"] = "remove" if v.verdict == "WRONG_PERSON" else "keep"
                    base.setdefault("joined_by", "episode id + voice")
                    joined.append(base)
                    if base["action"] != label_action:
                        disagreements.append(base)

    matrix: collections.Counter = collections.Counter(
        (r["label_action"], r["action"]) for r in joined
    )
    return {
        "labelled_under_floor_names": labelled_under_floor,
        "joined": len(joined),
        "unjoinable": {"count": len(unjoinable), "rows": unjoinable},
        "label_outcome_counts": dict(collections.Counter(r["label_outcome"] for r in joined)),
        "host_read_ads_among_joined": sum(1 for r in joined if r["host_read_ad"]),
        "label_action_vs_verdict": {
            f"label_{a}__verdict_{b}": n for (a, b), n in sorted(matrix.items())
        },
        "disagreements": {"count": len(disagreements), "rows": disagreements},
        "joined_rows": joined,
    }


# --------------------------------------------------------------------------- main


def summary(ev: Dict[str, Any]) -> List[str]:
    c2, c12, c13 = ev["criterion_2"], ev["criterion_12"], ev["criterion_13"]
    rec = c12["reconciliation"]
    lines = [
        f"inputs: list {ev['list_entries']} entries, removals {ev['removals']}, "
        f"verdicts sha256 {ev['verdicts_sha256'][:12]}",
        f"  frozen list vs the floor recomputed on the before corpus: "
        f"{ev['list_vs_recomputed']['only_in_list']} only in list, "
        f"{ev['list_vs_recomputed']['only_recomputed']} only recomputed",
        "criterion 2",
        f"  (a) named voices >= floor {c2['a_named_voices_at_or_above_floor']} "
        f"in {c2['a_named_voices_at_or_above_floor_episodes']} episodes; "
        f"under-floor named {c2['a_under_floor_voices_named']}, "
        f"under-floor already unnamed {c2['a_under_floor_voices_already_unnamed']}",
        f"  (b) placeholder-labelled under-floor "
        f"{c2['b_placeholder_labelled_under_floor_voices']}, "
        f"under-floor with no label {c2['b_under_floor_voices_with_no_label']}",
        f"  (c) {c2['c_post_snapshot_episodes']}",
        f"  (d) removals with real voice unnamed: "
        f"{c2['d_removed_name_real_voice_unnamed']['voices']} "
        f"voices in {c2['d_removed_name_real_voice_unnamed']['episodes']} episodes "
        f"(of {c2['d_removed_name_real_voice_unnamed']['of']}); all listed: "
        f"{c2['d_all_listed_name_real_voice_unnamed']['voices']} voices in "
        f"{c2['d_all_listed_name_real_voice_unnamed']['episodes']} episodes "
        f"(of {c2['d_all_listed_name_real_voice_unnamed']['of']})",
    ]
    for label in ("e_listed", "e_removed"):
        e = c2[label]
        a, b = e["same_name_on_two_under_floor_voices"], e["name_also_held_at_or_above_floor"]
        lines.append(
            f"  (e, {label[2:]}) one name on two under-floor voices: {a['voices']} voices / "
            f"{a['episodes']} episodes; name also held at/above floor: {b['voices']} voices / "
            f"{b['episodes']} episodes"
        )
    lines += [
        "criterion 12",
        f"  insights on removed voices {c12['insights_on_removed_voices']} "
        f"(in {c12['removed_voices_with_insights']} of {c12['removed_voices']} voices): "
        + ", ".join(f"{k} {v}" for k, v in c12["class_totals"].items()),
        f"  placed by the char_end fallback "
        f"{c12['insights_placed_by_char_end_fallback']['count']}; shown insights that stay shown "
        f"{c12['shown_insights_that_stay_shown']['count']}",
        f"  predicted corpus delta {rec['predicted_from_removed_voices']}",
        f"  measured corpus delta  {rec['measured_corpus_wide_delta']} "
        f"(predicted == measured: {rec['predicted_equals_measured']})",
    ]
    if "gate_report_delta" in rec:
        lines.append(
            f"  gate report delta      {rec['gate_report_delta']} "
            f"(gate == measured: {rec['gate_equals_measured']}, gate == predicted: "
            f"{rec['gate_equals_predicted']})"
        )
    lines += [
        f"  insights changed outside removed voices: "
        f"{c12['insights_changed_outside_removed_voices']['count']}; "
        f"unplaced in removed episodes: "
        f"{c12['insights_changed_unplaced_in_removed_episodes']['count']}; "
        f"missing after: {c12['insights_missing_after']['count']}; "
        f"added after: {c12['insights_added_after']['count']}",
        "criterion 13",
        f"  persons losing their last voice {c13['persons_losing_last_voice']} "
        f"by surface {c13['by_surface']}; removed voices whose name another voice holds "
        f"{c13['removed_voices_whose_name_is_held_by_another_voice']['count']}; "
        f"with no person node {c13['removed_voices_with_no_person_node_on_any_surface']['count']}; "
        f"vanished non-candidates {c13['persons_vanished_that_were_not_candidates']['count']}",
        f"  kept reasons {c13['kept_reasons']}",
    ]
    c7 = ev.get("criterion_7")
    if isinstance(c7, dict) and "joined" in c7:
        lines += [
            "criterion 7",
            f"  labelled under-floor names {c7['labelled_under_floor_names']}, "
            f"joined {c7['joined']}, "
            f"unjoinable {c7['unjoinable']['count']}, "
            f"host-read ads {c7['host_read_ads_among_joined']}",
            f"  label outcomes {c7['label_outcome_counts']}",
            f"  label action vs verdict {c7['label_action_vs_verdict']}",
            f"  disagreements {c7['disagreements']['count']}",
        ]
        for r in c7["disagreements"]["rows"]:
            lines.append(
                f"    DISAGREE case {r['case']} {r['key']} label {r['label_outcome']}"
                f"{' (host-read ad)' if r['host_read_ad'] else ''} -> {r['label_action']}, "
                f"verdict {r['verdict']}/{r['basis']} -> {r['action']}"
            )
        for r in c7["unjoinable"]["rows"]:
            lines.append(
                f"    NOT JOINABLE case {r['case']} voice {r['voice']} "
                f"label {r['label_outcome']}: {r['cause']}"
            )
    else:
        lines += ["criterion 7", f"  {c7}"]
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--before", type=Path, required=True, help="corpus before 0034")
    ap.add_argument("--after", type=Path, required=True, help="the same corpus with 0034 applied")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate", type=Path, default=GATE_DEFAULT, help="the change gate's report")
    ap.add_argument(
        "--labels-rows",
        action="append",
        default=[],
        help="comma-separated label row files (.jsonl.gz); repeat, paired by position with "
        "--labels-cases",
    )
    ap.add_argument(
        "--labels-cases", action="append", default=[], help="segment file listing the case ids"
    )
    ap.add_argument("--eval-repo", type=Path, default=EVAL_REPO_DEFAULT)
    a = ap.parse_args(argv)
    if len(a.labels_rows) != len(a.labels_cases):
        ap.error("--labels-rows and --labels-cases must be given the same number of times")

    mig = m34.TalkFloorNamesRemovedMigration()
    removed, missing = mig._reviewed()
    listed = m34.load_list()
    verdicts = mig._verdicts()
    before, after = index_corpus(a.before), index_corpus(a.after)

    recomputed: Dict[Key, Any] = {}
    for eid, ep in before.items():
        if ep.transcript_rel.endswith(".txt"):
            rows = m34._segments(ep.meta_path, ep.transcript_rel)
            for voice, entry in m34.floor_changes(rows).items():
                recomputed[(eid, voice)] = entry
    gate = json.loads(a.gate.read_text()) if a.gate.is_file() else None

    ev: Dict[str, Any] = {
        "before": str(a.before),
        "after": str(a.after),
        "list_entries": len(listed),
        "removals": len(removed),
        "listed_without_verdict": len(missing),
        "verdicts_sha256": verdicts.sha256,
        "gate_verdicts_sha256": (gate or {}).get("verdicts_sha256"),
        "verdicts_match_gate": bool(gate) and gate.get("verdicts_sha256") == verdicts.sha256,
        "list_vs_recomputed": {
            "recomputed": len(recomputed),
            "only_in_list": len(set(listed) - set(recomputed)),
            "only_recomputed": len(set(recomputed) - set(listed)),
            "only_recomputed_keys": keys_str(set(recomputed) - set(listed)),
        },
        "episodes_before": len(before),
        "episodes_after": len(after),
        "criterion_2": criterion2(before, listed, removed),
        "criterion_12": criterion12(before, after, removed, gate),
        "criterion_13": criterion13(before, after, removed),
    }
    if a.labels_rows:
        ev["criterion_7"] = criterion7(
            a.before,
            recomputed,
            before,
            listed,
            verdicts,
            list(zip(a.labels_rows, a.labels_cases)),
            a.eval_repo,
        )
    else:
        ev["criterion_7"] = "not run: no --labels-rows/--labels-cases given"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(ev, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(summary(ev)))
    print(f"verdicts file matches the gate report: {ev['verdicts_match_gate']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
