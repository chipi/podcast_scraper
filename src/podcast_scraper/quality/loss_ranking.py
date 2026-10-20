"""The loss ranking (INGESTION-ARC Stage 2, 2.2 and 2.3): why names are wrong, why insights hide.

Read-only. Two populations, each counted in full and classed once, by definitions fixed in the
change record ``docs/wip/changes/loss_ranking.md`` (step 5, A1–A11):

- **Gold, wrong names.** Every ``S7_wrong`` row the committed S7 code (``gold_insights``) yields,
  classed by how the stored name relates to the episode's labels: ``binding`` (another labelled
  voice in the episode owns the name; sub-classes ``swapped``, ``duplicated``, ``rotated``),
  ``spelling`` (the labelled person, spelled as the matcher allows per token) or ``other_person``.
  Wrong names outside ``S7_wrong`` (a name where the label supports none, an undecidable or an
  unplaceable quote under a real name) are counted beside them.
- **Corpus, hidden insights.** Every S1 insight (``surfaceable`` false, voice type ``unknown``),
  classed by what the episode's own record states: nobody named, a stated person ``placed:
  false`` (with the insight on a principal voice or not, and the one-to-one case), every stated
  person placed, or a record that does not say who was placed.
- **The seat check.** On the gold, the one-to-one case's seat (the one unplaced stated person on
  the one unnamed principal voice) against the label. Elimination alone is not a fix: this is its
  measured precision.

Conservation is asserted: the classes are exactly S7's ``S7_wrong`` ids and the scoreboard's S1
nodes, each once. Gold-derived text (names) goes only to the caller's output files, never to the
record.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..kg.speaker_coherence import _drop_honorifics, _fold, same_person
from . import corpus_scoreboard as cs
from .gold_insights import _RAW

#: A voice holding at least this share of the talk is principal (the S3 denominator).
PRINCIPAL_SHARE = cs.PRINCIPAL_SHARE
GOLD_SETS = ("dev_v1", "val_v1", "val_v2")


def _real(name: Any) -> bool:
    return isinstance(name, str) and bool(name.strip()) and not _RAW.match(name.strip())


def _tokens(name: str) -> List[str]:
    return _drop_honorifics(_fold(name)).split()


def is_spelling(label: str, stored: str) -> bool:
    """The stored name is the labelled person as the matcher allows it token by token: no more
    tokens than the label, each paired with a different label token, equal or ``same_person`` at
    its own threshold. No thresholds of its own."""
    lt, st = _tokens(label), _tokens(stored)
    if not st or len(st) > len(lt):
        return False

    def pair(i: int, used: Set[int]) -> bool:
        if i == len(st):
            return True
        for j, tok in enumerate(lt):
            if j not in used and (tok == st[i] or same_person(tok, st[i])):
                if pair(i + 1, used | {j}):
                    return True
        return False

    return pair(0, set())


def _labels(gold: Path, gset: str, ldir_name: str, case: str) -> Dict[str, Dict[str, Any]]:
    path = gold / gset / ldir_name / f"{case}.json"
    data = cs._load(path) or {}
    return {str(v.get("voice")): v for v in data.get("voices") or [] if isinstance(v, dict)}


def gold_wrong_names(report: Dict[str, Any], gold: Path, root: Path) -> Dict[str, Any]:
    """Every ``S7_wrong`` row of an S7 report classed (A3, A4), plus the A2 counts. Asserts that
    the classed ids are exactly the report's ``S7_wrong`` ids."""
    rows = report["rows"]
    published: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    has_rows: Set[Tuple[str, str, str]] = set()
    for r in rows:
        if r.get("voice"):
            key = (r["set"], str(r["case"]), str(r["voice"]))
            has_rows.add(key)
            if _real(r.get("stored")):
                published[key].add(_fold(str(r["stored"])))
    ldirs = {s: str(report["sets"][s]["labels"]) for s in report["sets"]}
    cases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    side: Counter = Counter()
    for r in rows:
        bucket = r.get("bucket")
        if bucket in ("scored_no_name", "S8_undecidable", "quote_not_placeable"):
            if _real(r.get("stored")):
                side[
                    {
                        "scored_no_name": "name_without_support",
                        "S8_undecidable": "undecidable_named",
                        "quote_not_placeable": "unplaceable_named",
                    }[bucket]
                ] += 1
            continue
        if bucket != "S7_wrong":
            continue
        gset, case, voice = r["set"], str(r["case"]), str(r["voice"])
        labels = _labels(gold, gset, ldirs[gset], case)
        label, stored = str(r["label"]), str(r["stored"])
        owners = sorted(
            v
            for v, lab in labels.items()
            if v != voice and _real(lab.get("name")) and _fold(str(lab["name"])) == _fold(stored)
        )
        row = {
            k: r.get(k)
            for k in (
                "set",
                "case",
                "episode_id",
                "insight",
                "voice",
                "hidden",
                "label",
                "stored",
                "segment_label",
                "later_quote_on_stored",
            )
        }
        row["owners"] = owners
        row["owner_published"] = {o: sorted(published.get((gset, case, o), set())) for o in owners}
        row["owner_label_weak"] = any(
            labels[o].get("confidence") == "low" or labels[o].get("role") == "unknown"
            for o in owners
        )
        row["same_person_owner_sensitivity"] = (not owners) and any(
            v != voice and _real(lab.get("name")) and same_person(str(lab["name"]), stored)
            for v, lab in labels.items()
        )
        if owners:
            mine, theirs = _fold(label), [published.get((gset, case, o), set()) for o in owners]
            if any(mine in p for p in theirs):
                sub = "swapped"
            elif any(_fold(stored) in p for p in theirs):
                sub = "duplicated"
            elif any(p for p in theirs):
                sub = "rotated/third_name"
            elif any((gset, case, o) in has_rows for o in owners):
                sub = "rotated/owner_unnamed"
            else:
                sub = "rotated/owner_no_insight"
            row.update(cls="binding", sub=sub)
        elif is_spelling(label, stored):
            row.update(cls="spelling", sub=None)
        else:
            row.update(cls="other_person", sub=None)
            row["shares_label_token"] = bool(set(_tokens(label)) & set(_tokens(stored)))
            ckey = (gset, case)
            if ckey not in cases:
                cases[ckey] = cs._load(root / _case_meta(gold, gset, case)) or {}
            row["stored_on_another_voice_unnamed_label"] = _on_unnamed_label_voice(
                cases[ckey], stored, voice, labels
            )
        out.append(row)
    wrong_ids = {
        (r["set"], str(r["case"]), r["insight"]) for r in rows if r["bucket"] == "S7_wrong"
    }
    classed = [(r["set"], str(r["case"]), r["insight"]) for r in out]
    if len(classed) != len(set(classed)) or set(classed) != wrong_ids:
        raise AssertionError("gold: the classes are not exactly S7's S7_wrong insights")
    return {
        "rows": out,
        "summary": _gold_summary(out),
        "outside_s7_wrong": dict(sorted(side.items())),
    }


def _case_meta(gold: Path, gset: str, case: str) -> str:
    data = cs._load(gold / gset / "cases" / f"{case}.json") or {}
    return str(data.get("meta_relpath") or "")


def _on_unnamed_label_voice(
    meta: Dict[str, Any], stored: str, voice: str, labels: Dict[str, Dict[str, Any]]
) -> bool:
    for e in (meta.get("content") or {}).get("speakers") or []:
        if isinstance(e, dict) and _real(e.get("name")) and _fold(str(e["name"])) == _fold(stored):
            for v in e.get("voices") or []:
                if str(v) != voice and str(v) in labels and not _real(labels[str(v)].get("name")):
                    return True
    return False


def _gold_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Any] = {}
    for gset in sorted({r["set"] for r in rows}):
        mine = [r for r in rows if r["set"] == gset]
        by[gset] = {
            "total": len(mine),
            "classes": dict(sorted(Counter(r["cls"] for r in mine).items())),
            "binding": dict(
                sorted(Counter(r["sub"] for r in mine if r["cls"] == "binding").items())
            ),
        }
    return {
        "total": len(rows),
        "classes": dict(sorted(Counter(r["cls"] for r in rows).items())),
        "binding": dict(sorted(Counter(r["sub"] for r in rows if r["cls"] == "binding").items())),
        "flags": {
            "owner_label_weak": sum(1 for r in rows if r.get("owner_label_weak")),
            "two_or_more_owners": sum(1 for r in rows if len(r.get("owners") or []) >= 2),
            "same_person_owner_sensitivity": sum(
                1 for r in rows if r.get("same_person_owner_sensitivity")
            ),
            "shares_label_token": sum(1 for r in rows if r.get("shares_label_token")),
            "stored_on_another_voice_unnamed_label": sum(
                1 for r in rows if r.get("stored_on_another_voice_unnamed_label")
            ),
        },
        "by_set": by,
    }


class _Episode:
    """One served episode's record, talk shares and insight voices, as the scoreboard reads them."""

    def __init__(self, meta_path: Path) -> None:
        self.meta_path = meta_path
        self.meta = cs._load(meta_path)
        self.ok = isinstance(self.meta, dict)
        if not self.ok:
            return
        run = meta_path.parent.parent
        stem = meta_path.name[: -len(".metadata.json")]
        self.gi = cs._load(run / "metadata" / f"{stem}.gi.json")
        tr = Path(str((self.meta.get("content") or {}).get("transcript_file_path") or "")).name
        self.texts = cs._Texts(run / "transcripts")
        diag = (
            cs._load(run / "transcripts" / (tr[: -len(".txt")] + ".speakers.diagnostics.json"))
            if tr.endswith(".txt")
            else None
        )
        self.has_diag = isinstance(diag, dict)
        self.share = cs._talk_share(diag) if self.has_diag else {}
        entries = [
            e
            for e in (self.meta.get("content") or {}).get("speakers") or []
            if isinstance(e, dict) and _real(e.get("name"))
        ]
        self.named = entries
        keyed = [("placed" in e) for e in entries]
        if not entries:
            self.placed_recorded = "no entries"
        elif all(keyed):
            self.placed_recorded = "all"
        elif not any(keyed):
            self.placed_recorded = "none"
        else:
            self.placed_recorded = "mixed"
        self.unplaced = [e for e in entries if e.get("placed") is False]
        placed_voices = {
            str(v) for e in entries if e.get("placed") is not False for v in e.get("voices") or []
        }
        self.unnamed_principal = sorted(
            v for v, s in self.share.items() if s >= PRINCIPAL_SHARE and v not in placed_voices
        )

    def hidden_insights(self) -> Iterable[Tuple[str, Optional[str]]]:
        """``(insight id, its voice or None)`` for every S1 node."""
        gi = self.gi if isinstance(self.gi, dict) else {}
        nodes = {str(n.get("id")): n for n in gi.get("nodes") or [] if isinstance(n, dict)}
        first: Dict[str, str] = {}
        for e in gi.get("edges") or []:
            if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
                first.setdefault(str(e.get("from")), str(e.get("to")))
        for nid, n in nodes.items():
            p = n.get("properties") or {}
            if n.get("type") != "Insight" or p.get("surfaceable") is not False:
                continue
            if p.get("speaker_voice_type") != "unknown":
                continue
            q = nodes.get(first.get(nid, ""))
            yield nid, (cs._voice_of_quote(q, self.texts) if q else None)


def corpus_hidden(root: Path) -> Dict[str, Any]:
    """Every S1 insight of the served corpus classed (A6, A7); asserts it equals the scoreboard."""
    from ..upgrade.corpus_selection import select_served_artifacts

    root = Path(root)
    out: List[Dict[str, Any]] = []
    for meta_path in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta_path)
        if not ep.ok:
            continue
        if ep.placed_recorded == "mixed":
            raise AssertionError(f"{meta_path.name}: placed recorded on some entries only")
        for iid, voice in ep.hidden_insights():
            share = ep.share.get(voice, 0.0) if voice else 0.0
            principal = bool(voice) and share >= PRINCIPAL_SHARE
            if not ep.has_diag:
                cls = "gap/no_diagnostics"
            elif voice is None:
                cls = "gap/voice_not_found"
            elif not ep.named:
                cls = "nobody_named"
            elif ep.placed_recorded == "none":
                cls = f"placed_not_recorded_{'principal' if principal else 'minor'}"
            elif ep.unplaced:
                cls = f"unplaced_{'principal' if principal else 'minor'}"
            else:
                cls = f"none_unplaced_{'principal' if principal else 'minor'}"
            one_to_one = (
                cls == "unplaced_principal"
                and len(ep.unplaced) == 1
                and ep.unnamed_principal == [voice]
            )
            out.append(
                {
                    "episode_id": str(
                        ((ep.meta or {}).get("episode") or {}).get("episode_id") or ""
                    ),
                    "meta": str(meta_path.relative_to(root)),
                    "insight": iid,
                    "voice": voice,
                    "share": round(share, 4),
                    "principal": principal,
                    "placed_recorded": ep.placed_recorded,
                    "n_unplaced": len(ep.unplaced),
                    "unplaced_roles": sorted({str(e.get("role") or "?") for e in ep.unplaced}),
                    "n_unnamed_principal": len(ep.unnamed_principal),
                    "one_to_one": one_to_one,
                    "cls": cls,
                }
            )
    s1 = cs.score_corpus(root)["total"]["S1_hidden_unknown"]
    ids = [(r["meta"], r["insight"]) for r in out]
    if len(ids) != len(set(ids)) or len(out) != s1:
        raise AssertionError(f"corpus: {len(out)} classed S1 insights, the scoreboard says {s1}")
    classes = Counter(r["cls"] for r in out)
    return {
        "rows": out,
        "summary": {
            "total": len(out),
            "classes": dict(sorted(classes.items())),
            "one_to_one": sum(1 for r in out if r["one_to_one"]),
            "one_to_one_episodes": len({r["meta"] for r in out if r["one_to_one"]}),
        },
    }


def seat_check(root: Path, gold: Path, ldirs: Dict[str, str], served: Set[str]) -> Dict[str, Any]:
    """The one-to-one seat on every served gold case that has one, against its label (A8)."""
    out: List[Dict[str, Any]] = []
    for gset, ldir in sorted(ldirs.items()):
        for cf in sorted((gold / gset / "cases").glob("*.json")):
            case = cs._load(cf) or {}
            rel = str(case.get("meta_relpath") or "")
            ep = _Episode(root / rel) if rel in served else None
            if ep is None or not ep.ok:
                continue
            if len(ep.unplaced) != 1 or len(ep.unnamed_principal) != 1:
                continue
            voice = ep.unnamed_principal[0]
            candidate = str(ep.unplaced[0]["name"])
            lab = _labels(gold, gset, ldir, cf.stem).get(voice)
            if lab is None:
                verdict = "no_label_entry"
            elif not _real(lab.get("name")):
                verdict = "label_supports_no_name"
            elif same_person(str(lab["name"]), candidate) or (
                str(lab["name"]).lower() == candidate.lower()
            ):
                verdict = "right"
            else:
                verdict = "another_person"
            on_voice = sum(1 for _iid, v in ep.hidden_insights() if v == voice)
            out.append(
                {
                    "set": gset,
                    "case": cf.stem,
                    "served": True,
                    "candidate": candidate,
                    "label_name": (lab or {}).get("name"),
                    "role": (lab or {}).get("role"),
                    "confidence": (lab or {}).get("confidence"),
                    "verdict": verdict,
                    "hidden_insights_on_voice": on_voice,
                }
            )
    return {
        "rows": out,
        "summary": {
            "cases": len(out),
            "verdicts": dict(sorted(Counter(r["verdict"] for r in out).items())),
            "hidden_insights_by_verdict": dict(
                sorted(
                    (k, sum(r["hidden_insights_on_voice"] for r in out if r["verdict"] == k))
                    for k in {r["verdict"] for r in out}
                )
            ),
        },
    }


def write(out_dir: Path, name: str, rows: List[Dict[str, Any]]) -> Path:
    """One JSONL file of rows under ``out_dir``."""
    path = Path(out_dir) / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return path
