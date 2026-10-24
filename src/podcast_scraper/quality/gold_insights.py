"""S7 / S8: insight-weighted correctness on the naming gold sets (method §3, step 0).

For every gold case whose labelled run is still SERVED, each insight is placed on a voice by the
pipeline's own rule: its FIRST ``SUPPORTED_BY`` edge (the GI file's edge order) gives a quote; the
quote's ``char_start`` falls in a segment of the transcript it cites (``gi.pipeline.
_segment_char_spans``); that segment's RAW ``speaker`` is the voice. Never the stored label and
never the insight's ``speaker``: one name on two voices would count their insights twice (the
v0.1 defect), and a speakerless insight would drop out. The voice's gold label then decides:

- ``no_grounded_quote``: the insight has no SUPPORTED_BY edge;
- ``quote_not_placeable`` (by reason: unresolved ref, no offsets, outside every segment, segment
  without a raw speaker, spans refused by the alignment guard);
- ``voice_not_labelled``; ``S8_undecidable`` (label not scored: role unknown or confidence low);
- ``scored_no_name`` (by role: a scored label with no name — ad, promo, clip, a nameless host);
- ``S7_correct`` / ``S7_wrong`` / ``S7_unnamed`` by the stored name (``properties.speaker``; empty
  or ``SPEAKER_NN`` is unnamed) against the label, with ``naming_gate._same``.

Every bucket is split shown / hidden (``surfaceable is False`` = hidden). S7 correct is split by
the matcher's tier: fold-exact, token-subset, fuzzy or title (``same_person``'s rules in order),
because the token-subset rule counts a bare "Josh" as "Josh Elman", and ASR spellings ("Josh
Dossi" / "Josh Dawsey") count as wrong. Conservation by insight id is asserted: every insight of
an included case lands in exactly one bucket. Read-only. Criteria:
``docs/wip/changes/s7_s8_gold_insight_weighted.acceptance.md``.

On an eval run's output (a relabel or a rediarize renumbers the voices and writes a new run), the
labels' voice ids are the labelled run's, so a lookup by the after voice id names the wrong person.
``voice_maps`` (the eval scorer's ``<rows>.voicemap.json``: per case the after run's metadata path
and every labelled-run voice's after voice) follows each case to its after run and re-keys the
labels. Two labelled voices on one after voice whose labels differ is ``voice_map_ambiguous``: its
insights belong to neither label alone; labels that agree (one person the labelled run split) apply.
A case the map does not cover is skipped and named.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BUCKETS = (
    "no_grounded_quote",
    "quote_not_placeable",
    "voice_not_labelled",
    "S8_undecidable",
    "scored_no_name",
    "S7_correct",
    "S7_wrong",
    "S7_unnamed",
    "voice_map_ambiguous",
)
_RAW = re.compile(r"^SPEAKER(_\d+)?$")
_LABELS = re.compile(r"^labels_v(\d+)$")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_scored(label: Dict[str, Any]) -> bool:
    """The naming gate's rule (``scripts/measure/naming_gate.py``): role unknown or confidence low
    is not scored. Kept equal to it by a test."""
    role = str(label.get("role") or "unknown")
    return role != "unknown" and str(label.get("confidence") or "") != "low"


def same_name(a: str, b: str) -> bool:
    """The naming gate's ``_same``: equal ignoring case, or ``same_person``."""
    from ..kg.speaker_coherence import same_person

    return a.strip().lower() == b.strip().lower() or same_person(a, b)


def newest_labels(gold_set: Path) -> Optional[Path]:
    """The ``labels_vN`` directory with the highest N (numeric, so v10 > v2)."""
    found = [(int(m.group(1)), p) for p in gold_set.iterdir() if (m := _LABELS.match(p.name))]
    return max(found)[1] if found else None


def resolve_ref(ref: str, root: Path, run: Path) -> Path:
    """A quote's ``transcript_ref`` as a path on this corpus: ``/app/output/<rest>`` is the corpus
    root, anything else is relative to the episode's run directory; ``..`` normalised."""
    if ref.startswith("/app/output/"):
        p = root / ref[len("/app/output/") :]
    elif ref.startswith("/"):
        p = Path(ref)
    else:
        p = run / ref
    parts: List[str] = []
    for part in p.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return Path(*parts) if parts else p


def tier(label_name: str, stored: str) -> str:
    """Which of ``same_person``'s rules matched: fold-exact, token-subset, or fuzzy-or-title."""
    from ..kg.speaker_coherence import _fold

    fa, fb = _fold(label_name), _fold(stored)
    if label_name.strip().lower() == stored.strip().lower() or (fa and fa == fb):
        return "fold_exact"
    ta, tb = set(fa.split()), set(fb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return "token_subset"
    return "fuzzy_or_title"


class _Texts:
    """Transcript texts and segment spans by resolved path (cached)."""

    def __init__(self) -> None:
        self._spans: Dict[Path, Tuple[str, Optional[list]]] = {}

    def spans(self, txt: Path) -> Tuple[str, Optional[list]]:
        from ..gi.pipeline import _segment_char_spans

        if txt not in self._spans:
            if not txt.is_file():
                self._spans[txt] = ("unresolved_ref", None)
            else:
                seg = _load(txt.with_name(txt.name[: -len(".txt")] + ".segments.json"))
                rows = seg if isinstance(seg, list) else (seg or {}).get("segments") or []
                rows = [r for r in rows if isinstance(r, dict)]
                body = txt.read_text(encoding="utf-8")
                spans = _segment_char_spans(body, rows) if rows else None
                self._spans[txt] = ("ok", spans) if spans is not None else ("spans_none", None)
        return self._spans[txt]


def place(
    quote: Dict[str, Any], root: Path, run: Path, texts: _Texts
) -> Tuple[str, Optional[str], Optional[str]]:
    """``(reason, raw voice, segment label)``: reason ``ok``, or why the quote cannot be placed."""
    props = quote.get("properties") or {}
    ref, start = props.get("transcript_ref"), props.get("char_start")
    if not isinstance(start, int):
        return "no_offsets", None, None
    if not isinstance(ref, str) or not ref:
        return "unresolved_ref", None, None
    state, spans = texts.spans(resolve_ref(ref, root, run))
    if state != "ok" or spans is None:
        return state, None, None
    for a, b, seg in spans:
        if a <= start < b:
            voice = seg.get("speaker")
            label = seg.get("speaker_label")
            return ("ok", str(voice), label) if voice else ("no_raw_speaker", None, None)
    return "outside_segments", None, None


def load_voice_maps(path: Path) -> Dict[str, Any]:
    """The eval scorer's voice-map file: ``{"after": <corpus root>, "cases": {case id: {...}}}``."""
    data = _load(Path(path))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), dict):
        raise SystemExit(f"{path}: not a voice-map file (no 'cases')")
    return data


def relabel(label: Dict[str, Any], vmap: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The label keyed by the after run's voices, and what the re-keying could not carry: label
    voices the map lacks (``unmapped``) and after voices two labels map to (``ambiguous``)."""
    m = vmap.get("map") or {}
    by_after: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unmapped: List[str] = []
    for v in label.get("voices") or []:
        if not isinstance(v, dict):
            continue
        hit = m.get(str(v.get("voice")))
        if not hit or not hit[0]:
            unmapped.append(str(v.get("voice")))
            continue
        by_after[str(hit[0])].append(
            dict(v, voice=str(hit[0]), voice_before=str(v.get("voice")), map_share=hit[1])
        )
    voices: List[Dict[str, Any]] = []
    ambiguous: Dict[str, List[str]] = {}
    merged: Dict[str, List[str]] = {}
    for a, vs in by_after.items():
        # one person split in two by the labelled run's diarization and merged by the after run
        # (a solo host labelled twice): every label says the same thing, so it applies
        same = {
            (str(x.get("name") or "").strip().lower(), str(x.get("role") or ""), is_scored(x))
            for x in vs
        }
        if len(vs) > 1 and len(same) > 1:
            ambiguous[a] = sorted(x["voice_before"] for x in vs)
            continue
        if len(vs) > 1:
            merged[a] = sorted(x["voice_before"] for x in vs)
        voices.append(dict(vs[0], voice_before="+".join(sorted(x["voice_before"] for x in vs))))
    lost = {"unmapped": sorted(unmapped), "ambiguous": ambiguous, "merged_same_label": merged}
    return dict(label, voices=voices), lost


def score_case(
    case: Dict[str, Any],
    label: Dict[str, Any],
    root: Path,
    texts: _Texts,
    ambiguous: Optional[Dict[str, List[str]]] = None,
) -> Iterable[Dict[str, Any]]:
    """One row per Insight node of the served case."""
    meta_path = root / case["meta_relpath"]
    run = meta_path.parent.parent
    meta = _load(meta_path) or {}
    lang = str((meta.get("episode") or {}).get("language") or "unknown")
    gi = _load(Path(str(meta_path)[: -len(".metadata.json")] + ".gi.json")) or {}
    nodes = {str(n.get("id")): n for n in gi.get("nodes") or [] if isinstance(n, dict)}
    first: Dict[str, str] = {}
    later: Dict[str, List[str]] = defaultdict(list)
    for e in gi.get("edges") or []:
        if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
            src = str(e.get("from"))
            if src in first:
                later[src].append(str(e.get("to")))
            else:
                first[src] = str(e.get("to"))
    labels = {str(v.get("voice")): v for v in label.get("voices") or [] if isinstance(v, dict)}
    from ..server.app_gi_view import insights_from_gi

    served_ids = {str(i.id) for i in insights_from_gi(gi) or []}
    for iid, node in nodes.items():
        if node.get("type") != "Insight":
            continue
        props = node.get("properties") or {}
        stored = str(props.get("speaker") or "")
        row: Dict[str, Any] = {
            "case": case.get("case_id"),
            "insight": iid,
            "language": lang,
            "hidden": props.get("surfaceable") is False,
            "stored": stored,
            "speakerless": not stored,
            "voice": None,
            "label": None,
            "reason": None,
            "tier": None,
            "episode_id": str((meta.get("episode") or {}).get("episode_id") or ""),
            "excluded_unflagged": props.get("surfaceable") is not False and iid not in served_ids,
        }
        qid = first.get(iid)
        if qid is None or qid not in nodes:
            row["bucket"] = "no_grounded_quote"
            yield row
            continue
        reason, voice, seg_label = place(nodes[qid], root, run, texts)
        row["voice"] = voice
        row["segment_label"] = seg_label
        if reason != "ok":
            row.update(bucket="quote_not_placeable", reason=reason)
            yield row
            continue
        if ambiguous and str(voice) in ambiguous:
            row.update(bucket="voice_map_ambiguous", reason="+".join(ambiguous[str(voice)]))
            yield row
            continue
        lab = labels.get(str(voice))
        if lab is not None and "voice_before" in lab:
            row.update(voice_before=lab["voice_before"], map_share=lab.get("map_share"))
        if lab is None:
            row["bucket"] = "voice_not_labelled"
            yield row
            continue
        row["label"] = lab.get("name")
        if not is_scored(lab):
            row["bucket"] = "S8_undecidable"
        elif not lab.get("name"):
            row.update(bucket="scored_no_name", reason=str(lab.get("role") or "unknown"))
        elif not stored or _RAW.match(stored):
            row["bucket"] = "S7_unnamed"
        elif same_name(str(lab["name"]), stored):
            row.update(bucket="S7_correct", tier=tier(str(lab["name"]), stored))
        else:
            row["bucket"] = "S7_wrong"
            later_voices = [
                place(nodes[q], root, run, texts)[1] for q in later.get(iid, []) if q in nodes
            ]
            row["later_quote_on_stored"] = any(
                (labels.get(str(v)) or {}).get("name")
                and same_name(str(labels[str(v)]["name"]), stored)
                for v in later_voices
                if v is not None
            )
        yield row


def run(
    root: Path,
    gold: Path,
    sets: List[str],
    snapshot: str,
    labels: Optional[Dict[str, str]] = None,
    voice_maps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The S7/S8 report over the given gold sets on ``root``. Asserts conservation."""
    from ..upgrade.corpus_selection import select_served_artifacts

    root = Path(root)
    if voice_maps is not None and Path(str(voice_maps.get("after"))).resolve() != root.resolve():
        raise SystemExit(
            f"the voice map was made for {voice_maps.get('after')}, not {root}: voice ids differ"
        )
    served_list = sorted(
        str(m.relative_to(root)) for m in select_served_artifacts(root, ".metadata.json")[0]
    )
    served = set(served_list)
    texts = _Texts()
    report: Dict[str, Any] = {
        "snapshot": snapshot,
        "corpus_root": str(root),
        "served_metadata_sha256": hashlib.sha256("\n".join(served_list).encode()).hexdigest(),
        # What the corpus's own ledger records. Not proof of state: a migration applied by a
        # script (0030 on the 10-16 clone) is not in it, which is why ``snapshot`` names it.
        "ledger_applied": sorted(
            str(a.get("id", ""))
            for a in ((_load(root / "upgrade_ledger.json") or {}).get("applied") or [])
            if isinstance(a, dict)
        ),
        "matcher_limits": (
            "token-subset matching counts a bare first name as the full name (some wrong-person "
            "cases count as correct); ASR spellings and junk names count as wrong. "
            "Strict S7 correct = fold_exact."
        ),
        "sets": {},
    }
    if voice_maps is not None:
        report["voice_map"] = {
            "after": voice_maps.get("after"),
            "before": voice_maps.get("before"),
            "cases": len(voice_maps["cases"]),
            "sha256": hashlib.sha256(json.dumps(voice_maps, sort_keys=True).encode()).hexdigest(),
        }
    rows_out: List[Dict[str, Any]] = []
    for gset in sets:
        gdir = gold / gset
        ldir = gdir / labels[gset] if labels and gset in labels else newest_labels(gdir)
        if ldir is None or not ldir.is_dir():
            raise SystemExit(f"{gset}: no label directory")
        files_h = hashlib.sha256()
        skipped, included, no_label, misses, gi_missing = [], 0, 0, 0, 0
        no_map: List[str] = []
        unmapped: Dict[str, List[str]] = {}
        merged: Dict[str, Dict[str, List[str]]] = {}
        same_label: Dict[str, Dict[str, List[str]]] = {}
        rows: List[Dict[str, Any]] = []
        relpaths: Dict[str, str] = {}
        for cf in sorted((gdir / "cases").glob("*.json")):
            case = json.loads(cf.read_text(encoding="utf-8"))
            files_h.update(cf.read_bytes())
            vmap = None
            if voice_maps is not None:
                vmap = voice_maps["cases"].get(str(case.get("case_id")))
                if vmap is None:
                    no_map.append(str(case.get("case_id")))
                    continue
                case = dict(case, meta_relpath=vmap.get("after_relpath") or case["meta_relpath"])
            if case.get("meta_relpath") not in served:
                skipped.append(case.get("case_id"))
                continue
            lf = ldir / cf.name
            if not lf.is_file():
                no_label += 1
                continue
            files_h.update(lf.read_bytes())
            label = json.loads(lf.read_text(encoding="utf-8"))
            ambiguous: Dict[str, List[str]] = {}
            if vmap is not None:
                label, lost = relabel(label, vmap)
                ambiguous = lost["ambiguous"]
                if lost["unmapped"]:
                    unmapped[str(case.get("case_id"))] = lost["unmapped"]
                if ambiguous:
                    merged[str(case.get("case_id"))] = ambiguous
                if lost["merged_same_label"]:
                    same_label[str(case.get("case_id"))] = lost["merged_same_label"]
            relpaths[cf.name] = case["meta_relpath"]
            included += 1
            if not (root / (case["meta_relpath"][: -len(".metadata.json")] + ".gi.json")).is_file():
                gi_missing += 1
            case_rows = list(score_case(case, label, root, texts, ambiguous))
            voices_here = {r["voice"] for r in case_rows if r["voice"]}
            misses += sum(
                1
                for v in label.get("voices") or []
                if isinstance(v, dict)
                and v.get("voice") not in voices_here
                and not _voice_in_segments(root, case, str(v.get("voice")))
            )
            for r in case_rows:
                r["set"] = gset
            rows += case_rows
        ids = [(r["case"], r["insight"]) for r in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{gset}: an insight counted twice")
        total = _insight_total(
            root, gdir, served, ldir, relpaths if voice_maps is not None else None
        )
        if len(rows) != total:
            raise AssertionError(f"{gset}: {len(rows)} rows for {total} insights")
        report["sets"][gset] = _summary(rows) | {
            "labels": ldir.name,
            "files_sha256": files_h.hexdigest(),
            "included": included,
            "skipped_not_served": sorted(str(s) for s in skipped),
            "no_label": no_label,
            "gi_missing": gi_missing,
            "labelled_voices_missing_from_segments": misses,
        }
        if voice_maps is not None:
            report["sets"][gset] |= {
                "skipped_no_voice_map": sorted(no_map),
                "labelled_voices_unmapped": unmapped,
                "labelled_voices_merged": merged,
                "labelled_voices_merged_same_label": same_label,
            }
        rows_out += rows
    report["rows"] = sorted(rows_out, key=lambda r: (r["set"], str(r["case"]), r["insight"]))
    return report


def entry_voices(report: Dict[str, Any], keys: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
    """Per (episode id, voice) of a migration's list that gold covers: bucket counts, shown/hidden.
    Run on the before and the after corpus and compare (a list migration's own population)."""
    want = set(keys)
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in report["rows"]:
        k = (r["episode_id"], str(r["voice"]))
        if k in want:
            key = f"{r['set']}/{r['case']} {r['voice']}"
            out[key][r["bucket"]] += 1
            out[key][f"{r['bucket']}_{'hidden' if r['hidden'] else 'shown'}"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _voice_in_segments(root: Path, case: Dict[str, Any], voice: str) -> bool:
    meta = _load(root / case["meta_relpath"]) or {}
    ref = str((meta.get("content") or {}).get("transcript_file_path") or "")
    run = (root / case["meta_relpath"]).parent.parent
    txt = resolve_ref(ref, root, run) if ref else None
    if txt is None:
        return False
    seg = _load(txt.with_name(txt.name[: -len(".txt")] + ".segments.json"))
    rows = seg if isinstance(seg, list) else (seg or {}).get("segments") or []
    return any(isinstance(r, dict) and str(r.get("speaker")) == voice for r in rows)


def _insight_total(
    root: Path, gdir: Path, served: set, ldir: Path, relpaths: Optional[Dict[str, str]] = None
) -> int:
    n = 0
    for cf in sorted((gdir / "cases").glob("*.json")):
        case = json.loads(cf.read_text(encoding="utf-8"))
        rel = case.get("meta_relpath") if relpaths is None else relpaths.get(cf.name)
        if not isinstance(rel, str) or rel not in served or not (ldir / cf.name).is_file():
            continue
        gi = _load(root / (rel[: -len(".metadata.json")] + ".gi.json")) or {}
        n += sum(
            1 for x in gi.get("nodes") or [] if isinstance(x, dict) and x.get("type") == "Insight"
        )
    return n


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    return {
        "total": _counts(rows),
        "per_language": {k: _counts(v) for k, v in sorted(by_lang.items())},
    }


def _counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    c: Dict[str, Any] = {"insights": len(rows)}
    for b in BUCKETS:
        mine = [r for r in rows if r["bucket"] == b]
        c[b] = len(mine)
        c[f"{b}_hidden"] = sum(1 for r in mine if r["hidden"])
        c[f"{b}_shown"] = len(mine) - c[f"{b}_hidden"]
    c["quote_not_placeable_by_reason"] = dict(
        sorted(Counter(r["reason"] for r in rows if r["bucket"] == "quote_not_placeable").items())
    )
    c["scored_no_name_by_role"] = dict(
        sorted(Counter(r["reason"] for r in rows if r["bucket"] == "scored_no_name").items())
    )
    c["S7_correct_by_tier"] = dict(
        sorted(Counter(r["tier"] for r in rows if r["bucket"] == "S7_correct").items())
    )
    c["sensitivity_wrong_with_later_quote_on_stored"] = sum(
        1 for r in rows if r["bucket"] == "S7_wrong" and r.get("later_quote_on_stored")
    )
    c["sensitivity_wrong_stored_not_segment_label"] = sum(
        1
        for r in rows
        if r["bucket"] == "S7_wrong" and (r.get("segment_label") or "") != (r.get("stored") or "")
    )
    c["flag_absent_excluded_by_insights_from_gi"] = sum(1 for r in rows if r["excluded_unflagged"])
    c["speakerless_by_bucket"] = dict(
        sorted(Counter(r["bucket"] for r in rows if r["speakerless"]).items())
    )
    return c
