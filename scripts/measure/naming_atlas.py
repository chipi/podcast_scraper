#!/usr/bin/env python3
"""The naming FAILURE ATLAS: every gold-labelled voice we get wrong, and why. Read-only.

`naming_gate.py` counts outcomes (correct / missing / wrong / spurious / role error) old vs new.
This lists the failures of ONE code variant voice by voice, with what decides whether a rule could
ever fix them:

- ``where``: where the right name was available — ``known_host`` (the feed's stated hosts),
  ``metadata`` (episode metadata), ``llm`` (the model's per-voice verdict), ``stated_voice`` (a
  self-introduction), ``title``, ``surname_in_transcript``; empty when it is nowhere;
- ``label_name_on_other_voice``: the voices that carry this voice's real name (a SWAP when set);
- ``rungs``: the decision trace's record for the voice (which rung named it, or why none did).

    python scripts/measure/naming_atlas.py --corpus /app/output \\
        --cases naming_gold/val_v1/cases --labels naming_gold/val_v1/labels_v1 \\
        --variant roster=src/podcast_scraper/providers/ml/diarization/roster.py \\
        --signatures --out atlas_val.jsonl

Prints a summary by outcome x label role x where the name was; writes one JSON line per failure.
First used on 2026-10-12 (#2304): it showed that the episodes-with-no-named-voice count mostly
measured causes outside the naming ladder, and ranked what was left (val: 43 swaps, 140 stated but
unplaced, 33 title-only, 238 with the name nowhere).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import naming_gate as G  # noqa: E402
import roster_replay as R  # noqa: E402

FAILURES = ("missing_name", "wrong_name", "role_error", "spurious_name")
_STATED = {"known_host", "metadata", "llm", "stated_voice"}


def where_available(label_name: str, inputs: Dict[str, Any], texts: str, title: str) -> List[str]:
    """Where the labelled name could have come from (see the module docstring)."""
    if not label_name:
        return []
    out = []
    if any(G._same(label_name, x) for x in inputs.get("known_hosts") or []):
        out.append("known_host")
    if any(G._same(label_name, x) for x in inputs.get("metadata_named") or []):
        out.append("metadata")
    if any(G._same(label_name, x) for x in (inputs.get("llm_voice_names") or {}).values()):
        out.append("llm")
    if any(G._same(label_name, x) for x in (inputs.get("stated_voice_names") or {}).values()):
        out.append("stated_voice")
    if label_name.lower() in (title or "").lower():
        out.append("title")
    if label_name.split()[-1].lower() in texts.lower():
        out.append("surname_in_transcript")
    return out


def availability(where: List[str]) -> str:
    w = set(where)
    if w & _STATED:
        return "name stated or proposed"
    if "title" in w:
        return "name in the title only"
    if "surname_in_transcript" in w:
        return "name only heard"
    return "name nowhere"


def atlas_rows(
    corpus: Path, labelled: List[Any], variant: Dict[str, Any], sig: Any
) -> List[Dict[str, Any]]:
    from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace

    rows: List[Dict[str, Any]] = []
    for case, labels in labelled:
        try:
            ep = R.load_episode(corpus / case["meta_relpath"])
            trace = NamingTrace()
            res = R.replay(variant["roster"], ep, sig, trace)
        except Exception:  # noqa: BLE001 - one bad episode must not stop the atlas
            continue
        t = trace.to_dict()
        texts: Dict[str, List[str]] = {}
        dur: Counter[str] = Counter()
        for s in ep["segs"]:
            texts.setdefault(str(s.get("speaker")), []).append(str(s.get("text") or ""))
            dur[str(s.get("speaker"))] += float(s.get("end", 0) or 0) - float(
                s.get("start", 0) or 0
            )
        total = sum(dur.values()) or 1.0
        alltext = " ".join(" ".join(v) for v in texts.values())
        published = {v: G._published(res, v) for v in res.by_voice}
        title = (case.get("episode") or {}).get("title") or ""
        for lab in labels.get("voices") or []:
            if not G.is_scored(lab):
                continue
            v = str(lab.get("voice"))
            name, role = G._published(res, v)
            score = G.outcome(lab, name, role)
            if score not in FAILURES:
                continue
            ln = lab.get("name") or ""
            where = where_available(ln, t.get("inputs") or {}, alltext, title)
            rows.append(
                {
                    "case": case["case_id"],
                    "feed": (case.get("feed") or {}).get("title"),
                    "episode": title,
                    "voice": v,
                    "share": round(dur[v] / total, 3),
                    "label": {"name": ln, "role": lab.get("role")},
                    "published": {
                        "name": name,
                        "role": role,
                        "source": getattr(res.by_voice.get(v), "source", None),
                    },
                    "score": score,
                    "where": where,
                    "label_name_on_other_voice": [
                        u
                        for u, (pn, _) in published.items()
                        if u != v and pn and ln and G._same(pn, ln)
                    ],
                    "rungs": (t.get("voices") or {}).get(v, []),
                }
            )
    return rows


def summary(rows: List[Dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        kind = r["score"]
        if kind == "wrong_name" and r["label_name_on_other_voice"]:
            kind = "wrong_name (swap)"
        c[f"{kind} | {r['label']['role']} | {availability(r['where'])}"] += 1
    return c


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--variant", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--signatures", action="store_true")
    ap.add_argument("--out", type=Path, help="write one JSON line per failure here")
    args = ap.parse_args(argv)
    labelled = G.load_labelled(args.cases, args.labels)
    R.index_siblings(args.corpus)
    variant = R.load_variant(R._variant_arg(args.variant))
    sig = R._signatures(variant.get("ad_signatures"), args.corpus) if args.signatures else None
    rows = atlas_rows(args.corpus, labelled, variant, sig)
    if args.out:
        args.out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    print(f"labelled episodes: {len(labelled)}  failures: {len(rows)}")
    for k, n in summary(rows).most_common():
        print(f"{n:5d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
