#!/usr/bin/env python3
"""Replay the episode SPEAKER RECORD (placed + unplaced people), old code vs new code. Read-only.

``roster_replay.py`` compares per-voice names. A person listed twice on one episode — once on a
voice as the ASR spelt them, once unplaced as the feed spells them ("Professor Hannah Frye" placed,
"Hannah Fry" unplaced) — is not a per-voice change, so it never shows there. This rebuilds what
``metadata_generation._build_speaker_record`` publishes from each variant's replayed roster: the
placed people, then ``_unplaced_speakers`` over that variant's own diagnostics, with the variant's
roster module bound where ``_unplaced_speakers`` imports its same-person check from.

Not reproduced (the record builder reads them from disk, the replay has no disk to read): the
corpus organisation vote, the show-name filter on placed names, the pre-listening ``hint`` names.

    python scripts/measure/speaker_record_replay.py --corpus /app/output \\
        --old roster=/tmp/old_roster.py metadata_generation=/tmp/old_mg.py \\
        --new roster=src/.../roster.py metadata_generation=src/.../metadata_generation.py

One JSON line per episode whose record changed, plus every episode whose NEW record still lists one
person twice (``kind: duplicate``), then a summary. Exit 1 when the new side has a replay error the
old side did not, when the new side lists someone twice on an episode the old side did not, when
more than ``--max-new-duplicates`` episodes still do, or when nothing was replayed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roster_replay as rr  # noqa: E402

MG = "podcast_scraper.workflow.metadata_generation"


def _load(files: Dict[str, Path]) -> Dict[str, Any]:
    import importlib

    mods: Dict[str, Any] = dict(
        rr.load_variant({k: v for k, v in files.items() if k in rr.MODULES})
    )
    saved = importlib.import_module(MG)
    try:
        mods["mg"] = rr._exec_module(
            MG, files.get("metadata_generation") or Path(str(saved.__file__))
        )
    finally:
        sys.modules[MG] = saved
    return mods


def record(mods: Dict[str, Any], ep: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    roster_mod, mg = mods["roster"], mods["mg"]
    kw = rr.roster_inputs(ep, roster_mod)
    dz, text = kw["diarization"], kw["transcript_text"]
    res = rr.replay(roster_mod, ep)
    diag = roster_mod.build_speaker_diagnostics(
        dz,
        res,
        transcript_text=text,
        voice_texts=kw["voice_texts"],
        detected_guests=kw["detected_guests"],
        known_hosts=kw["known_hosts"],
        metadata_named=kw["metadata_named"],
    )
    diag["tried"] = dict(diag.get("tried") or {}, known_hosts=kw["known_hosts"])
    by_name: Dict[str, Any] = {}
    for v, r in res.by_voice.items():
        if not r.named:
            continue
        sp = by_name.get(r.name)
        if sp is None:
            by_name[r.name] = mg.SpeakerInfo(
                id=f"p{len(by_name)}",
                name=r.name,
                role=r.role,
                placed=True,
                voices=[v],
                source=r.source,
            )
        else:
            sp.voices.append(v)
    placed = list(by_name.values())
    installed = sys.modules.get(rr.MODULES["roster"])
    sys.modules[rr.MODULES["roster"]] = roster_mod
    try:
        unplaced = mg._unplaced_speakers(
            placed,
            diagnostics=diag,
            detected_hosts=None,
            detected_guests=None,
            feed_title=(ep["meta"].get("feed") or {}).get("title"),
        )
    finally:
        if installed is not None:
            sys.modules[rr.MODULES["roster"]] = installed
    return [(s.name, s.role, True) for s in placed] + [(s.name, s.role, False) for s in unplaced]


def duplicate_pairs(names: List[str]) -> List[Tuple[str, str]]:
    """The census predicate of 2026-10-08: same name after titles, or one last-token respelling."""
    import difflib

    from podcast_scraper.providers.ml.diarization import roster as R

    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a == b:
                out.append((a, b))
                continue
            ta, tb = R._strip_titles(a), R._strip_titles(b)
            if ta == tb or (
                len(ta) == len(tb) >= 2
                and ta[:-1] == tb[:-1]
                and difflib.SequenceMatcher(None, ta[-1], tb[-1]).ratio() >= 0.70
            ):
                out.append((a, b))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--old", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--new", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--only", nargs="*", default=[], help="substrings of metadata paths to replay")
    ap.add_argument(
        "--max-new-duplicates",
        type=int,
        default=None,
        help="exit 1 when more episodes than this still list one person twice on the new side",
    )
    args = ap.parse_args(argv)

    def files(pairs: List[str]) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for p in pairs:
            k, _, v = p.partition("=")
            if k not in rr.MODULES and k != "metadata_generation":
                raise SystemExit(f"bad variant file {p!r}")
            out[k] = Path(v)
        return out

    rr.index_siblings(args.corpus)
    new = _load(files(args.new))
    old = _load(files(args.old))
    counts: Counter = Counter()
    for ep in rr.corpus_episodes(args.corpus):
        if args.only and not any(s in ep["meta_path"] for s in args.only):
            continue
        stored = [
            (str(s["name"]), str(s.get("role")), s.get("placed") is not False)
            for s in ((ep["meta"].get("content") or {}).get("speakers") or [])
            if isinstance(s, dict) and s.get("name")
        ]
        recs = {}
        for side, mods in (("old", old), ("new", new)):
            try:
                recs[side] = record(mods, ep)
            except Exception as exc:  # noqa: BLE001 — one bad episode must not stop the replay
                counts[f"{side}_error"] += 1
                counts[f"{side}_error:{type(exc).__name__}"] += 1
        if "old" not in recs or "new" not in recs:
            if "old" in recs:
                counts["new_only_error"] += 1
            continue
        counts["episodes"] += 1
        if counts["episodes"] % 200 == 0:
            print(f"progress: {counts['episodes']} episodes", file=sys.stderr, flush=True)
        a, b = recs["old"], recs["new"]
        if sorted(a) == sorted(stored):
            counts["old_equals_stored"] += 1
        da, db = duplicate_pairs([n for n, _, _ in a]), duplicate_pairs([n for n, _, _ in b])
        counts["old_duplicate_eps"] += bool(da)
        counts["new_duplicate_eps"] += bool(db)
        # A duplicate the old code did not have is a regression, whatever the totals say.
        counts["duplicates_introduced"] += bool(db and not da)
        counts["stored_duplicate_eps"] += bool(duplicate_pairs([n for n, _, _ in stored]))
        changed = sorted(a) != sorted(b)
        if changed or db:
            kind = "changed" if changed else "duplicate"
            if changed and db:
                kind = "changed+duplicate"
            counts[kind] += 1
            print(
                json.dumps(
                    {
                        "kind": kind,
                        "meta": ep["meta_path"].split("/feeds/")[-1][:90],
                        "feed": (ep["meta"].get("feed") or {}).get("title"),
                        "episode": (ep["meta"].get("episode") or {}).get("title"),
                        "stored": stored,
                        "old": a,
                        "new": b,
                        "new_dups": db,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(json.dumps({"summary": dict(counts)}, ensure_ascii=False))
    failed = bool(counts.get("new_only_error") or counts.get("duplicates_introduced"))
    if args.max_new_duplicates is not None:
        failed |= counts["new_duplicate_eps"] > args.max_new_duplicates
    if not counts["episodes"]:
        failed = True  # an empty replay proves nothing
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
