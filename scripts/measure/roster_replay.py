#!/usr/bin/env python3
"""Replay the speaker roster over STORED episodes, old code vs new code. Read-only.

Every naming change of 2026-10-01/02 was decided with this: re-run production's own roster over
each stored episode, fed exactly what the diarization pipeline handed it — voice texts and turns
from ``.segments.json``, the ad intervals, the feed's recurring script, the shared voice cleaning,
and the LLM's per-voice names/roles as stored in ``.speakers.diagnostics.json`` — so no LLM is
called and nothing is written. Replay fidelity measured 124/125 voices on fresh episodes.

Two code VARIANTS are compared. A variant is a set of module source files that replace the
installed ones while that variant runs (the roster, the host/name rules, the resolution guards,
the ad signatures). Take the old side from git and the new side from the working tree:

    git show HEAD~1:src/podcast_scraper/providers/ml/diarization/roster.py > /tmp/old_roster.py
    python scripts/measure/roster_replay.py --corpus /app/output \\
        --old roster=/tmp/old_roster.py \\
        --new roster=src/podcast_scraper/providers/ml/diarization/roster.py \\
        --signatures > changes.jsonl

Output: one JSON line per voice whose published name changed (kind, feed, episode, voice, talk
share, old name, new name, old/new voice type, opening words), then a summary line.

Rule of use: read every changed voice. A replay that only reports counts proves nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

_PKG = "podcast_scraper"
#: Short names for the modules a variant may replace.
MODULES = {
    "roster": f"{_PKG}.providers.ml.diarization.roster",
    "hosts": f"{_PKG}.speaker_detectors.hosts",
    "resolution": f"{_PKG}.speaker_detectors.resolution",
    "ad_signatures": f"{_PKG}.providers.ml.diarization.ad_signatures",
}


def load_episode(meta: Path) -> Dict[str, Any]:
    """The stored artifacts of one episode: metadata, segments, speaker diagnostics."""
    doc = json.loads(meta.read_text(encoding="utf-8"))
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    run = meta.parent.parent
    segs = json.loads((run / rel.replace(".txt", ".segments.json")).read_text(encoding="utf-8"))
    segs = segs if isinstance(segs, list) else segs.get("segments", [])
    diag_path = run / rel.replace(".txt", ".speakers.diagnostics.json")
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}
    return {"meta": doc, "segs": segs, "diag": diag, "feed_dir": run.parent}


def roster_inputs(ep: Dict[str, Any], roster: Any, ad_signatures: Any = None) -> Dict[str, Any]:
    """What ``providers/ml/diarization/pipeline.py`` hands ``resolve_speaker_roster``."""
    from podcast_scraper.providers.ml.diarization import pipeline as P
    from podcast_scraper.providers.ml.diarization.base import (
        DiarizationResult,
        DiarizationSegment,
    )

    segs = [s for s in ep["segs"] if s.get("speaker")]
    turns = [(str(s["speaker"]), str(s.get("text") or "")) for s in segs]
    chunks: Dict[str, List[str]] = {}
    for v, t in turns:
        chunks.setdefault(v, []).append(t)
    voice_texts = {v: " ".join(x) for v, x in chunks.items()}
    dz = DiarizationResult(
        segments=[
            DiarizationSegment(float(s.get("start") or 0), float(s.get("end") or 0), s["speaker"])
            for s in segs
        ],
        num_speakers=len(voice_texts),
    )
    ad_intervals = P._ad_intervals(ep["segs"])
    recurring = P._feed_recurring_text(SimpleNamespace(output_dir=str(ep["feed_dir"])))
    extra = {"ad_signatures": ad_signatures} if ad_signatures is not None else {}
    cleaning = roster.classify_voices(
        dz,
        ad_intervals,
        voice_texts=voice_texts,
        ordered_turns=turns,
        recurring_text=recurring,
        diarization_provider="tailnet_dgx",
        cameo_max_talk_s=20.0,
        **extra,
    )
    voices = ep["diag"].get("voices") or []
    llm = [v for v in voices if v.get("source") == "llm_resolution"]
    tried = ep["diag"].get("tried") or {}
    episode = ep["meta"].get("episode") or {}
    return dict(
        diarization=dz,
        transcript_text=" ".join(t for _, t in turns),
        detected_guests=tried.get("detected_guests") or [],
        known_hosts=tried.get("known_hosts") or [],
        voice_texts=voice_texts,
        ordered_turns=turns,
        ad_intervals=ad_intervals,
        metadata_named=tried.get("metadata_named") or [],
        llm_voice_names={v["voice"]: v["resolved_name"] for v in llm if v.get("resolved_name")}
        or None,
        llm_voice_roles={v["voice"]: v["role"] for v in llm if v.get("role")} or None,
        cleaning=cleaning,
        recurring_text=recurring,
        diarization_provider="tailnet_dgx",
        episode_text=" ".join(x for x in (episode.get("title"), episode.get("description")) if x)
        or None,
    )


def replay(roster: Any, ep: Dict[str, Any], ad_signatures: Any = None) -> Any:
    kw = roster_inputs(ep, roster, ad_signatures)
    dz, text = kw.pop("diarization"), kw.pop("transcript_text")
    return roster.resolve_speaker_roster(dz, text, **kw)


def _exec_module(name: str, path: Path) -> types.ModuleType:
    import importlib

    base = importlib.import_module(name) if name in sys.modules or _importable(name) else None
    mod = types.ModuleType(name)
    mod.__dict__.update(
        {
            "__name__": name,
            "__package__": name.rsplit(".", 1)[0],
            "__file__": getattr(base, "__file__", str(path)),
        }
    )
    sys.modules[name] = mod  # dataclasses resolve their module during exec
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def load_variant(files: Dict[str, Path]) -> Dict[str, types.ModuleType]:
    """Build the modules of one variant. Dependencies first, so the roster binds to them.

    The installed modules are restored afterwards; the returned roster keeps references to the
    variant's own dependency modules, captured at its import time.
    """
    saved = {MODULES[k]: sys.modules.get(MODULES[k]) for k in MODULES}
    built: Dict[str, types.ModuleType] = {}
    try:
        for key in ("ad_signatures", "hosts", "resolution", "roster"):
            if key in files:
                built[key] = _exec_module(MODULES[key], files[key])
            elif key == "roster":
                built[key] = _exec_module(MODULES[key], Path(sys.modules[MODULES[key]].__file__))
            elif key == "ad_signatures" and _importable(MODULES[key]):
                built[key] = sys.modules.get(MODULES[key]) or __import__(
                    MODULES[key], fromlist=["*"]
                )
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
    return built


def compare(
    episodes: Iterable[Dict[str, Any]],
    old: Dict[str, types.ModuleType],
    new: Dict[str, types.ModuleType],
    *,
    signatures_old: Any = None,
    signatures_new: Any = None,
) -> Iterable[Dict[str, Any]]:
    """Yield one record per voice whose published name changed, then a summary record."""
    counts: Counter = Counter()
    for ep in episodes:
        try:
            a = replay(old["roster"], ep, signatures_old)
            b = replay(new["roster"], ep, signatures_new)
        except Exception as exc:  # noqa: BLE001 — one bad episode must not stop the replay
            counts["replay_error"] += 1
            counts[f"error:{type(exc).__name__}"] += 1
            continue
        counts["episodes"] += 1
        segs = [s for s in ep["segs"] if s.get("speaker")]
        total = sum(s["end"] - s["start"] for s in segs) or 1.0
        for v in sorted(set(a.by_voice) | set(b.by_voice)):
            x, y = a.by_voice.get(v), b.by_voice.get(v)
            xn = x.name if x and x.named else None
            yn = y.name if y and y.named else None
            if xn == yn:
                continue
            kind = "gained" if not xn else ("lost" if not yn else "renamed")
            counts[kind] += 1
            talk = sum(s["end"] - s["start"] for s in segs if s.get("speaker") == v)
            yield {
                "kind": kind,
                "feed": (ep["meta"].get("feed") or {}).get("title"),
                "episode": (ep["meta"].get("episode") or {}).get("title"),
                "voice": v,
                "share": round(talk / total, 3),
                "old": xn,
                "new": yn,
                "old_type": getattr(x, "voice_type", None),
                "new_type": getattr(y, "voice_type", None),
                "opening": " ".join(str(s.get("text") or "") for s in segs if s["speaker"] == v)[
                    :120
                ],
            }
    yield {"summary": dict(counts)}


def corpus_episodes(corpus: Path) -> Iterable[Dict[str, Any]]:
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
        discover_metadata_files,
    )

    for meta in dedupe_metadata_paths_newest_run_per_episode(
        corpus, discover_metadata_files(corpus)
    ):
        try:
            yield load_episode(Path(meta))
        except (OSError, ValueError, KeyError):
            continue


def _variant_arg(pairs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for pair in pairs or []:
        key, _, path = pair.partition("=")
        if key not in MODULES or not path:
            raise SystemExit(f"bad variant file {pair!r}; use one of {sorted(MODULES)}=PATH")
        out[key] = Path(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--old", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument("--new", nargs="*", default=[], metavar="MODULE=PATH")
    ap.add_argument(
        "--signatures",
        action="store_true",
        help="build cross-show ad signatures in memory from the corpus (each side with its own "
        "ad_signatures module) and pass them to classify_voices",
    )
    args = ap.parse_args(argv)
    old = load_variant(_variant_arg(args.old))
    new = load_variant(_variant_arg(args.new))
    sig_old = sig_new = None
    if args.signatures:
        sig_old = _signatures(old.get("ad_signatures"), args.corpus)
        sig_new = _signatures(new.get("ad_signatures"), args.corpus)
    for rec in compare(
        corpus_episodes(args.corpus), old, new, signatures_old=sig_old, signatures_new=sig_new
    ):
        print(json.dumps(rec, ensure_ascii=False))
    return 0


def _signatures(module: Optional[types.ModuleType], corpus: Path) -> Any:
    if module is None:
        return None
    doc = module.build(module._corpus_episodes(corpus))
    return module.AdSignatures(frozenset(doc["recurring"]), frozenset(doc["ad_languages"]))


if __name__ == "__main__":
    sys.exit(main())
