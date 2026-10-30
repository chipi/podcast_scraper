#!/usr/bin/env python3
"""Every derived surface a migration's writes leave stale, from its receipts (method D17).

    python scripts/measure/derived_surface_census.py --corpus <after-state clone> \\
        --receipts named_voices_shown.jsonl

A migration rewrites some of an episode's artifacts. Other files are BUILT from those and are not
rewritten by it, so they are stale until refreshed and the run steps must name each one. Staleness
is decided by what each surface is built from (``DEPENDS``), never by file times: a file older than
the writes is stale only if it reads something the migration changed.

Per episode (keyed by its metadata path), for each surface in ``DEPENDS``: stale when its input was
written for that episode and the surface itself was not. For the ad map, which holds character
offsets into the transcript, it also checks the offsets against the transcript on disk
(``source_length`` against the text's length). Corpus-level enrichments and the request-time
readers are listed with their refresh. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

#: surface suffix -> (the written suffixes it is built from, how it is refreshed)
DEPENDS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    ".insight_density.json": ((".gi.json",), "re-enrich the episode (deterministic, GI+KG)"),
    ".insight_sentiment.json": ((".gi.json",), "re-enrich the episode (deterministic, GI+KG)"),
    ".manifest.json": ((".txt", ".cleaned.txt", ".adfree.txt"), "rewrite the transcript manifest"),
    ".anon.txt": ((".txt",), "re-render the anonymised transcript"),
    ".adfree.admap.json": ((".txt",), "rebuild the ad map from the transcript (offsets)"),
    ".context.json": ((".metadata.json", ".kg.json"), "rebuild the context digest"),
    ".bridge.json": ((".gi.json", ".kg.json"), "rebuild the bridge"),
    ".turns.json": ((".txt", ".segments.json"), "rebuild the turns"),
}
SUFFIXES = sorted(
    {s for s in DEPENDS}
    | {d for deps, _ in DEPENDS.values() for d in deps}
    | {".segments.json", ".adfree.segments.json", ".speakers.diagnostics.json"},
    key=len,
    reverse=True,
)
#: Corpus-level outputs built from every episode's GI/KG, and how they are refreshed.
CORPUS_LEVEL = "the nightly's corpus enrichers (deterministic) and insight clusters"
#: Request-time readers of insight visibility / speaker names. Their caches key on
#: ``perf_cache.corpus_mtime``, which includes ``upgrade_ledger.json``: an apply invalidates them.
READERS = (
    "server/gi_view.py (insights_from_gi: the surfaceable gate)",
    "server/routes/app_episodes.py (episode insights, _related_cache)",
    "server/app_auto_picks.py, app_board_view.py, app_captured_rail.py, app_episode_notes.py, "
    "app_moments.py, app_recap_view.py, relational_view.py, app_trending_episodes.py",
    "og/build.py (share cards); mcp/tools/gi.py; search/corpus_search.py (speaker scope)",
)


def _split(name: str) -> Tuple[str, str]:
    """``(episode stem, suffix)`` by the known suffixes, longest first (titles contain dots)."""
    for sfx in SUFFIXES:
        if name.endswith(sfx):
            return name[: -len(sfx)], sfx
    return name, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--receipts", required=True)
    args = ap.parse_args()
    root: Path = args.corpus.resolve()
    written: Dict[Tuple[Path, str], Set[str]] = defaultdict(set)  # (run, stem) -> suffixes written
    for line in open(root / args.receipts, encoding="utf-8"):
        rel = json.loads(line).get("relpath")
        if not rel:
            continue
        p = (root / rel).resolve()
        stem, sfx = _split(p.name)
        written[(p.parent.parent, stem)].add(sfx)
    stale: Counter = Counter()
    admap_off: List[str] = []
    for (run, stem), sfxs in sorted(written.items()):
        for surface, (deps, _how) in DEPENDS.items():
            sub = (
                "metadata/enrichments"
                if surface.startswith(".insight_")
                else ("metadata" if surface in (".context.json", ".bridge.json") else "transcripts")
            )
            path = run / sub / f"{stem}{surface}"
            if path.is_file() and surface not in sfxs and any(d in sfxs for d in deps):
                stale[surface] += 1
                if surface == ".adfree.admap.json":
                    admap = json.loads(path.read_text(encoding="utf-8"))
                    txt = run / "transcripts" / f"{stem}.txt"
                    n = admap.get("source_length")
                    if (
                        txt.is_file()
                        and isinstance(n, int)
                        and n != len(txt.read_text(encoding="utf-8"))
                    ):
                        size = len(txt.read_text(encoding="utf-8"))
                        admap_off.append(f"{txt.name}: source_length {n} vs text {size}")
    print(f"episodes written: {len(written)}")
    print("per-episode surfaces built from what was written, and not rewritten (stale):")
    for surface, (deps, how) in DEPENDS.items():
        frm = "/".join(deps)
        print(f"  {surface:26} stale {stale[surface]:>4}   from {frm:34} refresh: {how}")
    print(f"ad maps whose offsets no longer match their transcript: {len(admap_off)}")
    for line in admap_off:
        print(f"    {line}")
    retext = sum(1 for sfxs in written.values() if ".txt" in sfxs)
    print(
        f"search index: transcript chunk and quote rows of the {retext} episode(s) whose "
        "transcript text was rewritten (incremental index of those episodes); insight rows "
        "carry no speaker or visibility field"
    )
    print(f"corpus level: {CORPUS_LEVEL}")
    print("request-time readers (re-read on the next request, no refresh step):")
    for r in READERS:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
