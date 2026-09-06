#!/usr/bin/env python3
"""#1976 — per-feed ad-excision coverage across a corpus. Read-only, no GPU, no LLM.

The question this answers
-------------------------
**Is there a feed whose ads are not being detected?**

That is knowable *before* anything leaks into a summary, and it is a much stronger signal than
waiting for a leak: a feed whose episodes show near-zero excision while comparable feeds cut
regularly is a feed the detector is missing. The alternative — string-matching finished summaries
for "brought to you by" — cannot work, because the LLM paraphrases a sponsor read into a generic
claim carrying no markers (``gi.filters`` caught **0 of 1,200 insights**, see ``gi/ad_regions``).
So "no marker found" never meant "no ad content".

Why this needs no pipeline change
---------------------------------
Per-episode excision data is **already on disk**. ``workflow/adfree_transcript.py`` writes
``<base>.adfree.admap.json`` next to each transcript with the full ``AdRegionMetadata``, and it
writes an identity record even when nothing was cut — so denominators exist, not just positives.
``save_adfree_transcript`` defaults to True.

For episodes predating the sidecar, the detector is re-run in ``dry_run`` mode straight off the
raw ``.txt``. That is deterministic regex over the first/last 5,000 chars — no model, no cost —
so coverage is fully retroactive.

Reading the output
------------------
``cut%`` is the share of a feed's episodes where an ad region was actually excised. There is no
universal "right" number: a show with no sponsors should read 0%, and that is correct. What
matters is a feed sitting far below its peers, or a feed at 0% whose episodes are long and
commercial. The report therefore prints the corpus median so each feed can be read against it.

``raw`` counts episodes with no sidecar, measured by dry-run recompute. A high raw count is not a
defect — it just means those episodes predate the sidecar.

Usage
-----
    python scripts/audit/ad_excision_coverage.py <corpus_root> [--json out.json] [--quiet]

Exit codes
----------
- 0 — report produced.
- 2 — corpus root missing or unreadable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.gi.ad_regions import excise_ad_regions  # noqa: E402

ADMAP_SUFFIX = ".admap.json"


def _feed_of(path: Path, corpus_root: Path) -> str:
    """Feed directory name for a transcript path, or ``(top-level)``."""
    try:
        rel = path.relative_to(corpus_root)
    except ValueError:
        return "(outside corpus)"
    parts = rel.parts
    if "feeds" in parts:
        i = parts.index("feeds")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "(top-level)"


def _read_admap(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _recompute(transcript: Path) -> Optional[Dict[str, Any]]:
    """Dry-run the detector on a transcript with no sidecar. Deterministic, no model."""
    try:
        text = transcript.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        _cleaned, _segments, meta = excise_ad_regions(text, dry_run=True)
    except Exception:  # noqa: BLE001 - an audit must never take the corpus down with it
        return None
    return meta.to_dict()


def collect(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """Per-feed excision stats. Prefers the on-disk sidecar; recomputes when absent."""
    per_feed: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "cut": 0, "chars_removed": 0, "from_sidecar": 0, "recomputed": 0}
    )

    admaps = {p for p in corpus_root.rglob(f"*{ADMAP_SUFFIX}")}
    covered_bases = {str(p)[: -len(ADMAP_SUFFIX)].removesuffix(".adfree") for p in admaps}

    for admap in sorted(admaps):
        meta = _read_admap(admap)
        if meta is None:
            continue
        feed = _feed_of(admap, corpus_root)
        row = per_feed[feed]
        row["episodes"] += 1
        row["from_sidecar"] += 1
        removed = int(meta.get("chars_removed") or 0)
        row["chars_removed"] += removed
        if removed > 0:
            row["cut"] += 1

    for transcript in sorted(corpus_root.rglob("*.txt")):
        name = transcript.name
        if ".adfree" in name or ".segments" in name:
            continue
        if str(transcript)[: -len(".txt")] in covered_bases:
            continue
        meta = _recompute(transcript)
        if meta is None:
            continue
        feed = _feed_of(transcript, corpus_root)
        row = per_feed[feed]
        row["episodes"] += 1
        row["recomputed"] += 1
        removed = int(meta.get("chars_removed") or 0)
        row["chars_removed"] += removed
        if removed > 0:
            row["cut"] += 1

    return dict(per_feed)


def _rate(row: Dict[str, Any]) -> float:
    return (row["cut"] / row["episodes"]) if row["episodes"] else 0.0


def report(stats: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not stats:
        return ["no transcripts found — is this a corpus root?"]

    rates = [_rate(r) for r in stats.values() if r["episodes"] >= 3]
    median = statistics.median(rates) if rates else 0.0

    lines.append(f"{'feed':<44}{'eps':>5}{'cut':>5}{'cut%':>7}{'chars':>10}{'src':>14}")
    lines.append("-" * 86)
    for feed, row in sorted(stats.items(), key=lambda kv: _rate(kv[1])):
        src = f"{row['from_sidecar']}map/{row['recomputed']}raw"
        lines.append(
            f"{feed[:43]:<44}{row['episodes']:>5}{row['cut']:>5}"
            f"{100 * _rate(row):>6.0f}%{row['chars_removed']:>10,}{src:>14}"
        )

    total_eps = sum(r["episodes"] for r in stats.values())
    total_cut = sum(r["cut"] for r in stats.values())
    lines.append("")
    lines.append(
        f"corpus: {total_eps} episodes, {total_cut} with an ad region excised "
        f"({100 * total_cut / total_eps:.0f}%); per-feed median {100 * median:.0f}%"
    )

    # A feed well below its peers is the signal worth acting on. Feeds with <3 episodes are
    # excluded: one episode proves nothing either way.
    suspects = [(f, r) for f, r in stats.items() if r["episodes"] >= 3 and _rate(r) < median * 0.34]
    lines.append("")
    if suspects:
        lines.append("FAR BELOW the per-feed median — the detector may be missing these:")
        for feed, row in sorted(suspects, key=lambda kv: _rate(kv[1])):
            lines.append(
                f"  {feed[:50]:<52}{100 * _rate(row):>4.0f}%  ({row['cut']}/{row['episodes']})"
            )
        lines.append("")
        lines.append(
            "A 0% feed is not automatically a defect — a show with no sponsors reads 0% and that "
            "is correct. Check whether these episodes actually carry ads before changing anything."
        )
    else:
        lines.append("No feed sits far below the per-feed median.")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus_root", help="corpus directory (the one containing feeds/)")
    ap.add_argument("--json", dest="json_out", help="also write the raw stats here")
    ap.add_argument("--quiet", action="store_true", help="suppress the table; JSON only")
    args = ap.parse_args(argv)

    root = Path(args.corpus_root).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    stats = collect(root)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, indent=2, sort_keys=True), "utf-8")
    if not args.quiet:
        print(os.linesep.join(report(stats)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
