"""Does every episode's metadata point at ITS OWN transcript? (#2082)

On a reprocess, ``idx`` used to be assigned from the number in the episode's FILENAME. Every run
directory numbers its episodes from ``0001``, so a feed with fourteen run dirs has fourteen
"episode 1"s — and ``idx`` keys per-episode state and output filenames, so they collided. The
result is metadata written for one episode carrying a DIFFERENT episode's transcript path, and the
roster computed from that other episode's words.

Measured on the 2026-09-14 production snapshot: **147 of 2,256** metadata files (6.5%) point
elsewhere, every one of them a whisper transcription. Of those carrying real speaker names, 119 had
a roster matching the WRONG transcript and none matched their own. "DHH's new way of writing code"
credited to Addy Osmani; "How Kent Beck shapes the software engineering industry" to Grady Booch.

THE FIRST NUMBER PUBLISHED HERE WAS 275, AND IT WAS WRONG. A plain stem equality test called 128
episodes mispaired that are not: the metadata filename truncates the title to 32 characters and the
transcript filename does not, so ``0006 - This Funding Model is Helping Fi_<guid>`` and ``0006 -
This Funding Model is Helping Fight Climate Change_<guid>`` are the SAME episode and compared
unequal. ``_names_the_same_episode`` below is the corrected test. It rescues 128 downloaded
transcripts and NOT ONE of the 147 whisper mismatches — so the damage this audit exists to find is
the size it always was; only the false positives are gone. Anything scoped off the old 275 (repair
batches, GPU estimates, #2082's headline) is scoped off a number that was 87% too large.

The code fix makes ``idx`` unique per run and stops new damage. It repairs nothing already written,
which is why this exists: the post-deploy runbook must know the number BEFORE it runs a migration,
because every measurement it takes reads ``content.speakers`` — on an affected episode that field
describes a different episode.

Three outcomes per mismatched file, because they need different work:

``roster_matches_wrong``
    The speaker names appear in the transcript it points AT and not in its own. Confirmed
    misattribution: these episodes credit their quotes to another episode's people.

``roster_matches_own``
    Names appear in its own transcript. The pointer is stale but the roster is right.

``inconclusive``
    Names in neither or in both — usually a roster built from metadata rather than the transcript.
    These are NOT proven safe; they are unclassified, and nobody should quote the confirmed count
    as a total until they are looked at.

And two repair routes:

``repairable``
    The episode's own transcript (and its ``.segments.json``) is still on disk, so a scoped
    ``relabel_only`` fixes it. Cheap: no audio, no GPU.

``needs_reingest``
    Its transcript does not exist anywhere in the feed. ``relabel_only`` OVERWRITES the transcript
    it picks, so where episode A was relabelled onto B's file, A's was never written and B's was
    overwritten with A's labels. Only a re-download + re-ASR + re-diarize recovers those.

Read-only. Nothing here writes to the corpus. Exits 1 when any mismatch is found, so it can gate a
deploy step; ``--quiet-ok`` prints nothing when the corpus is clean.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

#: A screenplay line: ``LABEL: text``. The label is the resolved speaker name once naming has run.
_LINE = re.compile(r"^([^:\n]{1,60}): ")

#: Variant suffixes that are not the base transcript.
_VARIANTS = (".adfree.", ".cleaned.")


def _metadata_files(corpus: Path) -> List[Path]:
    """Every ``*.metadata.json`` under the corpus, in both layouts the corpus uses."""
    seen: Dict[str, Path] = {}
    for pattern in (
        "run_*/metadata/*.metadata.json",
        "metadata/*.metadata.json",
        "feeds/*/run_*/metadata/*.metadata.json",
        "feeds/*/metadata/*.metadata.json",
    ):
        for p in corpus.glob(pattern):
            seen[str(p)] = p
    return sorted(seen.values())


def _labels(path: Optional[Path]) -> Set[str]:
    """Speaker labels in a screenplay transcript."""
    if path is None or not path.is_file():
        return set()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    out: Set[str] = set()
    for line in text.split("\n"):
        m = _LINE.match(line)
        if m:
            out.add(m.group(1).strip())
    return out


def _own_transcripts(feed_dir: Path) -> Dict[str, Path]:
    """``{stem: transcript}`` for every base transcript anywhere in this feed."""
    out: Dict[str, Path] = {}
    for t in feed_dir.rglob("transcripts/*.txt"):
        if any(v in t.name for v in _VARIANTS):
            continue
        out[t.name[: -len(".txt")]] = t
    return out


def _feed_root(meta_path: Path, corpus: Path) -> Path:
    """The feed workspace a metadata file belongs to (``<corpus>/feeds/<slug>/``).

    Falls back to the corpus root for the flat single-feed layout.
    """
    for parent in meta_path.parents:
        if parent.parent.name == "feeds":
            return parent
    return corpus


def _names_the_same_episode(meta_stem: str, transcript_stem: str) -> bool:
    """Do these two filenames name the same episode, allowing for a truncated title?

    The metadata filename truncates the episode title; the transcript filename does not. Both keep
    the same trailing identifier — a per-episode guid for a downloaded transcript, a per-RUN
    timestamp for a transcribed one. So strip the common tail and the remainder is the title as each
    file spells it; if one spells a prefix of the other, they are the same episode.

    This is deliberately NOT "skip ``direct_download`` episodes". Excluding by transcript source
    would hide a genuine mispairing on a downloaded transcript — and there is exactly one in the
    production snapshot, which this test still reports.

    The residual risk is stated rather than hidden: two episodes in the SAME run whose titles agree
    for the first 32 characters would compare equal here. Nothing in the snapshot does.
    """
    if meta_stem == transcript_stem:
        return True
    n = 0
    while (
        n < min(len(meta_stem), len(transcript_stem))
        and meta_stem[-1 - n] == transcript_stem[-1 - n]
    ):
        n += 1
    a = meta_stem[: len(meta_stem) - n].rstrip()
    b = transcript_stem[: len(transcript_stem) - n].rstrip()
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def audit(corpus: Path) -> Tuple[List[dict], int]:
    """``(findings, total_examined)``. A finding is one metadata file pointing elsewhere."""
    findings: List[dict] = []
    total = 0
    own_cache: Dict[str, Dict[str, Path]] = {}

    for meta in _metadata_files(corpus):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        content = data.get("content") or {}
        rel = str(content.get("transcript_file_path") or "").strip()
        if not rel:
            continue
        total += 1
        stem = meta.name[: -len(".metadata.json")]
        if _names_the_same_episode(stem, Path(rel).name[: -len(".txt")]):
            continue

        feed_root = _feed_root(meta, corpus)
        key = str(feed_root)
        if key not in own_cache:
            own_cache[key] = _own_transcripts(feed_root)
        own = own_cache[key].get(stem)

        names = {
            str(s.get("name") or "") for s in (content.get("speakers") or []) if isinstance(s, dict)
        }
        names = {n for n in names if n and not n.startswith("SPEAKER_")}

        pointed = (meta.parent.parent / rel).resolve()
        in_pointed = bool(names & _labels(pointed))
        in_own = bool(names & _labels(own))
        if not names:
            verdict = "inconclusive"
        elif in_pointed and not in_own:
            verdict = "roster_matches_wrong"
        elif in_own and not in_pointed:
            verdict = "roster_matches_own"
        else:
            verdict = "inconclusive"

        segments = own.with_name(own.name[: -len(".txt")] + ".segments.json") if own else None
        findings.append(
            {
                "metadata": str(meta),
                "feed": str((data.get("feed") or {}).get("title") or feed_root.name),
                "episode": str((data.get("episode") or {}).get("title") or ""),
                "points_at": Path(rel).name,
                "verdict": verdict,
                "repairable": bool(own and segments and segments.is_file()),
                "speakers": sorted(names)[:4],
            }
        )
    return findings, total


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-dir", required=True, help="Corpus parent path")
    ap.add_argument("--json", action="store_true", help="Emit findings as JSON")
    ap.add_argument(
        "--quiet-ok", action="store_true", help="Print nothing when the corpus is clean"
    )
    ap.add_argument("--limit", type=int, default=10, help="Examples to show per verdict")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus_dir)
    if not corpus.is_dir():
        print(f"not a directory: {corpus}", file=sys.stderr)
        return 2

    findings, total = audit(corpus)
    if args.json:
        print(json.dumps({"examined": total, "findings": findings}, indent=2))
        return 1 if findings else 0

    if not findings:
        if not args.quiet_ok:
            print(f"transcript pairing: OK — {total} metadata files, every one points at its own.")
        return 0

    by_verdict = collections.Counter(f["verdict"] for f in findings)
    repairable = sum(1 for f in findings if f["repairable"])
    by_feed = collections.Counter(f["feed"] for f in findings)

    print("TRANSCRIPT PAIRING AUDIT (#2082)\n")
    print(f"  metadata files examined                      : {total}")
    print(
        f"  pointing at ANOTHER episode's transcript     : {len(findings)}"
        f"  ({len(findings)/total:.1%})"
        if total
        else ""
    )
    print(f"    confirmed misattribution (roster is wrong) : {by_verdict['roster_matches_wrong']}")
    print(f"    pointer stale but roster right             : {by_verdict['roster_matches_own']}")
    print(f"    unclassified — NOT proven safe             : {by_verdict['inconclusive']}")
    print()
    print(f"  repairable by scoped relabel_only            : {repairable}")
    print(f"  needs re-download + re-ASR + re-diarize      : {len(findings) - repairable}")

    print("\n  by feed:")
    for feed, n in by_feed.most_common(15):
        print(f"    {n:5}  {feed[:52]}")

    worst = [f for f in findings if f["verdict"] == "roster_matches_wrong"][: args.limit]
    if worst:
        print("\n  confirmed misattribution — these credit another episode's people:")
        for f in worst:
            print(f"    {f['episode'][:54]!r}")
            print(f"       speakers  : {f['speakers']}")
            print(f"       points at : {f['points_at'][:54]!r}")

    print("\n  This audit is READ-ONLY. See #2082 for the repair routes; run it again after.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
