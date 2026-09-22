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
unequal. ``utils.filesystem.names_the_same_episode`` is the corrected test — it lives in the
package, not here, because the PIPELINE needs the same answer: ``_transcript_beside_metadata``
uses it to refuse a stored pointer that names another episode, and two implementations would be
free to disagree about which episodes are damaged. It rescues 128 downloaded
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.search.corpus_scope import (  # noqa: E402
    dedupe_metadata_paths_newest_run_per_episode,
)
from podcast_scraper.utils.filesystem import (  # noqa: E402
    names_the_same_episode as _names_the_same_episode,
)

#: A screenplay line: ``LABEL: text``. The label is the resolved speaker name once naming has run.
_LINE = re.compile(r"^([^:\n]{1,60}): ")

#: Variant suffixes that are not the base transcript.
_VARIANTS = (".adfree.", ".cleaned.")


def _metadata_files(corpus: Path, newest_run_only: bool = False) -> List[Path]:
    """Every ``*.metadata.json`` under the corpus, in both layouts the corpus uses.

    TWO DENOMINATORS, AND THEY ANSWER DIFFERENT QUESTIONS. Undeduped (the default) is
    what the MIGRATIONS see: m0007/m0009/m0010 all ``root.rglob`` with no dedupe, so a
    superseded run's copy is read and rewritten like any other, and this number is the
    one that says how many artifacts m0009 will read with a corrupted roster.

    ``newest_run_only`` applies the central corpus-membership rule instead — one winner
    per ``(feed_id, episode_id)`` — which is what the SERVING layer and
    ``_on_disk_guid_index`` use. That number says how much damage is actually rendered
    and how much of it ``relabel_only`` can reach: a superseded copy can never be
    repaired, because the repair resolves through the newest-run index.

    Measured on prod 2026-09-22: 2,312 undeduped vs 2,002 newest-run-only. Reporting
    only one of them is how "147 mispaired" and "49 mispaired" both become true
    statements about the same corpus.
    """
    seen: Dict[str, Path] = {}
    for pattern in (
        "run_*/metadata/*.metadata.json",
        "metadata/*.metadata.json",
        "feeds/*/run_*/metadata/*.metadata.json",
        "feeds/*/metadata/*.metadata.json",
    ):
        for p in corpus.glob(pattern):
            seen[str(p)] = p
    files = sorted(seen.values())
    if newest_run_only:
        files = sorted(dedupe_metadata_paths_newest_run_per_episode(corpus, files))
    return files


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


#: A run stamp in a filename: ``_20260729-003748`` plus the optional feed-hash suffix.
_RUN_STAMP = re.compile(r"_\d{8}-\d{6}(?:_[0-9a-f]{6,10})?")


def _same_episode_allowing_stale_stamp(meta_stem: str, transcript_stem: str) -> bool:
    """Same episode, tolerating a transcript filename that kept an OLDER run's stamp.

    THE 43 "NEEDS RE-DOWNLOAD + RE-ASR + RE-DIARIZE" WERE ALL THIS, AND ALL FALSE.
    A reprocess can carry a transcript forward into the new run directory under its
    ORIGINAL name, so the file sits in the right place with a stale stamp:

        metadata : 0012 - AI Agents and the Future of Glob_20260805-174108_7a69fc41
        file     : 0012 - AI Agents and the Future of Glob_20260729-003748_ba02775e
                   ^^^^^^^^^^^^^ same title ^^^^^^^^^^^^^^ ^^^^ differs ^^^^

    ``names_the_same_episode`` strips the common TAIL and compares prefixes, which
    handles a truncated title within one run but not this: the stamps differ mid-string,
    neither stem prefixes the other, and the record is reported as pointing at another
    episode's transcript. It points at its own.

    Verified by content, not by name: the pointed transcript for the episode above is
    labelled ``Kuo Zhang`` 22 times — the episode's own guest — against a title of
    "AI Agents and the Future of Global Trade with Alibaba's Kuo Zhang".

    Measured on `snapshot-prod-20260920` AND on live prod, independently: of the serving
    view's findings, the same-name-modulo-stamp class is exactly the population marked
    "needs re-ingest", and NONE of it is repairable. Stripping stamps turns those
    findings into zero and leaves the genuinely cross-episode damage untouched — which
    is the whole difference between a GPU bill and no GPU bill.
    """
    if _names_the_same_episode(meta_stem, transcript_stem):
        return True
    return _RUN_STAMP.sub("", meta_stem) == _RUN_STAMP.sub("", transcript_stem)


def audit(corpus: Path, newest_run_only: bool = False) -> Tuple[List[dict], int]:
    """``(findings, total_examined)``. A finding is one metadata file pointing elsewhere."""
    findings: List[dict] = []
    total = 0
    own_cache: Dict[str, Dict[str, Path]] = {}

    for meta in _metadata_files(corpus, newest_run_only=newest_run_only):
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
        if _same_episode_allowing_stale_stamp(stem, Path(rel).name[: -len(".txt")]):
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
        episode_block = data.get("episode") if isinstance(data.get("episode"), dict) else {}
        findings.append(
            {
                "metadata": str(meta),
                "feed": str((data.get("feed") or {}).get("title") or feed_root.name),
                "episode": str(episode_block.get("title") or ""),
                # The REPAIR needs an id, not a title. `--reprocess-episode-ids` matches on
                # `episode_id` and falls back to the guid, so emit whichever this record has —
                # without it the 42 repairable episodes have to be looked up by hand.
                "episode_id": str(
                    episode_block.get("episode_id") or episode_block.get("guid") or ""
                ),
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
    ap.add_argument(
        "--newest-run-only",
        action="store_true",
        help=(
            "Count only the newest run per episode — the SERVING view, and the only "
            "population `relabel_only` can repair. Default counts every run, which is "
            "what the migrations actually read. See _metadata_files."
        ),
    )
    ap.add_argument(
        "--worklist",
        type=Path,
        default=None,
        help=(
            "Write the REPAIRABLE episode ids here, one per line, for "
            "`--reprocess-episode-ids`. Only the episodes whose own transcript is still on disk: "
            "the rest need a re-ingest and must not ride along in a cheap relabel batch."
        ),
    )
    args = ap.parse_args(argv)

    corpus = Path(args.corpus_dir)
    if not corpus.is_dir():
        print(f"not a directory: {corpus}", file=sys.stderr)
        return 2

    findings, total = audit(corpus, newest_run_only=args.newest_run_only)

    if args.worklist is not None:
        ids = sorted(
            {str(f["episode_id"]) for f in findings if f["repairable"] and f["episode_id"]}
        )
        missing = sum(1 for f in findings if f["repairable"] and not f["episode_id"])
        args.worklist.parent.mkdir(parents=True, exist_ok=True)
        args.worklist.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        print(
            f"work-list: {len(ids)} repairable episode id(s) -> {args.worklist}"
            + (
                f"  ({missing} repairable record(s) carry NO id and are NOT listed)"
                if missing
                else ""
            ),
            file=sys.stderr,
        )
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
