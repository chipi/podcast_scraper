"""Which episodes were transcribed from UNPREPROCESSED audio? (#18 / #558)

THE DAMAGE
Audio preprocessing used to run under a flat 300-second budget regardless of input size. On a
long episode it hit that wall, produced nothing, and the pipeline fell back to uploading the
ORIGINAL file — unnormalised, un-trimmed, full-size — to the STT provider. The transcript that
came back is the artifact, and every downstream artifact derives from it.

WHY THIS NEEDS A DETECTOR
The GI repair (``gi.repair``) rebuilds insights from the transcript. It cannot fix a transcript.
``reprocess-corpus-from-transcripts`` runs with ``transcribe=off`` by design. So transcript-layer
damage survives every repair we have, and until something can NAME the affected episodes,
"should we re-transcribe?" is unanswerable — which is how it stays unfixed.

THE SIGNATURE, from run-level ``metrics.json`` that every run already writes:

    preprocessing_attempts >= 1  AND  preprocessing_count == 0

Preprocessing was asked for and produced nothing, so the original file is what went to the
provider. Corroborated by ``avg_preprocessing_wall_ms`` sitting at the old flat budget and by
zero size reduction.

MEASURED 2026-08-17 on two local corpora, and the split is stark:

    pre-#558  (2026-08-15, 15 runs)   9 DAMAGED (60 %), wall_ms 297,064-300,845
    post-#558 (2026-08-16, 14 runs)   0 damaged, reductions 50-90 %

One post-fix run took 324,114 ms — it would have been killed outright by the old flat 300 s
budget. That is the fix working, and it is also why the pre-fix episodes cluster so tightly at
the wall.

SCOPE LIMIT — READ BEFORE TRUSTING A COUNT
``metrics.json`` is RUN-level, not per-episode. A run that processed ONE episode attributes
exactly; a run that processed several reports only how many attempted versus completed, so it
can say "this run has N damaged episodes" but not always WHICH. Per-episode attribution needs
the ``audio_preprocessing`` stage-ledger row added in #22, which by construction only exists on
runs made after that change — i.e. never on the damaged ones. The report states which runs are
ambiguous rather than guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The flat budget #558 replaced. A wall time within ~10 % of it is the fingerprint of the old
#: timeout rather than a coincidental slow run.
LEGACY_FLAT_BUDGET_MS = 300_000


@dataclass
class RunPreprocessing:
    """One run's preprocessing metrics, plus the episode ids that run wrote.

    Carries ``episode_ids`` alongside the counters because ``metrics.json`` is RUN-level: the
    counters say how many episodes were attempted and completed, and only the id list can say
    WHICH — and then only when the run held exactly one episode (``attribution_is_exact``).
    """

    run_dir: str
    attempts: Optional[int]
    completed: Optional[int]
    wall_ms: Optional[float]
    size_reduction_pct: Optional[float]
    saved_bytes: Optional[int]
    episodes_in_run: int
    episode_ids: List[str] = field(default_factory=list)

    @property
    def damaged(self) -> bool:
        """Preprocessing was asked for and did not complete for EVERY episode it was asked for.

        ``completed < attempts``, not ``completed == 0``. The original rule only fired when a run
        preprocessed NOTHING, which is correct on the one-episode-per-run corpora it was written
        against and silently wrong everywhere else: a production run of 50 episodes where 45
        preprocessed and 5 hit the wall records attempts=50/completed=45 and read as HEALTHY,
        hiding all 5.

        That is the same failure I have now made four times — validating a rule only in the
        degenerate case where one run means one episode. Production runs are batches.

        Cache hits count as completed: ``record_preprocessing_time`` fires on a hit, so a reused
        preprocessed file is a legitimate completion, not a miss.
        """
        if not self.attempts or self.attempts < 1:
            return False
        return (self.completed or 0) < self.attempts

    @property
    def partially_damaged(self) -> bool:
        """Some episodes in the run preprocessed and some did not — attribution is impossible.

        Distinguished from a total failure because it changes what an operator can conclude:
        every episode in such a run is a suspect, and none can be cleared.
        """
        return self.damaged and (self.completed or 0) > 0

    @property
    def hit_legacy_wall(self) -> bool:
        """Wall time sits at the old flat 300 s budget — the #558 fingerprint."""
        if self.wall_ms is None:
            return False
        return abs(self.wall_ms - LEGACY_FLAT_BUDGET_MS) <= LEGACY_FLAT_BUDGET_MS * 0.10

    @property
    def attribution_is_exact(self) -> bool:
        """One episode in the run means the run-level metric names that episode exactly."""
        return self.episodes_in_run == 1


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _episode_ids_in_run(run_dir: Path) -> List[str]:
    out: List[str] = []
    for meta in sorted((run_dir / "metadata").glob("*.metadata.json")):
        doc = _read_json(meta)
        if not doc:
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id") or doc.get("episode_id")
        if isinstance(eid, str) and eid.strip():
            out.append(eid.strip())
    return out


def _current_run_dirs(corpus_root: Path) -> Optional[set]:
    """Run dirs that hold at least one CURRENT episode, per the corpus membership rule.

    Returns ``None`` when the rule cannot be applied, so callers can fall back to scanning
    everything and say so.
    """
    try:
        from ..search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

        all_meta = sorted(corpus_root.rglob("*.metadata.json"))
        members = dedupe_metadata_paths_newest_run_per_episode(corpus_root, all_meta)
        return {str(Path(p).parent.parent) for p in members}
    except Exception as exc:  # noqa: BLE001 - the gate must still run without the search extra
        logger.debug("corpus membership rule unavailable (%s); scanning all runs", exc)
        return None


def assess_preprocessing(
    corpus_root: Path,
    *,
    current_only: bool = True,
) -> List[RunPreprocessing]:
    """One record per run that wrote ``metrics.json``.

    ``current_only`` (default) restricts to runs holding at least one CURRENT episode.

    WHY THAT DEFAULT — found by an end-to-end repair test 2026-08-17. A run's ``metrics.json`` is
    immutable history: after an episode is successfully re-transcribed into a NEW run dir, the old
    run's metrics still read ``attempts=1 completed=0 wall=297183ms`` forever. Counting every run
    therefore means the audit can NEVER go green after a repair — the identical flaw the
    placeholder gate had, reintroduced here in a different file.

    Scoping to corpus members asks the question that matters: is the copy of this episode the
    corpus actually SERVES built from unpreprocessed audio? Superseded runs are history, not
    damage.

    Pass ``current_only=False`` for a forensic view of every run ever made.
    """
    keep = _current_run_dirs(corpus_root) if current_only else None

    runs: List[RunPreprocessing] = []
    for metrics_path in sorted(corpus_root.rglob("metrics.json")):
        d = _read_json(metrics_path)
        if d is None:
            continue
        run_dir = metrics_path.parent
        if keep is not None and str(run_dir) not in keep:
            continue
        ids = _episode_ids_in_run(run_dir)
        runs.append(
            RunPreprocessing(
                run_dir=str(run_dir),
                attempts=d.get("preprocessing_attempts"),
                completed=d.get("preprocessing_count"),
                wall_ms=d.get("avg_preprocessing_wall_ms"),
                size_reduction_pct=d.get("avg_preprocessing_size_reduction_percent"),
                saved_bytes=d.get("total_preprocessing_saved_bytes"),
                episodes_in_run=len(ids),
                episode_ids=ids,
            )
        )
    return runs


def damaged_episode_ids(corpus_root: Path) -> List[str]:
    """The work-list: episode_ids to feed ``--reprocess-episode-ids``.

    Includes episodes from AMBIGUOUS runs (several episodes, one run-level metric). That is
    deliberate: the run demonstrably transcribed from raw audio and the metric cannot say which
    episode, so treating every episode in it as suspect over-repairs rather than under-repairs.
    Re-transcribing a healthy episode wastes money; skipping a damaged one leaves the corpus
    wrong, and nothing downstream would ever reveal it.
    """
    out: List[str] = []
    for run in assess_preprocessing(corpus_root):
        if run.damaged:
            out.extend(run.episode_ids)
    return sorted(set(out))


def episode_durations_seconds(corpus_root: Path) -> Dict[str, float]:
    """``{episode_id: duration_seconds}`` for every episode on disk that states one.

    Used to price a work-list before anyone runs it. Episodes whose metadata carries no usable
    duration are simply ABSENT from the mapping rather than defaulted — a missing duration must
    read as "unknown", never as "free", because a zero would quietly make an expensive episode
    look affordable.
    """
    out: Dict[str, float] = {}
    for meta in sorted(corpus_root.rglob("*.metadata.json")):
        doc = _read_json(meta)
        if not doc:
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id") or doc.get("episode_id")
        if not (isinstance(eid, str) and eid.strip()):
            continue
        raw = ep.get("duration_seconds")
        try:
            seconds = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            out.setdefault(eid.strip(), seconds)
    return out


def chunk_ids_by_cost(
    ids: Sequence[str],
    durations: Dict[str, float],
    *,
    budget_usd: float,
    usd_per_minute: float,
) -> Tuple[List[List[str]], List[str]]:
    """Pack ``ids`` into batches each estimated to cost at most ``budget_usd``.

    Returns ``(chunks, unpriced)``. ``unpriced`` are ids with no known duration; they are
    returned SEPARATELY rather than distributed into chunks, because an episode that cannot be
    priced cannot be shown to fit and silently padding a batch with them is how a "$5 batch"
    becomes a $20 one.

    Greedy in the given order, and deliberately so: the order is the audit's, which groups
    related episodes, and re-sorting to pack tighter would scramble a list an operator reads.
    A single episode costing more than the whole budget still gets its own chunk — refusing to
    emit it would silently drop work from the list.
    """
    chunks: List[List[str]] = []
    unpriced: List[str] = []
    current: List[str] = []
    current_cost = 0.0

    for eid in ids:
        seconds = durations.get(eid)
        if seconds is None:
            unpriced.append(eid)
            continue
        cost = (seconds / 60.0) * usd_per_minute
        if current and (current_cost + cost) > budget_usd:
            chunks.append(current)
            current, current_cost = [], 0.0
        current.append(eid)
        current_cost += cost

    if current:
        chunks.append(current)
    return chunks, unpriced


def _work_list_header(corpus_root: Path) -> str:
    return (
        "# Episodes transcribed from UNPREPROCESSED audio (#18/#558).\n"
        f"# Corpus: {corpus_root}\n"
        "# Feed to: podcast-scraper --reprocess-episode-ids <this file>\n"
        "# This re-runs ASR (cost!) and cascades diarization/GI/KG.\n"
    )


def write_work_list(
    corpus_root: Path,
    destination: Path,
    *,
    chunk_budget_usd: Optional[float] = None,
    usd_per_minute: float = 0.0043,
) -> int:
    """Write the damaged episode_ids one per line. Returns how many were written.

    With ``chunk_budget_usd`` set, writes ``<destination>.001``, ``.002``, … instead, each sized
    so its estimated ASR cost fits the budget, and each stating that estimate in its header.

    WHY CHUNKING EXISTS. The unit of work that matters is a COST, not a count: a 3-hour episode
    costs 3.5x a 51-minute one, so "16 episodes" is not a budget. Splitting by hand is arithmetic
    an operator should not have to do at 1am, and getting it wrong is what the per-run cap then
    refuses — correctly, but only after the dispatch. ``usd_per_minute`` defaults to the deepgram
    nova-3 rate that the 2026-08-18 bill matched; pass the rate for whatever provider is
    configured. This function names no provider.
    """
    ids = damaged_episode_ids(corpus_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = _work_list_header(corpus_root)

    if not chunk_budget_usd or chunk_budget_usd <= 0:
        destination.write_text(header + "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        return len(ids)

    durations = episode_durations_seconds(corpus_root)
    chunks, unpriced = chunk_ids_by_cost(
        ids, durations, budget_usd=float(chunk_budget_usd), usd_per_minute=usd_per_minute
    )

    for n, chunk in enumerate(chunks, start=1):
        minutes = sum(durations[e] for e in chunk) / 60.0
        part = destination.with_name(f"{destination.name}.{n:03d}")
        part.write_text(
            header + f"# BATCH {n} of {len(chunks)}: {len(chunk)} episode(s), "
            f"{minutes / 60:.1f} audio-hours, est. ${minutes * usd_per_minute:.2f}\n"
            + "\n".join(chunk)
            + "\n",
            encoding="utf-8",
        )

    if unpriced:
        # Their own file, never folded into a priced batch: these cannot be shown to fit.
        part = destination.with_name(f"{destination.name}.unpriced")
        part.write_text(
            header
            + f"# {len(unpriced)} episode(s) with NO known duration — cost UNKNOWN, not zero.\n"
            "# Run these deliberately and watch the spend; they are not in any batch above.\n"
            + "\n".join(unpriced)
            + "\n",
            encoding="utf-8",
        )

    return len(ids)


def check_corpus_preprocessing(corpus_root: Path) -> Tuple[bool, str]:
    """``(ok, report)`` — ``ok`` is False when any run transcribed from unpreprocessed audio.

    Reports rather than repairs: the fix is re-transcription, which is expensive and is the
    operator's call. The point is to make the number KNOWN.
    """
    runs = assess_preprocessing(corpus_root)
    damaged = [r for r in runs if r.damaged]
    exact = [r for r in damaged if r.attribution_is_exact]
    ambiguous = [r for r in damaged if not r.attribution_is_exact]
    at_wall = [r for r in damaged if r.hit_legacy_wall]

    lines = [
        f"Corpus: {corpus_root}",
        f"  runs with metrics        : {len(runs)}",
        f"  runs DAMAGED             : {len(damaged)}",
        f"    of which at the 300s wall: {len(at_wall)}  (the #558 flat-budget fingerprint)",
        "",
    ]

    if damaged:
        lines.append("  Episodes transcribed from UNPREPROCESSED audio:")
        for r in exact:
            eid = r.episode_ids[0] if r.episode_ids else "(no episode_id)"
            wall = f"{r.wall_ms:.0f}ms" if r.wall_ms is not None else "-"
            lines.append(f"    {eid}   wall={wall}   {r.run_dir}")
        if ambiguous:
            lines.append("")
            lines.append(
                "  Runs where preprocessing failed but the run held SEVERAL episodes — the"
            )
            lines.append(
                "  run-level metric cannot say which. Treat every episode in these as suspect:"
            )
            for r in ambiguous:
                lines.append(
                    f"    {r.run_dir}  episodes={r.episodes_in_run} "
                    f"attempts={r.attempts} completed={r.completed}"
                )

    lines.append("")
    lines.append("  NOT COVERED BY THIS VERDICT")
    lines.append(
        "    Damage is in the TRANSCRIPT. gi-repair rebuilds insights FROM the transcript and"
    )
    lines.append(
        "    cannot fix it; reprocess-corpus-from-transcripts runs transcribe=off by design."
    )
    lines.append(
        "    The only repair is re-transcription (make redo-diarization / --reprocess-source),"
    )
    lines.append("    which re-runs ASR and cascades GI/KG. That is an explicit cost decision.")
    no_metrics = [r for r in runs if r.attempts is None]
    lines.append(
        f"    runs with no preprocessing metrics at all: {len(no_metrics)} "
        "(cannot be judged either way)"
    )
    idle = [r for r in runs if not r.attempts and r.episodes_in_run]
    lines.append(
        f"    runs that attempted NO preprocessing at all: {len(idle)} (reported as not damaged,"
    )
    lines.append(
        "      because a run that never preprocessed damaged nothing — but a repair run served"
    )
    lines.append(
        "      entirely from the transcript cache looks identical here, having repaired nothing."
    )
    lines.append(
        "      After a repair, assert positively that attempts >= 1 (runbook step 8), and pass"
    )
    lines.append("      --no-transcript-cache so the cache cannot replay the damaged transcript.")

    lines.append("")
    lines.append(f"VERDICT: {'PASS' if not damaged else 'FAIL'}")
    if not runs:
        lines.append("  NOTE: no metrics.json found — check CORPUS_DIR points at a corpus.")

    return (not damaged), "\n".join(lines)
