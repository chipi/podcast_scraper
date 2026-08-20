"""Which episodes lost their summary, how many times, and which now need a human (#1686).

Marko, 2026-08-20: "it's never acceptable to have an episode without the summary of a single one
... ideally requeue and try to reprocess the episode at least once if there's some mechanism
around it" — and, on the terminal case: "if episode fails on re-queue then it is finally failed
for pipeline and we need to manually investigate."

So there are three states, and collapsing any two of them is how this issue happened:

    healthy            no `summarization: failed` in the ledger
    RETRYABLE          failed on one run — requeue it
    TERMINAL           failed on two or more runs — the requeue was already tried and did not
                       work, so dispatching it again just spends money to fail again. A human
                       looks at it.

WHY THE ATTEMPT COUNT IS DERIVED, NOT STORED
Every pipeline run writes a fresh ``run_<ts>/``, and old runs stay on disk. "How many runs
recorded this episode's summary as failed" is therefore already a fact about the corpus — no
counter to keep in sync, nothing to reset, and it stays true across restores from backup. A
stored counter would be a second source of truth for something the artifacts already say.

This module reports and produces a work-list. It never re-summarises: that costs provider money
and is the operator's call, which is the same line ``preprocessing/audit.py`` draws for
re-transcription.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Attempts after which an episode stops being requeued and becomes a human's problem. One
#: automatic retry, then stop — matching the in-run retry policy rather than inventing a second.
MAX_SUMMARY_ATTEMPTS = 2

STAGE = "summarization"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _summary_failure(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The `summarization: failed` ledger record in *doc*, or None.

    A ``degraded`` outcome is deliberately NOT a failure here: that is the
    ``deadline_exceeded_but_completed`` case — a summary that was slow but real — and also the
    in-flight ``retrying_*`` marker. Neither means the episode lost its summary.
    """
    processing = doc.get("processing")
    if not isinstance(processing, dict):
        return None
    ledger = processing.get("stage_ledger")
    if not isinstance(ledger, dict):
        return None
    record = ledger.get(STAGE)
    if not isinstance(record, dict) or record.get("outcome") != "failed":
        return None
    return record


def _episode_id(doc: Dict[str, Any]) -> Optional[str]:
    episode = doc.get("episode")
    if isinstance(episode, dict):
        eid = episode.get("episode_id")
        if isinstance(eid, str) and eid.strip():
            return eid.strip()
    eid = doc.get("episode_id")
    return eid.strip() if isinstance(eid, str) and eid.strip() else None


def assess_summaries(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """``episode_id -> {attempts, reasons, terminal, latest_has_summary}`` for every episode
    that lost its summary on at least one run.

    Scans EVERY run, not just the newest, because the attempt count is the whole point. An
    episode whose latest run produced a summary is excluded even if an older run failed — it
    recovered, and listing recovered episodes on a repair work-list is how a work-list becomes
    something nobody reads.
    """
    per_episode: Dict[str, Dict[str, Any]] = {}
    latest_ok: Dict[str, Tuple[str, bool]] = {}
    seen: set = set()

    for meta_path in sorted(corpus_root.rglob("*.metadata.json")):
        doc = _read_json(meta_path)
        if not doc:
            continue
        eid = _episode_id(doc)
        if not eid:
            continue
        run_key = str(meta_path.parent.parent.name)
        has_summary = isinstance(doc.get("summary"), dict)
        prev = latest_ok.get(eid)
        if prev is None or run_key >= prev[0]:
            latest_ok[eid] = (run_key, has_summary)

        seen.add(eid)
        record = _summary_failure(doc)
        if record is None:
            continue
        slot = per_episode.setdefault(eid, {"attempts": 0, "reasons": Counter()})
        slot["attempts"] += 1
        reason = record.get("reason")
        if isinstance(reason, str):
            slot["reasons"][reason] += 1

    # THE DEFECT CONDITION IS "the newest run of this episode has no summary" — not "the ledger
    # says failed". The ledger supplies the CAUSE and the attempt count; it must not be what
    # decides whether an episode counts, because then any path that loses a summary without
    # recording it stays invisible, which is precisely the bug this module exists for. An
    # episode with no summary and no ledger record predates #1647 (or found a path nobody
    # tagged) — still serving without a summary, cause unknown, and reported as `unattributed`
    # rather than dropped.
    out: Dict[str, Dict[str, Any]] = {}
    for eid in sorted(seen):
        _run, has_summary = latest_ok.get(eid, ("", False))
        if has_summary:
            continue  # recovered on a later run — not a defect any more
        slot = per_episode.get(eid) or {"attempts": 0, "reasons": Counter({"unattributed": 1})}
        out[eid] = {
            "attempts": slot["attempts"],
            "reasons": dict(slot["reasons"]),
            "terminal": slot["attempts"] >= MAX_SUMMARY_ATTEMPTS,
            "latest_has_summary": has_summary,
        }
    return out


def retryable_episode_ids(corpus_root: Path) -> List[str]:
    """Episode ids worth ONE requeue — sorted, so the work-list is stable across runs."""
    assessed = assess_summaries(corpus_root)
    return sorted(eid for eid, row in assessed.items() if not row["terminal"])


def terminal_episode_ids(corpus_root: Path) -> List[str]:
    """Episode ids the pipeline has given up on. These need a person, not another dispatch."""
    assessed = assess_summaries(corpus_root)
    return sorted(eid for eid, row in assessed.items() if row["terminal"])


def _terminal_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".terminal")


def previously_terminal(destination: Path) -> set:
    """Ids a previous pass already gave up on.

    THIS IS WHAT MAKES THE LOOP GUARD DURABLE, and leaving it out made the guard oscillate
    instead of hold. Escalating an id moves it OUT of the main work-list and into the terminal
    sidecar; without reading the sidecar back, the next pass no longer recognises the id as
    "seen before", treats it as new, and dispatches it again — dispatch, escalate, dispatch,
    escalate, forever. Caught by the local end-to-end run (three passes), not by the unit tests,
    which only ever did two.
    """
    out: set = set()
    try:
        body = _terminal_path(destination).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eid = line.partition("#")[0].strip()
        if eid:
            out.add(eid)
    return out


def _previous_attempts(destination: Path) -> Dict[str, int]:
    """Attempt counts recorded in a previously written work-list, if one is there.

    The work-list is its own state file. Each line is ``<episode_id>  # attempts=<n>``, so the
    next run can tell "this episode has been dispatched before and got nowhere" from "this is
    the first time I have seen it" — without a database, and without a counter that a corpus
    restore would silently reset.
    """
    out: Dict[str, int] = {}
    try:
        body = destination.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eid, _, comment = line.partition("#")
        eid = eid.strip()
        if not eid:
            continue
        # -1 means "this id was listed before, but the line carries no provable attempt count"
        # (hand-edited file, stripped comments, an older format). That must escalate, not reset:
        # an unprovable count is the same situation as no progress, and defaulting it to 0 would
        # hand a loop back to anything that rewrites this file. Guard hard.
        n = -1
        marker = "attempts="
        if marker in comment:
            try:
                n = int(comment.split(marker, 1)[1].split()[0])
            except (ValueError, IndexError):
                n = -1
        out[eid] = n
    return out


def write_work_list(corpus_root: Path, destination: Path) -> int:
    """Write the retryable ids one per line for ``--reprocess-episode-ids``. Returns the count.

    Terminal episodes are written to ``<destination>.terminal`` instead of being silently
    dropped: "not on the list" and "the pipeline gave up on this" must not look the same, which
    is the mistake this whole issue is about.

    LOOP GUARD. ``assess_summaries`` derives the attempt count from run dirs, which is sound as
    long as every requeue leaves evidence. A requeue that dies BEFORE writing metadata leaves the
    corpus byte-identical, so the same episode would be re-emitted with the same count forever —
    an automated dispatcher would then retry it endlessly at provider expense. So an episode that
    was already on the previous work-list and whose attempt count has NOT risen is escalated to
    terminal: no new evidence after a dispatch is itself the evidence that dispatching is not
    working. Marko, 2026-08-20: "we need to guard against looping hard."
    """
    assessed = assess_summaries(corpus_root)
    previous = _previous_attempts(destination)
    already_terminal = previously_terminal(destination)

    retryable, terminal = [], []
    for eid, row in sorted(assessed.items()):
        if row["terminal"]:
            terminal.append(eid)
        elif eid in already_terminal:
            # Given up on by an earlier pass. Note this is only reached when the episode is
            # STILL missing a summary — a recovered one never gets here, because
            # `assess_summaries` drops it. So "terminal" is durable without being permanent.
            row["terminal"] = True
            row["no_progress"] = True
            terminal.append(eid)
        elif eid in previous and (previous[eid] < 0 or row["attempts"] <= previous[eid]):
            row["terminal"] = True
            row["no_progress"] = True
            terminal.append(eid)
        else:
            retryable.append(eid)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"# Episodes persisted WITHOUT a summary, worth one requeue (#1686).\n"
        f"# Corpus: {corpus_root}\n"
        f"# Feed to: podcast-scraper --reprocess-episode-ids <this file>\n"
        f"# Re-summarisation costs provider money — read the list first.\n"
        + "\n".join(f"{eid}  # attempts={assessed[eid]['attempts']}" for eid in retryable)
        + ("\n" if retryable else ""),
        encoding="utf-8",
    )
    if terminal:
        _terminal_path(destination).write_text(
            f"# TERMINAL: failed on {MAX_SUMMARY_ATTEMPTS}+ runs (#1686). The requeue was already\n"
            f"# tried and did not work — dispatching these again spends money to fail again.\n"
            f"# These need a person: check the `reason` slug in each episode's stage ledger.\n"
            f"# `no-progress` means it was dispatched and the corpus gained no new evidence —\n"
            f"# the requeue is not reaching the episode at all, which is a different bug.\n"
            f"# Corpus: {corpus_root}\n"
            + "\n".join(
                eid + ("  # no-progress" if assessed[eid].get("no_progress") else "")
                for eid in terminal
            )
            + "\n",
            encoding="utf-8",
        )
    return len(retryable)


def check_corpus_summaries(corpus_root: Path) -> Tuple[bool, str]:
    """``(ok, report)`` — ok is False when ANY episode is serving without a summary.

    Not "few enough to tolerate". One episode without a summary is one episode the product
    misdescribes, and a threshold here is how 8 of them became normal.
    """
    assessed = assess_summaries(corpus_root)
    if not assessed:
        return True, "Summary coverage: every episode has a summary."

    retryable = {e: r for e, r in assessed.items() if not r["terminal"]}
    terminal = {e: r for e, r in assessed.items() if r["terminal"]}
    causes: Counter = Counter()
    for row in assessed.values():
        causes.update(row["reasons"])

    lines = [
        f"Summary coverage: {len(assessed)} episode(s) have NO summary.",
        f"  retryable (requeue): {len(retryable)}",
        f"  TERMINAL (needs a person): {len(terminal)}",
        "  causes:",
    ]
    lines += [f"    {slug}: {n}" for slug, n in causes.most_common()]
    if terminal:
        lines.append("  terminal episode ids:")
        lines += [f"    {eid} ({assessed[eid]['attempts']} attempts)" for eid in sorted(terminal)]
    return False, "\n".join(lines)
