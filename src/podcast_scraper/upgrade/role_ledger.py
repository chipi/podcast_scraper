"""What m0009 changed, and how to put it back (#2069).

m0009's demotions were irreversible, and that one fact set the bar for everything around them:
every predicate had to be provably correct BEFORE the run. That is why the show-name guard took
three attempts, why demotions need a human to read them, and why "can we migrate production" kept
meaning "prove zero damage in advance".

Some of those predicates CANNOT be perfected. The feed title genuinely cannot distinguish
``Lex Fridman Podcast`` (host is a person) from ``Latent Space: The AI Engineer Podcast`` (the show
really is called that). A ledger moves the bar to "bound the damage and keep the receipt".

FORMAT: append-only JSONL, one row per changed node, at the corpus root beside
``upgrade_ledger.json``. Append-only because the first version used ``os.replace`` and a second
migration run silently discarded the first run's rows — while the docstring promised the next
migration could append to it. Each row carries ``run_id``; undo defaults to the newest run.

THIS IS A FACT ABOUT A RUN, not about an episode, which is why it lives at the corpus root rather
than in per-episode sidecars. (Provenance — #2070 — is the opposite: a fact about an episode's
VALUE, so it belongs inside the artifact.) The objection to a root file is that it can go stale
against the artifacts it describes; ``file_sha_after`` closes that, because every row can be
validated against the file it claims to have written.

WHAT THE UNDO REFUSES, AND WHY IT IS A FILE HASH RATHER THAN A ROLE. Role-equality detects
"someone moved this node to a DIFFERENT role" and is blind to "someone rewrote this file and
happened to agree". That blindness is the LIKELY production sequence, not a corner case: runbook
step 3 is a re-enrich, ``rederive_only`` / ``rediarize_only`` cascade to GI/KG, and the rebuilt
graph now reads the roster — so it writes ``host`` for most of the same nodes m0009 promoted. An
undo afterwards would demote all of them, report zero refusals, and call it a clean rollback while
destroying an independently-derived answer. So the ledger records each file's sha256 AS IT LEFT THE
MIGRATION, and a changed file refuses the whole episode.

THREE OUTCOMES, not two. ``restored`` / ``skipped`` (already back where it started — an undo run
twice, or an episode the migration never actually wrote because it crashed first) / ``refused``
(something else owns this file now). The first version reported its own previous work as "something
else wrote this node", which sends an operator hunting a concurrent writer that does not exist.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Corpus-root sibling of ``upgrade_ledger.json``. Named for what it holds rather than for m0009,
#: because the next role-writing migration appends to it rather than inventing a second format.
LEDGER_FILE = "speaker_roles_ledger.jsonl"

MIGRATION_ID = "0009_backfill_speaker_roles"

#: Sentinel for "this node had no ``role`` key at all". NOT the string "none": the first version
#: recorded that, wrote it back verbatim on undo, and left the node holding a role that is not in
#: ``_PROMOTABLE`` — so a re-run could never repair it. The undo made the node WORSE than not
#: undoing at all, and every fixture set a role, so the suite never saw it.
ABSENT = None


@dataclass(frozen=True)
class RoleChange:
    """One Person node's role transition, with enough context to judge it without the corpus."""

    episode: str  #: corpus-relative path of the ``.kg.json`` that was rewritten
    node_id: str
    name: str
    #: ``None`` means the node carried no ``role`` key. Stored RAW — not lower-cased — so an undo
    #: restores exactly what was there.
    role_before: Optional[str]
    role_after: str
    #: Which rule made the change — the first question when a demotion looks wrong.
    #: ``not_a_person`` / ``roster_denies`` / ``promote``.
    route: str
    feed_title: str = ""
    #: sha256 of the artifact as the migration left it. Undo refuses the episode if it differs.
    file_sha_after: str = ""
    #: Index of the node within ``nodes``, so same-id duplicates in one file stay addressable.
    node_index: int = -1
    run_id: str = ""


def file_sha(path: Path) -> str:
    """sha256 of *path*, or ``""`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def new_run_id() -> str:
    """A stable id for one migration run, so undo can scope to the newest."""
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def append_ledger(root: Path | str, changes: Sequence[RoleChange]) -> Path:
    """Append *changes* to the corpus ledger. Returns the path.

    Append-only and flushed per call, so a crash mid-migration leaves a ledger describing the
    episodes that HAD been written rather than no ledger at all. The first version wrote once at
    the end: a crash on the second episode left the first one rewritten with nothing recording it,
    and a later undo then reported a complete rollback having missed it entirely.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LEDGER_FILE
    with path.open("a", encoding="utf-8") as fh:
        for change in changes:
            fh.write(json.dumps(asdict(change), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_ledger(root: Path | str, *, run_id: Optional[str] = None) -> List[RoleChange]:
    """Every recorded change, newest run first when *run_id* is omitted.

    Raises ``ValueError`` on a corrupt ledger. A missing ledger is ``[]`` — missing and unreadable
    are different facts, and the first version collapsed them, so a corrupt ledger printed
    "nothing to undo" and exited 0.
    """
    path = Path(root) / LEDGER_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: List[RoleChange] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        try:
            rows.append(
                RoleChange(
                    episode=str(row["episode"]),
                    node_id=str(row["node_id"]),
                    name=str(row.get("name") or ""),
                    role_before=row.get("role_before"),
                    role_after=str(row["role_after"]),
                    route=str(row.get("route") or ""),
                    feed_title=str(row.get("feed_title") or ""),
                    file_sha_after=str(row.get("file_sha_after") or ""),
                    node_index=int(row.get("node_index", -1)),
                    run_id=str(row.get("run_id") or ""),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{lineno} is missing {exc}") from exc
    if run_id is not None:
        return [r for r in rows if r.run_id == run_id]
    return rows


def latest_run_id(root: Path | str) -> Optional[str]:
    """The most recently appended ``run_id``, or ``None``."""
    rows = read_ledger(root)
    return rows[-1].run_id if rows else None


def _node_at(payload: dict, row: RoleChange) -> Optional[dict]:
    """The node this row addresses — by index when recorded, else by id."""
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return None
    if 0 <= row.node_index < len(nodes):
        node = nodes[row.node_index]
        if isinstance(node, dict) and str(node.get("id") or "") == row.node_id:
            return node
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id") or "") == row.node_id:
            return node
    return None


def undo_from_ledger(
    root: Path | str, *, run_id: Optional[str] = None
) -> Tuple[int, List[str], List[str]]:
    """Restore every recorded ``role_before``. Returns ``(restored, skipped, refused)``.

    Scoped to the newest run unless *run_id* says otherwise.

    An episode is REFUSED WHOLE when its file's sha256 differs from what the migration wrote —
    see the module docstring. SKIPPED means the node is already at ``role_before``: an undo run
    twice, or an episode the migration recorded but never wrote because it crashed first.
    """
    root = Path(root)
    if run_id is None:
        run_id = latest_run_id(root)
    if run_id is None:
        return 0, [], []
    changes = read_ledger(root, run_id=run_id)
    if not changes:
        return 0, [], []

    # HOLD THE CORPUS LOCK. Undo is a read-modify-write over many artifacts; an ingest running
    # concurrently rewrites some of them underneath it and the undo loses the race silently. The
    # per-file sha check narrows that window — a file rewritten BEFORE we read it is refused — but
    # it cannot close a rewrite that lands between our read and our write. The lock does.
    #
    # WHAT MAY BE SWALLOWED, AND WHAT MAY NOT. `corpus_parent_lock` raises RuntimeError when a
    # LIVE process holds the lock — which is the ingest-is-running case, the entire reason this
    # lock is here. An earlier version wrapped the whole thing in `except Exception` and fell back
    # to running UNLOCKED, so the one situation the lock existed for was the one situation it was
    # skipped in. Contention now propagates: refusing loudly is the correct answer, because the
    # rollback is not safe to perform and the operator can stop the ingest and re-run.
    # Only "there is no lock to take" (module missing, unwritable lock dir) degrades to unlocked.
    #
    # `_undo_locked` is deliberately OUTSIDE the try. Inside it, any error it raised — an
    # unreadable artifact mid-write would do it — re-entered the fallback and ran the ENTIRE undo
    # a second time, unlocked, before raising from the second pass.
    with contextlib.ExitStack() as stack:
        try:
            from ..utils.corpus_lock import corpus_parent_lock

            stack.enter_context(corpus_parent_lock(root))
        except (ImportError, OSError):
            pass
        return _undo_locked(root, changes)


def _undo_locked(root: Path, changes: List[RoleChange]) -> Tuple[int, List[str], List[str]]:
    """The body of :func:`undo_from_ledger`, with the corpus lock already held."""

    by_episode: Dict[str, List[RoleChange]] = {}
    for change in changes:
        by_episode.setdefault(change.episode, []).append(change)

    restored = 0
    skipped: List[str] = []
    refused: List[str] = []
    for relpath, rows in sorted(by_episode.items()):
        path = root / relpath
        expected = next((r.file_sha_after for r in rows if r.file_sha_after), "")
        actual = file_sha(path)
        if not actual:
            refused.append(f"{relpath}: unreadable")
            continue
        if expected and actual != expected:
            # Something else owns this file now — a re-enrich, a later migration, or a previous
            # undo. Refusing the WHOLE episode, not per node: a file we did not write last is a
            # file whose node indices and contents we cannot reason about.
            if all(
                (_node_at(json.loads(path.read_text(encoding="utf-8")), r) or {})
                .get("properties", {})
                .get("role")
                == r.role_before
                for r in rows
            ):
                skipped.append(f"{relpath}: already restored ({len(rows)} node(s))")
            else:
                refused.append(
                    f"{relpath}: file changed since the migration wrote it "
                    f"(sha {actual[:12]} != {expected[:12]}) — something else owns it; left alone"
                )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        touched = False
        for row in rows:
            node = _node_at(payload, row)
            if node is None:
                refused.append(f"{row.node_id} in {relpath}: node is gone")
                continue
            props = node.setdefault("properties", {})
            current = props.get("role")
            if current == row.role_before:
                skipped.append(f"{row.node_id} in {relpath}: already restored")
                continue
            if current != row.role_after:
                refused.append(
                    f"{row.node_id} in {relpath}: role is {current!r}, ledger expected "
                    f"{row.role_after!r}; left alone"
                )
                continue
            if row.role_before is ABSENT:
                # The node had NO role key. Restoring the string "none" would leave it holding a
                # role no promoter recognises — unrepairable by a re-run.
                props.pop("role", None)
            else:
                props["role"] = row.role_before
            restored += 1
            touched = True
        if touched:
            # BYTE-FOR-BYTE the serialisation m0009 used (`_write_atomic`: indent=2 + newline), so
            # a file the migration wrote comes back byte-identical and "did the undo work?" is
            # answerable by comparing hashes. NOTE: a kg.json last written by the PIPELINE uses
            # `kg/io.py` (ensure_ascii=False, no trailing newline), so those come back
            # role-identical but not byte-identical — such files are refused by the sha check
            # above anyway.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".undo-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, indent=2) + "\n")
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    if restored:
        # A ROLLBACK THE REST OF THE SYSTEM CANNOT SEE IS NOT A ROLLBACK.
        #
        # `upgrade_ledger.json` still recorded 0009 as applied, so `runner.status()` — which
        # computes pending as "not in applied" — would skip it on the next `upgrade run`. The
        # ledger claimed 2.7.3 while the roles were pre-2.7.3, and nothing in the system could tell.
        #
        # It also moves `perf_cache.corpus_mtime`, which tokens on that file: without this, every
        # in-process projection keeps serving POST-migration roles after the rollback — S6
        # re-created by the undo.
        try:
            from .state import FilesystemStateStore

            FilesystemStateStore(root).record_reverted(MIGRATION_ID)
        except Exception:  # noqa: BLE001 — the roles are already restored; never undo the undo
            pass
    return restored, skipped, refused
