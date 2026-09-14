"""What m0009 changed, and how to put it back (#2065, reversibility).

m0009's demotions were irreversible, and that one fact set the bar for everything around them:
every predicate had to be provably correct BEFORE the run, because a wrong demotion could not be
taken back. That is why the show-name guard took three attempts, why demotions need a human to read
them, and why "can we run this on production" kept meaning "prove zero damage in advance".

Some of those predicates CANNOT be perfected. The feed title genuinely cannot distinguish
``Lex Fridman Podcast`` (host is a person) from ``Latent Space: The AI Engineer Podcast`` (the show
really is called that) — the difference is world knowledge. Chasing certainty there is chasing
something unobtainable.

A ledger moves the bar to "bound the damage and keep the receipt", which is both cheaper and more
honest. The data already existed and was being thrown away: ``apply()`` computes every
``(episode, node, role_before, role_after)`` transition and kept only the counts.

ONE FILE, TWO JOBS — deliberately. It answers "what did it do" and "put it back". A separate audit
artifact would be a second instrument free to drift from the first, which is the failure this whole
arc keeps rediscovering.

UNDO MUST NOT CLOBBER NEWER WORK. If a ``relabel_only`` re-enrich ran after the migration, a node's
role may be newer and better than anything here. Undo therefore refuses any node whose CURRENT role
is not the ``role_after`` we recorded, and reports it rather than forcing — a rollback that
overwrites a later fix is a regression wearing a rollback's clothes.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

#: Sits at the corpus root beside ``upgrade_ledger.json``. Named for what it holds rather than for
#: the migration, because the next role-writing migration should append to it rather than invent a
#: second format.
LEDGER_FILE = "speaker_roles_ledger.json"

MIGRATION_ID = "0009_backfill_speaker_roles"


@dataclass(frozen=True)
class RoleChange:
    """One Person node's role transition, with enough context to judge it without the corpus."""

    episode: str  #: corpus-relative path of the ``.kg.json`` that was rewritten
    node_id: str
    name: str
    role_before: str
    role_after: str
    #: Which rule made the change — the first question when a demotion looks wrong.
    #: ``not_a_person`` (show name / role word), ``roster_denies`` (roster accounted for every
    #: voice and this person was not among them), ``promote``.
    route: str
    feed_title: str = ""


def write_ledger(root: Path | str, changes: Sequence[RoleChange]) -> Path:
    """Write *changes* to the corpus ledger, atomically. Returns the path."""
    root = Path(root)
    path = root / LEDGER_FILE
    payload = {
        "migration": MIGRATION_ID,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "changes": [asdict(c) for c in changes],
    }
    # Same atomic-replace shape the migration uses for artifacts: a half-written ledger is worse
    # than none, because it would make undo restore a subset and call it a rollback.
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(root), prefix=".ledger-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_ledger(root: Path | str) -> List[RoleChange]:
    """Every recorded change, or ``[]`` when there is no ledger (which is not an error)."""
    path = Path(root) / LEDGER_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: List[RoleChange] = []
    for row in payload.get("changes") or []:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                RoleChange(
                    episode=str(row["episode"]),
                    node_id=str(row["node_id"]),
                    name=str(row.get("name") or ""),
                    role_before=str(row["role_before"]),
                    role_after=str(row["role_after"]),
                    route=str(row.get("route") or ""),
                    feed_title=str(row.get("feed_title") or ""),
                )
            )
        except KeyError:
            continue
    return out


def undo_from_ledger(root: Path | str) -> Tuple[int, List[str]]:
    """Restore every recorded ``role_before``. Returns ``(restored, refused)``.

    A node is REFUSED — not forced — when its current role is not the ``role_after`` this ledger
    recorded. That means something else has written the node since: a re-enrich, a later migration,
    or a previous undo. Replaying over it would overwrite work that is newer and probably better
    than anything here.

    Refusal is therefore also what makes a second undo a no-op rather than a corruption: the first
    undo leaves every node at ``role_before``, which no longer matches ``role_after``.
    """
    root = Path(root)
    changes = read_ledger(root)
    if not changes:
        return 0, []

    by_episode: dict[str, List[RoleChange]] = {}
    for change in changes:
        by_episode.setdefault(change.episode, []).append(change)

    restored = 0
    refused: List[str] = []
    for relpath, rows in sorted(by_episode.items()):
        path = root / relpath
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            refused.extend(f"{r.node_id} in {relpath}: unreadable ({exc})" for r in rows)
            continue
        nodes = {
            str(n.get("id") or ""): n
            for n in (payload.get("nodes") or [])
            if isinstance(n, dict) and n.get("type") == "Person"
        }
        touched = False
        for row in rows:
            node = nodes.get(row.node_id)
            if node is None:
                refused.append(f"{row.node_id} in {relpath}: node is gone")
                continue
            props = node.setdefault("properties", {})
            current = str(props.get("role") or "")
            if current != row.role_after:
                refused.append(
                    f"{row.node_id} in {relpath}: role is {current!r}, ledger expected "
                    f"{row.role_after!r} — something else wrote this node; left alone"
                )
                continue
            props["role"] = row.role_before
            restored += 1
            touched = True
        if touched:
            # BYTE-FOR-BYTE the same serialisation the migration used (`_write_atomic`:
            # `indent=2` plus a trailing newline). An undo that restores every role but rewrites
            # every file's formatting is not a clean rollback — it shows up as a whole-corpus diff
            # in git or rsync, and makes "did the undo work?" unanswerable by comparing hashes.
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
    return restored, refused
