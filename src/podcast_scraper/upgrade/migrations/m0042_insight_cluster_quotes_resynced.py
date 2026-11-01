"""0042 — each insight cluster's quotes match their GI quotes again (text only).

RFC-131 D2 item 5, and a derived surface of 0041. Record:
``docs/wip/changes/0042_insight_cluster_quotes_resynced.md``.

``search/insight_clusters.json`` copies each supporting quote's ``text``, ``speaker_id``,
``char_start`` and ``char_end`` from the GI when it is built, and nothing resyncs it. 0038 took a
raw-label ``speaker_id`` off GI quotes (prod's D1 count found one copy left here), and 0041 moves
quote offsets when a speaker prefix changes length. This takes each copied field from the member's
own GI quote, matched by ``(episode_id, insight_id, quote_id)`` in a GI file holding both the
insight and the quote; with several runs, the newest served one. A ``speaker_id`` the GI quote no
longer has becomes ``null``, as a fresh build writes it. Membership, order, scores and every other
field stay; a quote with no GI match is kept as it is and counted. UNDO restores from
``.podcast_scraper/upgrade-backups/0042/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...utils.corpus_walk import corpus_rglob
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0042_insight_cluster_quotes_resynced"
RECEIPTS_FILE = "insight_cluster_quotes_resynced.jsonl"
BACKUP_TAG = "0042"
CLUSTERS_REL = Path("search") / "insight_clusters.json"
FIELDS = ("text", "speaker_id", "char_start", "char_end")

_Key = Tuple[str, str, str]


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gi_quotes(root: Path) -> Tuple[Dict[_Key, Dict[str, Any]], int]:
    """``({(episode_id, insight_id, quote_id): the GI quote's copied fields}, duplicated keys)``.

    Every ``*.gi.json`` of the corpus is read (``corpus_rglob``), as the cluster builder reads
    them; a key found in several files takes the newest served run's
    (``select_served_artifacts``)."""
    served = {
        str(m)[: -len(".metadata.json")] + ".gi.json"
        for m in select_served_artifacts(root, ".metadata.json")[0]
    }
    out: Dict[_Key, Dict[str, Any]] = {}
    from_served: Dict[_Key, bool] = {}
    duplicated = 0
    for gi_path in sorted(corpus_rglob(root, "*.gi.json")):
        gi = _load(gi_path)
        if not isinstance(gi, dict):
            continue
        episode_id = str(gi.get("episode_id", gi_path.stem))
        nodes = {n.get("id"): n for n in gi.get("nodes") or [] if isinstance(n, dict)}
        is_served = str(gi_path) in served
        for edge in gi.get("edges") or []:
            if not isinstance(edge, dict) or edge.get("type") != "SUPPORTED_BY":
                continue
            insight, quote = nodes.get(edge.get("from")), nodes.get(edge.get("to"))
            if not insight or not quote or quote.get("type") != "Quote":
                continue
            key = (episode_id, str(insight["id"]), str(quote["id"]))
            props = quote.get("properties") or {}
            fields = {f: props.get(f) for f in FIELDS}
            if key in out:
                duplicated += 1
                if from_served.get(key) or not is_served:
                    continue
            out[key] = fields
            from_served[key] = is_served
    return out, duplicated


def resynced(payload: Any, quotes: Dict[_Key, Dict[str, Any]]) -> Tuple[Any, Dict[str, int]]:
    """``(payload with each member quote's copied fields from its GI quote, counts)``; the input is
    not modified."""
    out = json.loads(json.dumps(payload))
    counts: Dict[str, int] = {"quotes": 0, "unmatched": 0, "quotes_resynced": 0}
    for cluster in out.get("clusters") or [] if isinstance(out, dict) else []:
        for member in cluster.get("members") or []:
            for q in member.get("supporting_quotes") or []:
                counts["quotes"] += 1
                key = (
                    str(member.get("episode_id")),
                    str(member.get("insight_id")),
                    str(q.get("quote_id")),
                )
                gi = quotes.get(key)
                if gi is None:
                    counts["unmatched"] += 1
                    continue
                changed = [f for f in FIELDS if q.get(f) != gi[f]]
                for f in changed:
                    q[f] = gi[f]
                    counts[f"{f}_changed"] = counts.get(f"{f}_changed", 0) + 1
                counts["quotes_resynced"] += bool(changed)
    return out, counts


def _plan(root: Path) -> Tuple[Optional[Any], Dict[str, int]]:
    path = root / CLUSTERS_REL
    payload = _load(path)
    if payload is None:
        return None, {}
    quotes, duplicated = gi_quotes(root)
    new, counts = resynced(payload, quotes)
    counts["gi_keys_in_several_runs"] = duplicated
    return (new if counts["quotes_resynced"] else None), counts


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore the file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class InsightClusterQuotesResyncedMigration(Migration):
    """The insight clusters' copied quote fields resynced from the GI on disk."""

    id = MIGRATION_ID
    to_version = "2.7.36"
    description = (
        "search/insight_clusters.json's supporting quotes take their GI quotes' current text, "
        "speaker_id and offsets (0038's removed ids, 0041's moved offsets); membership kept"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Count the quotes apply() would resync; pure read."""
        new, counts = _plan(ctx.corpus_root)
        return f"insight cluster quotes plan: {counts or 'no clusters file'}"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when no cluster quote differs from its GI quote (or there is no clusters file)."""
        new, counts = _plan(ctx.corpus_root)
        if new is not None:
            return False, f"{counts['quotes_resynced']} cluster quote(s) still differ from the GI"
        return True, "every insight cluster quote matches its GI quote"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite the clusters file when any quote differs, with a backup and a receipt."""
        root = ctx.corpus_root
        new, counts = _plan(root)
        receipts: List[Dict[str, str]] = []
        if new is not None and not ctx.dry_run:
            receipts.append(write_with_backup(root, BACKUP_TAG, root / CLUSTERS_REL, new))
            append_receipts(root, RECEIPTS_FILE, {"migration": self.id, "counts": counts}, receipts)
        verb = "would resync" if ctx.dry_run else "resynced"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {counts.get('quotes_resynced', 0)} cluster quote(s); {counts}",
            details={"counts": counts, "files_written": len(receipts)},
        )


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row for the clusters file, with its counts and every file its plan read."""
    from ..read_recorder import recording

    root = Path(root)
    with recording(root) as rec:
        new, counts = _plan(root)
    if new is None and not counts:
        return
    yield {"meta": str(CLUSTERS_REL), "refused": None, "counts": counts, "inputs": rec.inputs()}
