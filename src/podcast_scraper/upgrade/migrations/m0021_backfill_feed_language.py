"""0021 — fetch each show's declared RSS ``<language>`` and backfill it onto its episodes (#2173).

WHY A BACKFILL IS NEEDED. Before #2172 nothing parsed the RSS ``<language>``; what sits in
``feed.language`` on disk is the RUN CONFIG written back out, so every existing episode claims the
profile's language whatever its publisher declared, and no episode carries a language of its own.
``skip_existing`` is GUID-keyed, so a normal pipeline run never rewrites an already-processed
episode — this migration is the only path.

PER SHOW, NOT PER EPISODE. The language is a property of the feed, so one fetch covers every
episode under it, and the report is per show.

IT FETCHES. This is the first migration here that touches the network, and that is deliberate.
The first draft split the fetch into an operator-run script and left this step applying a mapping
file — which is broken in the context migrations actually run in: ``runner.py:128`` records a
migration only ``if result.applied``, so a step that found no input file would return
``applied=False``, stay pending, and re-log "run the script first" on every single upgrade for
ever, while the backfill never happened. A migration that cannot complete itself is a false
promise. ``upgrade run`` is an explicit deploy/operator step (``Makefile:1543``), not a hot path,
so ~20 sequential HTTP GETs are affordable there.

HOW IT CONVERGES, since a fetch can fail. ``applied`` is True only when no show is left in a
retryable state, so a run that resolves 14 of 20 writes those 14 (idempotently), returns False,
stays pending, and picks up the remaining 6 on the next upgrade. Two states are NOT retryable and
must not block completion for ever:

* a feed the corpus has no ``url`` for — it can never be fetched, and the registered CI migration
  fixture is exactly that shape, so treating it as retryable would mean the migration never
  completes in CI;
* a feed that fetches fine and declares no ``<language>`` — a definite answer, just not a language.

Both are reported and skipped. Nothing is ever guessed: an episode whose show could not be
resolved is left exactly as it was.

IT IS UNDOABLE, which it was not when first written. Every write went through a bare
``path.write_text``, so the ONLY record that a value had been replaced was the new value itself —
and this migration overwrites five fields on every served episode in the corpus, fetching the
replacement over the network. A publisher serving a wrong or changed ``<language>`` (they are
routinely wrong, which is why ``language_override`` exists) would have been stamped across
hundreds of artifacts with no way back short of reprocessing. ``m0012`` and ``m0014`` already had
the answer — ``file_rewrite``'s backup + receipt + :func:`undo` — and this one simply did not use
it. Receipts are appended PER SHOW rather than once at the end, so a crash or a killed upgrade
loses at most the current show's receipts instead of every one taken so far.

ONE THING IT MUST NOT TOUCH: an episode whose ``language_source`` is already ``override``. The
per-feed override is the remedy for a publisher declaring the wrong tag, so those are the exact
episodes where the ``<language>`` this migration fetches is known to be worse than what is on
disk. Overwriting them would invert the precedence ``resolve_episode_language`` defines and undo
an operator's correction at deploy time. See :func:`_is_operator_override`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ... import overrides as ov
from ...languages import normalize_language_tag, SOURCE_OVERRIDE, SOURCE_RSS
from ...rss.parser import _channel_language
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

#: Per-feed HTTP timeout, seconds. Overridable via ``ctx.options["fetch_timeout"]``.
DEFAULT_FETCH_TIMEOUT = 20.0

MIGRATION_ID = "0021_backfill_feed_language"
RECEIPTS_FILE = "backfill_feed_language.jsonl"
BACKUP_TAG = "0021"


def _block(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The named top-level object, or an empty dict when absent or the wrong shape.

    A helper rather than an inline ternary at each site: the ternary form does not narrow for
    the type checker, and the alternative was eleven ignores.
    """
    value = payload.get(name)
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, str(exc)
    return (payload, "") if isinstance(payload, dict) else (None, "not a JSON object")


def _shows_and_episodes(
    paths: List[Path],
) -> Tuple[Dict[str, str], Dict[str, List[Path]], List[str]]:
    """``({feed_id: url}, {feed_id: [metadata paths]}, unparsable)``.

    A show with no ``url`` still gets an episode list — it is reported as unfetchable, not
    dropped, so the count in the message accounts for every episode in the corpus.
    """
    urls: Dict[str, str] = {}
    episodes: Dict[str, List[Path]] = {}
    unparsable: List[str] = []
    for path in paths:
        payload, err = _load_json(path)
        if payload is None:
            unparsable.append(f"{path.name}: {err}")
            continue
        feed = _block(payload, "feed")
        feed_id = str(feed.get("feed_id") or "").strip()
        if not feed_id:
            unparsable.append(f"{path.name}: no feed.feed_id")
            continue
        episodes.setdefault(feed_id, []).append(path)
        url = str(feed.get("url") or "").strip()
        if url and feed_id not in urls:
            urls[feed_id] = url
    return urls, episodes, unparsable


def _fetch_language(url: str, timeout: float) -> Tuple[Optional[str], str]:
    """``(language_raw, error)`` — the channel ``<language>`` as the publisher declared it."""
    try:
        import httpx

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            body = resp.content
    except Exception as exc:  # noqa: BLE001 - any failure is retryable, never a guess
        return None, f"{type(exc).__name__}: {exc}"
    return _channel_language(body), ""


def _is_operator_override(payload: Dict[str, Any]) -> bool:
    """True when this episode's language was set by the per-feed operator override.

    OFF LIMITS TO THIS MIGRATION. The override exists precisely for the case where a publisher
    declares the wrong tag, so an episode carrying it is the one place where the feed's
    ``<language>`` is known to be WORSE than what is already on disk.
    ``resolve_episode_language`` ranks override above the feed tag; a backfill that overwrote it
    would invert that precedence and silently undo the correction at deploy time.

    Either block is enough. The pipeline writes both, but a hand-repaired artifact may carry only
    one, and the safe reading of a half-marked override is to leave it alone.
    """
    return SOURCE_OVERRIDE in {
        _block(payload, "feed").get("language_source"),
        _block(payload, "episode").get("language_source"),
    }


def _plan_for_episode(payload: Dict[str, Any], raw: str, normalized: str) -> Dict[Any, Any]:
    """The field writes this episode needs, or ``{}`` when it is already correct.

    Idempotent by comparison rather than by the ledger, so a manual re-run is harmless and the
    report can distinguish "updated" from "already correct".
    """
    feed = _block(payload, "feed")
    episode = _block(payload, "episode")
    wanted = {
        ("feed", "language"): normalized,
        ("feed", "language_raw"): raw,
        ("feed", "language_source"): SOURCE_RSS,
        ("episode", "language"): normalized,
        ("episode", "language_source"): SOURCE_RSS,
    }
    current = {
        ("feed", "language"): feed.get("language"),
        ("feed", "language_raw"): feed.get("language_raw"),
        ("feed", "language_source"): feed.get("language_source"),
        ("episode", "language"): episode.get("language"),
        ("episode", "language_source"): episode.get("language_source"),
    }
    return {k: v for k, v in wanted.items() if current.get(k) != v}


def _plan_for_override(payload: Dict[str, Any], language: str) -> Dict[Any, Any]:
    """The writes that put an operator's feed-level language override on this episode.

    ``language_raw`` is left as it is: it records what the publisher declared, and the override is
    the operator saying that declaration is wrong — not that it was never made.
    """
    feed = _block(payload, "feed")
    episode = _block(payload, "episode")
    wanted = {
        ("feed", "language"): language,
        ("feed", "language_source"): SOURCE_OVERRIDE,
        ("episode", "language"): language,
        ("episode", "language_source"): SOURCE_OVERRIDE,
    }
    current = {
        ("feed", "language"): feed.get("language"),
        ("feed", "language_source"): feed.get("language_source"),
        ("episode", "language"): episode.get("language"),
        ("episode", "language_source"): episode.get("language_source"),
    }
    return {k: v for k, v in wanted.items() if current.get(k) != v}


def _feed_language_overrides(root: Path) -> Dict[str, str]:
    """``{feed url: ISO code}`` for every feed whose ``overrides.json`` entry sets a language.

    THE OPERATOR'S CORRECTION OUTRANKS THE PUBLISHER, here as in the pipeline. Without this the
    migration fetched every show's ``<language>`` and stamped it onto every existing episode even
    when an operator had overridden that show precisely because its tag is wrong (English-path
    audit, 2026-10-05). A broken ``overrides.json`` RAISES (``load_overrides``): applying the
    publisher's tag because the corrections could not be read would be the same failure.

    Feed level only. ``metadata.json`` records no ``<guid>``, so an episode-level override cannot
    be matched to its artifact here; the pipeline applies those on the episode's next run.
    """
    doc = ov.load_overrides(root)
    out: Dict[str, str] = {}
    for url, entry in doc.feeds.items():
        language = entry.fields.language
        if language:
            out[ov.feed_key(url)] = language
    return out


def _apply_plan(payload: Dict[str, Any], plan: Dict[Any, Any]) -> None:
    for (block, field), value in plan.items():
        target = payload.setdefault(block, {})
        if isinstance(target, dict):
            target[field] = value


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``.

    Same contract as ``m0012``/``m0014``: a file changed since the migration wrote it is REFUSED
    rather than overwritten, because the thing being undone is this migration's write and nothing
    here can know what the later change meant.
    """
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class BackfillFeedLanguageMigration(Migration):
    """Replace the run-config language on pre-#2172 artifacts with the show's declared one.

    Every artifact written before #2172 carries whatever language the RUN was configured with,
    which for a non-English show is simply wrong. The declared `<language>` on the feed is the
    measurement, so this fetches it and writes it down with `language_source: "rss"`.

    It skips any episode carrying a per-feed operator override. An override is a human decision
    that outranks the feed, and stamping `"rss"` over it would not just lose the value — it would
    record a provenance that never happened.
    """

    id = MIGRATION_ID
    to_version = "2.7.15"
    description = (
        "Fetch each show's declared RSS <language> and backfill it onto the show and every "
        "episode under it, replacing the run-config value every pre-#2172 artifact carries. "
        "Episodes carrying a per-feed operator override are left untouched."
    )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Fetch each show's declared language and write it onto the show and its episodes.

        Counts per feed rather than globally, so a single show whose RSS is unreachable is
        reported as that show failing instead of as a lower overall success rate. Feeds with no
        URL, no declared language, or a failed fetch are each listed separately — they need
        different fixes, and one "skipped" bucket would hide that.
        """
        timeout = float(ctx.options.get("fetch_timeout") or DEFAULT_FETCH_TIMEOUT)
        root = ctx.corpus_root
        served, _superseded = select_served_artifacts(root, ".metadata.json")
        urls, episodes, unparsable = _shows_and_episodes(served)
        receipts: List[Dict[str, str]] = []
        files_written = 0

        per_feed: Dict[str, Dict[str, Any]] = {}  # counters plus the resolved labels
        no_url: List[str] = []
        no_language: List[str] = []
        fetch_failed: Dict[str, str] = {}
        updated = 0
        already = 0
        overridden = 0
        feed_overrides = _feed_language_overrides(root)
        from_override: List[str] = []

        for feed_id in sorted(episodes):
            url = urls.get(feed_id)
            if not url:
                # Unfetchable for ever, so it must not keep the migration pending.
                no_url.append(feed_id)
                ctx.log(f"  {feed_id}: SKIPPED — corpus has no feed.url for this show")
                continue

            override_language = feed_overrides.get(ov.feed_key(url))
            if override_language:
                # The operator's language for this show; the publisher's tag is not even fetched.
                raw: Optional[str] = None
                normalized: Optional[str] = override_language
                from_override.append(feed_id)
            else:
                raw, err = _fetch_language(url, timeout)
                if err:
                    fetch_failed[feed_id] = err
                    ctx.log(f"  {feed_id}: FETCH FAILED ({err}) — will retry on the next upgrade")
                    continue
                if not raw:
                    no_language.append(feed_id)
                    ctx.log(f"  {feed_id}: declares no <language> — left as it is")
                    continue
                normalized = normalize_language_tag(raw)
                if not normalized:
                    no_language.append(feed_id)
                    ctx.log(
                        f"  {feed_id}: declares {raw!r}, which is not a usable tag — left as it is"
                    )
                    continue

            counts: Dict[str, int] = {"updated": 0, "already": 0, "overridden": 0}
            for path in episodes[feed_id]:
                payload, err2 = _load_json(path)
                if payload is None:
                    unparsable.append(f"{path.name}: {err2}")
                    continue
                if override_language:
                    plan = _plan_for_override(payload, override_language)
                    if not plan:
                        counts["already"] += 1
                        already += 1
                        continue
                    counts["updated"] += 1
                    updated += 1
                    if not ctx.dry_run:
                        _apply_plan(payload, plan)
                        receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                    continue
                if _is_operator_override(payload):
                    # Per EPISODE, not per show: the override is recorded on the artifact, so
                    # skipping the whole show would strand every episode processed before the
                    # operator set it.
                    counts["overridden"] += 1
                    overridden += 1
                    continue
                assert raw is not None and normalized is not None  # the fetch branch set both
                plan = _plan_for_episode(payload, raw, normalized)
                if not plan:
                    counts["already"] += 1
                    already += 1
                    continue
                counts["updated"] += 1
                updated += 1
                if ctx.dry_run:
                    continue
                _apply_plan(payload, plan)
                # Backed up and receipted, never a bare write: this replaces five fields with a
                # value fetched from a third party, and `undo()` below is the only route back.
                # `write_with_backup` serialises with `dump_json`, which is byte-identical to the
                # `json.dumps(..., indent=2, ensure_ascii=False) + "\n"` this used to do — a
                # different encoding would make every migrated file differ from its own
                # regenerated form.
                receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
            if receipts:
                # PER SHOW. One append at the end of the whole migration would mean a crash on
                # show 12 of 20 left eleven shows' files rewritten with no receipt naming them,
                # and `undo` restores only what a receipt points at.
                append_receipts(
                    root,
                    RECEIPTS_FILE,
                    {"migration": MIGRATION_ID, "feed_id": feed_id, "language": normalized},
                    receipts,
                )
                files_written += len(receipts)
                receipts = []
            per_feed[feed_id] = {
                **counts,
                "language": normalized,
                "raw": raw,
                "source": SOURCE_OVERRIDE if override_language else SOURCE_RSS,
            }
            declared = "operator override" if override_language else repr(raw)
            ctx.log(
                f"  {feed_id}: {declared} -> {normalized!r} — {counts['updated']} updated, "
                f"{counts['already']} already correct, "
                f"{counts['overridden']} left to the operator override"
            )

        # Only a fetch failure is retryable. A missing url and a feed that declares nothing are
        # both final answers, so they must not hold the migration pending for ever.
        complete = not fetch_failed
        message = (
            f"{len(served)} episode(s) across {len(episodes)} show(s): "
            f"{updated} updated, {already} already correct, "
            f"{overridden} override(s) left alone, {len(from_override)} show(s) set from "
            f"overrides.json; "
            f"{len(no_url)} show(s) with no url, {len(no_language)} declaring no language, "
            f"{len(fetch_failed)} fetch failure(s), {len(unparsable)} unparsable"
        )
        if fetch_failed:
            message += " — INCOMPLETE, stays pending so the next upgrade retries"
        ctx.log(message)

        return MigrationResult(
            migration_id=self.id,
            # Recorded (and so never re-run) only when nothing is left to retry. Note this is
            # True even when 0 episodes changed: a corpus already carrying the right languages is
            # a COMPLETE migration, and `bool(updated)` would have left it pending for ever.
            applied=complete and not ctx.dry_run,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "episodes_scanned": len(served),
                "shows": len(episodes),
                "updated": updated,
                "already_correct": already,
                "overridden": overridden,
                "per_feed": per_feed,
                # So an operator can tell at a glance whether `undo` has anything to restore.
                "files_written": files_written,
                "shows_without_url": sorted(no_url),
                "shows_declaring_no_language": sorted(no_language),
                "fetch_failures": fetch_failed,
                "unparsable": unparsable,
                "complete": complete,
            },
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Every episode whose show declares a language carries it, sourced from the feed.

        Verification reads the CORPUS only — it does not re-fetch. A verify that went back to the
        network would fail on a publisher outage long after the migration was correctly applied,
        which is a verify that reports someone else's availability as our defect.
        """
        served, _superseded = select_served_artifacts(ctx.corpus_root, ".metadata.json")
        stale: List[str] = []
        for path in served:
            payload, _err = _load_json(path)
            if payload is None:
                continue
            if _is_operator_override(payload):
                # `apply` never touches these, so measuring them against the feed tag would
                # report a correct corpus as a partial backfill. Note they DO reach here: the
                # raw recorded beside an override is the override's own value, so the
                # "was this show backfilled" test below is satisfied by them.
                continue
            raw = _block(payload, "feed").get("language_raw")
            if not isinstance(raw, str) or not raw.strip():
                # No backfill happened for this show (no url, no declared language, or a fetch
                # that has not succeeded yet). Reported by `apply`, not a verify failure.
                continue
            normalized = normalize_language_tag(raw)
            if normalized and _plan_for_episode(payload, raw, normalized):
                stale.append(path.name)
        if stale:
            return False, f"{len(stale)} episode(s) carry a partial backfill: {stale[:5]}"
        return True, "every backfilled show is consistent across its episodes"
