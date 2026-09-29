"""0011 — fetch each show's declared RSS ``<language>`` and backfill it onto its episodes (#2173).

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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...languages import normalize_language_tag, SOURCE_RSS
from ...rss.parser import _channel_language
from ..corpus_selection import select_served_artifacts
from ..migration import Migration, MigrationContext, MigrationResult

#: Per-feed HTTP timeout, seconds. Overridable via ``ctx.options["fetch_timeout"]``.
DEFAULT_FETCH_TIMEOUT = 20.0


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


def _apply_plan(payload: Dict[str, Any], plan: Dict[Any, Any]) -> None:
    for (block, field), value in plan.items():
        target = payload.setdefault(block, {})
        if isinstance(target, dict):
            target[field] = value


class BackfillFeedLanguageMigration(Migration):
    id = "0015_backfill_feed_language"
    to_version = "2.7.9"
    description = (
        "Fetch each show's declared RSS <language> and backfill it onto the show and every "
        "episode under it, replacing the run-config value every pre-#2172 artifact carries."
    )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        timeout = float(ctx.options.get("fetch_timeout") or DEFAULT_FETCH_TIMEOUT)
        served, _superseded = select_served_artifacts(ctx.corpus_root, ".metadata.json")
        urls, episodes, unparsable = _shows_and_episodes(served)

        per_feed: Dict[str, Dict[str, Any]] = {}  # counters plus the resolved labels
        no_url: List[str] = []
        no_language: List[str] = []
        fetch_failed: Dict[str, str] = {}
        updated = 0
        already = 0

        for feed_id in sorted(episodes):
            url = urls.get(feed_id)
            if not url:
                # Unfetchable for ever, so it must not keep the migration pending.
                no_url.append(feed_id)
                ctx.log(f"  {feed_id}: SKIPPED — corpus has no feed.url for this show")
                continue

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
                ctx.log(f"  {feed_id}: declares {raw!r}, which is not a usable tag — left as it is")
                continue

            counts: Dict[str, int] = {"updated": 0, "already": 0}
            for path in episodes[feed_id]:
                payload, err2 = _load_json(path)
                if payload is None:
                    unparsable.append(f"{path.name}: {err2}")
                    continue
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
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            per_feed[feed_id] = {**counts, "language": normalized, "raw": raw}
            ctx.log(
                f"  {feed_id}: {raw!r} -> {normalized!r} — {counts['updated']} updated, "
                f"{counts['already']} already correct"
            )

        # Only a fetch failure is retryable. A missing url and a feed that declares nothing are
        # both final answers, so they must not hold the migration pending for ever.
        complete = not fetch_failed
        message = (
            f"{len(served)} episode(s) across {len(episodes)} show(s): "
            f"{updated} updated, {already} already correct; "
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
                "per_feed": per_feed,
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
