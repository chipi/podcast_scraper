"""The ``enrich-edges`` command: derive HAS_EPISODE / MENTIONS / SPOKEN_BY into each ``gi.json``
(moved verbatim from ``search.cli_handlers``, E04 S4; reached through ``cli_hooks``).
"""

from __future__ import annotations

import argparse
import json
import logging
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from podcast_scraper.search.cli_handlers import EXIT_INVALID_ARGS, EXIT_LEDGER_FAILED, EXIT_SUCCESS
from podcast_scraper.search.corpus_scope import discover_metadata_files
from podcast_scraper.utils.log_redaction import format_exception_for_log


def _speaker_infos(raw: Any) -> List[SimpleNamespace]:
    """Persisted ``content.speakers`` dicts as attribute-bearing objects (#2062).

    The artifact on disk carries plain dicts; ``_speaker_lists_for_graph`` reads ``.name`` /
    ``.role`` so the pipeline and this pass can share ONE merge rule instead of two that drift.
    """
    out: List[SimpleNamespace] = []
    for s in raw or []:
        if isinstance(s, dict):
            # `placed` MUST survive the conversion (#2075). Stripping it made every person the
            # record lists as only named look like a placed voice to the merge rule — which would
            # then credit quotes to someone no voice was matched to.
            out.append(
                SimpleNamespace(
                    name=s.get("name") or "",
                    role=s.get("role") or "",
                    placed=s.get("placed"),
                )
            )
    return out


def parse_enrich_edges_argv(argv: Sequence[str]) -> Namespace:
    """Parse ``enrich-edges`` args (#874: derive relational edges into each gi.json)."""
    parser = argparse.ArgumentParser(prog="podcast_scraper enrich-edges")
    parser.add_argument("--output-dir", required=True, help="Corpus root (parent of feeds/).")
    parser.add_argument(
        "--replace-speakers",
        action="store_true",
        help=(
            "Rebuild Quote->SPOKEN_BY->Person from scratch instead of only adding missing edges "
            "(#2062). Without this, a re-run cannot correct an episode that was attributed wrongly "
            "the first time: the pass skips edges that already exist and removes none, so it "
            "reports SPOKEN_BY=0 and leaves the wrong speaker in place. Use after an attribution "
            "fix; it rewrites historical artifacts, so pair it with --retro-audit."
        ),
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="Skip SPOKEN_BY (diarization-dependent); emit only show + entity edges.",
    )
    parser.add_argument(
        "--use-ner",
        action="store_true",
        help=(
            "#1076 chunk 4-A. Augment the typed-MENTIONS post-pass with a "
            "spaCy PERSON detection pass — catches BART-paraphrased name "
            "fragments (KG 'Maya Hutchinson' matched by Insight text "
            "'Maya'). Default off so re-runs are bit-identical to today's "
            "output until you opt in."
        ),
    )
    parser.add_argument(
        "--retro-audit",
        action="store_true",
        help=(
            "#1076 chunk 4-A retro-audit hook. When set, stamp "
            "_retro_audit on every mutated GI artifact + emit a summary "
            "JSON listing per-episode edge deltas. Use when running "
            "--use-ner against existing prod corpora so the audit trail "
            "survives the in-place rewrite."
        ),
    )
    parser.add_argument(
        "--retro-marker",
        default=None,
        help="Custom marker string for --retro-audit (default '#1076-ner').",
    )
    parser.add_argument(
        "--retro-summary",
        default=None,
        help=(
            "Path to write the --retro-audit summary JSON. Default: "
            "<corpus>/_retro_audit_<marker>.json."
        ),
    )
    args = parser.parse_args(list(argv))
    args.command = "enrich-edges"
    return args


def _apply_bare_name_scope(gi_payload: dict, kg_payload: Any, episode_id: str) -> int:
    """Episode-scope any bare single-token person id this pass just minted (#2062).

    Delegates to ``identity.bare_name_scope`` — the same functions the pipeline runs over finished
    payloads and the m0007 migration runs over the corpus — so enrich-edges cannot drift into its
    own answer about who "Sam" is. Mutates *gi_payload* in place; returns how many ids changed.

    ``candidate_ids`` stays narrow (node-backed only) for the same reason #1868 gives: healing
    writes a REAL person's id onto content and has no cheap undo, so an id with no node behind it
    is a dangling reference, not evidence.

    Mutates BOTH payloads. The plan is computed from the union of the two rosters, so applying it
    to only one of them guarantees they disagree about an id the pass itself chose — see the
    comment at the rewrite below.
    """
    if not episode_id:
        return 0
    try:
        from podcast_scraper.identity.bare_name_scope import (
            person_ids_in,
            person_node_ids_in,
            plan_bare_name_ids,
            rewrite_ids,
        )
    except ImportError:  # pragma: no cover - defensive
        return 0
    roster = set(person_ids_in(gi_payload))
    candidates = set(person_node_ids_in(gi_payload))
    if isinstance(kg_payload, dict):
        roster |= set(person_ids_in(kg_payload))
        candidates |= set(person_node_ids_in(kg_payload))
    id_map = plan_bare_name_ids(roster, episode_id, heal=True, candidate_ids=candidates)
    if not id_map:
        return 0
    rewritten, changes = rewrite_ids(gi_payload, id_map)
    if changes:
        gi_payload.clear()
        gi_payload.update(rewritten)

    # THE PLAN IS BUILT FROM BOTH LAYERS, SO IT HAS TO BE APPLIED TO BOTH. Scoping GI alone
    # against a roster that INCLUDES kg.json's ids leaves KG holding the bare id this pass just
    # decided was unsafe — one human under two ids across the two artifacts, which is #1862's
    # desync arriving through the CLI instead of the pipeline. Harmless only when m0007 happened
    # to run first; that is a schedule, not a guarantee.
    #
    # The caller writes gi.json unconditionally, so the KG copy is returned separately for it to
    # persist — see `_scoped_kg_payload` in the caller. A KG change the caller cannot write is
    # reported rather than silently dropped.
    if isinstance(kg_payload, dict):
        kg_rewritten, kg_changes = rewrite_ids(kg_payload, id_map)
        if kg_changes:
            kg_payload.clear()
            kg_payload.update(kg_rewritten)
            changes += kg_changes
    return int(changes)


def _stable_json(payload: Any) -> str:
    """A comparable snapshot of a payload, or ``""`` when there is none."""
    if not isinstance(payload, dict):
        return ""
    try:
        return json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError):
        return ""


def _persist_scoped_kg(
    kg_path: Path,
    kg_payload: Any,
    before: str,
    episode_id: str,
    totals: Dict[str, int],
    logger: logging.Logger,
) -> None:
    """Write kg.json when the scoping pass changed it. BOTH LAYERS OR NEITHER (#1862).

    ``_apply_bare_name_scope`` plans from the UNION of the GI and KG rosters, so applying that
    plan to gi.json alone leaves kg.json holding the bare id the pass just decided was unsafe —
    one human under two ids across the two artifacts, arriving through the CLI instead of the
    pipeline. It was harmless only when m0007 happened to have run first, which is a schedule
    rather than a guarantee.

    A failure is COUNTED and logged at error level rather than swallowed: gi.json has already been
    written by the caller, so a silent failure here is the desync itself, not a missed enrichment.
    """
    if not isinstance(kg_payload, dict) or not kg_path.is_file():
        return
    if _stable_json(kg_payload) == before:
        return
    # `kg.io`, NOT `gi.io`. They are different validators for different schemas, and the wrong one
    # here does not fail loudly — it fails on EVERY episode, in the direction that recreates the
    # bug this function exists to prevent. `gi/schema` requires `model_version` / `prompt_version`
    # and `schema_version in ("3.0","3.1")`; measured on the production snapshot, all 2,256 kg.json
    # are `schema_version 2.1` and carry NEITHER key, so every write raised, was caught, counted
    # as a failure and logged — after gi.json had already been written. That is the #1862 desync
    # with a log line attached.
    #
    # It stayed invisible because the scope plan touches KG ids on 0 episodes of that snapshot
    # (m0007 already ran there), so the path is latent until the first episode that needs it, and
    # then fails 100%.
    from podcast_scraper.kg.io import write_artifact

    try:
        write_artifact(kg_path, kg_payload, validate=True)
        totals["kg_scoped"] += 1
    except (OSError, ValueError) as exc:
        totals["kg_write_failed"] += 1
        logger.error(
            "enrich-edges: gi.json was scoped but kg.json could not be written (%s) — "
            "the layers now disagree for %s: %s",
            kg_path.name,
            episode_id,
            exc,
        )


def _log_quote_alignment(stats: Dict[str, int], logger: logging.Logger) -> None:
    """One line for the whole enrich-edges pass (see `add_spoken_by_edges` ``stats``)."""
    if stats.get("misaligned"):
        logger.warning(
            "enrich-edges: %d quote(s) in %d episode(s) have a char_start that does not index into "
            "the transcript and text that is absent or ambiguous; their speaker is not attributed "
            "(%d other quote(s) were re-anchored by their text).",
            stats["misaligned"],
            stats.get("episodes", 0),
            stats.get("reanchored", 0),
        )


def _write_edges_stamp(corpus: Path, totals: Dict[str, int], logger: logging.Logger) -> None:
    """BUMP THE CACHE TOKEN after enrich-edges rewrote artifacts.

    This command rewrites gi.json -- including SPOKEN_BY, which insight attribution and
    ``get_corpus_graph(derive_speaker_links=True)`` read -- but it is neither an ingest nor a
    migration, so it touches none of the files ``perf_cache.corpus_mtime`` watches. Without this
    stamp the API keeps serving the OLD edges until the next ingest or a process restart. Called
    last, so the stamp cannot claim work that did not finish.

    Only when something was actually rewritten: a no-op pass that bumped the token anyway made
    every api rebuild its projections (the cache warmer's minutes-long entity-map build) for
    nothing.
    """
    if not (totals.get("gi_rewritten") or totals.get("kg_scoped")):
        return
    try:
        (corpus / "corpus_edges_stamp.json").write_text(
            json.dumps({"applied_at": datetime.now(timezone.utc).isoformat(), **totals}) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:  # a stamp we cannot write must not fail the enrichment itself
        logger.warning("enrich-edges: could not write the cache-invalidation stamp: %s", exc)


def _write_corpus_followups(
    corpus: Path, totals: Dict[str, int], logger: logging.Logger, *, defer: bool
) -> bool:
    """Stamp the cache token and refresh the identity ledger after a pass; True if the ledger
    refresh failed.

    ``defer``: a multi-feed batch child walks only its run dir, so both would land in the run
    dir, not the corpus root. The batch's ``corpus_run_summary.json`` moves the cache token, and
    the batch (or the enrichment it queues) refreshes the ledger.
    """
    if defer:
        return False
    _write_edges_stamp(corpus, totals, logger)
    if not (totals.get("gi_rewritten") or totals.get("kg_scoped")):
        return False
    from podcast_scraper.identity.ledger import refresh_ledger

    return not refresh_ledger(corpus, after="enrich-edges", log=logger)


def run_enrich_edges_cli(args: Namespace, logger: logging.Logger) -> int:
    """Derive relational edges into each gi.json (#874): Podcast→HAS_EPISODE→Episode,
    Insight→MENTIONS→Entity, and (unless --no-speaker) Quote→SPOKEN_BY→Person.

    A cross-layer, idempotent corpus pass — re-runnable any time; reprocess after the
    new diarization (#876) lands to fill SPOKEN_BY corpus-wide.
    """
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("enrich-edges: --output-dir is required")
        return EXIT_INVALID_ARGS

    from podcast_scraper.gi.io import write_artifact
    from podcast_scraper.gi.relational_edges import (
        add_episode_show_edges,
        add_insight_entity_edges,
        kg_entity_index,
    )
    from podcast_scraper.gi.speakers import add_spoken_by_edges
    from podcast_scraper.search.corpus_scope import episode_root_from_metadata_path
    from podcast_scraper.search.indexer import _gi_path, _load_metadata_file, _transcript_path
    from podcast_scraper.speaker_stage.record import _speaker_lists_for_graph

    # #1076 chunk 4-A — operator-controlled flag. ``--use-ner`` on the
    # CLI flips the typed-MENTIONS post-pass to also run a spaCy PERSON
    # detection pass. Defaults off so re-running enrich-edges on
    # historical corpora is bit-identical to today's output until the
    # operator opts in.
    use_ner = bool(getattr(args, "use_ner", False))
    nlp = None
    if use_ner:
        try:
            from podcast_scraper.config import Config
            from podcast_scraper.providers.ml.speaker_detection import get_ner_model

            nlp = get_ner_model(Config())
        except Exception as ner_load_exc:
            logger.warning(
                "enrich-edges: --use-ner set but spaCy load failed (%s); "
                "falling back to regex-only typed-MENTIONS",
                format_exception_for_log(ner_load_exc),
            )
            nlp = None

    # #1076 chunk 4-A — retro-audit hook. When ``--retro-audit`` is set
    # AND we're actively flipping behavior on existing artifacts, stamp
    # ``_retro_audit`` on every mutated GI and emit a summary JSON so
    # the audit trail survives the in-place rewrite (per memory
    # [[feedback_never_mutate_historical_artifacts]]).
    retro_audit = bool(getattr(args, "retro_audit", False))
    retro_marker = getattr(args, "retro_marker", None) or "#1076-ner"
    retro_summary_path_arg = getattr(args, "retro_summary", None)
    retro_rows: list = []
    retro_applied_at: str | None = None
    if retro_audit:
        # No local `from datetime import ...` here. The module already imports both at the top,
        # and a function-scoped import binds the name LOCALLY for the whole function — so every
        # other use of `datetime` in this function raised UnboundLocalError whenever this branch
        # did not run. Harmless while this was the only use; it broke 10 tests the moment a
        # second one was added below.
        retro_applied_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info(
            "enrich-edges: --retro-audit on (marker=%s applied_at=%s)",
            retro_marker,
            retro_applied_at,
        )

    corpus = Path(output_dir)
    include_speaker = not bool(getattr(args, "no_speaker", False))
    replace_speakers = bool(getattr(args, "replace_speakers", False))
    if replace_speakers:
        logger.info(
            "enrich-edges: --replace-speakers on — SPOKEN_BY will be REBUILT, not "
            "merely topped up; wrong speakers from earlier runs are removed"
        )
    totals = {
        "episodes": 0,
        "has_episode": 0,
        "mentions": 0,
        "spoken_by": 0,
        "scoped": 0,
        "kg_scoped": 0,
        "kg_write_failed": 0,
        "gi_rewritten": 0,
    }
    # One summary for the whole pass instead of one warning per episode on every run (#2276).
    quote_alignment: Dict[str, int] = {}
    for meta_path in discover_metadata_files(corpus):
        doc = _load_metadata_file(meta_path)
        if not doc:
            continue
        episode_root = episode_root_from_metadata_path(meta_path)
        gi_path = _gi_path(episode_root, meta_path, doc)
        if not (gi_path and gi_path.is_file()):
            continue
        try:
            artifact = json.loads(gi_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("enrich-edges: skip %s (%s)", gi_path, format_exception_for_log(exc))
            continue
        # Write the gi.json back only when this pass CHANGED it. It used to rewrite every episode
        # in the corpus on every run -- a 1-episode job's finalize rewrote 2,447 gi.json on prod
        # (2026-10-06), and each rewrite also reset the file's mode. The kg.json side already
        # compared before/after (`_persist_scoped_kg`); this is the same rule for gi.json.
        _gi_before = _stable_json(artifact)
        episode_id_for_scope = (
            artifact.get("episode_id") or (doc.get("episode") or {}).get("episode_id") or ""
        )
        kg_artifact_for_scope = None
        show_title = (doc.get("feed") or {}).get("title") or ""
        per_episode_has_episode = add_episode_show_edges(artifact, show_title)
        totals["has_episode"] += per_episode_has_episode
        kg_path = Path(str(gi_path).replace(".gi.json", ".kg.json"))
        per_episode_mentions = 0
        if kg_path.is_file():
            try:
                kg_artifact = json.loads(kg_path.read_text(encoding="utf-8"))
                kg_artifact_for_scope = kg_artifact
                per_episode_mentions = add_insight_entity_edges(
                    artifact, kg_entity_index(kg_artifact), nlp=nlp
                )
                totals["mentions"] += per_episode_mentions
            except (OSError, ValueError):
                pass
        per_episode_spoken_by = 0
        if include_speaker:
            transcript_path = _transcript_path(episode_root, doc)
            if transcript_path and transcript_path.is_file():
                content = doc.get("content") or {}
                # #2062: `content.speakers` is the DIARIZATION ROSTER (present on 97.2% of a
                # 330-episode production sample).
                # feed_title is REQUIRED for the show-name refusal (#2064): `names_the_show`
                # returns False on an empty title by design ("no title, no opinion"), so omitting
                # it silently disabled the guard on this path — measured: `Africa Tech Summit`,
                # `Machine Learning Street` and `Latent.Space` were all KEPT as hosts here while
                # the pipeline path refused them. Two paths, two answers, which is the drift this
                # arc keeps paying for.
                # #2075: the speaker record only. No `detected_*` fallback — that field was the
                # pre-listening guess (and, on pre-1.2.0 artifacts, a projection of `speakers`), so
                # reading it here would credit quotes to people no voice was matched to.
                roster_hosts, roster_guests = _speaker_lists_for_graph(
                    _speaker_infos(content.get("speakers")),
                    feed_title=str((doc.get("feed") or {}).get("title") or ""),
                    language=(doc.get("feed") or {}).get("language") or None,
                )
                per_episode_spoken_by = add_spoken_by_edges(
                    artifact,
                    transcript_path.read_text(encoding="utf-8", errors="replace"),
                    hosts=roster_hosts,
                    guests=roster_guests,
                    replace=replace_speakers,
                    stats=quote_alignment,
                )
                totals["spoken_by"] += per_episode_spoken_by
        # Retro-audit marker — stamp ONLY when --retro-audit set AND we
        # actually added edges. Idempotent re-runs (zero-op) don't stamp.
        if retro_audit and (per_episode_has_episode + per_episode_mentions + per_episode_spoken_by):
            audit_entry = {
                "marker": retro_marker,
                "applied_at": retro_applied_at,
                "use_ner": bool(use_ner),
                "edges_added": {
                    "has_episode": per_episode_has_episode,
                    "mentions": per_episode_mentions,
                    "spoken_by": per_episode_spoken_by,
                },
            }
            audit_log = artifact.setdefault("_retro_audit", [])
            if isinstance(audit_log, list):
                audit_log.append(audit_entry)
            retro_rows.append(
                {
                    "gi_path": str(gi_path.resolve().relative_to(corpus.resolve())),
                    **audit_entry,
                }
            )
        # #2062: this CLI runs AFTER the pipeline's bare-name scoping pass, so any speaker id it
        # just minted is UNSCOPED and sits beside the already-scoped node for the same human —
        # `person:twiggy` next to `person:unresolved-twiggy-{ep}`, one voice with two identities and
        # a full set of SPOKEN_BY edges on each (a fresh ingest produced 116 edges for 59 quotes).
        #
        # The rule is deliberately NOT applied inside any mint function: it needs the episode's
        # whole roster, and three mint families would drift if one of them freelanced. So run the
        # SAME pass the pipeline and the m0007 migration run, over the finished payloads.
        _kg_before = _stable_json(kg_artifact_for_scope)
        _scoped = _apply_bare_name_scope(artifact, kg_artifact_for_scope, str(episode_id_for_scope))
        if _scoped:
            totals["scoped"] += _scoped
        if _stable_json(artifact) != _gi_before:
            write_artifact(gi_path, artifact, validate=True)
            totals["gi_rewritten"] += 1
        _persist_scoped_kg(
            kg_path,
            kg_artifact_for_scope,
            _kg_before,
            str(episode_id_for_scope),
            totals,
            logger,
        )
        totals["episodes"] += 1

    msg = (
        f"enrich-edges: episodes={totals['episodes']} HAS_EPISODE={totals['has_episode']} "
        f"MENTIONS={totals['mentions']} SPOKEN_BY={totals['spoken_by']} "
        f"scoped_bare_names={totals['scoped']} gi_rewritten={totals['gi_rewritten']} "
        f"kg_rewritten={totals['kg_scoped']} "
        f"kg_write_failed={totals['kg_write_failed']}"
    )
    logger.info(msg)
    _log_quote_alignment(quote_alignment, logger)
    print(msg)

    ledger_failed = _write_corpus_followups(
        corpus, totals, logger, defer=bool(getattr(args, "defer_corpus_writes", False))
    )

    if retro_audit and retro_rows:
        if retro_summary_path_arg:
            summary_path = Path(retro_summary_path_arg)
        else:
            summary_path = (
                corpus / f"_retro_audit_{retro_marker.replace('#', '').replace('/', '-')}.json"
            )
        summary_path.write_text(
            json.dumps(
                {
                    "marker": retro_marker,
                    "applied_at": retro_applied_at,
                    "use_ner": bool(use_ner),
                    "corpus": str(corpus),
                    "totals": totals,
                    "per_episode": retro_rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"enrich-edges: retro-audit summary → {summary_path}")
    if ledger_failed:
        print("enrich-edges: identity ledger refresh FAILED; the previous ledger stays")
        return EXIT_LEDGER_FAILED
    return EXIT_SUCCESS
