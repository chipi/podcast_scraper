"""The passes that join GI and KG, then the bridge and digest (lifted, E04 S2)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.workflow.metadata_generation import _resolve_scope_write_targets

logger = logging.getLogger(__name__)


def run_gi_kg_passes(  # noqa: C901 - the block as generate_episode_metadata held it
    *,
    bridge_gi_payload: Any,
    bridge_kg_payload: Any,
    cfg: Any,
    episode: Any,
    episode_description: Any,
    episode_id: Any,
    gi_artifact_path: Any,
    gi_path: Any,
    kg_path: Any,
    metadata_path: Any,
    nlp: Any,
    pipeline_metrics: Any,
) -> Tuple[Any, ...]:
    """Once both layers exist: the KG organisation post-pass, person-id scoping and the
    intra-episode merges, typed mentions, GI topics aligned with the KG, then the bridge and the
    context digest. Lifted verbatim from ``generate_episode_metadata``; returns the names it
    reassigns that the metadata record reads after it."""
    # RFC-097 v3.0 chunk-4 — typed-MENTIONS post-pass. The live GI emit
    # path doesn't materialize MENTIONS_PERSON / MENTIONS_ORG cross-layer
    # edges (it lacks the KG entity index at build time). Here, with both
    # payloads available, run the post-pass and re-write gi.json so the
    # disk artifact carries the typed descriptive edges that the viewer
    # graph + search relational queries expect.
    # #1058 chunk 1 — KG Organization post-pass under LLM-free profiles.
    # Runs BEFORE the GI typed-MENTIONS post-pass so any MENTIONS_ORG
    # edges that downstream chunks resolve will see the Organization
    # nodes that just landed. Skipped when the flag is off (default).
    if (
        bridge_gi_payload is not None
        and bridge_kg_payload is not None
        and getattr(cfg, "kg_organizations_use_ner", False)
    ):
        try:
            from podcast_scraper.kg.ner_postpass import apply_org_postpass_to_kg_artifact

            added_orgs = apply_org_postpass_to_kg_artifact(
                bridge_kg_payload, bridge_gi_payload, nlp
            )
            if added_orgs > 0:
                # Re-write the KG artifact to persist the lifted
                # Organization nodes. Strict validation stays on.
                from podcast_scraper.kg.io import write_artifact as _kg_write_artifact

                _kg_path = Path(kg_path) if "kg_path" in locals() and kg_path is not None else None
                if _kg_path is not None:
                    _kg_write_artifact(_kg_path, bridge_kg_payload, validate=True)
                logger.info(
                    "[%s] Added %d Organization nodes to KG artifact (NER post-pass)",
                    episode.idx if hasattr(episode, "idx") else episode_id,
                    added_orgs,
                )
        except Exception as org_postpass_exc:
            logger.warning(
                "[%s] KG Organization post-pass failed (non-fatal): %s",
                episode.idx if hasattr(episode, "idx") else episode_id,
                org_postpass_exc,
                exc_info=True,
            )

    # #1685 — scope bare person names BEFORE typed mentions run, so mentions attach to the
    # FINAL ids rather than to ids that are about to be rewritten underneath them.
    #
    # A single-token person name identifies someone inside this episode and nobody globally.
    # Minting `person:jensen` as a global id makes three different Jensens on three different
    # shows into one followable node, and `POST /interests/{token}` accepts it while derived
    # interests mint it straight from `entities_from_kg` with no click at all. Production:
    # 208 occurrences of 172 such ids, of which 196 have no full name anywhere in their episode.
    #
    # This runs HERE, over the finished payloads, rather than inside `entity_node_id()`, because
    # the rule needs the episode's whole roster (which only exists once both layers have
    # extracted) and because there are three mint families — `entity_node_id`, `person_node_id`
    # and `identity.slugify.person_id`, the last used by the GI speaker path. One pass covers
    # all three. It is also the SAME function the backfill migration runs, so the pipeline and
    # the migration cannot drift into disagreeing about who "Sam" is.
    if bridge_gi_payload is not None and bridge_kg_payload is not None:
        # SNAPSHOT BEFORE THE PASS, so a failure can put the payloads BACK. Everything below
        # reassigns `bridge_gi_payload` / `bridge_kg_payload` in place, and the `except` used to
        # only log — leaving the in-memory GI rewritten while KG was not. That is not inert:
        # `apply_typed_mentions_and_rewrite_gi` and the topic-alignment step further down write
        # GI again on essentially every fresh episode, so a KG-side failure here shipped a
        # half-scoped GI to disk and the stage ledger recorded the episode as left unscoped —
        # which was false for one of the two layers.
        #
        # This is the shape of the 2026-08-31 incident (10 episodes, see
        # `_write_scoped_artifacts_both_or_neither`). That incident's trigger was fixed; this
        # path was not, because the on-disk both-or-neither restore happens INSIDE the try and
        # the later writers are outside it.
        _pre_scope_gi, _pre_scope_kg = bridge_gi_payload, bridge_kg_payload
        try:
            from podcast_scraper.identity.bare_name_scope import (
                person_ids_in,
                person_node_ids_in,
                plan_bare_name_ids,
                rewrite_ids,
            )
            from podcast_scraper.identity.intra_episode_merge import (
                apply_display_names,
                plan_display_names,
                plan_heal_display_names,
                plan_intra_episode_merges,
            )

            # ROSTER is wide (everything `rewrite_ids` can write, incl. quote `speaker_id` and
            # edge endpoints) so nothing is left unscoped. CANDIDATES are narrow (node-backed
            # only) because healing writes a REAL person's id onto content and has no cheap undo
            # — an id with no node is a dangling reference, not evidence. See #1868.
            _roster = person_ids_in(bridge_gi_payload) | person_ids_in(bridge_kg_payload)
            _candidates = person_node_ids_in(bridge_gi_payload) | person_node_ids_in(
                bridge_kg_payload
            )
            _id_map = plan_bare_name_ids(
                _roster,
                str(episode_id),
                heal=bool(getattr(cfg, "bare_name_heal", True)),
                candidate_ids=_candidates,
            )
            _gi_changes = _kg_changes = 0
            if _id_map:
                bridge_gi_payload, _gi_changes = rewrite_ids(bridge_gi_payload, _id_map)
                bridge_kg_payload, _kg_changes = rewrite_ids(bridge_kg_payload, _id_map)
                # THE HEAL HAS TO UNITE THE LABEL TOO, not just the id. `rewrite_ids` gives the GI
                # node the resolved id and keeps its own name — the bare label that needed
                # resolving — so `person:kashmir-hill` read "Kashmir" in gi.json and "Kashmir
                # Hill" in kg.json. Measured: 42 ids disagree across the two artifacts,
                # 13 with the GI name a strict prefix of KG's, which is this exact signature.
                # Unlike a merge, this is not two plausible spellings — it is a bare label that was
                # just resolved TO the fuller name, so KG's resolved name wins outright.
                _heal_renames = plan_heal_display_names(
                    bridge_gi_payload, bridge_kg_payload, _id_map
                )
                if _heal_renames:
                    bridge_gi_payload, _gi_heal_renames = apply_display_names(
                        bridge_gi_payload, _heal_renames
                    )
                    bridge_kg_payload, _kg_heal_renames = apply_display_names(
                        bridge_kg_payload, _heal_renames
                    )
                    _gi_changes += _gi_heal_renames
                    _kg_changes += _kg_heal_renames

            # #2056: one human, two ids, ONE episode — the speaker path minted `person:aaron-levy`
            # from ASR of the spoken name while the entity extractor minted `person:aaron-levie`
            # from the text. `entity_clusters._are_xep_variants` is the CROSS-episode test and is
            # never asked about two ids inside one episode, so the resolver that "already matches
            # the pair" never runs on this shape.
            #
            # Runs AFTER the bare-name pass, deliberately: that pass creates the scoped
            # `person:unresolved-<name>-<episode>` ids, and this one skips them so the two rules
            # cannot both rewrite the same id and drift about who a person is.
            _merge_map = plan_intra_episode_merges(bridge_gi_payload, bridge_kg_payload)
            if _merge_map:
                # WHICH SPELLING SURVIVES is decided by the feed's own prose, not by which layer
                # minted the id. The title/description are human-written — not transcribed, not
                # generated — so they are authoritative where neither ASR nor the LLM is. Measured
                # over 287 artifacts, provenance alone gets this wrong more often than right.
                _ep_prose = " ".join(
                    str(part or "")
                    for part in (
                        getattr(episode, "title", ""),
                        locals().get("episode_description") or "",
                    )
                )
                _renames = plan_display_names(
                    bridge_gi_payload,
                    bridge_kg_payload,
                    _merge_map,
                    episode_text=_ep_prose,
                )
                bridge_gi_payload, _gi_merge_changes = rewrite_ids(bridge_gi_payload, _merge_map)
                bridge_kg_payload, _kg_merge_changes = rewrite_ids(bridge_kg_payload, _merge_map)
                # The id follows who SPOKE; the display name follows the FEED's spelling. Both
                # payloads get the same `_renames` map, and that is the point: `rewrite_ids` keeps
                # whichever node is FIRST in each payload's own node list, and GI and KG order
                # theirs differently — so without this the two artifacts show different names for
                # the id they just agreed on. Measured on the sample: 8 of 11 merges did exactly
                # that. Applying it to both is what makes the merge single-valued.
                bridge_gi_payload, _gi_rename_changes = apply_display_names(
                    bridge_gi_payload, _renames
                )
                bridge_kg_payload, _kg_rename_changes = apply_display_names(
                    bridge_kg_payload, _renames
                )
                # A RENAME IS A CHANGE. The write gate below decides per layer from these counts,
                # and counting only ID rewrites meant a layer that was renamed but not re-ID'd
                # looked unchanged and was never written. That is the normal shape here: the
                # merged-away id usually lives only in KG, so GI gets 0 id changes — and on the
                # production snapshot the decided name was dropped from gi.json on 14 of the 117
                # episodes this pass touches, leaving one id with two names across the layers.
                _gi_changes += _gi_merge_changes + _gi_rename_changes
                _kg_changes += _kg_merge_changes + _kg_rename_changes

            if _id_map or _merge_map:
                # #1862 defect 1: write each layer only if it changed, and refuse a planned KG
                # change with no bound target (``generate_kg`` off) rather than desync the layers.
                _kg_write_path = kg_path if "kg_path" in locals() and kg_path else None
                _gi_target, _kgp = _resolve_scope_write_targets(
                    gi_artifact_path, bool(_gi_changes), _kg_write_path, bool(_kg_changes)
                )
                _write_scoped_artifacts_both_or_neither(
                    _gi_target,
                    bridge_gi_payload,
                    _kgp,
                    bridge_kg_payload,
                )
            if _id_map:
                logger.info(
                    "[%s] Scoped %d bare person id(s): %s",
                    episode.idx if hasattr(episode, "idx") else episode_id,
                    len(_id_map),
                    ", ".join(f"{k}->{v}" for k, v in sorted(_id_map.items())[:5]),
                )
            if _merge_map:
                logger.info(
                    "[%s] Merged %d duplicate person id(s) within the episode: %s",
                    episode.idx if hasattr(episode, "idx") else episode_id,
                    len(_merge_map),
                    ", ".join(f"{k}->{v}" for k, v in sorted(_merge_map.items())[:5]),
                )
        except Exception as bare_name_exc:  # noqa: BLE001 - never lose the episode over this
            # BOTH LAYERS GO BACK. Partial in-memory state is worse than no state: the later GI
            # writers cannot tell a half-applied plan from a completed one, and the result is one
            # human under two ids across kg.json and gi.json — the defect this whole pass exists
            # to remove, caused by the pass failing rather than by it not running.
            bridge_gi_payload, bridge_kg_payload = _pre_scope_gi, _pre_scope_kg
            logger.warning(
                "[%s] Bare-name scoping failed (non-fatal, ids left as minted): %s",
                episode.idx if hasattr(episode, "idx") else episode_id,
                bare_name_exc,
                exc_info=True,
            )
            # MARK IT. A degradation whose only trace is a log line is precisely the defect
            # #1686 fixed for summaries the same day this landed — the episode keeps unscoped
            # global ids and nothing downstream can tell that from a corpus with none. Recorded
            # on the same stage ledger, so `outcome == "failed"` is queryable rather than
            # something a re-audit has to infer.
            if pipeline_metrics is not None:
                try:
                    pipeline_metrics.record_stage_outcome(
                        "bare_name_scoping",
                        episode.idx,
                        "failed",
                        reason="scoping_pass_raised",
                        detail={"message": format_exception_for_log(bare_name_exc)},
                    )
                except Exception:  # noqa: BLE001 - telemetry must not cost the episode
                    pass

    if (
        bridge_gi_payload is not None
        and bridge_kg_payload is not None
        and gi_artifact_path is not None
    ):
        try:
            from podcast_scraper.gi.relational_edges import apply_typed_mentions_and_rewrite_gi

            # #1076 chunk 4-A — pass the cached spaCy NER model when the
            # operator-gated flag is on. ``nlp`` was loaded at line ~3093
            # for speaker-detection purposes; reuse it here so we don't
            # pay a second model-load cost. ``None`` keeps the legacy
            # whole-word substring behavior (the default).
            typed_mentions_nlp = nlp if getattr(cfg, "gi_typed_mentions_use_ner", False) else None
            added_typed = apply_typed_mentions_and_rewrite_gi(
                bridge_gi_payload,
                bridge_kg_payload,
                gi_artifact_path,
                nlp=typed_mentions_nlp,
            )
            if added_typed > 0:
                logger.info(
                    "[%s] Added %d typed MENTIONS_PERSON/MENTIONS_ORG edges to GI artifact" "%s",
                    episode.idx if hasattr(episode, "idx") else episode_id,
                    added_typed,
                    " (NER pass on)" if typed_mentions_nlp is not None else "",
                )
        except Exception as typed_mentions_exc:
            logger.warning(
                "[%s] Typed MENTIONS post-pass failed (non-fatal): %s",
                episode.idx if hasattr(episode, "idx") else episode_id,
                typed_mentions_exc,
                exc_info=True,
            )

    # Align GI topic nodes with KG topic labels (#585): when both GI and KG
    # ran, replace GI's bullet-derived Topic nodes (sentence-style slugs) with
    # KG's noun-phrase topic labels so the bridge can merge them by exact ID.
    if bridge_gi_payload is not None and bridge_kg_payload is not None:
        try:
            # Extracted to ``gi.topic_alignment`` so the corpus repair (``gi.repair``), which
            # rebuilds artifacts OUTSIDE this function, applies the identical alignment. A
            # second copy of these lines would drift the first time either changed, and the
            # symptom would be an episode whose topics silently stop merging in the bridge.
            from podcast_scraper.gi.topic_alignment import align_gi_topics_with_kg

            aligned_count = align_gi_topics_with_kg(bridge_gi_payload, bridge_kg_payload)
            if aligned_count:
                logger.info(
                    "[%s] Aligned %d GI topic nodes with KG labels for CIL bridge",
                    episode.idx if hasattr(episode, "idx") else episode_id,
                    aligned_count,
                )
                # #653 Part D follow-up: gi.json was written at build time (line ~4061) with the
                # staged-mode bullet-derived Topic nodes. bridge_gi_payload IS that same payload
                # object, now carrying KG canonical topics — re-persist gi.json so the on-disk
                # artifact matches the bridge instead of keeping sentence-fragment topic labels.
                try:
                    from podcast_scraper.gi import write_artifact as _gi_write_artifact

                    _gi_write_artifact(Path(gi_path), bridge_gi_payload, validate=True)
                except Exception as _gi_rewrite_exc:
                    logger.warning(
                        "[%s] gi.json topic-alignment re-write failed (non-fatal): %s",
                        episode.idx if hasattr(episode, "idx") else episode_id,
                        _gi_rewrite_exc,
                    )
        except Exception as align_exc:
            logger.warning(
                "GI/KG topic alignment for bridge failed (non-fatal): %s",
                align_exc,
                exc_info=True,
            )

    if bridge_gi_payload is not None or bridge_kg_payload is not None:
        try:
            from podcast_scraper.builders.bridge_artifact_paths import (
                bridge_json_path_adjacent_to_metadata,
            )
            from podcast_scraper.builders.bridge_builder import build_bridge

            bridge_path = bridge_json_path_adjacent_to_metadata(metadata_path)
            bridge_doc = build_bridge(
                str(episode_id),
                bridge_gi_payload,
                bridge_kg_payload,
            )
            bridge_parent = os.path.dirname(bridge_path) or "."
            os.makedirs(bridge_parent, exist_ok=True)
            with open(bridge_path, "w", encoding="utf-8") as bf:
                json.dump(
                    bridge_doc,
                    bf,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            logger.debug("[%s] Generated bridge artifact: %s", episode.idx, bridge_path)
        except Exception as bridge_exc:
            logger.warning(
                "[%s] Bridge artifact generation failed (non-fatal): %s",
                episode.idx,
                bridge_exc,
                exc_info=True,
            )
    return bridge_gi_payload, bridge_kg_payload


def _write_scoped_artifacts_both_or_neither(
    gi_target: Optional[Path],
    gi_payload: Dict[str, Any],
    kg_target: Optional[Path],
    kg_payload: Dict[str, Any],
) -> None:
    """Write the (already id-rewritten) GI and KG artifacts as an all-or-nothing pair (#1862 def.1).

    The two artifacts are separate files, so there is no single atomic commit across both. Before
    this, GI was written first and a failure of the KG write left GI scoped and KG not — the two
    graph layers permanently disagreeing about who a person is (prod: 6/6 placeholders GI-only).
    Two guards make it both-or-neither:

    1. **Validate both payloads before writing either.** A ``validate=True`` schema failure was the
       common cause; validating up front aborts before any file is touched.
    2. **Back up GI; restore it if the KG write fails on I/O.** Restores the prior bytes (or removes
       a file we just created), so a failed pass is a no-op rather than a half-applied scope — which
       also keeps the caller's "ids left as minted" warning accurate.

    ``None`` target means "nothing to write for that layer" (no changes / no target); a genuinely
    missing-but-needed KG target is the caller's error to raise, not silently skip.
    """
    from podcast_scraper.gi.io import write_artifact
    from podcast_scraper.gi.schema import validate_artifact as validate_gi_artifact
    from podcast_scraper.kg.schema import validate_artifact as validate_kg_artifact

    # EACH layer gets ITS OWN validator. Both were previously checked with the GI one, so any
    # scoping pass with a KG layer to write failed on ``GIL artifact missing required key:
    # 'model_version'`` — a key the KG schema does not have and should not: its required set is
    # (schema_version, episode_id, extraction, nodes, edges).
    #
    # The failure was invisible in the worst way. Guard 1 above validates BOTH payloads before
    # writing EITHER, precisely so a schema failure aborts before any file is touched — so this
    # rejected the pass every time, the caller logged "Bare-name scoping failed (non-fatal, ids
    # left as minted)", and the episode kept UNSCOPED person ids. Production 2026-08-31: 10
    # episodes, with the message naming the GI artifact for a defect entirely about the KG one.
    # The both-or-neither guard worked exactly as designed; it was being fed a false failure.
    if gi_target is not None:
        validate_gi_artifact(gi_payload, strict=False)
    if kg_target is not None:
        validate_kg_artifact(kg_payload, strict=False)

    gi_backup: Optional[bytes] = None
    if gi_target is not None:
        gi_backup = gi_target.read_bytes() if gi_target.exists() else None
        write_artifact(gi_target, gi_payload, validate=False)  # already validated above
    if kg_target is not None:
        try:
            write_artifact(kg_target, kg_payload, validate=False)
        except Exception:
            if gi_target is not None:
                if gi_backup is not None:
                    gi_target.write_bytes(gi_backup)
                else:
                    gi_target.unlink(missing_ok=True)
            raise
