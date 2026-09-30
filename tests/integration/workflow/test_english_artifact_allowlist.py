"""The instrument Phase 0's acceptance is judged by (#2171 / slice S0.10).

"English artifacts are byte-identical" was the stated criterion for Phase 0 and there was **no
command behind it** — no artifact-shape check existed in the repo. Unqualified byte-identity is
also not achievable, because the arc deliberately adds language fields. So the criterion becomes
**unchanged outside a declared allow-list**, and this is the thing that checks it.

TWO DIRECTIONS, NOT ONE. The issue originally asked for ``added_keys ⊆ ALLOWLIST``. A subset
check only catches *additions*. The measured baseline has **no optional paths at all** — every
key path is on every episode — so there is no reason to accept the weaker form, and the more
dangerous direction is the other one: a key that silently **disappears**. A consumer reading a
vanished key gets ``None`` and usually carries on without error. So:

    nothing in the baseline may vanish        (baseline - observed) == {}
    anything new must be declared             (observed - baseline) ⊆ ALLOWLIST

WHICH SHAPE — AND WHY NOT A FIXTURE. The first version of this test read the committed
``app-validation-corpus/v3`` metadata. That was blind, for the reason Phase 0 is itself warned
about in §3.1: a committed corpus is a pre-built **output**, so when Phase 0 makes the pipeline
write ``episode.language`` the fixture does not change, the test keeps passing, and it has
proved nothing about the code. It pinned the fixture's shape, not the writer's.

It reads the writer now. ``EpisodeMetadataDocument`` (``metadata_generation.py:866``) is dumped
with ``model_dump_json(indent=2, exclude_none=False)`` at ``:3757`` — ``exclude_none=False``, so
**every declared field is written, including the None ones**. That makes the model's field tree
exactly the on-disk shape: no optional-field ambiguity, and nothing to go stale.

The fixture was also not even the right shape. It has five top-level keys
(``content``/``episode``/``feed``/``schema_version``/``summary``) while the model has seven and
requires ``processing``, which the fixture lacks entirely — because
``build_app_validation_corpus.py`` hand-builds a simplified approximation rather than running
the real writer.

THE MANIFEST HAS TO BE GENERATED. No fixture anywhere carries a per-episode
``<base>.manifest.json`` — the 45 ``*manifest*`` files in the tree are all different artifacts
(``run_manifest.json``, ``corpus_manifest.json``, LanceDB's ``_versions/*.manifest``). So the
manifest half builds one the way the pipeline does, which is also what gives the composition-hash
tests direct control over which stages are recorded.

Regenerate the baseline with::

    REGENERATE_ENGLISH_ARTIFACT_BASELINE=1 .venv/bin/python -m pytest \
        tests/integration/workflow/test_english_artifact_allowlist.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import (
    Any,
    cast,
    Dict,
    get_args,
    get_origin,
    Iterator,
    List,
    Optional,
    Set,
    Type,
    Union,
)

import pytest
from pydantic import BaseModel

from podcast_scraper.workflow import processing_manifest as pm
from podcast_scraper.workflow.metadata_generation import EpisodeMetadataDocument

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]
_BASELINE = _REPO / "tests" / "fixtures" / "goldens" / "english_artifact_shape.golden.json"

#: The writer whose output shape this pins. See the module docstring for why not a fixture.
_WRITER_SOURCE = _REPO / "src" / "podcast_scraper" / "workflow" / "metadata_generation.py"

#: The ONLY key paths Phase 0 is permitted to add. `feed.language` is deliberately absent — the
#: model already declares it (and the fixtures carry `"en-us"`, the value that trips `is_english`
#: today, see #2174).
#:
#: ALL FOUR ARE NOW IN THE BASELINE TOO, as of the S1.2 regeneration: Phase 0 shipped, so its
#: additions are the English shape rather than a pending exception. That makes this list
#: currently redundant and the guard strictly stronger — a path in the baseline may not vanish,
#: while an allowlist-only path could have disappeared without complaint. The list stays because
#: Phase 2 adds fields to the same documents and will need it again; it is kept rather than
#: emptied so the next addition is declared here instead of regenerated in silently.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "feed.language_raw",
        "feed.language_source",
        "episode.language",
        "episode.language_source",
    }
)

#: The stages recorded for an English episode, and the hash of that stage graph. Pinned
#: literally so a change to the graph has to be acknowledged here rather than discovered later
#: by a reprocessing decision — `pipeline_composition_version` is what "reprocess below version
#: X" and the prod-state pin key on.
#:
#: THE STAGES THE PIPELINE RECORDS. Not "the English stages" — there is no such thing. One
#: pipeline runs for every episode in every language, and a stage that finds nothing to do on a
#: given episode records that it found nothing rather than vanishing from the ledger. Naming this
#: `_ENGLISH_STAGES` is what made a per-language stage set look reasonable for one slice.
#:
#: ``translation`` is therefore HERE, and an English episode records it with ``ran=False``.
#:
#: ``turns`` (RFC-123 / S1.2) is here too, but is deliberately NOT in ``CANONICAL_STAGE_ORDER``
#: — it is a structural view of an artifact rather than a stage in the graph, so it does not
#: enter the composition hash. Asserted directly in ``TestCompositionVersion``.
#:
#: WHAT THIS LIST CANNOT DO. It is pinned literally, so it pins the shape of the stages it is
#: TOLD about — it does not discover the pipeline's actual stage set. When S1.2 started writing a
#: `turns` block this guard stayed green until the list was edited by hand. That is a real limit
#: of the instrument, recorded here rather than left to be rediscovered: adding a stage requires
#: editing this line, and nothing fails if you forget.
_PIPELINE_STAGES: tuple[str, ...] = (
    "asr",
    "diarization",
    "naming",
    "turns",
    "translation",
    "summary",
    "gi",
    "kg",
)


def _key_paths(obj: Any, prefix: str = "") -> Iterator[str]:
    """Every leaf path, with list indices collapsed to ``[]``.

    Collapsing matters: a list of 40 speaker dicts is one shape, not 40 paths, and without it
    the baseline would churn whenever a fixture gained an entry.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _key_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _key_paths(v, f"{prefix}[]")
    else:
        yield prefix or "<root>"


#: Fields typed ``Dict[str, Any]``, whose CONTENTS this instrument cannot see.
#:
#: The walker derives the on-disk shape from the model's declared fields, and a
#: ``Dict[str, Any]`` declares no shape at all — so every key inside one is invisible and could
#: appear, change or vanish without failing this test. Found by a whole-branch review on
#: 2026-09-30, along with the more serious case below.
#:
#: They are ENUMERATED rather than silently skipped: a new dict-typed field fails
#: ``test_no_undeclared_opaque_fields`` until someone states why its contents are unmeasurable
#: here. The blindness is then a recorded limitation instead of an accident of the walker.
OPAQUE_DICT_FIELDS: Dict[str, str] = {
    "processing.config_snapshot{}": (
        "records the run Config verbatim, so its keys ARE the Config's fields — pinning them "
        "here would make this test fail on every unrelated config addition, which is what the "
        "config-snapshot tests cover instead"
    ),
    "summary.prefilled_extraction{}": (
        "a provider's raw bundled-extraction payload, whose shape belongs to the provider "
        "response schema rather than to this artifact"
    ),
    "summary.timestamps[]{}": (
        "Whisper segment dicts passed through unchanged; their shape is the ASR provider's"
    ),
    "processing.stage_ledger{}.detail{}": (
        "per-stage free-form detail; each stage chooses its own keys, and the translation "
        "stage's are asserted directly in the translation tests rather than by shape here"
    ),
}


def _model_of(annotation: Any) -> Union[Type[BaseModel], tuple, None]:
    """Peel ``Optional`` / ``Union`` / ``list`` / ``Dict`` down to a ``BaseModel``, if any.

    Returns the model, ``("[]", inner)`` for a list, ``("{}", inner)`` for a dict, or ``None``
    for a leaf.

    THE DICT BRANCH IS THE POINT. It did not exist until 2026-09-30, and without it
    ``Dict[str, StageOutcome]`` peeled to ``None`` — a LEAF. So ``processing.stage_ledger`` was
    a single key path and every field of ``StageOutcome`` was invisible to the instrument that
    Phase 0's acceptance is judged by. The stage ledger is where the translation stage records
    its outcome, so the one artifact subtree this arc adds most to was the one subtree not
    being measured.
    """
    origin = get_origin(annotation)
    if origin is Union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            got = _model_of(arg)
            if got is not None:
                return got
        return None
    if origin in (list, set, tuple):
        args = get_args(annotation)
        return ("[]", _model_of(args[0])) if args else None
    if origin is dict:
        # Keys are arbitrary (a stage name, an episode id), so the KEY is collapsed to `{}` the
        # way list indices collapse to `[]`. The VALUE's shape is what matters and is walked.
        args = get_args(annotation)
        return ("{}", _model_of(args[1])) if len(args) == 2 else ("{}", None)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _metadata_key_paths(
    model: Type[BaseModel] = EpisodeMetadataDocument, prefix: str = "", depth: int = 0
) -> Set[str]:
    """Every key path the writer emits, derived from the model's declared fields.

    Safe to read as "the on-disk shape" ONLY because the writer passes
    ``exclude_none=False`` — asserted by ``test_the_writer_still_emits_none_fields`` below,
    since the moment that changes this whole approach silently measures the wrong thing.
    """
    assert depth <= 8, f"model nesting deeper than expected at {prefix}"
    out: Set[str] = set()
    for name, field in model.model_fields.items():
        here = f"{prefix}.{name}" if prefix else name
        got = _model_of(field.annotation)
        if isinstance(got, tuple):
            # Containers NEST — `List[Dict[str, Any]]` is `("[]", ("{}", None))` — so the
            # markers are composed rather than only the outermost one taken. Without this,
            # `summary.timestamps` emitted `summary.timestamps[]` and lost the `{}` that says
            # its contents are unmeasurable, which put it outside the opaque-field audit below
            # while still being just as opaque.
            marker, inner = got
            while isinstance(inner, tuple):
                marker += inner[0]
                inner = inner[1]
            if inner is not None:
                out |= _metadata_key_paths(inner, f"{here}{marker}", depth + 1)
            else:
                # A container of leaves. `Dict[str, Any]` lands here and its contents are
                # unmeasurable by construction; declared in `OPAQUE_DICT_FIELDS`.
                out.add(f"{here}{marker}")
        elif got is not None:
            out |= _metadata_key_paths(got, here, depth + 1)
        else:
            out.add(here)
    return out


def _stage_metrics(stage: str) -> Optional[Dict[str, Any]]:
    """The real metrics for the stages that report any; ``None`` for the rest."""
    if stage == "turns":
        return _turns_metrics()
    if stage == "translation":
        return _translation_metrics()
    return None


def _turns_metrics() -> Dict[str, Any]:
    """The real ``turns`` metrics shape, from the reporter the pipeline uses (RFC-123 §Monitoring).

    Both variants present and both available, because that is the shape an English diarized
    episode produces; the ``unavailable_reason`` key is conditional and therefore deliberately
    absent from the pinned shape.
    """
    from podcast_scraper.workflow.turns_artifact import turns_manifest_metrics, TurnsOutcome

    return turns_manifest_metrics(
        TurnsOutcome(
            relpath="transcripts/01 - ep.turns.json", count=42, backchannels=3, median_turn_s=8.4
        ),
        TurnsOutcome(
            relpath="transcripts/01 - ep.adfree.turns.json",
            count=38,
            backchannels=3,
            median_turn_s=8.1,
        ),
    )


def _translation_metrics() -> Dict[str, Any]:
    """The real ``translation`` metrics shape, from the stage's own reporter (RFC-124 / S2.2).

    Built from an English episode's outcome, because that is the one every episode in the
    corpus produces today — the stage ran and found nothing to translate.
    """
    from podcast_scraper.workflow.translation_stage import TranslationOutcome

    return TranslationOutcome(
        status="skipped",
        source_language="en",
        language_source="profile_default",
        reason="already_english",
    ).to_metrics()


def _generated_manifest(stages: tuple[str, ...] = _PIPELINE_STAGES) -> Dict[str, Any]:
    """A manifest built the way the pipeline builds one, in a throwaway directory."""
    with tempfile.TemporaryDirectory() as d:
        rel = "transcripts/01 - ep.txt"
        (Path(d) / "transcripts").mkdir(parents=True)
        for stage in stages:
            pm.update_stage(
                d,
                rel,
                stage,
                pm.stage_block(
                    ran=True,
                    method_version=f"{stage}-1",
                    # A block whose metrics are absent pins three key paths instead of ten, so
                    # every stage that reports metrics is built from its own real reporter.
                    metrics=_stage_metrics(stage),
                ),
                episode_id="ep1",
                feed_id="f1",
                run_id="r1",
            )
        raw = Path(pm.manifest_path(d, rel)).read_text(encoding="utf-8")
        return cast(Dict[str, Any], json.loads(raw))


def _manifest_key_paths() -> Set[str]:
    return set(_key_paths(_generated_manifest()))


def _build() -> Dict[str, Any]:
    return {
        "_readme": (
            "The English artifact shape Phase 0 must not disturb. Generated by "
            "tests/integration/workflow/test_english_artifact_allowlist.py. A key may only "
            "appear here if it is in ALLOWLIST; a key may never disappear."
        ),
        "episode_metadata_key_paths": sorted(_metadata_key_paths()),
        "manifest_key_paths": sorted(_manifest_key_paths()),
        "pipeline_composition_version": pm.pipeline_composition_version(_PIPELINE_STAGES),
    }


def _baseline() -> Dict[str, Any]:
    assert _BASELINE.is_file(), (
        f"{_BASELINE.relative_to(_REPO)} is missing — regenerate with "
        "REGENERATE_ENGLISH_ARTIFACT_BASELINE=1"
    )
    return cast(Dict[str, Any], json.loads(_BASELINE.read_text(encoding="utf-8")))


def _verdict(observed: Set[str], baseline: List[str]) -> tuple[Set[str], Set[str]]:
    """``(vanished, undeclared)`` — the two ways a shape can be wrong."""
    base = set(baseline)
    return base - observed, (observed - base) - ALLOWLIST


def test_regenerate_baseline_when_asked() -> None:
    if not os.environ.get("REGENERATE_ENGLISH_ARTIFACT_BASELINE"):
        pytest.skip("set REGENERATE_ENGLISH_ARTIFACT_BASELINE=1 to rewrite the baseline")
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(_build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TestArtifactShape:
    def test_episode_metadata_shape_is_unchanged_outside_the_allowlist(self) -> None:
        vanished, undeclared = _verdict(
            _metadata_key_paths(), _baseline()["episode_metadata_key_paths"]
        )
        assert not vanished, (
            f"episode metadata LOST key paths: {sorted(vanished)}. A consumer reading a "
            "vanished key gets None and carries on, so this is the quieter failure of the two."
        )
        assert not undeclared, (
            f"episode metadata gained undeclared key paths: {sorted(undeclared)}. Add them to "
            f"ALLOWLIST deliberately, or stop adding them. Currently declared: {sorted(ALLOWLIST)}"
        )

    def test_manifest_shape_is_unchanged_outside_the_allowlist(self) -> None:
        vanished, undeclared = _verdict(_manifest_key_paths(), _baseline()["manifest_key_paths"])
        assert not vanished, f"manifest LOST key paths: {sorted(vanished)}"
        assert not undeclared, f"manifest gained undeclared key paths: {sorted(undeclared)}"

    def test_the_turns_block_does_not_move_the_composition_hash(self) -> None:
        """RFC-123's block is recorded without disturbing the reprocess query key.

        ``pipeline_composition_version`` is what "reprocess everything below version X" and the
        prod-state pin key on. Putting ``turns`` in ``CANONICAL_STAGE_ORDER`` would rewrite it for
        every episode in the corpus — invalidating those queries for a sidecar no consumer reads
        yet — so it is excluded there, and this is the assertion that keeps it excluded.
        """
        core = tuple(s for s in _PIPELINE_STAGES if s != "turns")
        assert pm.pipeline_composition_version(_PIPELINE_STAGES) == (
            pm.pipeline_composition_version(core)
        )
        assert "turns" not in pm.CANONICAL_STAGE_ORDER

    def test_the_writer_still_emits_none_fields(self) -> None:
        """The assumption the whole metadata half rests on.

        Reading the model as "the on-disk shape" is only valid while the writer passes
        ``exclude_none=False``. Flip it to ``True`` and optional fields stop being written, the
        model over-states the shape, and this instrument measures something that is no longer
        on disk — silently, because the model still declares them.

        Asserted against the source text deliberately. There is no way to observe the flag from
        outside the call, and an assumption that cannot be observed is exactly the kind that
        should be pinned loudly rather than trusted.
        """
        src = _WRITER_SOURCE.read_text(encoding="utf-8")
        assert "model_dump_json(indent=2, exclude_none=False)" in src, (
            "the episode-metadata writer no longer dumps with exclude_none=False. The model's "
            "field tree is then NOT the on-disk shape, and _metadata_key_paths() over-states "
            "it. Derive the baseline from a dumped instance instead."
        )

    def test_the_model_declares_the_blocks_the_arc_touches(self) -> None:
        """A cheap sanity check that the walker reached the whole document.

        A walker that silently stopped early would produce a small baseline, and a small
        baseline makes the `vanished` check weak rather than failing.
        """
        top = {p.split(".")[0].split("[")[0] for p in _metadata_key_paths()}
        assert {"feed", "episode", "content", "processing"} <= top, top
        assert "feed.language" in _metadata_key_paths(), "the field Phase 0 normalizes"


class TestCompositionVersion:
    """The hash `reprocess below version X` and the prod-state pin key on.

    `pipeline_composition_version` is NOT simply "the stages present": it intersects the
    recorded stage keys with `CANONICAL_STAGE_ORDER`, so a stage must pass both gates to
    affect the hash.

    THIS CLASS PREVIOUSLY ENCODED THE OPPOSITE OF WHAT IT NOW ASSERTS, and the correction is
    the point of the rewrite. It used to say the "safe shape" for Phase 2 was to declare
    `translation` in the order but never RECORD it for an English episode, so the English
    hash would not move. That advice was mine, written in Phase 0, and in S2.2 I cited it back
    as though it were an independent measurement and let it override the arc's plan. It was an
    opinion in the shape of a test.

    It was also wrong on the merits. There is ONE pipeline — ASR, diarization, naming,
    translation, summary, GI, KG — and it runs for every episode in every language. Withholding
    the record made the hash a function of the EPISODE'S LANGUAGE instead of the code: two
    episodes off the same commit hashed differently because one was Spanish. A provenance hash
    that varies with content is not a stable hash, it is a broken one.
    """

    def test_the_stage_graph_hash_is_pinned(self) -> None:
        assert (
            pm.pipeline_composition_version(_PIPELINE_STAGES)
            == _baseline()["pipeline_composition_version"]
        )

    def test_the_hash_does_not_depend_on_the_episodes_language(self) -> None:
        """THE INVARIANT THIS CLASS EXISTS FOR, and the one that would have caught the defect.

        Same code, same pipeline, different episode language — the composition hash must be
        identical, because the pipeline is identical. Driven through the real
        `run_translation_stage` and a real written manifest, not through set arithmetic, since
        the claim is about what the pipeline WRITES.
        """
        from podcast_scraper import config
        from podcast_scraper.workflow.translation_stage import run_translation_stage

        def hash_for(language: str) -> str:
            with tempfile.TemporaryDirectory() as d:
                rel = "transcripts/01 - ep.txt"
                (Path(d) / "transcripts").mkdir(parents=True)
                for stage in (s for s in _PIPELINE_STAGES if s != "translation"):
                    pm.update_stage(d, rel, stage, pm.stage_block(ran=True, method_version="x"))
                run_translation_stage(
                    config.Config(rss="https://e.com/f.xml", language=language),
                    transcript_relpath=rel,
                    effective_output_dir=d,
                )
                data = json.loads(Path(pm.manifest_path(d, rel)).read_text(encoding="utf-8"))
                return str(data["pipeline_composition_version"])

        assert hash_for("en") == hash_for("es") == _baseline()["pipeline_composition_version"]

    def test_every_language_records_the_translation_stage(self) -> None:
        """The mechanism behind the invariant above, so a regression names its own cause.

        An English episode records `translation` with `ran=False` — the stage ran and found
        nothing to do. That is a RESULT. Absence would have meant "this episode predates the
        translation stage", which is the measured-vs-defaulted confusion `language_source`
        exists to prevent.
        """
        from podcast_scraper import config
        from podcast_scraper.workflow.translation_stage import run_translation_stage

        for language, expected_reason in (("en", "already_english"), ("es", "flag_off")):
            with tempfile.TemporaryDirectory() as d:
                rel = "transcripts/01 - ep.txt"
                (Path(d) / "transcripts").mkdir(parents=True)
                run_translation_stage(
                    config.Config(rss="https://e.com/f.xml", language=language),
                    transcript_relpath=rel,
                    effective_output_dir=d,
                )
                data = json.loads(Path(pm.manifest_path(d, rel)).read_text(encoding="utf-8"))

            block = data["stages"]["translation"]
            assert block["ran"] is False, language
            assert block["metrics"]["reason"] == expected_reason
            assert block["metrics"]["source_language"] == language

    def test_a_stage_outside_the_canonical_order_cannot_move_the_hash(self) -> None:
        """Gate one. `turns` is outside the tuple deliberately — it is a structural view of an
        artifact, not a stage in the graph — so recording it changes nothing."""
        assert "turns" not in pm.CANONICAL_STAGE_ORDER
        assert pm.pipeline_composition_version(
            [*_PIPELINE_STAGES, "turns"]
        ) == pm.pipeline_composition_version(_PIPELINE_STAGES)

    def test_a_stage_leaving_the_recorded_set_does_move_the_hash(self) -> None:
        """Gate two, and the inverse that makes the pin mean something.

        A test that only ever confirms "unchanged" cannot tell a stable hash from one that never
        moves at all. Dropping a real stage from the recorded set must move it.
        """
        pinned = _baseline()["pipeline_composition_version"]
        without_translation = [s for s in _PIPELINE_STAGES if s != "translation"]
        assert pm.pipeline_composition_version(without_translation) != pinned

    def test_ran_false_does_not_keep_a_stage_out_of_the_hash(self) -> None:
        """The mechanic the old advice was built on, kept because it is still true and still
        surprising: the hash keys on the stage being PRESENT in the recorded set, and `ran` is
        just a field inside the block.

        What changed is the conclusion drawn from it. It used to read as "therefore do not
        record a skipped stage"; it now reads as "therefore recording it is what puts the stage
        in the graph, which is exactly what we want it to say."
        """
        with tempfile.TemporaryDirectory() as d:
            rel = "transcripts/01 - ep.txt"
            (Path(d) / "transcripts").mkdir(parents=True)
            for stage in (s for s in _PIPELINE_STAGES if s != "translation"):
                pm.update_stage(d, rel, stage, pm.stage_block(ran=True, method_version="x"))
            path = Path(pm.manifest_path(d, rel))
            before = json.loads(path.read_text(encoding="utf-8"))

            pm.update_stage(d, rel, "translation", pm.stage_block(ran=False))
            after = json.loads(path.read_text(encoding="utf-8"))

        assert after["stages"]["translation"]["ran"] is False, "recorded, and honestly skipped"
        assert after["pipeline_composition_version"] != before["pipeline_composition_version"]
        assert after["pipeline_composition_version"] == _baseline()["pipeline_composition_version"]


class TestTheCheckFires:
    """Prove the instrument can fail. See MULTILINGUAL_ARC.md §3.1.

    Every assertion above is satisfiable by a check that can never fail, and a check that has
    only ever been green is indistinguishable from no check.
    """

    def test_an_undeclared_added_key_is_named(self) -> None:
        observed = set(_baseline()["episode_metadata_key_paths"]) | {"feed.sneaked_in"}
        vanished, undeclared = _verdict(observed, _baseline()["episode_metadata_key_paths"])
        assert not vanished
        assert undeclared == {"feed.sneaked_in"}

    def test_an_allowlisted_added_key_is_accepted(self) -> None:
        observed = set(_baseline()["episode_metadata_key_paths"]) | set(ALLOWLIST)
        vanished, undeclared = _verdict(observed, _baseline()["episode_metadata_key_paths"])
        assert not vanished and not undeclared

    def test_a_vanished_key_is_named(self) -> None:
        base = _baseline()["episode_metadata_key_paths"]
        observed = set(base) - {"content.transcript_file_path"}
        vanished, undeclared = _verdict(observed, base)
        assert vanished == {"content.transcript_file_path"}
        assert not undeclared

    def test_a_key_swapped_for_an_allowlisted_one_still_fails(self) -> None:
        """The case a one-directional check waves through.

        Removing a real key while adding a declared one nets to zero under
        `added ⊆ ALLOWLIST`, and is exactly the silent shape this test exists to catch.
        """
        base = _baseline()["episode_metadata_key_paths"]
        observed = (set(base) - {"feed.language"}) | {"feed.language_raw"}
        vanished, undeclared = _verdict(observed, base)
        assert vanished == {"feed.language"}
        assert not undeclared, "the addition is declared; the REMOVAL is what must fail this"


class TestTheInstrumentCanSeeWhatItClaims:
    """The instrument's own blind spots, made explicit rather than left to be rediscovered.

    Found by a whole-branch review on 2026-09-30. `_model_of` peeled `Optional`, `Union` and
    `list` but not `Dict`, so any dict-typed field became a LEAF and everything inside it was
    invisible to the check that Phase 0's acceptance is judged by.

    For `Dict[str, Any]` that is unavoidable — there is no declared shape to walk — but it was
    silent. For `Dict[str, SomeModel]` it was simply wrong, and that is the case that mattered:
    `processing.stage_ledger` is `Dict[str, StageOutcome]`, so the entire stage-ledger shape
    (`outcome`, `reason`, `duration_seconds`, `detail`) was unmeasured — the one subtree this
    arc writes most to. Those four paths appear in the baseline for the first time here; they
    were always written, never watched.
    """

    def test_no_undeclared_opaque_fields(self) -> None:
        """A new `Dict[str, Any]` field must state why its contents cannot be measured here.

        This is what turns the blindness from an accident of the walker into a recorded
        limitation: the field still is not checked, but nobody can add one without saying so.
        """
        opaque = {p for p in _metadata_key_paths() if p.endswith("{}")}
        undeclared = opaque - set(OPAQUE_DICT_FIELDS)
        assert undeclared == set(), (
            f"dict-typed field(s) whose contents this test cannot see: {sorted(undeclared)}. "
            "Add each to OPAQUE_DICT_FIELDS with the reason its shape is unmeasurable here, or "
            "type it as a model so the walker can check it."
        )

    def test_no_stale_opaque_declarations(self) -> None:
        """A declaration for a field that is no longer opaque is a standing excuse; it would let
        a real dict field slip in later under an existing line."""
        opaque = {p for p in _metadata_key_paths() if p.endswith("{}")}
        assert sorted(set(OPAQUE_DICT_FIELDS) - opaque) == []

    def test_every_opaque_field_states_a_reason(self) -> None:
        for path, reason in OPAQUE_DICT_FIELDS.items():
            assert reason.strip(), f"{path} has no stated reason"
            assert len(reason.split()) >= 6, f"{path}: {reason!r} is not a reason"

    def test_the_stage_ledger_shape_is_now_MEASURED(self) -> None:
        """The regression that motivated the fix. `Dict[str, StageOutcome]` peeled to a leaf, so
        a field added to or removed from `StageOutcome` changed nothing here."""
        paths = _metadata_key_paths()
        for field in ("outcome", "reason", "duration_seconds"):
            assert f"processing.stage_ledger{{}}.{field}" in paths

    def test_a_dict_of_models_is_walked_not_treated_as_a_leaf(self) -> None:
        """Directly, on a throwaway model, so the property holds independently of whatever the
        real document happens to declare today."""

        class Inner(BaseModel):
            a: int
            b: Optional[str] = None

        class Outer(BaseModel):
            rows: Dict[str, Inner]

        assert _metadata_key_paths(Outer) == {"rows{}.a", "rows{}.b"}

    def test_nested_containers_keep_every_marker(self) -> None:
        """`List[Dict[str, Any]]` must not lose its `{}`. It did, which put
        `summary.timestamps` outside the opaque audit while being exactly as opaque."""

        class Outer(BaseModel):
            rows: List[Dict[str, Any]]

        assert _metadata_key_paths(Outer) == {"rows[]{}"}

    def test_a_list_of_models_is_still_walked(self) -> None:
        """The pre-existing behaviour, pinned so the dict branch did not disturb it."""

        class Inner(BaseModel):
            a: int

        class Outer(BaseModel):
            rows: List[Inner]

        assert _metadata_key_paths(Outer) == {"rows[].a"}

    def test_an_optional_dict_of_models_is_also_walked(self) -> None:
        """`Optional[Dict[str, Model]]` is the shape most of these fields actually have."""

        class Inner(BaseModel):
            a: int

        class Outer(BaseModel):
            rows: Optional[Dict[str, Inner]] = None

        assert _metadata_key_paths(Outer) == {"rows{}.a"}
