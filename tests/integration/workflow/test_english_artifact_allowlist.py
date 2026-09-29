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
from typing import Any, cast, Dict, get_args, get_origin, Iterator, List, Set, Type, Union

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
#: ``translation`` is deliberately ABSENT: an English episode records no translation block, which
#: is what keeps its composition hash where it was (see ``TestCompositionVersion``). A non-English
#: episode does get one, and that episode's hash differs — correctly, because its pipeline ran a
#: different set of stages.
#:
#: ``turns`` (RFC-123 / S1.2) is here because an English episode records it, but it is
#: deliberately NOT in ``CANONICAL_STAGE_ORDER`` — so the composition hash below is unchanged by
#: its presence, which is the whole point and is asserted directly in
#: ``test_the_turns_block_does_not_move_the_composition_hash``.
#:
#: WHAT THIS LIST CANNOT DO. It is pinned literally, so it pins the shape of the stages it is
#: TOLD about — it does not discover the pipeline's actual stage set. When S1.2 started writing a
#: `turns` block this guard stayed green until the list was edited by hand. That is a real limit
#: of the instrument, recorded here rather than left to be rediscovered: adding a stage requires
#: editing this line, and nothing fails if you forget.
_ENGLISH_STAGES: tuple[str, ...] = (
    "asr",
    "diarization",
    "naming",
    "turns",
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


def _model_of(annotation: Any) -> Union[Type[BaseModel], tuple, None]:
    """Peel ``Optional`` / ``Union`` / ``list`` down to a ``BaseModel``, if there is one.

    Returns the model, or ``("[]", inner)`` for a list, or ``None`` for a leaf.
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
            _, inner = got
            if inner is not None and not isinstance(inner, tuple):
                out |= _metadata_key_paths(inner, f"{here}[]", depth + 1)
            else:
                out.add(f"{here}[]")
        elif got is not None:
            out |= _metadata_key_paths(got, here, depth + 1)
        else:
            out.add(here)
    return out


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


def _generated_manifest(stages: tuple[str, ...] = _ENGLISH_STAGES) -> Dict[str, Any]:
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
                    # The turns block carries metrics, and a block whose metrics are absent pins
                    # three key paths instead of ten — so it is built from the real reporter.
                    metrics=_turns_metrics() if stage == "turns" else None,
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
        "pipeline_composition_version": pm.pipeline_composition_version(_ENGLISH_STAGES),
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
        core = tuple(s for s in _ENGLISH_STAGES if s != "turns")
        assert pm.pipeline_composition_version(_ENGLISH_STAGES) == (
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

    Corrected from the issue's original framing, which named the wrong lever.
    `pipeline_composition_version` is NOT simply "the stages present": it intersects the
    recorded stage keys with `CANONICAL_STAGE_ORDER`, so there are TWO gates and a stage must
    pass both to affect the hash.
    """

    def test_the_english_stage_graph_hash_is_pinned(self) -> None:
        assert (
            pm.pipeline_composition_version(_ENGLISH_STAGES)
            == _baseline()["pipeline_composition_version"]
        )

    def test_a_stage_outside_the_canonical_order_cannot_move_the_hash(self) -> None:
        """Gate one. Recording a stage changes nothing while it is not in the tuple.

        The example used to be ``translation``. S2.2 put translation IN the tuple deliberately
        (a pipeline with a translation step is not the pipeline without one), so the example is
        now ``turns`` — which is outside the tuple, also deliberately, for the opposite reason:
        it is a structural view of an artifact rather than a stage in the graph.
        """
        assert "turns" not in pm.CANONICAL_STAGE_ORDER
        assert pm.pipeline_composition_version(
            [*_ENGLISH_STAGES, "turns"]
        ) == pm.pipeline_composition_version(_ENGLISH_STAGES)

    def test_declaring_the_stage_but_never_recording_it_is_the_safe_shape(self) -> None:
        """Gate two, and the shape Phase 2 DID adopt — no longer a recommendation.

        ``translation`` is declared in ``CANONICAL_STAGE_ORDER`` as of S2.2, and that alone is
        harmless: the hash for an English episode only moves if the stage is RECORDED for it,
        including recorded as skipped. So ``translation_stage`` does not write a block when the
        reason is ``already_english``, and the arc's S2.2 text ("every English episode's ledger
        gains translation: skipped") was corrected to match this measurement.

        The monkeypatch is gone because the stage is really in the tuple now — the assertion is
        against the shipped order, not a simulated one.
        """
        assert "translation" in pm.CANONICAL_STAGE_ORDER
        assert (
            pm.pipeline_composition_version(_ENGLISH_STAGES)
            == _baseline()["pipeline_composition_version"]
        )

    def test_the_english_episode_really_gets_no_translation_block(self) -> None:
        """The decision above, asserted through the stage itself rather than about it.

        Gate two is only safe while the writer actually declines to write. This drives the real
        ``run_translation_stage`` for an English episode and requires the manifest to come back
        with no ``translation`` key and the pinned English hash intact.
        """
        from podcast_scraper import config
        from podcast_scraper.workflow.translation_stage import run_translation_stage

        with tempfile.TemporaryDirectory() as d:
            rel = "transcripts/01 - ep.txt"
            (Path(d) / "transcripts").mkdir(parents=True)
            for stage in _ENGLISH_STAGES:
                pm.update_stage(d, rel, stage, pm.stage_block(ran=True, method_version="x"))

            run_translation_stage(
                config.Config(rss="https://e.com/f.xml", language="en"),
                transcript_relpath=rel,
                effective_output_dir=d,
            )
            data = json.loads(Path(pm.manifest_path(d, rel)).read_text(encoding="utf-8"))

        assert "translation" not in data["stages"]
        assert data["pipeline_composition_version"] == _baseline()["pipeline_composition_version"]

    def test_a_non_english_episode_does_get_one(self) -> None:
        """The inverse, so the decision above is a choice and not an inability to write.

        A stage that never records anything would pass the test above for the wrong reason.
        """
        from podcast_scraper import config
        from podcast_scraper.workflow.translation_stage import run_translation_stage

        with tempfile.TemporaryDirectory() as d:
            rel = "transcripts/01 - ep.txt"
            (Path(d) / "transcripts").mkdir(parents=True)
            run_translation_stage(
                config.Config(rss="https://e.com/f.xml", language="es"),
                transcript_relpath=rel,
                effective_output_dir=d,
            )
            data = json.loads(Path(pm.manifest_path(d, rel)).read_text(encoding="utf-8"))

        assert data["stages"]["translation"]["metrics"]["source_language"] == "es"
        assert data["stages"]["translation"]["metrics"]["reason"] == "flag_off"

    def test_recording_the_stage_does_move_the_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The inverse, so the pin above means something.

        A test that only ever confirms "unchanged" cannot distinguish a stable hash from a hash
        that never moves at all.
        """
        monkeypatch.setattr(pm, "CANONICAL_STAGE_ORDER", (*pm.CANONICAL_STAGE_ORDER, "translation"))
        assert (
            pm.pipeline_composition_version([*_ENGLISH_STAGES, "translation"])
            != _baseline()["pipeline_composition_version"]
        )

    def test_recording_a_stage_as_ran_false_still_moves_the_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap, demonstrated end to end through a real written manifest.

        `stage_block(ran=False)` is the obvious way to record "this stage was skipped", and it
        appears NOWHERE in `src/` or `tests/` today — so translation would be the first use.
        It is also precisely wrong here: the hash keys on the stage being **present in the
        recorded set**, and `ran` is just a field inside the block. Recording a skipped
        translation therefore moves the composition hash for every English episode.

        Asserted through `update_stage` rather than by calling the hash function with a hand-made
        list, because the claim is about what the pipeline WRITES, not about set arithmetic.
        """
        monkeypatch.setattr(pm, "CANONICAL_STAGE_ORDER", (*pm.CANONICAL_STAGE_ORDER, "translation"))
        pinned = _baseline()["pipeline_composition_version"]

        with tempfile.TemporaryDirectory() as d:
            rel = "transcripts/01 - ep.txt"
            (Path(d) / "transcripts").mkdir(parents=True)
            for stage in _ENGLISH_STAGES:
                pm.update_stage(d, rel, stage, pm.stage_block(ran=True, method_version="x"))
            path = Path(pm.manifest_path(d, rel))
            before = json.loads(path.read_text(encoding="utf-8"))
            assert before["pipeline_composition_version"] == pinned, "English baseline intact"

            # The one line Phase 2 must not write for an English episode.
            pm.update_stage(d, rel, "translation", pm.stage_block(ran=False))
            after = json.loads(path.read_text(encoding="utf-8"))

        assert after["stages"]["translation"]["ran"] is False, "recorded, and honestly skipped"
        assert after["pipeline_composition_version"] != pinned, (
            "a stage recorded as ran=False still entered the hash — which is the whole point: "
            "`ran` does not keep it out"
        )


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
