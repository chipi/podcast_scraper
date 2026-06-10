"""Integration tests for v3 fixture generator (issue #921).

These tests assert the v3 corpus produced by ``scripts/build_v3_fixtures.py``:

1. Exercises every failure mode in ``FAILURE_MODES`` (≥ 1 episode each).
2. Is deterministic: re-running the generator produces identical bytes.
3. Wraps cleanly into the autoresearch dataset format
   (``data/eval/datasets/curated_5feeds_smoke_v3.json``).
4. Does NOT alter v2 paths (additive only).
5. Ground-truth labels are consistent with the rendered transcripts
   (every recorded surface form appears verbatim in the transcript).

Run::

    pytest tests/integration/eval/test_v3_fixtures.py
    pytest -p no:randomly tests/integration/eval/test_v3_fixtures.py  # determinism
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_generator():
    """Load ``scripts/build_v3_fixtures.py`` as a module.

    ``scripts/`` isn't a package so importing the dotted path fails; load by
    spec so the test rig works without touching ``pyproject``/``sys.path``.
    """
    path = PROJECT_ROOT / "scripts" / "build_v3_fixtures.py"
    spec = importlib.util.spec_from_file_location("build_v3_fixtures", path)
    assert spec and spec.loader, "Failed to load scripts/build_v3_fixtures.py"
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclasses.dataclass() can look up
    # the module's __dict__ when it walks type hints. Otherwise dataclass()
    # raises ``AttributeError: 'NoneType' object has no attribute '__dict__'``
    # (Python 3.11 dataclasses.py:712).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v3():
    return _load_generator()


@pytest.fixture(scope="module")
def v3_summary(v3):
    """In-memory render of the v3 corpus (dry-run; no disk writes)."""
    podcasts = v3.build_v3_spec()
    return v3.emit_corpus(podcasts, dry_run=True)


@pytest.fixture(scope="module")
def manifest_on_disk():
    """Read the persisted ``tests/fixtures/v3/manifest.json``.

    The repo-tracked manifest is the source of truth for downstream tools;
    if a generator change leaves disk out of sync, the v3 build is broken
    and the test must surface that.
    """
    manifest_path = PROJECT_ROOT / "tests" / "fixtures" / "v3" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(
            "tests/fixtures/v3/manifest.json missing — run "
            "`python scripts/build_v3_fixtures.py` first."
        )
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Coverage assertion: every failure mode is exercised by ≥ 1 episode.
# ---------------------------------------------------------------------------


def test_failure_mode_vocabulary_covered_by_episodes(v3, v3_summary):
    """Every entry in ``FAILURE_MODES`` is tagged on ≥ 1 episode.

    This is the central coverage test for #921. If a new failure mode is
    added to the vocabulary, the spec must add at least one episode that
    tags it, or this test fails — preventing dead vocabulary entries.
    """
    missing = [mode for mode in v3.FAILURE_MODES if v3_summary["failure_mode_coverage"][mode] == 0]
    assert not missing, f"Failure modes not exercised by any episode: {missing}"


def test_failure_mode_tags_are_in_vocabulary(v3, v3_summary):
    """No episode tags an out-of-vocabulary failure mode (catches typos)."""
    vocab = set(v3.FAILURE_MODES)
    for ep in v3_summary["episodes"]:
        unknown = [m for m in ep["failure_modes"] if m not in vocab]
        assert not unknown, f"Episode {ep['episode_id']} tags unknown modes: {unknown}"


# ---------------------------------------------------------------------------
# Determinism: same spec → same transcripts → same hashes.
# ---------------------------------------------------------------------------


def test_generator_is_deterministic(v3):
    """Running the generator twice produces bit-identical episode transcripts
    + ground truth.

    Uses the in-memory ``render_episode`` path so this doesn't depend on
    the filesystem; rendering must be a pure function of (podcast, ep).
    """
    podcasts_a = v3.build_v3_spec()
    podcasts_b = v3.build_v3_spec()
    for pod_a, pod_b in zip(podcasts_a, podcasts_b):
        assert pod_a.pod_id == pod_b.pod_id
        for ep_a, ep_b in zip(pod_a.episodes, pod_b.episodes):
            text_a, truth_a = v3.render_episode(pod_a, ep_a)
            text_b, truth_b = v3.render_episode(pod_b, ep_b)
            assert text_a == text_b, f"non-deterministic transcript for {pod_a.pod_id}_{ep_a.ep_id}"
            assert (
                truth_a == truth_b
            ), f"non-deterministic ground truth for {pod_a.pod_id}_{ep_a.ep_id}"


def test_summary_round_trip_is_stable(v3):
    """``emit_corpus(dry_run=True)`` is idempotent within a single process.

    A weaker but cheaper check than the per-episode determinism test —
    runs the whole emit pipeline twice and compares the summary dict.
    """
    podcasts = v3.build_v3_spec()
    first = v3.emit_corpus(podcasts, dry_run=True)
    second = v3.emit_corpus(podcasts, dry_run=True)
    # Drop path strings from comparison — they're absolute and OS-dependent
    # but identical within a single test run.
    for key in ("fixtures_manifest_path", "dataset_yaml_path", "dataset_json_path"):
        first.pop(key, None)
        second.pop(key, None)
    assert first == second


# ---------------------------------------------------------------------------
# Disk-side checks (persisted manifest matches generator state).
# ---------------------------------------------------------------------------


def test_persisted_manifest_matches_generator_state(manifest_on_disk, v3_summary, v3):
    """The committed manifest covers the same failure modes as the live spec.

    Catches the failure mode where someone updates the generator's spec
    without re-running it, leaving disk stale.
    """
    disk_modes_per_ep = {
        ep["episode_id"]: set(ep["failure_modes"]) for ep in manifest_on_disk["episodes"]
    }
    live_modes_per_ep = {
        ep["episode_id"]: set(ep["failure_modes"]) for ep in v3_summary["episodes"]
    }
    assert disk_modes_per_ep == live_modes_per_ep, (
        "tests/fixtures/v3/manifest.json is stale; run "
        "`python scripts/build_v3_fixtures.py` to refresh."
    )


def test_each_failure_mode_present_in_manifest(manifest_on_disk, v3):
    """Every vocabulary entry is also present on at least one manifest episode.

    Duplicate of the coverage test but against on-disk state, so a stale
    or hand-edited manifest can't silently drop coverage.
    """
    seen: set[str] = set()
    for ep in manifest_on_disk["episodes"]:
        seen.update(ep["failure_modes"])
    missing = [m for m in v3.FAILURE_MODES if m not in seen]
    assert not missing, f"Manifest is missing coverage for: {missing}"


# ---------------------------------------------------------------------------
# Ground-truth consistency: surface forms appear in transcripts.
# ---------------------------------------------------------------------------


def _read_transcript(episode_id: str) -> str:
    p = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v3" / f"{episode_id}.txt"
    return p.read_text(encoding="utf-8")


def _read_ground_truth(episode_id: str) -> dict[str, Any]:
    p = PROJECT_ROOT / "tests" / "fixtures" / "v3" / "ground_truth" / f"{episode_id}.json"
    return cast(dict[str, Any], json.loads(p.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "episode_id",
    [
        "p01_e01",
        "p02_e02",
        "p03_e03",
        "p04_e04",
        "p05_e02",
        "p05_e03",
        "p06_e01",
        "p07_e02",
        "p09_e03",
    ],
)
def test_ground_truth_surface_forms_appear_in_transcript(episode_id):
    """Every recorded surface form is verbatim in its rendered transcript.

    If ground truth claims ``Liam Vandermeer`` is the alias in p01_e03, the
    string must literally appear in the transcript — otherwise downstream
    eval scoring can't match the claim against text spans.
    """
    transcript = _read_transcript(episode_id)
    truth = _read_ground_truth(episode_id)
    for sf in truth["surface_forms"]:
        assert sf["surface"] in transcript, (
            f"surface form {sf['surface']!r} not found in {episode_id}.txt "
            f"(canonical_id={sf['canonical_id']}, kind={sf['kind']})"
        )


def test_sponsor_block_kinds_recorded_per_episode():
    """Every persisted ground-truth file lists at least one sponsor block.

    The shape we expect: opening + closing template ads on every episode,
    plus optional native_ad / enthusiastic_recommendation / template_midroll.
    """
    truth_dir = PROJECT_ROOT / "tests" / "fixtures" / "v3" / "ground_truth"
    if not truth_dir.exists():
        pytest.skip("ground truth not on disk; run the generator first")
    for json_path in truth_dir.glob("*.json"):
        truth = json.loads(json_path.read_text(encoding="utf-8"))
        assert truth["sponsor_blocks"], f"{json_path.name}: no sponsor blocks recorded"
        kinds = {b["kind"] for b in truth["sponsor_blocks"]}
        assert "template_opening" in kinds, f"{json_path.name}: missing template_opening"


def test_enthusiastic_recommendation_marked_as_real_content():
    """Sponsor-shaped real content is marked with the explicit 'real' note.

    The cleaning baseline scoring (#905) reads this field to penalize
    cleaners that strip enthusiastic_recommendation spans.
    """
    truth_dir = PROJECT_ROOT / "tests" / "fixtures" / "v3" / "ground_truth"
    found_any = False
    for json_path in truth_dir.glob("*.json"):
        truth = json.loads(json_path.read_text(encoding="utf-8"))
        for block in truth["sponsor_blocks"]:
            if block["kind"] == "enthusiastic_recommendation":
                found_any = True
                assert "note" in block and "NOT a paid sponsor" in block["note"]
    assert found_any, "no enthusiastic_recommendation blocks across the v3 corpus"


# ---------------------------------------------------------------------------
# v2 additive check — v2 must still exist + be unmodified by v3 generator.
# ---------------------------------------------------------------------------


def test_v2_transcripts_untouched():
    """v3 generation must NOT delete or modify v2 transcripts."""
    v2_dir = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v2"
    assert v2_dir.is_dir(), "v2 transcripts dir missing — additive contract violated"
    v2_files = list(v2_dir.glob("*.txt"))
    assert len(v2_files) >= 30, f"v2 transcripts disappeared (found {len(v2_files)})"


def test_fixtures_version_still_v2():
    """Default fixtures version stays at v2 until v3 is feature-complete.

    Per #921 plan: bump FIXTURES_VERSION to v3 only when downstream tests
    are verified to pass on v3 — not in this PR.
    """
    fv = PROJECT_ROOT / "tests" / "fixtures" / "FIXTURES_VERSION"
    if not fv.exists():
        pytest.skip("FIXTURES_VERSION file missing")
    assert fv.read_text(encoding="utf-8").strip() == "v2"


# ---------------------------------------------------------------------------
# Dataset wrapper format checks.
# ---------------------------------------------------------------------------


def test_smoke_dataset_json_well_formed():
    """The flat-file dataset JSON loads cleanly and has 5 smoke episodes."""
    p = PROJECT_ROOT / "data" / "eval" / "datasets" / "curated_5feeds_smoke_v3.json"
    if not p.exists():
        pytest.skip("v3 smoke dataset not on disk; run the generator first")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["dataset_id"] == "curated_5feeds_smoke_v3"
    assert data["num_episodes"] == 5
    assert len(data["episodes"]) == 5
    # Every smoke entry carries failure_modes (may be empty list but present).
    for ep in data["episodes"]:
        assert "failure_modes" in ep, f"smoke ep {ep['episode_id']} missing failure_modes"


def test_smoke_dataset_yaml_emitted():
    p = PROJECT_ROOT / "data" / "eval" / "datasets" / "curated_5feeds_smoke_v3" / "manifest.yaml"
    if not p.exists():
        pytest.skip("v3 dataset dir not on disk; run the generator first")
    text = p.read_text(encoding="utf-8")
    assert text.startswith("dataset_id: curated_5feeds_smoke_v3"), "yaml header malformed"
    # Each smoke episode should carry a transcript_hash.
    assert "transcript_hash:" in text


# ---------------------------------------------------------------------------
# Smoke-shape parity with v2 autoresearch loader.
# ---------------------------------------------------------------------------


def test_v3_smoke_has_v2_shape_compatible_fields():
    """The v3 smoke flat file carries every field v2's smoke loader expects.

    Don't break the v2 autoresearch reader — v3 must be a superset.
    Compared against the live v2 smoke dataset; if v2's schema evolves, this
    test catches divergence.
    """
    v2_path = PROJECT_ROOT / "data" / "eval" / "datasets" / "curated_5feeds_smoke_v2.json"
    v3_path = PROJECT_ROOT / "data" / "eval" / "datasets" / "curated_5feeds_smoke_v3.json"
    if not (v2_path.exists() and v3_path.exists()):
        pytest.skip("v2 or v3 smoke dataset missing")
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v2_top_level = set(v2.keys())
    v3_top_level = set(v3.keys())
    # v3 may add fields; must not drop any.
    missing = v2_top_level - v3_top_level
    assert not missing, f"v3 smoke dataset dropped v2 top-level fields: {missing}"
    # Each v3 episode must have at minimum the v2 episode fields.
    v2_ep_keys = set(v2["episodes"][0].keys())
    for ep in v3["episodes"]:
        missing_ep = v2_ep_keys - set(ep.keys())
        # transcript_path may legitimately differ (v3 lives elsewhere); we
        # only require the *key* to exist.
        assert not missing_ep, f"v3 ep {ep['episode_id']} missing v2-shape fields: {missing_ep}"
