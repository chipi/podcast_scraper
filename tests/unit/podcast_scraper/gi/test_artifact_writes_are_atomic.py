"""A failed artifact write must not destroy the artifact it was replacing.

THE DEFECT
``gi.io.write_artifact`` (and its identical twin ``kg.io.write_artifact``) opened the destination
with ``open(path, "w")`` and serialized straight into it. ``"w"`` truncates on open, so the
previous contents are gone the instant the write BEGINS. Anything that stops the write partway —
a kill, a full disk, a serialization error — leaves a truncated, unparseable file at the real
artifact path.

WHY IT IS WORSE THAN A NORMAL LOST WRITE
``gi.repair`` reads the existing artifact before rewriting it, and REFUSES to proceed on one it
cannot parse (that refusal is deliberate: it is what stops the repair from overwriting an artifact
it does not understand). So a kill during a repair leaves an episode that the repair tool will
decline to touch from then on — permanently unrepairable by the only tool that repairs it, and
broken by that same tool. The failure is self-locking.

THE REPRO here is a serialization error rather than a kill: ``allow_nan=False`` with a NaN buried
in the payload makes ``json.dump`` raise partway through, after it has already written the opening
chunks. That is the same shape as a kill (partial bytes, exception on the way out) and it is
deterministic, which a kill is not.

THE FIX is the idiom already hand-rolled in six places in this repo (monitor/status.py,
utils/audio_cache.py, utils/storage_backend.py, three upgrade migrations): serialize to a temp
file in the SAME directory, fsync, then ``os.replace``. Same directory because a rename is only
atomic within one filesystem. Promoted to ``utils.atomic_io`` so this is the seventh USE, not the
seventh COPY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.gi import io as gi_io
from podcast_scraper.kg import io as kg_io

pytestmark = [pytest.mark.unit]


def _good_gi_payload() -> Dict[str, Any]:
    return {
        "schema_version": "3.0",
        "model_version": "test-model",
        "prompt_version": "v1",
        "episode_id": "ep-atomic-001",
        "nodes": [
            {
                "id": "insight-1",
                "type": "Insight",
                "properties": {"text": "A real insight that must survive a failed rewrite."},
            }
        ],
        "edges": [],
    }


def _good_kg_payload() -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "episode_id": "ep-atomic-001",
        "extraction": {"model": "test-model"},
        "nodes": [{"id": "entity-1", "type": "Entity", "properties": {"name": "Ada"}}],
        "edges": [],
    }


def _poisoned(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Same payload with a NaN deep inside — serializes partway, then raises."""
    poisoned: Dict[str, Any] = json.loads(json.dumps(payload))
    poisoned["nodes"][0]["properties"]["score"] = float("nan")
    return poisoned


@pytest.mark.parametrize(
    "module,good_payload",
    [(gi_io, _good_gi_payload()), (kg_io, _good_kg_payload())],
    ids=["gi", "kg"],
)
def test_failed_write_leaves_the_previous_artifact_intact(tmp_path, module, good_payload):
    """THE contract: a write that fails must leave the OLD artifact readable.

    Before the fix this failed with a truncated file: ``read_artifact`` raised
    ``json.JSONDecodeError`` and the episode became permanently unrepairable.
    """
    path = tmp_path / "episode.artifact.json"
    module.write_artifact(path, good_payload, validate=False)
    original_bytes = path.read_bytes()

    with pytest.raises(ValueError):
        module.write_artifact(path, _poisoned(good_payload), validate=False)

    assert path.exists(), "the failed write deleted the artifact outright"
    assert path.read_bytes() == original_bytes, (
        "the failed write corrupted the previous artifact; gi.repair will now refuse this "
        "episode forever"
    )
    # The point of the byte comparison: it must still PARSE, which is what repair checks.
    assert module.read_artifact(path, validate=False)["episode_id"] == "ep-atomic-001"


@pytest.mark.parametrize(
    "module,good_payload",
    [(gi_io, _good_gi_payload()), (kg_io, _good_kg_payload())],
    ids=["gi", "kg"],
)
def test_failed_write_leaves_no_temp_file_behind(tmp_path, module, good_payload):
    """A repair run over hundreds of episodes must not litter the corpus with .tmp files.

    They would also be picked up by ``rglob`` scans in the integrity gate if named carelessly.
    """
    path = tmp_path / "episode.artifact.json"
    module.write_artifact(path, good_payload, validate=False)

    with pytest.raises(ValueError):
        module.write_artifact(path, _poisoned(good_payload), validate=False)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"


@pytest.mark.parametrize(
    "module,good_payload",
    [(gi_io, _good_gi_payload()), (kg_io, _good_kg_payload())],
    ids=["gi", "kg"],
)
def test_successful_write_still_round_trips(tmp_path, module, good_payload):
    """Regression guard: the atomic path must not change what a successful write produces."""
    path = tmp_path / "nested" / "episode.artifact.json"
    module.write_artifact(path, good_payload, validate=False)

    assert path.is_file()
    assert module.read_artifact(path, validate=False) == good_payload
    # Non-ASCII must survive: ensure_ascii=False was in the original writer and is load-bearing
    # for transcripts in any language other than English.
    good_payload["nodes"][0]["properties"]["text"] = "Ünïcödé — em-dash and ellipsis…"
    module.write_artifact(path, good_payload, validate=False)
    assert "Ünïcödé" in path.read_text(encoding="utf-8")


def test_write_creates_parent_directories(tmp_path):
    """``mkdir(parents=True)`` was in the original writer; the temp file needs it to exist too."""
    path = tmp_path / "a" / "b" / "c" / "episode.gi.json"
    gi_io.write_artifact(path, _good_gi_payload(), validate=False)
    assert path.is_file()


def test_shared_helper_is_the_one_used(tmp_path):
    """Both writers must go through ``utils.atomic_io``, not a seventh hand-rolled copy.

    Asserted by behaviour, not by import: the helper is what makes a failed write non-destructive,
    so if a writer stops using it the tests above fail. This case only pins the helper's own
    contract for the callers that use it directly.
    """
    from podcast_scraper.utils.atomic_io import write_json_atomic

    path = tmp_path / "thing.json"
    write_json_atomic(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    with pytest.raises(ValueError):
        write_json_atomic(path, {"a": float("nan")}, allow_nan=False)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["thing.json"]


def test_replacing_a_file_that_does_not_exist_yet(tmp_path):
    """First write of a new artifact — there is nothing to preserve, it must simply work."""
    path = Path(tmp_path) / "brand_new.gi.json"
    assert not path.exists()
    gi_io.write_artifact(path, _good_gi_payload(), validate=False)
    assert gi_io.read_artifact(path, validate=False)["episode_id"] == "ep-atomic-001"
