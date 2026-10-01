"""The #2200 audit puts each unbound stated name in the bucket that names its cause."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "audit" / "unbound_name_causes.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("unbound_name_causes", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _module()


def test_a_name_retrieval_finds_is_bucket_a() -> None:
    turns = [("S0", "Today we have Corin Berntsen with us."), ("S1", "Thanks for having me.")]
    assert M.classify_name("Corin Berntsen", turns, turns[0][1]) == M.BUCKET_A


def test_first_name_only_is_bucket_c() -> None:
    turns = [("S0", "Corin, welcome to the show."), ("S1", "Thanks.")]
    assert M.classify_name("Corin Berntsen", turns, turns[0][1]) == M.BUCKET_C


def test_absent_from_the_opening_is_bucket_d() -> None:
    turns = [("S0", "Welcome to the show."), ("S1", "Thanks.")]
    assert M.classify_name("Corin Berntsen", turns, "Welcome to the show.") == M.BUCKET_D


def test_one_word_name_is_single() -> None:
    assert M.classify_name("Corin", [("S0", "Corin is here.")], "Corin is here.") == M.SINGLE


def test_run_reads_only_served_episodes_with_hidden_insights(tmp_path: Path) -> None:
    run = tmp_path / "feeds" / "f1" / "run_a"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    stem = "ep1"
    (run / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": "f1"}, "episode": {"episode_id": "e1"}})
    )
    (run / "metadata" / f"{stem}.gi.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"type": "Insight", "properties": {"speaker_voice_type": "unknown"}},
                    {"type": "Insight", "properties": {"speaker_voice_type": "host"}},
                ]
            }
        )
    )
    (run / "transcripts" / f"{stem}.segments.json").write_text(
        json.dumps([{"start": 0.0, "speaker": "S0", "text": "Corin, welcome."}])
    )
    (run / "transcripts" / f"{stem}.speakers.diagnostics.json").write_text(
        json.dumps({"summary": {"unbound_names": ["Corin Berntsen"]}})
    )
    result = M.run(tmp_path)
    assert result == {
        M.BUCKET_C: {"episodes": 1, "hidden_insights": 1, "example": "Corin Berntsen"}
    }
