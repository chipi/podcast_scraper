"""Unit tests for RFC-057 Track A autoresearch helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from podcast_scraper.evaluation.autoresearch_track_a import (
    AutoresearchConfigError,
    combine_track_a_scalar,
    eval_n_from_env,
    extract_mean_rouge_l_f1,
    load_local_dotenv_files,
    merge_max_episodes_into_config_yaml,
    parse_judge_score_json,
    resolve_experiment_openai_key,
)


@pytest.mark.unit
class TestParseJudgeScoreJson:
    def test_plain_json(self) -> None:
        assert parse_judge_score_json('{"score": 0.75, "notes": "ok"}') == 0.75

    def test_fenced_json(self) -> None:
        text = '```json\n{"score": 1.0}\n```'
        assert parse_judge_score_json(text) == 1.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_score_json('{"score": 1.5}')


@pytest.mark.unit
class TestExtractMeanRougeLF1:
    def test_first_reference_blob(self) -> None:
        m = {
            "vs_reference": {
                "silver": {"rougeL_f1": 0.42, "rouge1_f1": 0.5},
            }
        }
        assert extract_mean_rouge_l_f1(m) == pytest.approx(0.42)

    def test_none_when_missing(self) -> None:
        assert extract_mean_rouge_l_f1({"vs_reference": None}) is None


@pytest.mark.unit
class TestCombineTrackAScalar:
    def test_contested_uses_rouge_only(self) -> None:
        out = combine_track_a_scalar(
            rouge_l_f1=0.5,
            judge_mean=0.9,
            contested=True,
            rouge_weight=0.4,
        )
        assert out == pytest.approx(0.5)

    def test_blend_when_ok(self) -> None:
        out = combine_track_a_scalar(
            rouge_l_f1=0.5,
            judge_mean=0.9,
            contested=False,
            rouge_weight=0.4,
        )
        assert out == pytest.approx(0.4 * 0.5 + 0.6 * 0.9)


@pytest.mark.unit
class TestMergeMaxEpisodes:
    def test_writes_max_episodes(self, tmp_path: Path) -> None:
        src = tmp_path / "in.yaml"
        src.write_text(
            yaml.dump(
                {
                    "id": "x",
                    "task": "summarization",
                    "data": {"dataset_id": "d1"},
                    "backend": {"type": "openai", "model": "gpt-4o-mini"},
                    "prompts": {"user": "openai/summarization/long_v1"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        dest = tmp_path / "out.yaml"
        merge_max_episodes_into_config_yaml(src, dest, 3)
        loaded = yaml.safe_load(dest.read_text(encoding="utf-8"))
        assert loaded["data"]["max_episodes"] == 3


@pytest.mark.unit
class TestEvalNFromEnv:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTORESEARCH_EVAL_N", raising=False)
        assert eval_n_from_env(7) == 7

    def test_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTORESEARCH_EVAL_N", "nope")
        with pytest.raises(AutoresearchConfigError):
            eval_n_from_env()


@pytest.mark.unit
class TestResolveExperimentOpenaiKey:
    def test_prefers_dedicated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", "sk-dedicated")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", raising=False)
        assert resolve_experiment_openai_key() == "sk-dedicated"

    def test_requires_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", raising=False)
        with pytest.raises(AutoresearchConfigError):
            resolve_experiment_openai_key()


@pytest.mark.unit
def test_load_local_dotenv_files_no_crash_under_pytest(tmp_path: Path) -> None:
    """Under pytest, dotenv loading is skipped (test env guard)."""
    load_local_dotenv_files(tmp_path)


@pytest.mark.unit
def test_metrics_json_fixture_roundtrip() -> None:
    """Sanity: metrics shape used by score.py parsing."""
    blob = {
        "vs_reference": {
            "silver_gpt4o_smoke_v1": {
                "rougeL_f1": 0.33,
            },
        },
    }
    s = json.dumps(blob)
    m = json.loads(s)
    assert extract_mean_rouge_l_f1(m) == pytest.approx(0.33)
