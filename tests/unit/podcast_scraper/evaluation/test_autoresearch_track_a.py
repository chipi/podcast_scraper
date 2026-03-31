"""Unit tests for RFC-057 Track A autoresearch helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from podcast_scraper.evaluation.autoresearch_track_a import (
    AutoresearchConfigError,
    combine_track_a_scalar,
    eval_n_from_env,
    extract_mean_rouge_l_f1,
    judge_one_episode,
    load_judge_config,
    load_local_dotenv_files,
    mean_judge_scores,
    merge_max_episodes_into_config_yaml,
    parse_judge_score_json,
    resolve_experiment_openai_key,
    resolve_judge_anthropic_key,
    resolve_judge_openai_key,
    rouge_weight_from_env,
    summary_text_from_prediction,
    transcripts_by_episode_id,
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


@pytest.mark.unit
def test_extract_mean_rouge_l_f1_skips_error_and_non_dict() -> None:
    m = {
        "vs_reference": {
            "bad": {"error": "x"},
            "nd": "not-a-dict",
            "ok": {"rougeL_f1": 0.2},
        }
    }
    assert extract_mean_rouge_l_f1(m) == pytest.approx(0.2)


@pytest.mark.unit
def test_load_judge_config_requires_mapping(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_judge_config(p)


@pytest.mark.unit
class TestResolveJudgeKeys:
    def test_openai_judge_dedicated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTORESEARCH_JUDGE_OPENAI_API_KEY", "sk-judge")
        monkeypatch.delenv("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", raising=False)
        assert resolve_judge_openai_key() == "sk-judge"

    def test_openai_judge_prod_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTORESEARCH_JUDGE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-prod")
        assert resolve_judge_openai_key() == "sk-prod"

    def test_anthropic_judge_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", raising=False)
        with pytest.raises(AutoresearchConfigError):
            resolve_judge_anthropic_key()


@pytest.mark.unit
class TestRougeWeightFromEnv:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTORESEARCH_SCORE_ROUGE_WEIGHT", raising=False)
        assert rouge_weight_from_env() == pytest.approx(0.4)

    def test_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTORESEARCH_SCORE_ROUGE_WEIGHT", "0.25")
        assert rouge_weight_from_env() == pytest.approx(0.25)

    def test_invalid_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTORESEARCH_SCORE_ROUGE_WEIGHT", "x")
        with pytest.raises(AutoresearchConfigError):
            rouge_weight_from_env()

    def test_out_of_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTORESEARCH_SCORE_ROUGE_WEIGHT", "2")
        with pytest.raises(AutoresearchConfigError):
            rouge_weight_from_env()


@pytest.mark.unit
def test_eval_n_from_env_too_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_EVAL_N", "0")
    with pytest.raises(AutoresearchConfigError):
        eval_n_from_env()


@pytest.mark.unit
def test_summary_text_from_prediction_variants() -> None:
    assert summary_text_from_prediction({"output": "plain"}) == "plain"
    assert summary_text_from_prediction({"output": {"summary_final": "sf"}}) == "sf"
    assert summary_text_from_prediction({"output": {"summary_long": "sl"}}) == "sl"
    assert summary_text_from_prediction({"output": {}}) == ""


@pytest.mark.unit
def test_transcripts_by_episode_id_reads_files(tmp_path: Path) -> None:
    base = tmp_path / "materialized" / "ds1"
    base.mkdir(parents=True)
    (base / "e1.txt").write_text("hello", encoding="utf-8")
    out = transcripts_by_episode_id(dataset_id="ds1", episode_ids=["e1"], eval_root=tmp_path)
    assert out["e1"] == "hello"


@pytest.mark.unit
def test_transcripts_by_episode_id_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcripts_by_episode_id(dataset_id="ds1", episode_ids=["missing"], eval_root=tmp_path)


@pytest.mark.unit
@patch("podcast_scraper.evaluation.autoresearch_track_a.call_anthropic_judge")
@patch("podcast_scraper.evaluation.autoresearch_track_a.call_openai_judge")
def test_judge_one_episode_contested(mock_openai: object, mock_anthropic: object) -> None:
    mock_openai.return_value = 1.0
    mock_anthropic.return_value = 0.0
    out = judge_one_episode(
        rubric="r",
        transcript="t",
        summary="s",
        judge_a_provider="openai",
        judge_a_model="m1",
        judge_b_provider="anthropic",
        judge_b_model="m2",
        openai_key="k1",
        anthropic_key="k2",
    )
    assert out.contested is True
    assert out.judge_a == 1.0
    assert out.judge_b == 0.0


@pytest.mark.unit
def test_merge_max_episodes_invalid_yaml(tmp_path: Path) -> None:
    src = tmp_path / "bad.yaml"
    src.write_text("[]", encoding="utf-8")
    dest = tmp_path / "out.yaml"
    with pytest.raises(ValueError, match="Invalid"):
        merge_max_episodes_into_config_yaml(src, dest, 2)


@pytest.mark.unit
@patch("podcast_scraper.evaluation.autoresearch_track_a.judge_one_episode")
def test_mean_judge_scores_skips_empty_summary(mock_judge: object, tmp_path: Path) -> None:
    base = tmp_path / "materialized" / "d1"
    base.mkdir(parents=True)
    (base / "e1.txt").write_text("tr", encoding="utf-8")
    preds = [{"episode_id": "e1", "output": ""}]
    cfg = {
        "judge_a": {"provider": "openai", "model": "m"},
        "judge_b": {"provider": "anthropic", "model": "m"},
        "rubric": "x",
    }
    with pytest.raises(RuntimeError, match="No episodes"):
        mean_judge_scores(
            predictions=preds,
            rubric="r",
            judge_cfg=cfg,
            dataset_id="d1",
            eval_root=tmp_path,
            openai_key="a",
            anthropic_key="b",
        )
    mock_judge.assert_not_called()
