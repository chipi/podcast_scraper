"""Unit tests for the pairwise dispatch layer in ``autoresearch_track_a``.

Covers the wire-adjacent glue that routes rubric+transcript+summaries through
OllamaChatJudge/VllmChatJudge ``raw()`` and back into
``parse_pairwise_verdict`` — plus the silver loader that fills the second
side of the pairwise comparison.

Transport-level HTTP is already covered in ``test_ollama_chat_judge`` and
``test_vllm_chat_judge``; here we stub those out and focus on the
dispatch + orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.evaluation.autoresearch_track_a import (
    _call_pairwise_judge,
    _load_silver_summaries,
    _silver_text_from_record,
    judge_one_episode_pairwise,
    mean_pairwise_scores,
)
from podcast_scraper.evaluation.pairwise import PairwiseVerdict

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _silver_text_from_record
# ---------------------------------------------------------------------------


def test_silver_text_extracts_summary_final_from_dict_output() -> None:
    rec = {"episode_id": "ep1", "output": {"summary_final": "the final summary"}}
    assert _silver_text_from_record(rec) == "the final summary"


def test_silver_text_falls_back_to_summary_long_when_final_missing() -> None:
    rec = {"episode_id": "ep1", "output": {"summary_long": "the long summary"}}
    assert _silver_text_from_record(rec) == "the long summary"


def test_silver_text_accepts_plain_string_output() -> None:
    rec = {"episode_id": "ep1", "output": "just a string"}
    assert _silver_text_from_record(rec) == "just a string"


def test_silver_text_returns_empty_for_unrecognized_output_shape() -> None:
    rec = {"episode_id": "ep1", "output": ["a", "list", "not", "supported"]}
    assert _silver_text_from_record(rec) == ""


def test_silver_text_returns_empty_when_output_missing() -> None:
    assert _silver_text_from_record({"episode_id": "ep1"}) == ""


# ---------------------------------------------------------------------------
# _load_silver_summaries
# ---------------------------------------------------------------------------


def test_load_silver_summaries_reads_jsonl_and_indexes_by_episode_id(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver_v2"
    silver_dir.mkdir()
    (silver_dir / "predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "ep1", "output": {"summary_final": "S1"}}),
                json.dumps({"episode_id": "ep2", "output": {"summary_final": "S2"}}),
            ]
        ),
        encoding="utf-8",
    )
    silvers = _load_silver_summaries(silver_dir)
    assert silvers == {"ep1": "S1", "ep2": "S2"}


def test_load_silver_summaries_raises_when_predictions_missing(tmp_path: Path) -> None:
    """The scalar-mode fallback ``no silver → skip judges`` doesn't apply to
    pairwise — pairwise NEEDS the silver by definition. Fail loudly."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="[Pp]airwise scoring requires"):
        _load_silver_summaries(empty_dir)


def test_load_silver_summaries_skips_malformed_lines_gracefully(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver_v2"
    silver_dir.mkdir()
    (silver_dir / "predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "ep1", "output": {"summary_final": "S1"}}),
                "not valid json",
                "",
                json.dumps({"episode_id": "ep3", "output": {"summary_final": "S3"}}),
            ]
        ),
        encoding="utf-8",
    )
    silvers = _load_silver_summaries(silver_dir)
    assert set(silvers.keys()) == {"ep1", "ep3"}


# ---------------------------------------------------------------------------
# _call_pairwise_judge dispatch
# ---------------------------------------------------------------------------


def test_call_pairwise_judge_routes_to_ollama_raw() -> None:
    """Provider=ollama must construct OllamaChatJudge, call ``raw()``, and
    parse the reply via :func:`parse_pairwise_verdict`. This is the
    contract that lets pairwise reuse the OllamaChatJudge retry policy."""
    with patch("podcast_scraper.evaluation.judges.ollama_chat.OllamaChatJudge") as MockCls:
        instance = MockCls.return_value
        instance.raw.return_value = '{"preference": "A", "magnitude": 3, "rationale": "clearer"}'
        verdict = _call_pairwise_judge(
            "ollama",
            model="gemma3:27b-it-q8_0",
            user_content="rubric...",
            candidate_slot="A",
        )
    MockCls.assert_called_once_with(model="gemma3:27b-it-q8_0", api_base=None)
    instance.raw.assert_called_once_with(user_content="rubric...")
    assert verdict.preference == "candidate"
    assert verdict.magnitude == 3


def test_call_pairwise_judge_routes_to_vllm_raw() -> None:
    with patch("podcast_scraper.evaluation.judges.vllm_chat.VllmChatJudge") as MockCls:
        instance = MockCls.return_value
        instance.raw.return_value = '{"preference": "B", "magnitude": 5, "rationale": "decisive"}'
        verdict = _call_pairwise_judge(
            "vllm",
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            user_content="rubric...",
            candidate_slot="B",
        )
    MockCls.assert_called_once_with(model="Qwen/Qwen3-30B-A3B-Instruct-2507", api_base=None)
    assert verdict.preference == "candidate"
    assert verdict.magnitude == 5


def test_call_pairwise_judge_forwards_api_base_override() -> None:
    """Multi-phase sweep passes a per-phase ``api_base`` so judge-a hits
    :8004 and judge-b hits :8005. The dispatcher must plumb it through
    to the transport."""
    with patch("podcast_scraper.evaluation.judges.vllm_chat.VllmChatJudge") as MockCls:
        instance = MockCls.return_value
        instance.raw.return_value = '{"preference": "A", "magnitude": 2, "rationale": "x"}'
        _call_pairwise_judge(
            "vllm",
            model="judge-a",
            user_content="...",
            candidate_slot="A",
            api_base="http://dgx:8004/v1",
        )
    MockCls.assert_called_once_with(model="judge-a", api_base="http://dgx:8004/v1")


def test_call_pairwise_judge_raises_for_openai_provider() -> None:
    """Cloud-API pairwise not implemented — clean NotImplementedError so
    misconfiguration surfaces immediately instead of silently degrading."""
    with pytest.raises(NotImplementedError, match="Currently supported: ollama, vllm"):
        _call_pairwise_judge(
            "openai",
            model="gpt-4o",
            user_content="rubric...",
            candidate_slot="A",
        )


def test_call_pairwise_judge_raises_for_anthropic_provider() -> None:
    with pytest.raises(NotImplementedError, match="Currently supported: ollama, vllm"):
        _call_pairwise_judge(
            "anthropic",
            model="claude-sonnet-4",
            user_content="rubric...",
            candidate_slot="A",
        )


# ---------------------------------------------------------------------------
# judge_one_episode_pairwise
# ---------------------------------------------------------------------------


def test_judge_one_episode_pairwise_calls_both_judges_with_slotted_message() -> None:
    """The SAME slot mapping must be used for both judges — no re-shuffle
    between judge_a and judge_b. Otherwise the two verdicts aren't
    comparing the same A/B layout, and the ``contested`` flag becomes
    meaningless."""
    calls = []

    def fake_call(provider, *, model, user_content, candidate_slot, api_base=None):
        calls.append((provider, model, candidate_slot, user_content))
        return PairwiseVerdict("candidate", 3, "stub")

    with patch(
        "podcast_scraper.evaluation.autoresearch_track_a._call_pairwise_judge",
        side_effect=fake_call,
    ):
        outcome = judge_one_episode_pairwise(
            transcript="the transcript",
            candidate_summary="CANDIDATE",
            silver_summary="SILVER",
            episode_id="ep-1",
            judge_a_provider="ollama",
            judge_a_model="gemma3:27b-it-q8_0",
            judge_b_provider="vllm",
            judge_b_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        )

    assert len(calls) == 2
    # Both judges must see the same slot assignment for this episode.
    assert calls[0][2] == calls[1][2]
    # Both judges must see the same slotted user_content — no
    # per-judge re-slotting.
    assert calls[0][3] == calls[1][3]
    assert outcome.judge_a.preference == "candidate"
    assert outcome.judge_b.preference == "candidate"
    assert outcome.contested is False  # both agree


# ---------------------------------------------------------------------------
# mean_pairwise_scores end-to-end (stubbed judges)
# ---------------------------------------------------------------------------


def _seed_materialized(tmp_path: Path, dataset_id: str, episode_texts: dict) -> Path:
    """Materialize per-episode transcript files so transcripts_by_episode_id resolves."""
    base = tmp_path / "materialized" / dataset_id
    base.mkdir(parents=True)
    for eid, text in episode_texts.items():
        (base / f"{eid}.txt").write_text(text, encoding="utf-8")
    return tmp_path


def _seed_silver(tmp_path: Path, per_episode_summary: dict) -> Path:
    silver_dir = tmp_path / "silver_v2"
    silver_dir.mkdir()
    lines = [
        json.dumps({"episode_id": eid, "output": {"summary_final": text}})
        for eid, text in per_episode_summary.items()
    ]
    (silver_dir / "predictions.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return silver_dir


def test_mean_pairwise_scores_aggregates_across_episodes(tmp_path: Path) -> None:
    """Two episodes, both judges agree candidate wins on each → mean_score
    is well above 0.5, no contests fire."""
    dataset_id = "curated_test_v1"
    eval_root = _seed_materialized(
        tmp_path,
        dataset_id,
        {"ep1": "transcript 1", "ep2": "transcript 2"},
    )
    silver_ref = _seed_silver(tmp_path, {"ep1": "silver 1", "ep2": "silver 2"})

    predictions = [
        {"episode_id": "ep1", "output": {"summary_final": "candidate 1"}},
        {"episode_id": "ep2", "output": {"summary_final": "candidate 2"}},
    ]
    judge_cfg = {
        "judge_a": {"provider": "ollama", "model": "j-a"},
        "judge_b": {"provider": "vllm", "model": "j-b"},
    }

    with patch(
        "podcast_scraper.evaluation.autoresearch_track_a._call_pairwise_judge",
        return_value=PairwiseVerdict("candidate", 4, "clear win"),
    ):
        mean_score, contested, summary = mean_pairwise_scores(
            predictions=predictions,
            judge_cfg=judge_cfg,
            dataset_id=dataset_id,
            eval_root=eval_root,
            silver_reference_path=silver_ref,
        )

    # candidate + magnitude 4 → per-verdict score 0.9; both judges same →
    # per-episode average 0.9; mean across 2 episodes = 0.9.
    assert mean_score == pytest.approx(0.9)
    assert contested is False
    assert summary["episodes"] == 2
    assert summary["contested_count"] == 0
    assert summary["judge_a"]["win_rate"] == pytest.approx(1.0)
    assert summary["judge_b"]["win_rate"] == pytest.approx(1.0)


def test_mean_pairwise_scores_flags_contested_run(tmp_path: Path) -> None:
    """When judges disagree on direction for 3 of 4 episodes (>40%
    CONTEST_FRACTION_THRESHOLD), the run is contested."""
    dataset_id = "curated_test_v1"
    eids = ["ep1", "ep2", "ep3", "ep4"]
    eval_root = _seed_materialized(tmp_path, dataset_id, {e: f"tr {e}" for e in eids})
    silver_ref = _seed_silver(tmp_path, {e: f"silver {e}" for e in eids})

    predictions = [{"episode_id": e, "output": {"summary_final": f"candidate {e}"}} for e in eids]
    judge_cfg = {
        "judge_a": {"provider": "ollama", "model": "j-a"},
        "judge_b": {"provider": "vllm", "model": "j-b"},
    }

    # Judge A always says candidate; judge B says silver on 3 of 4 → 3 contests.
    call_count = {"n": 0}

    def alternating(provider, *, model, user_content, candidate_slot, api_base=None):
        # judge_a call precedes judge_b for each episode.
        call_count["n"] += 1
        idx = call_count["n"] - 1
        is_judge_a = idx % 2 == 0
        episode_idx = idx // 2
        if is_judge_a:
            return PairwiseVerdict("candidate", 3, "")
        # judge_b: agree only on the last episode
        if episode_idx == 3:
            return PairwiseVerdict("candidate", 3, "")
        return PairwiseVerdict("silver", 3, "")

    with patch(
        "podcast_scraper.evaluation.autoresearch_track_a._call_pairwise_judge",
        side_effect=alternating,
    ):
        _, contested, summary = mean_pairwise_scores(
            predictions=predictions,
            judge_cfg=judge_cfg,
            dataset_id=dataset_id,
            eval_root=eval_root,
            silver_reference_path=silver_ref,
        )

    assert summary["contested_count"] == 3
    assert contested is True  # 3/4 = 75% > 40% threshold


def test_mean_pairwise_scores_skips_empty_candidate_summaries(tmp_path: Path) -> None:
    """An episode whose candidate summary is empty is skipped (not judged)
    — same policy as ``mean_judge_scores``. Prevents blank-summary
    candidates from silently voting themselves ``tie``."""
    dataset_id = "curated_test_v1"
    eval_root = _seed_materialized(tmp_path, dataset_id, {"ep1": "tr 1", "ep2": "tr 2"})
    silver_ref = _seed_silver(tmp_path, {"ep1": "silver 1", "ep2": "silver 2"})

    predictions = [
        {"episode_id": "ep1", "output": {"summary_final": ""}},  # empty, skipped
        {"episode_id": "ep2", "output": {"summary_final": "candidate 2"}},
    ]
    judge_cfg = {
        "judge_a": {"provider": "ollama", "model": "j-a"},
        "judge_b": {"provider": "vllm", "model": "j-b"},
    }

    with patch(
        "podcast_scraper.evaluation.autoresearch_track_a._call_pairwise_judge",
        return_value=PairwiseVerdict("candidate", 2, ""),
    ):
        _, _, summary = mean_pairwise_scores(
            predictions=predictions,
            judge_cfg=judge_cfg,
            dataset_id=dataset_id,
            eval_root=eval_root,
            silver_reference_path=silver_ref,
        )

    # Only ep2 got judged → 1 episode in the summary.
    assert summary["episodes"] == 1
