"""The summary use case's helpers (RFC-127): each is a no-op for the episode summary, so a
provider's ``summarize`` behaves exactly as before when no use case is passed."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.common.summary_use_case import (
    build_use_case_document,
    build_use_case_prompts,
    prompt_name,
    reply_budget,
    SummaryUseCase,
    use_case_from_params,
    use_case_input_kind,
    use_case_prompt_metadata,
    use_case_stage,
)

pytestmark = pytest.mark.unit

MATERIAL = {
    "episodes": [
        {"title": "T1", "show": "S", "summary": "Sum.", "key_points": ["k"], "insights": []},
        {"title": "T2", "show": "S", "summary": "Sum.", "key_points": [], "insights": []},
    ],
    "moments": [{"text": "A saved line.", "speaker": "Ada", "episode": "T1"}],
}


def _uc(max_length: int = 400) -> SummaryUseCase:
    return SummaryUseCase(
        system_prompt="summarization/board_system_v1",
        user_prompt="summarization/board_user_v1",
        stage="board_summary",
        max_length=max_length,
        params=MATERIAL,
    )


def test_the_episode_summary_has_no_use_case() -> None:
    assert use_case_from_params(None) is None
    assert use_case_from_params({"max_length": 10}) is None
    assert use_case_from_params({"use_case": {"stage": "x"}}) is None  # only the real type counts
    uc = _uc()
    assert use_case_from_params({"use_case": uc}) is uc


def test_the_budget_is_the_episodes_unless_a_use_case_brings_its_own() -> None:
    assert reply_budget(None, 800) == 800
    assert reply_budget(None, 800, lambda n: n * 10) == 800  # no clamp on the episode path here
    assert reply_budget(_uc(400), 800) == 400
    assert reply_budget(_uc(400), 800, lambda n: n + 1) == 401


def test_the_stage_defaults_for_the_episode_summary() -> None:
    assert use_case_stage(None) is None
    assert use_case_stage(_uc()) == "board_summary"


def test_names_resolve_under_the_provider_and_fall_back_to_shared() -> None:
    assert prompt_name("anthropic", "summarization/board_user_v1") == (
        "anthropic/summarization/board_user_v1"
    )
    system, user, sn, un = build_use_case_prompts("mistral", _uc())
    assert (sn, un) == (
        "mistral/summarization/board_system_v1",
        "mistral/summarization/board_user_v1",
    )
    assert "listener's board" in system
    assert "T1" in user and "T2" in user and "A saved line." in user and "Ada" in user


def test_metadata_names_the_templates_and_counts_the_material_without_copying_it() -> None:
    assert use_case_prompt_metadata("openai", None) is None
    meta = use_case_prompt_metadata("openai", _uc())
    assert meta is not None
    assert meta["system"]["file"] == "shared/summarization/board_system_v1.j2"
    assert meta["user"]["params"] == {"episodes": 2, "moments": 1}
    assert len(meta["system"]["sha256"]) == 64
    assert "A saved line." not in str(meta)


def test_a_seq2seq_model_gets_the_material_as_prose_without_instructions() -> None:
    from dataclasses import replace

    uc = replace(_uc(), document_prompt="summarization/board_document_v1")
    doc, name = build_use_case_document("transformers", uc)
    assert name == "transformers/summarization/board_document_v1"
    assert "T1, from S." in doc and "Ada: A saved line." in doc
    assert "listener's board" not in doc and "<episode" not in doc
    with pytest.raises(ValueError, match="document_prompt"):
        build_use_case_document("transformers", _uc())


def test_the_input_kind_follows_what_the_provider_can_render() -> None:
    class Chat:
        def _build_summarization_prompts(self) -> None: ...

    class Seq2Seq:
        def _build_use_case_document(self) -> None: ...

    assert use_case_input_kind(Chat()) == "chat"
    assert use_case_input_kind(Seq2Seq()) == "document"
    assert use_case_input_kind(object()) is None


def test_document_metadata_names_only_the_document_template() -> None:
    from dataclasses import replace

    uc = replace(_uc(), document_prompt="summarization/board_document_v1")
    meta = use_case_prompt_metadata("transformers", uc, kind="document")
    assert meta is not None and set(meta) == {"document"}
    assert meta["document"]["file"] == "shared/summarization/board_document_v1.j2"
    assert meta["document"]["params"] == {"episodes": 2, "moments": 1}
