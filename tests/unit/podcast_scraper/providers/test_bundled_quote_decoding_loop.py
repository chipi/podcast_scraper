"""The bundled-quote call against a decoding loop: prevent it, and survive it without bisecting.

Measured on prod 2026-10-02 (Latent Space, prod_dgx_full): 13 of 13 captured failures were the
model repeating speech filler inside one quote ("like, you know, like, you know, ...") until the
token budget ran out, with presence_penalty=1.5 already on. The bisect then retried 8 -> 4 -> 2
insights over the same transcript and every half looped again, each spending a full budget.

Two layers, each tested on its own:
  * PREVENT — vLLM gets a JSON schema whose bounds make the reply close within the budget.
  * SURVIVE — a loop that still happens keeps the quotes that closed and is not bisected.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.common.bundle_extract_parser import BundleOutputBudgetExceeded
from podcast_scraper.providers.common.bundled_prompts import (
    extract_quotes_bundled_json_schema,
    extract_quotes_bundled_max_tokens,
)
from podcast_scraper.providers.openai.openai_provider import (
    _is_decoding_loop,
    _salvage_closed_quotes,
    OpenAICompatibleProvider,
)
from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

pytestmark = pytest.mark.unit

_MODEL = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"
TRANSCRIPT = (
    "Host: The ports moved north because the river silted up over a decade. "
    "Guest: The merchants followed the trade within a generation, most of them anyway. "
    "Host: And the crown taxed whatever was left behind and called it reform."
)
LOOP = "like, you know, " * 400


def _vllm() -> VLLMProvider:
    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="vllm",
        speaker_detector_provider="vllm",
        generate_summaries=True,
        generate_metadata=True,
        vllm_api_base="http://dgx:8003/v1",
        vllm_summary_model=_MODEL,
        vllm_speaker_model=_MODEL,
    )
    p = VLLMProvider(Config(**base))
    p._summarization_initialized = True
    return p


def _reply(content: str, finish_reason: str, completion_tokens: int) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=completion_tokens),
    )


def _drive(p: VLLMProvider, content: str, finish_reason: str, insights: List[str]) -> Any:
    sent: List[Dict[str, Any]] = []

    def fake_chat_create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return _reply(content, finish_reason, extract_quotes_bundled_max_tokens(len(insights)))

    p._chat_create = fake_chat_create  # type: ignore[method-assign]
    out = p.extract_quotes_bundled(transcript=TRANSCRIPT, insight_texts=insights)
    return out, sent


# --- PREVENT: the schema --------------------------------------------------------------------


def test_the_schema_fits_its_worst_case_inside_the_budget() -> None:
    # Every insight writing 5 quotes at the full cap must still be under max_out tokens.
    for n in (1, 2, 4, 8):
        max_out = extract_quotes_bundled_max_tokens(n)
        schema = extract_quotes_bundled_json_schema(n, max_out)
        item = schema["properties"]["0"]
        worst_chars = n * item["maxItems"] * item["items"]["maxLength"]
        assert worst_chars / 4 < max_out, (n, worst_chars, max_out)


def test_the_schema_names_exactly_the_insights_sent() -> None:
    schema = extract_quotes_bundled_json_schema(3, 2048)
    assert schema["required"] == ["0", "1", "2"]
    assert set(schema["properties"]) == {"0", "1", "2"}
    assert schema["additionalProperties"] is False


def test_the_quote_cap_keeps_a_floor_for_very_large_batches() -> None:
    schema = extract_quotes_bundled_json_schema(1000, 5120)
    assert schema["properties"]["0"]["items"]["maxLength"] == 200


def test_vllm_sends_the_bounded_schema() -> None:
    _out, sent = _drive(_vllm(), json.dumps({"0": ["the river silted up"]}), "stop", ["Rivers"])
    fmt = sent[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["required"] == ["0"]


def test_other_openai_compatible_providers_keep_json_mode() -> None:
    # The schema bounds are verified on our vLLM only; nothing else changes.
    fmt = OpenAICompatibleProvider._bundled_quote_response_format(None, 3, 2048)  # type: ignore
    assert fmt == {"type": "json_object"}


# --- SURVIVE: the loop that still happens ----------------------------------------------------


def test_a_loop_is_recognised() -> None:
    assert _is_decoding_loop('{"0": ["' + LOOP)


def test_a_healthy_reply_is_not_a_loop() -> None:
    healthy = json.dumps(
        {str(i): [f"quote number {i} about the river and the trade"] for i in range(8)}
    )
    assert not _is_decoding_loop(healthy)


def test_salvage_keeps_closed_quotes_and_drops_the_one_that_never_closed() -> None:
    cut = '{\n  "0": ["the river silted up", "the merchants followed"],\n  "1": ["' + LOOP
    assert _salvage_closed_quotes(cut, 2) == {0: ["the river silted up", "the merchants followed"]}


def test_salvage_ignores_an_index_outside_the_batch() -> None:
    cut = '{"5": ["out of range"], "0": ["the river silted up"], "1": ["' + LOOP
    assert _salvage_closed_quotes(cut, 2) == {0: ["the river silted up"]}


def test_salvage_decodes_escaped_quotes() -> None:
    cut = '{"0": ["he said \\"reform\\" twice"], "1": ["' + LOOP
    assert _salvage_closed_quotes(cut, 2) == {0: ['he said "reform" twice']}


def test_a_looping_reply_returns_its_closed_quotes_instead_of_raising() -> None:
    cut = '{\n  "0": ["The ports moved north because the river silted up"],\n  "1": ["' + LOOP
    out, sent = _drive(_vllm(), cut, "length", ["Rivers", "Merchants"])
    assert len(sent) == 1
    assert [c.text for c in out[0]] == ["The ports moved north because the river silted up"]
    assert out[1] == []


def test_a_loop_whose_closed_strings_are_not_in_the_transcript_keeps_nothing() -> None:
    cut = '{"0": ["a sentence nobody said"], "1": ["' + LOOP
    out, _sent = _drive(_vllm(), cut, "length", ["Rivers", "Merchants"])
    assert out == {0: [], 1: []}


def test_a_cutoff_that_is_not_a_loop_still_asks_for_a_smaller_batch() -> None:
    # Genuine over-generation is what the bisect exists for; it must keep working.
    cut = '{"0": ["' + " ".join(f"word{i}" for i in range(3000))
    with pytest.raises(BundleOutputBudgetExceeded):
        _drive(_vllm(), cut, "length", ["Rivers", "Merchants"])
