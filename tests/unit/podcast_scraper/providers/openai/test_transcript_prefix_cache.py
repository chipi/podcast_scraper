"""RFC-111 Phase 1: transcript-prefix caching for the summary stage.

Every provider we use (except Gemini's OpenAI-compat endpoint) caches an identical LEADING token
prefix at ~0.1x price, but the legacy layout put each stage's *system* prompt first, so the
transcript — sent to ~5 stages per episode — never cached (probed: 0%). RFC-111 moves the
transcript to the byte-stable leading block of the system prompt so it caches across the stages of
an episode (and across reprocessing runs). Caching reuses the model's compute (KV state), never the
answer, so this changes cost/latency, not correctness.

These tests are the deterministic safety net (RFC-111 §6 tests 1-6, 10, 11): the layout builder,
the flag-off legacy fallback, content invariance (the model must see exactly what it saw before,
only reordered), auto-cache provider shaping, the end-to-end summarize wiring, cache-token
telemetry plumbing, and the cross-provider coverage guard. The paid live gate (§6 tests 7-9:
cache-hit assertion, blind quality-parity A/B, real cost drop) runs separately on episodes.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.common.transcript_cache import (
    cacheable_transcript_prefix as _cacheable_transcript_prefix,
    TRANSCRIPT_BLOCK_HEADER as _TRANSCRIPT_BLOCK_HEADER,
    TRANSCRIPT_MOVED_MARKER as _TRANSCRIPT_MOVED_MARKER,
)
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)

pytestmark = pytest.mark.unit


# A transcript long/unique enough to be an unambiguous single occurrence in the user prompt.
# Stripped (no surrounding whitespace) so the normalised block equals this exact string.
TRANSCRIPT = ("HOST: welcome to the show. GUEST: reliable software for small teams. " * 20).strip()
SYSTEM_INSTRUCTIONS = "You are an expert at summarizing podcast episodes."
# Mirrors the real user template (long_v2.j2): task text, then the transcript, then guidelines.
USER_PROMPT = (
    "Summarize the following podcast episode transcript.\n\n"
    "Transcript:\n"
    f"{TRANSCRIPT}\n\n"
    "Guidelines:\n- Write 4-6 paragraphs\n- Ignore ads"
)


def _provider(*, cache: bool = True, model: str = "gpt-4o-mini") -> OpenAIProvider:
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="openai",
        openai_summary_model=model,
        openai_api_key="sk-test-api-key-123",
        cache_transcript_prefix=cache,
    )
    return OpenAIProvider(cfg)


# ---------------------------------------------------------------------------------------------
# §6.1 — [unit] Layout builder
# ---------------------------------------------------------------------------------------------


def test_layout_builder_transcript_leads_system_and_task_stays_in_user() -> None:
    """system = transcript block + stage instructions; user = task with the transcript relocated."""
    msgs = _provider()._build_stage_messages(
        transcript=TRANSCRIPT, system_prompt=SYSTEM_INSTRUCTIONS, user_prompt=USER_PROMPT
    )
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]

    # Transcript is the LEADING block of the system prompt, stage instructions after it.
    assert system.startswith(_TRANSCRIPT_BLOCK_HEADER)
    assert TRANSCRIPT in system
    assert system.endswith(SYSTEM_INSTRUCTIONS)
    assert system.index(TRANSCRIPT) < system.index(SYSTEM_INSTRUCTIONS)

    # Transcript is NOT duplicated in the user message — a marker points at the system block.
    assert TRANSCRIPT not in user
    assert _TRANSCRIPT_MOVED_MARKER in user


def test_transcript_block_prefix_is_byte_stable_across_stages() -> None:
    """The cache only hits if the leading transcript block is byte-identical every stage; a
    per-stage difference before the transcript would shift the prefix and kill the cache."""
    p = _provider()
    summary = p._build_stage_messages(
        transcript=TRANSCRIPT, system_prompt="You are a SUMMARIZER.", user_prompt=USER_PROMPT
    )
    quotes = p._build_stage_messages(
        transcript=TRANSCRIPT,
        system_prompt="You are a QUOTE EXTRACTOR.",
        user_prompt=USER_PROMPT.replace("Summarize", "Extract quotes from"),
    )
    prefix = _cacheable_transcript_prefix(TRANSCRIPT)
    assert summary[0]["content"].startswith(prefix)
    assert quotes[0]["content"].startswith(prefix)
    # The shared prefix is identical to the byte; divergence begins only in the stage instructions.
    assert summary[0]["content"][: len(prefix)] == quotes[0]["content"][: len(prefix)]


# ---------------------------------------------------------------------------------------------
# §6.2 — [unit] Flag off = legacy
# ---------------------------------------------------------------------------------------------


def test_flag_off_emits_exact_legacy_layout() -> None:
    """cache_transcript_prefix=false must reproduce today's exact layout (transcript in user)."""
    msgs = _provider(cache=False)._build_stage_messages(
        transcript=TRANSCRIPT, system_prompt=SYSTEM_INSTRUCTIONS, user_prompt=USER_PROMPT
    )
    assert msgs == [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": USER_PROMPT},
    ]


def test_transcript_absent_from_user_falls_back_to_legacy() -> None:
    """If the transcript is not present verbatim (custom/truncated prompt) the builder must not
    corrupt the messages — it returns the legacy layout untouched."""
    p = _provider()
    msgs = p._build_stage_messages(
        transcript="a transcript that does not appear",
        system_prompt=SYSTEM_INSTRUCTIONS,
        user_prompt=USER_PROMPT,
    )
    assert msgs == [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": USER_PROMPT},
    ]


# ---------------------------------------------------------------------------------------------
# §6.3 — [unit] Content invariance (the core quality safety net)
# ---------------------------------------------------------------------------------------------


def test_content_invariance_only_reordered_nothing_dropped() -> None:
    """The information the model sees (transcript + instructions + task) is identical to legacy,
    only reordered. Assert every non-transcript span of the legacy prompt survives, and the
    transcript itself is present exactly once across the two messages."""
    p = _provider()
    cached = p._build_stage_messages(
        transcript=TRANSCRIPT, system_prompt=SYSTEM_INSTRUCTIONS, user_prompt=USER_PROMPT
    )
    combined = cached[0]["content"] + "\n" + cached[1]["content"]

    # The stage instructions and every task fragment (transcript excised) still reach the model.
    assert SYSTEM_INSTRUCTIONS in combined
    for fragment in USER_PROMPT.split(TRANSCRIPT):
        assert fragment.strip() in combined

    # The transcript is present exactly once — relocated, never duplicated, never dropped.
    assert combined.count(TRANSCRIPT) == 1


# ---------------------------------------------------------------------------------------------
# §6.4 — [unit] Provider shaping (auto-cache providers attach nothing extra)
# ---------------------------------------------------------------------------------------------


def test_autocache_provider_attaches_no_extra_api_fields() -> None:
    """For auto-prefix-cache providers (openai/deepseek/qwen/litellm/vllm) the LAYOUT alone enables
    caching — no cache_control / cached_content fields. (Anthropic + Gemini get provider-specific
    shaping in RFC-111 phases 3-4, in their own providers.)"""
    msgs = _provider()._build_stage_messages(
        transcript=TRANSCRIPT, system_prompt=SYSTEM_INSTRUCTIONS, user_prompt=USER_PROMPT
    )
    for m in msgs:
        assert set(m.keys()) == {"role", "content"}


# ---------------------------------------------------------------------------------------------
# §6.5 — [integration] The transcript is message[0] leading content on the real summarize() path
# ---------------------------------------------------------------------------------------------


def _mock_summarize_response(cached_tokens: int = 0) -> Mock:
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "A concise summary of the episode."
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 120
    # DeepSeek/OpenAI-compat cache-hit field shape (a subset of prompt_tokens).
    response.usage.prompt_cache_hit_tokens = cached_tokens
    details = Mock()
    details.cached_tokens = cached_tokens
    response.usage.prompt_tokens_details = details
    response.model = "gpt-4o-mini"
    response.id = "resp-test-1"
    return response


def _run_summarize(provider: OpenAIProvider, response: Mock) -> Mock:
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = response
    provider.client = mock_client
    provider._summarization_initialized = True
    provider.summarize(text=TRANSCRIPT, episode_title="An Episode")
    return mock_client


def test_summarize_puts_transcript_first_in_system_when_on() -> None:
    client = _run_summarize(_provider(cache=True), _mock_summarize_response())
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(_TRANSCRIPT_BLOCK_HEADER)
    assert TRANSCRIPT in messages[0]["content"]
    assert TRANSCRIPT not in messages[1]["content"]


def test_summarize_keeps_transcript_in_user_when_off() -> None:
    client = _run_summarize(_provider(cache=False), _mock_summarize_response())
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert TRANSCRIPT in messages[1]["content"]
    assert not messages[0]["content"].startswith(_TRANSCRIPT_BLOCK_HEADER)


# ---------------------------------------------------------------------------------------------
# §6.6 — [integration] Cache-field plumbing: the cache-read token count reaches llm_cost telemetry
# ---------------------------------------------------------------------------------------------


def test_summarize_forwards_cache_read_tokens_to_cost_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefix-cache saving must be OBSERVABLE: the provider forwards the raw response so the
    llm_cost event carries the normalised cached-read token count (extracted per-provider by
    token_accounting.extract_token_usage)."""
    captured: dict = {}

    import podcast_scraper.workflow.cost_monitoring as cm

    real_emit = cm.emit_llm_cost_event

    def _spy(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(
        "podcast_scraper.utils.provider_metrics.emit_llm_cost_event", _spy, raising=False
    )
    # provider_metrics imports emit_llm_cost_event lazily inside the function, so patch the source.
    monkeypatch.setattr(cm, "emit_llm_cost_event", _spy, raising=False)

    _run_summarize(_provider(cache=True), _mock_summarize_response(cached_tokens=850))

    assert "response" in captured and captured["response"] is not None
    from podcast_scraper.workflow.token_accounting import extract_token_usage

    usage = extract_token_usage("openai", captured["response"])
    assert usage.cached_input_tokens == 850


# ---------------------------------------------------------------------------------------------
# §6.10 / §6.11 — [regression] layout guard + cross-provider coverage
# ---------------------------------------------------------------------------------------------

# Every thin sibling that shares the OpenAI-compatible transport is an auto-cache provider (ADR-144)
# and MUST inherit the identical transcript-first layout — not silently redefine message assembly.
_SIBLING_IMPORTS = [
    ("podcast_scraper.providers.openai.openai_provider", "OpenAIProvider"),
    ("podcast_scraper.providers.deepseek.deepseek_provider", "DeepSeekProvider"),
    ("podcast_scraper.providers.qwen.qwen_provider", "QwenProvider"),
    ("podcast_scraper.providers.litellm.litellm_provider", "LiteLLMProvider"),
    ("podcast_scraper.providers.vllm.vllm_provider", "VLLMProvider"),
]


@pytest.mark.parametrize("module_path, class_name", _SIBLING_IMPORTS)
def test_every_autocache_sibling_inherits_the_base_layout(
    module_path: str, class_name: str
) -> None:
    import importlib

    sibling = getattr(importlib.import_module(module_path), class_name)
    assert issubclass(sibling, OpenAICompatibleProvider)
    # Reuse the base builder verbatim — a sibling override would fork the cache layout undetected.
    assert sibling._build_stage_messages is OpenAICompatibleProvider._build_stage_messages


# ---------------------------------------------------------------------------------------------
# Phase B — the cross-stage win: summary, GI insights and KG must share ONE cached transcript
# prefix. If their leading blocks are byte-identical, the provider caches the transcript once and
# stages 2-3 read it (the ~3x input-cost reduction the RFC targets).
# ---------------------------------------------------------------------------------------------


def _first_messages(client: Mock) -> list:
    return list(client.chat.completions.create.call_args_list[0].kwargs["messages"])


def test_summary_gi_kg_share_one_transcript_block() -> None:
    """The three stages must lead with a BYTE-IDENTICAL transcript block so the provider caches the
    transcript once and stages 2-3 read it. Uses a realistic transcript with a trailing newline —
    summary embeds it raw, GI/KG pre-strip it — proving normalisation keeps the block stable."""
    raw = TRANSCRIPT + "\n"  # real cleaned transcripts carry a trailing newline
    prefix = _cacheable_transcript_prefix(raw.strip())
    p = _provider(cache=True)
    p._summarization_initialized = True

    def run(kind: str) -> str:
        client = Mock()
        resp = Mock()
        resp.choices = [Mock()]
        # content that each parser tolerates without throwing before the captured call
        resp.choices[0].message.content = (
            '{"topics": [], "entities": []}' if kind == "kg" else "insight one\ninsight two"
        )
        resp.choices[0].finish_reason = "stop"
        resp.usage = Mock(prompt_tokens=100, completion_tokens=20)
        resp.usage.prompt_tokens_details = Mock(cached_tokens=0)
        resp.model = "gpt-4o-mini"
        resp.id = "r"
        client.chat.completions.create.return_value = resp
        p.client = client
        if kind == "summary":
            p.summarize(text=raw, episode_title="Ep")
        elif kind == "gi":
            p.generate_insights(text=raw, episode_title="Ep", max_insights=3)
        else:
            p.extract_kg_graph(text=raw, episode_title="Ep")
        return str(_first_messages(client)[0]["content"])

    summary_sys, gi_sys, kg_sys = run("summary"), run("gi"), run("kg")
    for name, sysmsg in [("summary", summary_sys), ("gi", gi_sys), ("kg", kg_sys)]:
        assert sysmsg.startswith(prefix), f"{name} stage did not lead with the transcript block"
    # The shared cache prefix must be byte-identical across the three stages.
    assert summary_sys[: len(prefix)] == gi_sys[: len(prefix)] == kg_sys[: len(prefix)]


def test_quote_stages_are_transcript_first() -> None:
    """The grounding quote stages (single + bundled) must also lead with the transcript block, so
    the prod quote path (gil_evidence_quote_mode: bundled) joins the shared cross-stage cache."""
    p = _provider(cache=True)
    p._summarization_initialized = True
    prefix = _cacheable_transcript_prefix(TRANSCRIPT)

    def messages_for(kind: str) -> list:
        client = Mock()
        resp = Mock()
        resp.choices = [Mock()]
        resp.choices[0].message.content = (
            '{"0": ["a quote"]}' if kind == "bundled" else '{"quotes": []}'
        )
        resp.choices[0].finish_reason = "stop"
        resp.usage = Mock(prompt_tokens=100, completion_tokens=20)
        resp.usage.prompt_tokens_details = Mock(cached_tokens=0)
        resp.model = "gpt-4o-mini"
        resp.id = "r"
        client.chat.completions.create.return_value = resp
        p.client = client
        if kind == "bundled":
            p.extract_quotes_bundled(TRANSCRIPT, ["insight one"])
        else:
            p.extract_quotes(TRANSCRIPT, "insight one")
        return _first_messages(client)

    for kind in ("single", "bundled"):
        msgs = messages_for(kind)
        assert msgs[0]["content"].startswith(prefix), f"{kind} quote stage not transcript-first"
        assert TRANSCRIPT not in msgs[1]["content"]


def test_relocation_normalises_surrounding_whitespace() -> None:
    """A transcript embedded raw (trailing newline) and the same transcript pre-stripped must yield
    the IDENTICAL block — otherwise stray whitespace silently splits the cross-stage cache."""
    core = TRANSCRIPT
    raw_msgs = _provider()._build_stage_messages(
        transcript=core + "\n", system_prompt="S", user_prompt=f"task\n{core}\nmore"
    )
    stripped_msgs = _provider()._build_stage_messages(
        transcript=core, system_prompt="S", user_prompt=f"task\n{core}\nmore"
    )
    assert raw_msgs[0]["content"] == stripped_msgs[0]["content"]
    assert raw_msgs[0]["content"].startswith(_cacheable_transcript_prefix(core))
