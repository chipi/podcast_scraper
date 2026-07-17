"""A bake-off compares MODELS only if the harness around each model is identical.

It was not. Measured before this file existed, the same pipeline stage ran different prompts on
different providers, and the differences were invisible because they were buried in string literals:

    * `complete_text` existed on gemini and ollama ONLY, so ADR-110 speaker resolution silently
      resolved NOBODY on the other five — and the arm would have scored as a weak model rather than
      a missing method.
    * The v3 speech-act extraction prompt ("Rodman ARGUES THAT prescribing is unsafe", not
      "prescribing is unsafe") was reachable by ollama alone; everyone else was pinned to v2, which
      flattens an opinion into a bare fact about the world.
    * openai's grounding prompt asked for `quote_text` — ONE quote — while its own parser read a
      LIST. It was structurally capped at one piece of evidence per insight.
    * Five providers built their grounding prompt from an INLINE string that asked for a list AND
      for "quote_text only", contradicting itself.
    * Only ollama carried the corrected ENTAILMENT wording. Asking for strict textual entailment is
      not the question the pipeline means, and asking it strictly cost 60% of the evidence a trusted
      annotator had accepted (#1179).

Any scoreboard built on that would have measured our integration effort and called it model quality.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

LLM_PROVIDERS = ["anthropic", "deepseek", "gemini", "grok", "mistral", "ollama", "openai"]

# Every capability the pipeline asks a provider for. A provider missing one does not fail loudly —
# the caller DEGRADES, and the model gets the blame.
CAPABILITIES = [
    "summarize",
    "generate_insights",
    "classify_insights",
    "complete_text",
    "extract_quotes",
    "score_entailment",
    "extract_kg_graph",
    "detect_speakers",
]

PROMPTS = Path("src/podcast_scraper/prompts")
SRC = Path("src/podcast_scraper/providers")


def _functions(provider: str) -> dict:
    src = (SRC / provider / f"{provider}_provider.py").read_text()
    tree = ast.parse(src)
    return {
        n.name: (ast.get_source_segment(src, n) or "")
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }


def _nodes(provider: str) -> dict:
    """The AST node for each method — walked directly, never re-parsed from a source slice."""
    src = (SRC / provider / f"{provider}_provider.py").read_text()
    tree = ast.parse(src)
    return {n.name: (n, src) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
@pytest.mark.parametrize("capability", CAPABILITIES)
def test_every_provider_has_every_capability(provider: str, capability: str) -> None:
    assert capability in _functions(provider), (
        f"{provider} cannot {capability}. The pipeline will degrade silently — ADR-110 resolves "
        f"nobody, the value gate keeps everything — and the MODEL will be blamed for a method we "
        f"never wrote."
    )


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
def test_every_provider_honors_its_api_base(provider: str) -> None:
    """Uniform contract: every provider reads ``<provider>_api_base`` to route the client.

    The Deepgram/Gemini incident: a configured base (the CI mock server or a self-hosted
    endpoint) that the SDK couldn't apply was SILENTLY dropped, and the client hit the
    REAL hosted API — a 401 in CI, and in prod audio leak + spend. The base-application
    code legitimately differs by SDK (OpenAI ``base_url`` vs Mistral ``server_url`` vs
    Gemini ``HttpOptions`` vs Deepgram env), but the contract must not: no provider may
    ignore its base, and when it can't honor a configured base it must fail loud rather
    than fall through to production.
    """
    src = (SRC / provider / f"{provider}_provider.py").read_text()
    assert f"{provider}_api_base" in src, (
        f"{provider} never reads {provider}_api_base — the CI mock / self-hosted base is "
        f"ignored, so the client silently hits the real hosted API (the Deepgram trap)."
    )


@pytest.mark.parametrize(
    "family",
    [
        "insight_extraction/v2",
        "insight_extraction/v3",
        "insight_extraction/system_v1",
        "insight_value_gate/v1",
        "evidence/extract_quote/v1",
        "evidence/entailment/v1",
    ],
)
def test_every_provider_has_every_prompt(family: str) -> None:
    missing = [p for p in LLM_PROVIDERS if not (PROMPTS / p / f"{family}.j2").exists()]
    assert not missing, (
        f"{family} is missing for {missing}. That provider runs a different prompt — or none — and "
        f"a bake-off would report the difference as model quality."
    )


@pytest.mark.parametrize(
    "family",
    [
        "evidence/extract_quote/v1",
        "evidence/entailment/v1",
        "insight_extraction/v3",
        "insight_extraction/v2",
    ],
)
def test_the_prompts_are_IDENTICAL_across_providers(family: str) -> None:
    """A provider may diverge DELIBERATELY. It must not diverge by accident."""
    texts = {p: (PROMPTS / p / f"{family}.j2").read_text() for p in LLM_PROVIDERS}
    distinct = set(texts.values())
    assert len(distinct) == 1, (
        f"{family} differs across providers: "
        f"{sorted({p for p in texts if texts[p] != texts[LLM_PROVIDERS[0]]})} have a different "
        f"prompt. openai once asked for ONE quote here while everyone else asked for a list."
    )


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
def test_the_extraction_prompt_version_is_a_TUNED_PARAMETER(provider: str) -> None:
    """The prompt decides what an insight IS. Hardcoding it made this the only part of the stage
    that could not be A/B tested — and left v3 reachable by ollama alone."""
    body = _functions(provider)["generate_insights"]
    assert (
        "gi_insight_prompt_version" in body
    ), f"{provider} hardcodes its extraction prompt version, so it can never run v3"


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
@pytest.mark.parametrize("method", ["generate_insights", "extract_quotes", "score_entailment"])
def test_no_prompt_is_BURIED_IN_CODE(provider: str, method: str) -> None:
    """A prompt in a file is versioned, diffable and reviewable. A prompt in code is a bug with
    nowhere to be seen — ollama's inline quote prompt hid THREE evidence-destroying defects at once
    (a 50k truncation below episode length, a system/user disagreement about the output shape, and a
    copyable example string that local models reproduced verbatim, #1179)."""
    node, src = _nodes(provider)[method]
    buried = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and isinstance(sub.value, (ast.Constant, ast.JoinedStr)):
            target = getattr(sub.targets[0], "id", "")
            if target in ("system", "user", "prompt", "system_prompt", "user_prompt"):
                seg = ast.get_source_segment(src, sub) or ""
                if len(seg) > 90:
                    buried.append(target)
    assert (
        not buried
    ), f"{provider}.{method} still builds {buried} as a literal instead of rendering a template"
