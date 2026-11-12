"""A bake-off compares MODELS only if the harness around each model is identical.

It was not. Measured before this file existed, the same pipeline stage ran different prompts on
different providers, and the differences were invisible because they were buried in string literals:

    * `complete_text` existed on gemini and ollama ONLY, so speaker resolution silently
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

# Every capability the pipeline asks a provider for (public's; the GI and naming ones are the
# extension's, checked where their methods live). A provider missing one does not fail loudly —
# the caller DEGRADES, and the model gets the blame.
CAPABILITIES = [
    "summarize",
    "complete_text",
    "extract_kg_graph",
]

PROMPTS = Path("src/podcast_scraper/prompts")
SRC = Path("src/podcast_scraper/providers")
# ADR-147: the shared OpenAI-compatible transport base. Thin siblings (deepseek/vllm/litellm/qwen)
# inherit the capability methods from here instead of redefining them, so a provider's own
# ``*_provider.py`` may legitimately hold only its overrides. Resolving capabilities against the
# provider file ALONE would falsely report "deepseek cannot summarize" the moment it became a thin
# sibling — so the scan folds in the base's methods for any provider that subclasses it.
_BASE_FILE = SRC / "openai" / "openai_provider.py"
_BASE_MARKER = "OpenAICompatibleProvider"


def _funcs_in(path: Path) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    return {
        n.name: (ast.get_source_segment(src, n) or "")
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }


def _nodes_in(path: Path) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    return {n.name: (n, src) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _is_thin_sibling(provider: str, src: str) -> bool:
    """A provider that inherits the capability methods from the shared base (ADR-147).

    Detected by a real base-class EDGE, not a substring: a provider that merely *mentions*
    ``OpenAICompatibleProvider`` in a docstring/comment must not be mistaken for a subclass of it.
    """
    if provider == "openai":
        return False
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == _BASE_MARKER:
                    return True
                if isinstance(base, ast.Attribute) and base.attr == _BASE_MARKER:
                    return True
    return False


def _with_methods_module(path: Path) -> list:
    """A provider module and, when present, the module its GI and naming methods moved to
    (``<stem>_gi_naming.py``, E04 split line S1); the class inherits them from there."""
    moved = path.with_name(f"{path.stem}_gi_naming.py")
    return [moved, path] if moved.exists() else [path]


def _merged(read, paths: list) -> dict:
    out: dict = {}
    for p in paths:  # later wins: the provider module's own definitions over its moved methods
        out.update(read(p))
    return out


def _functions(provider: str) -> dict:
    path = SRC / provider / f"{provider}_provider.py"
    own = _merged(_funcs_in, _with_methods_module(path))
    if _is_thin_sibling(provider, path.read_text()):
        # Own overrides win; the base fills in every inherited capability method.
        return {**_merged(_funcs_in, _with_methods_module(_BASE_FILE)), **own}
    return own


def _nodes(provider: str) -> dict:
    """The AST node for each method — walked directly, never re-parsed from a source slice.

    For a thin sibling the inherited methods come from the shared base file (ADR-147)."""
    path = SRC / provider / f"{provider}_provider.py"
    own = _merged(_nodes_in, _with_methods_module(path))
    if _is_thin_sibling(provider, path.read_text()):
        return {**_merged(_nodes_in, _with_methods_module(_BASE_FILE)), **own}
    return own


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
@pytest.mark.parametrize("capability", CAPABILITIES)
def test_every_provider_has_every_capability(provider: str, capability: str) -> None:
    assert capability in _functions(provider), (
        f"{provider} cannot {capability}. The pipeline will degrade silently — speaker "
        f"resolution resolves nobody, the value gate keeps everything — and the MODEL will be "
        f"blamed for a method we never wrote."
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
    # Direct literal read, OR the ADR-147 parameterized form: the OpenAI-compatible base
    # (OpenAICompatibleProvider) reads its base via getattr(cfg, f"{self._CONFIG_NS}_api_base"),
    # which honors <provider>_api_base for the provider's own namespace (openai -> openai_api_base,
    # vllm -> vllm_api_base). Still fails loud for any provider that reads NO api_base at all.
    honors_base = f"{provider}_api_base" in src or ("_CONFIG_NS" in src and "_api_base" in src)
    assert honors_base, (
        f"{provider} never reads {provider}_api_base — the CI mock / self-hosted base is "
        f"ignored, so the client silently hits the real hosted API (the Deepgram trap)."
    )


@pytest.mark.parametrize("provider", ["litellm", "vllm", "qwen", "groq"])
def test_thin_sibling_prompts_reuse_a_parity_covered_namespace(provider: str) -> None:
    """ADR-147 thin siblings (litellm/vllm/qwen/groq) ship no ``prompts/<provider>/`` dir — they
    REUSE another provider's prompt namespace (openai's). Fine, but it must be EXPLICIT, not luck:
    their summary + speaker prompt-name defaults must resolve to a namespace this suite actually
    checks (a member of ``LLM_PROVIDERS``). Otherwise a sibling could silently run an unvetted
    prompt with no parity guard — how ``qwen`` slipped past the finale (not in ``LLM_PROVIDERS``,
    covered only because its defaults happen to point at ``openai/``).
    """
    from podcast_scraper.config import Config

    for field in (
        f"{provider}_summary_system_prompt",
        f"{provider}_summary_user_prompt",
        f"{provider}_speaker_user_prompt",
    ):
        default = Config.model_fields[field].default
        if not default:  # e.g. a speaker_system_prompt that defaults None — nothing to cover
            continue
        namespace = str(default).split("/", 1)[0]
        assert namespace in LLM_PROVIDERS, (
            f"{field} = {default!r}; namespace {namespace!r} not in LLM_PROVIDERS, so "
            f"{provider}'s prompt is not covered by the cross-provider parity guard."
        )


# A summary other than the episode's (a board's) is a use case of ``summarize`` itself,
# so it gets the same retries, guardrails, cost record and prompt metadata on every provider. Each
# piece below is one place a provider could quietly fall back to the episode behaviour.
_USE_CASE_HOOKS = {
    "summarize": [
        "use_case_from_params",
        "reply_budget",
        "use_case_stage",
        "use_case_prompt_metadata",
    ],
    "_build_summarization_prompts": ["build_use_case_prompts"],
}


@pytest.mark.parametrize("provider", LLM_PROVIDERS)
@pytest.mark.parametrize("method", sorted(_USE_CASE_HOOKS))
def test_every_provider_runs_a_summary_use_case(provider: str, method: str) -> None:
    body = _functions(provider).get(method, "")
    missing = [hook for hook in _USE_CASE_HOOKS[method] if f"{hook}(" not in body]
    assert not missing, (
        f"{provider}.{method} does not call {missing}: a board summary on {provider} would run the "
        f"EPISODE prompts, budget or cost stage."
    )


def test_summllama_runs_a_summary_use_case_too() -> None:
    """The local chat model (ml/summllama_provider.py) is outside LLM_PROVIDERS but takes prompts,
    so it carries the use case as well. It records no cost (local, as for the episode summary)."""
    funcs = _funcs_in(SRC / "ml" / "summllama_provider.py")
    hooks = {
        "summarize": ["use_case_from_params", "reply_budget", "use_case_prompt_metadata"],
        "_build_summarization_prompts": ["build_use_case_prompts"],
    }
    for method, needed in hooks.items():
        missing = [h for h in needed if f"{h}(" not in funcs.get(method, "")]
        assert not missing, f"summllama.{method} does not call {missing}"


def test_the_seq2seq_provider_runs_a_summary_use_case_too() -> None:
    """BART/LED (ml/ml_provider.py) take a document, not instructions: the use case's material is
    written as prose and summarised by the same map/reduce."""
    funcs = _funcs_in(SRC / "ml" / "ml_provider.py")
    hooks = {
        "summarize": ["use_case_from_params", "_use_case_input", "use_case_prompt_metadata"],
        "_build_use_case_document": ["build_use_case_document"],
    }
    for method, needed in hooks.items():
        missing = [h for h in needed if f"{h}(" not in funcs.get(method, "")]
        assert not missing, f"transformers.{method} does not call {missing}"
