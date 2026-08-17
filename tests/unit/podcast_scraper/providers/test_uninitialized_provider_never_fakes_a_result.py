"""An unusable provider must SAY SO, not return a well-formed empty result (#34).

THE DEFECT
Across all six providers, the entire evidence stack — ``extract_quotes``,
``extract_quotes_bundled``, ``score_entailment``, ``score_entailment_bundled`` — returned an
empty result when ``_summarization_initialized`` was False: no exception, no log line. 24 sites.
Meanwhile 40 other guarded methods (``summarize``, ``clean_transcript``, ``classify_insights``,
``complete_text``, ...) RAISE on exactly the same condition. The project had already decided the
contract; the evidence stack quietly opted out of it.

WHY IT MATTERS MORE AFTER #1657
The old reasoning was defensible: evidence is optional, so degrade quietly. #1657 removed that
defence. "Nothing extracted means nothing returned" is now a LEGAL outcome — an episode may
honestly have zero insights and zero quotes. So an empty evidence result no longer distinguishes
"the model found nothing" from "the provider was never usable". The silence used to be
interpretable. It is not any more.

FOUND BY BEING FOOLED BY IT: a probe built a provider via ``create_summarization_provider()``,
skipped ``initialize()``, called ``extract_quotes_bundled`` and reported "parse ok: True, quotes
returned: 0". Adding ``initialize()`` turned the same episode into 114 quotes. A silent zero is
exactly as convincing as a real one.

NOT A LIVE PRODUCTION DEFECT: ``gi.deps.create_gil_evidence_providers`` initializes what it
builds, so the main pipeline path never reaches these guards. This closes a trap for direct
callers — of which ``gi.repair`` is now one.
"""

# mypy: disable-error-code="call-arg"
# Deliberate in this file: Config(rss_url=...) — the field declares alias="rss", so mypy's pydantic
# plugin
# only knows the alias while populate-by-name accepts either at runtime.
# Constructing the real types would pull in the machinery these tests isolate. The
# annotations on the helpers here are what make mypy check these bodies at all — most
# older test files are unannotated and therefore unchecked.

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple

import pytest

from podcast_scraper import config

pytestmark = [pytest.mark.unit]


#: (module path, class name) for every provider carrying the guard.
PROVIDERS: List[Tuple[str, str]] = [
    ("podcast_scraper.providers.openai.openai_provider", "OpenAICompatibleProvider"),
    ("podcast_scraper.providers.gemini.gemini_provider", "GeminiProvider"),
    ("podcast_scraper.providers.grok.grok_provider", "GrokProvider"),
    ("podcast_scraper.providers.mistral.mistral_provider", "MistralProvider"),
    ("podcast_scraper.providers.anthropic.anthropic_provider", "AnthropicProvider"),
    ("podcast_scraper.providers.ollama.ollama_provider", "OllamaProvider"),
]

#: The evidence stack: method name -> arguments that are otherwise VALID, so the only reason to
#: fail is the missing initialization.
EVIDENCE_CALLS: Dict[str, Callable[[Any], Any]] = {
    "extract_quotes": lambda p: p.extract_quotes("a real transcript here", "a real insight"),
    "extract_quotes_bundled": lambda p: p.extract_quotes_bundled(
        "a real transcript here", ["insight one", "insight two"]
    ),
    "score_entailment": lambda p: p.score_entailment("a premise", "a hypothesis"),
    "score_entailment_bundled": lambda p: p.score_entailment_bundled(
        [("a premise", "a hypothesis")]
    ),
}


#: Constructors validate credential PRESENCE, so a dummy value is enough to build an object.
#: Without these every case skipped, and a guardrail that skips proves nothing — the failure mode
#: this whole file is about.
_DUMMY_KEYS = {
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "GEMINI_API_KEY": "test-not-a-real-key",
    "GROK_API_KEY": "test-not-a-real-key",
    "MISTRAL_API_KEY": "test-not-a-real-key",
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "DEEPSEEK_API_KEY": "test-not-a-real-key",
    "LITELLM_API_KEY": "test-not-a-real-key",
}


@pytest.fixture(autouse=True)
def _dummy_credentials(monkeypatch):
    for name, value in _DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)


def _uninitialized(module_path: str, class_name: str) -> Any:
    """A provider instance that has NOT had initialize() called.

    Constructed WITHOUT touching the network: no initialize(), no API call. Only the guard is
    exercised, so the dummy keys above are never used for anything.
    """
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not present in {module_path}")
    cfg = config.Config(rss_url="https://example.com/feed.xml")
    try:
        provider = cls(cfg)
    except Exception as exc:  # pragma: no cover - e.g. ollama needs a live server to construct
        pytest.skip(f"cannot construct {class_name} in this environment: {exc}")
    assert not getattr(
        provider, "_summarization_initialized", False
    ), "precondition: the provider must start uninitialized or this test proves nothing"
    return provider


@pytest.mark.parametrize("module_path,class_name", PROVIDERS, ids=[p[1] for p in PROVIDERS])
@pytest.mark.parametrize("method_name", sorted(EVIDENCE_CALLS))
def test_evidence_methods_raise_when_uninitialized(module_path, class_name, method_name):
    """THE contract: an unusable provider raises rather than fabricating an empty result.

    A well-formed empty return here is indistinguishable from a genuine "found nothing", and
    under ``gi_require_grounding: true`` it silently strips an episode's whole evidence stack.
    """
    provider = _uninitialized(module_path, class_name)
    if not hasattr(provider, method_name):
        pytest.skip(f"{class_name} has no {method_name}")

    with pytest.raises(RuntimeError) as caught:
        EVIDENCE_CALLS[method_name](provider)

    message = str(caught.value).lower()
    assert (
        "initial" in message
    ), f"the error must name the cause so an operator can act on it; got: {caught.value!r}"


@pytest.mark.parametrize("module_path,class_name", PROVIDERS, ids=[p[1] for p in PROVIDERS])
def test_generate_insights_raises_when_uninitialized(module_path, class_name):
    """``generate_insights`` warned and returned ``[]``. Post-#1657 an empty insight list is a
    LEGAL outcome, so that return value now collides with a real one."""
    provider = _uninitialized(module_path, class_name)
    if not hasattr(provider, "generate_insights"):
        pytest.skip(f"{class_name} has no generate_insights")

    with pytest.raises(RuntimeError):
        provider.generate_insights("a real transcript here")


@pytest.mark.parametrize("module_path,class_name", PROVIDERS, ids=[p[1] for p in PROVIDERS])
def test_empty_INPUT_still_returns_empty_not_an_error(module_path, class_name):
    """The other half of the split, and the reason this is not a blanket 'raise on everything'.

    ``if not self._summarization_initialized or not transcript:`` conflated two unrelated
    conditions. An uninitialized provider is a programming error; an EMPTY TRANSCRIPT is a
    legitimate data outcome that must still return an empty result quietly.
    """
    provider = _uninitialized(module_path, class_name)
    if not hasattr(provider, "extract_quotes"):
        pytest.skip(f"{class_name} has no extract_quotes")
    provider._summarization_initialized = True  # usable provider, empty input

    assert provider.extract_quotes("", "an insight") == []
    assert provider.extract_quotes("a transcript", "") == []
