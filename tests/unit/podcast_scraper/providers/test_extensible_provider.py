"""A provider's stage methods are registered, not inherited (E04 split line, S1).

Behaviour-neutral while the methods are in-tree: ``OpenAIProvider(cfg)`` still answers
``generate_insights``, in the same lookup order a direct base gave, and its type still reads as
``OpenAIProvider``. Once the methods move, an extension registers them the same way, and a public
install without it builds the plain provider.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from podcast_scraper import builtin_stage_hooks
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.providers import extensible
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from podcast_scraper.providers.openai.openai_provider_gi_naming import (
    OpenAICompatibleProviderGiNamingMethods,
)
from podcast_scraper.stage_hooks import PipelineHooks
from tests.conftest import create_test_config

pytestmark = pytest.mark.unit


@pytest.fixture
def cfg() -> Any:
    return create_test_config(openai_api_key="sk-test")


def test_a_provider_gets_its_registered_methods_through_its_base(cfg: Any) -> None:
    provider = OpenAIProvider(cfg)
    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider, OpenAICompatibleProviderGiNamingMethods)
    assert hasattr(provider, "generate_insights")
    assert type(provider).__name__ == "OpenAIProvider"  # capabilities and logs read the name
    mro = type(provider).__mro__
    # The provider's own classes first, then the stage's, as a direct base ordered them.
    assert mro.index(OpenAICompatibleProvider) < mro.index(OpenAICompatibleProviderGiNamingMethods)


def test_the_composed_class_is_built_once_and_a_copy_is_not_composed_again(cfg: Any) -> None:
    a, b = OpenAIProvider(cfg), OpenAIProvider(cfg)
    assert type(a) is type(b)
    clone = copy.copy(a)
    assert type(clone) is type(a)


def test_without_an_owner_the_provider_is_plain(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        builtin_stage_hooks,
        "BUILTIN",
        dataclasses.replace(builtin_stage_hooks.BUILTIN, provider_methods={}),
    )
    provider = OpenAIProvider(cfg)
    assert type(provider) is OpenAIProvider
    assert not hasattr(provider, "generate_insights")


class _ExtraMethods:
    def extra_stage_call(self) -> str:
        return f"extra on {type(self).__name__}"


def test_an_extension_registers_methods_the_same_way(
    cfg: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        builtin_stage_hooks,
        "BUILTIN",
        dataclasses.replace(builtin_stage_hooks.BUILTIN, provider_methods={}),
    )
    key = "podcast_scraper.providers.openai.openai_provider.OpenAICompatibleProvider"
    path = f"{__name__}._ExtraMethods"
    ext = Extension(
        name="x", pipeline_hooks=PipelineHooks(name="x", provider_methods={key: (path,)})
    )
    with use_extensions([ext]):
        assert extensible.methods_for(OpenAIProvider) == (_ExtraMethods,)
        assert OpenAIProvider(cfg).extra_stage_call() == "extra on OpenAIProvider"


def test_class_level_access_and_patches_see_the_registered_methods(cfg: Any) -> None:
    """``OpenAIProvider.generate_insights`` and ``patch.object(OpenAIProvider, ...)`` work as with a
    direct base: the patch lands on the provider class, which the composed class finds first."""
    from unittest.mock import patch

    assert OpenAIProvider.generate_insights is (
        OpenAICompatibleProviderGiNamingMethods.generate_insights
    )
    assert "generate_insights" in dir(OpenAIProvider)
    with patch.object(OpenAIProvider, "generate_insights", return_value=["patched"]):
        assert OpenAIProvider(cfg).generate_insights("t") == ["patched"]
    assert "generate_insights" not in OpenAIProvider.__dict__  # the patch is undone
    assert OpenAIProvider(cfg).generate_insights is not None


def test_without_an_owner_the_class_has_no_stage_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        builtin_stage_hooks,
        "BUILTIN",
        dataclasses.replace(builtin_stage_hooks.BUILTIN, provider_methods={}),
    )
    assert not hasattr(OpenAIProvider, "generate_insights")
    assert "generate_insights" not in dir(OpenAIProvider)
