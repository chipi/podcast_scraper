"""The prompt store's search path: installed extensions' prompt directories, then this package's.

An extension that moves prompts out of this package (E04) adds its directory through
``Extension.prompt_dirs``. With none installed, what the store loads and the provenance it records
(``file``, ``sha256``) are what they were before the search path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.prompts import store

pytestmark = pytest.mark.unit

PUBLIC_NAME = "openai/summarization/long_v1"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    store.clear_cache()
    yield
    store.clear_cache()


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _ext(*dirs: Path) -> Extension:
    return Extension(name="prompts-test", prompt_dirs=lambda: list(dirs))


def test_without_an_extension_metadata_is_this_packages_file() -> None:
    public = store.get_prompt_dir() / f"{PUBLIC_NAME}.j2"
    with use_extensions([]):
        meta = store.get_prompt_metadata(PUBLIC_NAME)
    assert meta["file"] == f"{PUBLIC_NAME}.j2"
    assert meta["sha256"] == store.hash_text(public.read_text(encoding="utf-8"))


def test_an_extensions_template_is_found(tmp_path: Path) -> None:
    _write(tmp_path, "acme/ner/system_v1.j2", "private {{ who }}")
    with use_extensions([_ext(tmp_path)]):
        assert store.render_prompt("acme/ner/system_v1", who="prompt") == "private prompt"
        meta = store.get_prompt_metadata("acme/ner/system_v1")
    assert meta["file"] == "acme/ner/system_v1.j2"  # relative to the directory it was found in


def test_an_extensions_exact_template_comes_before_this_packages(tmp_path: Path) -> None:
    _write(tmp_path, f"{PUBLIC_NAME}.j2", "tuned")
    with use_extensions([_ext(tmp_path)]):
        assert store.get_prompt_source(PUBLIC_NAME) == "tuned"


def test_this_packages_exact_template_beats_an_extensions_shared_fallback(tmp_path: Path) -> None:
    # The shared summarization fallback is only tried once no directory has the exact name.
    _write(tmp_path, "shared/summarization/long_v1.j2", "extension shared")
    public = store.get_prompt_dir() / f"{PUBLIC_NAME}.j2"
    with use_extensions([_ext(tmp_path)]):
        assert store.get_prompt_source(PUBLIC_NAME) == public.read_text(encoding="utf-8")


def test_the_shared_fallback_is_still_found_in_an_extension(tmp_path: Path) -> None:
    _write(tmp_path, "shared/summarization/only_here_v1.j2", "shared in extension")
    with use_extensions([_ext(tmp_path)]):
        assert store.get_prompt_source("acme/summarization/only_here_v1") == "shared in extension"


def test_a_missing_template_names_every_directory_searched(tmp_path: Path) -> None:
    with use_extensions([_ext(tmp_path)]):
        with pytest.raises(store.PromptNotFoundError) as err:
            store.get_prompt_source("acme/none/missing_v1")
    assert str(tmp_path.resolve()) in str(err.value)
    assert str(store.get_prompt_dir()) in str(err.value)


def test_a_name_that_escapes_the_directory_is_refused(tmp_path: Path) -> None:
    with use_extensions([_ext(tmp_path)]):
        with pytest.raises(ValueError):
            store.get_prompt_source("../outside")


def test_ollamas_model_specific_template_is_found_in_an_extension(tmp_path: Path) -> None:
    """A tuned per-model template that moved to an extension is still preferred (E04 item 6).

    The lookup checked this package's directory only, so once the template moved the provider
    fell back to the generic one, silently.
    """
    from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

    _write(tmp_path, "ollama/tunedmodel_7b/ner/system_v9.j2", "tuned for this model")
    provider = OllamaProvider.__new__(OllamaProvider)
    fallback = "ollama/ner/system_v9"
    with use_extensions([_ext(tmp_path)]):
        chosen = provider._get_model_specific_prompt_path(
            "tunedmodel:7b", "ner", "system_v9", fallback
        )
    assert chosen == "ollama/tunedmodel_7b/ner/system_v9"
    with use_extensions([]):
        assert (
            provider._get_model_specific_prompt_path("tunedmodel:7b", "ner", "system_v9", fallback)
            == fallback
        )
