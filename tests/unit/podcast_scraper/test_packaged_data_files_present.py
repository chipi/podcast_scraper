"""Regression guard: runtime data files must ship with the wheel (#1657 acceptance item 8).

The same trap ``test_packaged_prompts_present`` was written for, one directory over. setuptools
strips every non-``.py`` file out of the distribution unless ``[tool.setuptools.package-data]``
lists it, and an editable install hides that completely — the source tree is right there, so
everything resolves locally and only the container is broken.

``src/podcast_scraper/data/known_models.yaml`` exists precisely to be the wheel-bundled fallback
for containers (``known_models._bundled_path``). It was added to the source tree and never to the
manifest, which listed ``data/pricing_assumptions.yaml`` file-by-file. So the pipeline image
logged::

    known_models.yaml not found; model allowlist validation is DISABLED

and ran every cloud call with no allowlist at all. That allowlist exists to reject an unknown or
typo'd model id BEFORE the request goes out (``UnknownModelError`` — "raised BEFORE any API
spend"); without it, a bad id reaches the provider and costs money to find out.

WHY THIS TEST ASSERTS ON THE MANIFEST rather than on ``importlib.resources``:
a resources-based check — the approach the prompts guard can afford — passes under the editable
install used for local runs and CI, because it finds the file in ``src/``. It would have been
green throughout the entire period the wheel was missing the file. The only thing that
distinguishes a packaged file from an unpackaged one, without building a wheel in a unit test, is
whether a ``package-data`` pattern covers it.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "src" / "podcast_scraper"
PYPROJECT = REPO / "pyproject.toml"


def _package_data_patterns() -> List[str]:
    """The ``podcast_scraper = [...]`` globs from ``[tool.setuptools.package-data]``."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    patterns = data["tool"]["setuptools"]["package-data"]["podcast_scraper"]
    assert isinstance(patterns, list) and patterns, "package-data manifest is missing or empty"
    return [str(p) for p in patterns]


def _covered(rel_path: str, patterns: List[str]) -> bool:
    """setuptools matches these as globs; ``**`` spans directories."""
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        # fnmatch treats ``*`` as crossing ``/``, so ``prompts/**/*.j2`` already matches
        # deep paths; this extra pass keeps a single-level pattern from matching too much.
        if (
            pat.endswith("/*")
            and rel_path.startswith(pat[:-1])
            and "/" not in rel_path[len(pat) - 1 :]
        ):
            return True
    return False


class TestEveryRuntimeDataFileIsPackaged:
    def test_known_models_yaml_is_covered(self) -> None:
        """The specific file whose absence disabled the allowlist in production images."""
        assert (PKG / "data" / "known_models.yaml").is_file(), "the bundled fallback is missing"
        assert _covered("data/known_models.yaml", _package_data_patterns())

    def test_no_yaml_under_data_is_left_behind(self) -> None:
        """Whole directory, not a named file. The manifest listed data files one by one, which
        is what let a newly-added one slip through silently."""
        patterns = _package_data_patterns()
        missing = [
            str(p.relative_to(PKG))
            for p in sorted((PKG / "data").glob("*.yaml"))
            if not _covered(str(p.relative_to(PKG)), patterns)
        ]
        assert not missing, f"data files not covered by package-data: {missing}"

    def test_the_enrichment_schemas_are_still_covered(self) -> None:
        """Guard the neighbours the same way — they have the identical failure mode."""
        patterns = _package_data_patterns()
        schemas = sorted((PKG / "enrichment" / "_schema").glob("*.json"))
        if not schemas:
            pytest.skip("no enrichment schemas in this tree")
        for s in schemas:
            assert _covered(str(s.relative_to(PKG)), patterns), f"{s.name} would not ship"


class TestTheContainerPathIsAlsoPopulated:
    """All THREE resolution paths missed in the image, not just the wheel.

    ``_resolve_path`` tries, in order: ``config/known_models.yaml`` relative to cwd (which is
    ``/app`` in the container, so ``/app/config/known_models.yaml``), then that same absolute
    container path, then the wheel-bundled copy. The Dockerfile copied ``config/profiles/`` and
    ``config/pricing_assumptions.yaml`` into ``/app/config/`` and never the allowlist — so the
    first two both missed for the same reason, and the third missed because of the manifest.
    Fixing only one would have left the outcome dependent on which path happened to win.
    """

    DOCKERFILE = REPO / "docker" / "pipeline" / "Dockerfile"

    def test_the_allowlist_is_copied_into_the_image(self) -> None:
        text = self.DOCKERFILE.read_text(encoding="utf-8")
        assert "config/known_models.yaml /app/config/known_models.yaml" in text

    def test_it_lands_where_the_loader_looks_first(self) -> None:
        """Not just "somewhere in the image": the destination has to be the cwd-relative path
        the resolver checks before anything else."""
        from podcast_scraper.providers import known_models

        assert known_models._DEFAULT_CONFIGURED_PATH == "config/known_models.yaml"
        assert Path("/app/config/known_models.yaml") in known_models._CONTAINER_FALLBACK_PATHS
        text = self.DOCKERFILE.read_text(encoding="utf-8")
        assert "/app/config/known_models.yaml" in text

    def test_the_source_file_the_dockerfile_copies_exists(self) -> None:
        """A COPY of a missing path fails the build, but only when someone rebuilds."""
        assert (REPO / "config" / "known_models.yaml").is_file()


class TestTheTwoCopiesDoNotDrift:
    """``config/known_models.yaml`` and the bundled fallback must stay identical.

    Found while fixing the packaging: they had ALREADY drifted. ``config/`` listed
    ``gpt-4.1-mini`` and the bundled copy did not — so a container falling back to the packaged
    file would have raised ``UnknownModelError`` for a model that is real, governed, and listed
    in the repo. A fallback that rejects valid models is worse than the missing file it replaces,
    because it fails a run instead of merely failing to guard one.

    Nothing generated the bundled copy; it was hand-copied once, and there was no check. This is
    that check. If you edit one, copy it to the other — they are the same document, one of which
    happens to live inside the wheel.
    """

    CONFIGURED = REPO / "config" / "known_models.yaml"
    BUNDLED = PKG / "data" / "known_models.yaml"

    def test_they_are_byte_identical(self) -> None:
        assert self.CONFIGURED.read_bytes() == self.BUNDLED.read_bytes(), (
            "config/known_models.yaml and src/podcast_scraper/data/known_models.yaml have "
            "drifted; copy one over the other"
        )

    def test_no_model_is_governed_in_one_copy_only(self) -> None:
        """The assertion that states the CONSEQUENCE, so a failure explains itself even if the
        byte comparison is later relaxed for formatting."""
        import yaml

        a = yaml.safe_load(self.CONFIGURED.read_text(encoding="utf-8"))
        b = yaml.safe_load(self.BUNDLED.read_text(encoding="utf-8"))
        assert set(a.get("governed_providers") or []) == set(b.get("governed_providers") or [])
        for provider in set(a.get("providers", {})) | set(b.get("providers", {})):
            only_configured = set(a.get("providers", {}).get(provider) or [])
            only_bundled = set(b.get("providers", {}).get(provider) or [])
            assert only_configured == only_bundled, (
                f"{provider}: models present in only one copy "
                f"{sorted(only_configured ^ only_bundled)} — a container on the fallback would "
                "reject them"
            )


class TestTheFallbackActuallyResolves:
    """Complements the manifest check: the file must also be loadable and well-formed. This part
    WOULD have passed while the wheel was broken — it is here for content, not for packaging."""

    def test_the_bundled_copy_parses_and_governs_something(self) -> None:
        import yaml

        payload = yaml.safe_load((PKG / "data" / "known_models.yaml").read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get(
            "governed_providers"
        ), "a fallback that governs nothing is not a fallback"
        assert isinstance(payload.get("providers"), dict) and payload["providers"]

    def test_the_loader_finds_a_path_in_this_tree(self) -> None:
        from podcast_scraper.providers import known_models

        assert known_models._resolve_path() is not None

    def test_the_allowlist_is_not_empty_when_loaded(self) -> None:
        """The silent-degradation shape: ``_load`` returns empty collections rather than raising
        when the file is absent, so "validation disabled" looks identical to "nothing governed"."""
        from podcast_scraper.providers import known_models

        known_models.clear_known_models_cache()
        governed, models = known_models._load()
        assert governed, "no governed providers — the allowlist would be inert"
        assert models
