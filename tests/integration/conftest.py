"""Pytest fixtures shared across the integration test suite.

Auto-generates the multi-run corpus fixture for the v2.6.1 hotfix tests
(``test_multi_run_corpus_fixture.py``) so contributors don't need a manual
generator step. The fixture is small + deterministic; regenerating it
takes ~50 ms.

Generation runs at conftest module-import time (before pytest collects
tests) so the test module's ``pytest.mark.skipif`` markers — evaluated at
collection — see the fixture present.
"""

from __future__ import annotations

import importlib.util
import os
import time as _real_time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

# Imported for its SIDE EFFECT, before any test module is imported. Do not remove.
#
# Several provider test modules install a module-level ``mock.patch.dict(sys.modules, …)`` at
# import time (see the "Mocking LLM SDKs at sys.modules" note in INTEGRATION_TESTING_GUIDE.md).
# ``patch.dict`` snapshots the dict on entry and RESTORES that snapshot on exit — which deletes
# every key added while it was active. Any module first imported inside that window is therefore
# evicted from ``sys.modules`` when the patch unwinds.
#
# For a pure-Python module that is harmless: the next import just re-executes it. For a C
# extension it is not. numpy's ``_multiarray_umath`` refuses to initialise twice in one process,
# so the re-import raises::
#
#     ImportError: cannot load module more than once per process
#
# which surfaced as ``test_embedding_loader.py::TestEncodeMocked::test_encode_return_numpy_*``
# failing in a full-suite run while passing when its own file is run alone — the two tests do
# ``import numpy`` inside the test body, i.e. after the eviction.
#
# Importing numpy HERE puts it in the snapshot every one of those patchers takes, so restoring
# the snapshot leaves it in place. That removes the precondition rather than hiding the symptom;
# the alternative — rewriting the module-level patchers those files deliberately use — is a much
# larger change for the same outcome. Verified 2026-08-17: 760 passed / 2 failed before, 762
# passed / 0 failed after, and the failures reproduce on this branch well before this file
# gained any of its stub helpers.
import numpy  # noqa: F401
import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "multi-run-corpus"
_GENERATOR_PATH = _REPO_ROOT / "scripts" / "tools" / "build_multi_run_fixture.py"


def _ensure_multi_run_corpus_fixture() -> None:
    """Generate the multi-run corpus fixture if missing."""
    if _FIXTURE_DIR.exists() and (_FIXTURE_DIR / "corpus_manifest.json").exists():
        return

    spec = importlib.util.spec_from_file_location("_build_multi_run_fixture", _GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generator at {_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_fixture(
        _FIXTURE_DIR,
        n_feeds=3,
        probe_episodes=1,
        middle_episodes=5,
        latest_episodes=5,
        overlap=3,
    )


# Module-level: runs at conftest import (before collection).
_ensure_multi_run_corpus_fixture()


def requires(*modules: str) -> "pytest.MarkDecorator":
    """Skip ONE test when an optional ML/search module is not importable.

    Integration tests may skip on a missing extra — unlike unit tests, where the same guard is
    banned (U1), because a unit test must not depend on a non-``[dev]`` extra in the first place.

    Applied per TEST, deliberately, never at module level. Every file that needs this also holds
    tests that pass without the extra — ``test_e2e_server.py`` is 2 ML-dependent tests beside 26
    that are not — so a module-level ``importorskip`` would have silenced about a hundred working
    integration tests to quiet a dozen failures. Skipping more than the thing that is actually
    unavailable is how a suite stops being evidence.

    Inert in CI: twelve jobs install ``.[dev,ml,llm,search]``, so the modules are present and every
    guarded test runs. It only bites on a machine that cannot install the ML stack at all — macOS
    x86_64, where torch/torchcodec publish no wheels.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"needs the [ml]/[search] extra — not importable: {', '.join(missing)}",
    )


def stub_transformers() -> dict[str, Any]:
    """A ``sys.modules`` overlay standing in for ``transformers``, or ``{}`` when it is installed.

    Use with ``@patch.dict(sys.modules, stub_transformers())``.

    Several tests need ``transformers`` present but never call it: they exist to assert OUR call
    shape — that a pinned ``revision`` reaches ``from_pretrained``, that ``unload_model`` is
    idempotent, that a loader failure surfaces as the right error — and they get there by patching
    deep library paths like ``transformers.models.auto.tokenization_auto.AutoTokenizer``. The import
    is required only so ``mock.patch`` can RESOLVE those strings. Patching our own module attribute
    is not available as an alternative: ``providers/ml/summarizer.py`` does ``from transformers
    import ...`` inside its methods, so there is nothing bound on our side — the import reads
    ``sys.modules`` at call time. Hence the overlay, the same technique the unit tier uses for torch.

    **Empty when transformers is genuinely installed**, so CI — where twelve jobs install
    ``.[dev,ml,llm,search]`` — keeps running these against the real library and this helper changes
    nothing there. It only fills in on a machine where the ML stack cannot be installed at all.
    """
    if importlib.util.find_spec("transformers") is not None:
        return {}

    def _mod(name: str, **attrs: Any) -> ModuleType:
        m = ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    # Only the paths the tests actually patch, plus GenerationConfig, which hybrid_ml_provider's
    # reduce() constructs for real before handing it to a mocked backend. Deliberately NOT a
    # blanket MagicMock package: a test patching a path this does not model must fail loudly at
    # patch time, not quietly pass against an auto-created attribute.
    tokenization_auto = _mod(
        "transformers.models.auto.tokenization_auto", AutoTokenizer=MagicMock()
    )
    modeling_auto = _mod(
        "transformers.models.auto.modeling_auto", AutoModelForSeq2SeqLM=MagicMock()
    )
    modeling_bart = _mod(
        "transformers.models.bart.modeling_bart", BartForConditionalGeneration=MagicMock()
    )
    auto = _mod(
        "transformers.models.auto", tokenization_auto=tokenization_auto, modeling_auto=modeling_auto
    )
    bart = _mod("transformers.models.bart", modeling_bart=modeling_bart)
    models = _mod("transformers.models", auto=auto, bart=bart)
    utils_logging = _mod("transformers.utils.logging", set_verbosity_error=lambda *a, **k: None)
    utils = _mod("transformers.utils", logging=utils_logging)
    root = _mod("transformers", models=models, utils=utils, GenerationConfig=MagicMock())

    # The root RESOLVES through to the submodules on every access, rather than holding its own
    # copies. That is what the real package does (``_LazyModule`` re-exports from
    # ``transformers.models.auto.*``), and it is load-bearing here: the tests patch the SUBMODULE
    # path, while the code under test does ``from transformers import AutoTokenizer``. Snapshot the
    # objects onto the root instead and the two would be different mocks — the code would call one,
    # the test would assert on the other, and it would look like the patch simply did not work.
    _EXPORTS = {
        "AutoTokenizer": (tokenization_auto, "AutoTokenizer"),
        "AutoModelForSeq2SeqLM": (modeling_auto, "AutoModelForSeq2SeqLM"),
        "BartForConditionalGeneration": (modeling_bart, "BartForConditionalGeneration"),
    }

    def _root_getattr(name: str) -> Any:
        if name in _EXPORTS:
            mod, attr = _EXPORTS[name]
            return getattr(mod, attr)
        raise AttributeError(f"stub transformers has no attribute {name!r}")

    root.__getattr__ = _root_getattr  # type: ignore[method-assign]  # PEP 562
    return {
        "transformers": root,
        "transformers.models": models,
        "transformers.models.auto": auto,
        "transformers.models.auto.tokenization_auto": tokenization_auto,
        "transformers.models.auto.modeling_auto": modeling_auto,
        "transformers.models.bart": bart,
        "transformers.models.bart.modeling_bart": modeling_bart,
        "transformers.utils": utils,
        "transformers.utils.logging": utils_logging,
    }


def stub_torch(*, cuda: bool = False, mps: bool = False) -> dict[str, Any]:
    """A ``sys.modules`` overlay standing in for ``torch``, or ``{}`` when it is installed.

    ``summarizer._resolve_device`` documents its own contract: *"Explicit device: no torch import
    (unit tests and minimal [dev] envs)"* — it only imports torch when asked to AUTO-DETECT. Tests
    that pass ``device=None`` are therefore choosing the auto-detect path on purpose, and a
    ``@patch("...summarizer.torch", create=True)`` cannot reach it: the function does a LOCAL
    ``import torch``, which reads ``sys.modules``, not a module attribute.

    Defaults to no accelerators, i.e. the ``cpu`` branch — the deterministic answer, and the one a
    CI runner gives anyway.

    Empty when torch is installed, so CI is untouched.
    """
    if importlib.util.find_spec("torch") is not None:
        return {}
    mod = ModuleType("torch")
    mod.cuda = SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    mod.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: mps)
    )
    mod.device = lambda name: f"device({name})"  # type: ignore[attr-defined]
    mod.no_grad = nullcontext  # type: ignore[attr-defined]
    return {"torch": mod}


#: The committed synthetic corpus. Its ``search/`` index is NOT committed (.gitignore) — it is
#: generated on demand by :func:`app_validation_search_index` below.
APP_VALIDATION_CORPUS = (
    Path(__file__).resolve().parents[1] / "fixtures" / "app-validation-corpus" / "v3"
)


@pytest.fixture(scope="session")
def app_validation_search_index() -> Path:
    """The app-validation corpus, guaranteed to carry a two-tier search index.

    ``search/metadata.json`` and ``search/lance_index`` are gitignored: a fresh checkout does not
    have them, and ``cli index-two-tier`` writes them into the committed fixture tree. Two test
    modules used to each build it themselves, and a third read it without saying so. Under xdist
    that is a race in both directions, and it bit twice on 2026-08-21:

      * READ vs WRITE — nightly ran the builder on gw0 and
        ``TestPoolIsInterestAware::test_index_maps_tokens_to_episodes`` on gw1 at the same 69%.
        The reader saw no sidecar, ``interest_episode_index`` returned ``{}`` (its documented
        answer for a corpus with no index), and the test failed. In the PR run the same two
        landed on ONE worker, builder first, so it passed. Nothing about the code differed.
      * WRITE vs WRITE — the two builders were never serialised against each other either, so
        two workers could run ``index-two-tier`` into the same LanceDB directory at once.

    So: one builder, session-scoped, behind a cross-process lock keyed on the corpus path. Every
    consumer requests this fixture and therefore SAYS it needs the index. The lock lives in the
    temp dir, not in the repo, so no build artifact needs a new ignore rule.

    Skips — never fails — when the index cannot be built: the embedding model is genuinely absent
    in model-less CI, and that is an environment fact, not a defect in the code under test.
    """
    import hashlib
    import json
    import logging
    import shutil
    import tempfile

    from filelock import FileLock, Timeout

    lance = APP_VALIDATION_CORPUS / "search" / "lance_index"
    sidecar = APP_VALIDATION_CORPUS / "search" / "metadata.json"

    def _present() -> bool:
        """Present AND built by a real embedder — a poisoned cache must not be served.

        2026-08-29 (#1874 W5): a run that stubbed the embedder left a 3-dimensional index in
        this on-disk cache. Every later run in the same checkout served it, because the check
        here saw a directory and skipped the rebuild, and eleven search tests failed with a
        dimension mismatch that read exactly like a code regression — an hour to trace to an
        ARTIFACT. A cache that cannot detect its own poisoning is a trap for whoever hits it
        next, so a stub-sized embedding dimension now triggers a rebuild instead.
        """
        if not (lance.is_dir() and sidecar.is_file()):
            return False
        meta_path = lance / "index_meta.json"
        if not meta_path.is_file():
            return True  # older index shape — leave previous behaviour alone
        try:
            dim = json.loads(meta_path.read_text(encoding="utf-8")).get("embed_dim")
        except (OSError, ValueError):
            return False
        if isinstance(dim, int) and 0 < dim < 32:
            # READ-ONLY here. The purge is destructive and this check runs BEFORE the
            # FileLock, so doing it inline let an xdist sibling observe a half-deleted index
            # (directory still present, index_meta.json already gone) and take the
            # "older index shape" branch — serving exactly the artifact we were removing.
            return False
        return True

    if _present():
        return APP_VALIDATION_CORPUS

    digest = hashlib.sha256(str(APP_VALIDATION_CORPUS).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"podcast-scraper-fixture-index-{digest}.lock"
    try:
        # Generous: whoever holds it is embedding 36 episodes, and the waiters must not give up
        # and start a second build — that is the write/write race this exists to prevent.
        with FileLock(str(lock_path), timeout=900):
            if _present():
                return APP_VALIDATION_CORPUS
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                from podcast_scraper.cli import main as cli_main

                # Under the lock, so no sibling can observe a half-deleted index: drop a
                # poisoned one wholesale before rebuilding.
                meta_path = lance / "index_meta.json"
                if meta_path.is_file():
                    try:
                        poisoned_dim = json.loads(meta_path.read_text(encoding="utf-8")).get(
                            "embed_dim"
                        )
                    except (OSError, ValueError):
                        poisoned_dim = None
                    if isinstance(poisoned_dim, int) and 0 < poisoned_dim < 32:
                        logging.getLogger(__name__).warning(
                            "fixture index at %s was built by a STUB embedder (embed_dim=%s); "
                            "rebuilding instead of serving a poisoned artifact",
                            lance,
                            poisoned_dim,
                        )
                        shutil.rmtree(lance, ignore_errors=True)
                        sidecar.unlink(missing_ok=True)

                # Clear the incremental ledger before ANY rebuild. With
                # episode_fingerprints.json in place the indexer sees all 36 episodes as
                # already indexed and writes nothing, so a checkout whose index was removed
                # (or purged as poisoned, above) would skip these tests forever. Same trap as
                # 90d092285 for prod reindex; verified here by watching a rebuild produce an
                # empty directory until the ledger was removed.
                (APP_VALIDATION_CORPUS / "search" / "episode_fingerprints.json").unlink(
                    missing_ok=True
                )
                rc = cli_main(["index-two-tier", "--output-dir", str(APP_VALIDATION_CORPUS)])
            except Exception as exc:  # noqa: BLE001 — any build failure => skip, not fail
                pytest.skip(f"could not build search index (embedding model offline?): {exc}")
            if rc not in (0, None) or not _present():
                pytest.skip("search index build produced no lance_index/metadata.json")
    except Timeout:  # pragma: no cover — only under a wedged concurrent build
        pytest.skip(f"timed out waiting for another worker to build {APP_VALIDATION_CORPUS}")
    return APP_VALIDATION_CORPUS


class _NoBackoffSleep:
    """Stand-in for the ``time`` module used by ``retry_with_metrics``: ``sleep`` is a no-op;
    everything else (``time()``, ``monotonic()``, …) delegates to the real module."""

    def sleep(self, *_args, **_kwargs):  # retry-backoff no-op
        return None

    def __getattr__(self, name):
        return getattr(_real_time, name)


@pytest.fixture(autouse=True)
def _instant_retry_backoff():
    """No-op retry-backoff sleeps across the whole integration suite so tests don't
    wall-clock-wait real exponential backoff.

    Provider error / resilience tests mock the upstream API to fail, then run
    ``retry_with_metrics()``'s real ``time.sleep()`` schedule — e.g. Gemini's *production*
    config is 6 retries @ up to 60s backoff, so one ``test_summarize_api_error`` used to sit
    for ~123s (``docs/wip/nightly-test-time-analysis.md``). That wait tests nothing here: the
    retry SCHEDULE (delays/jitter/cap/exhaustion) is asserted with a mocked clock in
    ``tests/unit/.../test_provider_metrics.py``. Here we only need the retry PATH to run
    (N attempts → correct terminal error), which it still does — instantly.

    Module-scoped on the retry util's own ``time`` reference, so global ``time.sleep`` and
    every other module are untouched. One fixture → ALL providers + integration resilience
    tests, no per-directory gaps.
    """
    from podcast_scraper.utils import provider_metrics

    with patch.object(provider_metrics, "time", _NoBackoffSleep()):
        yield
