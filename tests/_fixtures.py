"""Fixture path resolution with version awareness.

Test fixtures are stored under ``tests/fixtures/<subdir>/<version>/``. The
default version is read from ``tests/fixtures/FIXTURES_VERSION`` (single line,
e.g. ``v1`` or ``v2``). Tests that need a specific historical version pass
``version=`` explicitly; everything else picks up the current default.

Only ``transcripts`` and ``audio`` are version-split today. ``goldens``,
``viewer-validation-corpus``, etc. will be split when v2 goldens are
regenerated (Phase 7 of the v2 fixtures rebuild).
"""

from __future__ import annotations

from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_ROOT = _TESTS_DIR / "fixtures"

VERSIONED_SUBDIRS: frozenset[str] = frozenset({"transcripts", "audio", "viewer-validation-corpus"})


def _read_default_version() -> str:
    version_file = FIXTURES_ROOT / "FIXTURES_VERSION"
    return version_file.read_text(encoding="utf-8").strip()


DEFAULT_FIXTURE_VERSION: str = _read_default_version()


def fixtures_dir(subdir: str, version: str | None = None) -> Path:
    """Resolve a fixture subdirectory.

    For versioned subdirs (``transcripts``, ``audio``) this returns
    ``FIXTURES_ROOT/<subdir>/<version>``. For unversioned subdirs it returns
    ``FIXTURES_ROOT/<subdir>`` and ignores ``version``.
    """
    base = FIXTURES_ROOT / subdir
    if subdir in VERSIONED_SUBDIRS:
        return base / (version or DEFAULT_FIXTURE_VERSION)
    return base


# Placeholder DGX/tailnet host for tests — NEVER the operator's real tailnet suffix. Production
# resolvers read the host from env (``DGX_TAILNET_FQDN`` → ``vllm_chat._resolve_vllm_base``,
# ``scripts/ops/resolve_dgx_tailnet_host.sh``); tests inject THIS fake value instead of hardcoding
# a real FQDN. Keeps a ``dgx-`` prefix + ``.ts.net`` suffix so ``_classify_inference_target`` still
# yields ``"dgx-vllm"``.
TEST_DGX_FQDN: str = "dgx-test.tailnet.ts.net"
TEST_DGX_VLLM_BASE_URL: str = f"http://{TEST_DGX_FQDN}:8003/v1"
