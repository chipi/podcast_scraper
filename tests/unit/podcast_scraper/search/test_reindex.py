"""Isolation contract for corpus (re)indexing.

Indexing runs in a subprocess so a NATIVE crash in the Arrow/LanceDB layer (pyarrow's mimalloc
segfaulting in ``mi_thread_init`` under pthread-TLS pressure — observed at the end of a full run)
cannot take the whole pipeline down. These tests pin the contract: any subprocess outcome other
than a clean exit — a Python error (exit 1), a segfault (returncode -11), or a timeout — resolves
to ``False`` (non-fatal), never an exception in the parent.
"""

from __future__ import annotations

import signal
import subprocess
from types import SimpleNamespace

import pytest

from podcast_scraper.search import reindex

pytestmark = pytest.mark.unit


def _fake_cfg():
    # run_index_in_subprocess only needs .model_dump(mode="json") to serialize the config.
    return SimpleNamespace(model_dump=lambda mode=None: {"vector_search": True})


def test_arrow_memory_pool_pinned_to_system_on_import() -> None:
    # The cause fix: the module forces Arrow off mimalloc before pyarrow is imported.
    import os

    assert os.environ.get("ARROW_DEFAULT_MEMORY_POOL") == "system"


def test_clean_exit_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reindex.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert reindex.run_index_in_subprocess("/corpus", _fake_cfg()) is True


def test_nonzero_exit_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reindex.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    # a Python error in the child must NOT raise here
    assert reindex.run_index_in_subprocess("/corpus", _fake_cfg()) is False


def test_segfault_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    # THE isolation guarantee: a native crash (returncode == -SIGSEGV) is a non-fatal False,
    # never propagated as a process-killing signal in the parent.
    monkeypatch.setattr(
        reindex.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=-signal.SIGSEGV),
    )
    assert reindex.run_index_in_subprocess("/corpus", _fake_cfg()) is False


def test_timeout_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="reindex", timeout=1.0)

    monkeypatch.setattr(reindex.subprocess, "run", _raise_timeout)
    assert reindex.run_index_in_subprocess("/corpus", _fake_cfg(), timeout=1.0) is False


def test_temp_config_is_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _capture(argv, **k):
        # the serialized config path is the arg after --config-json
        idx = argv.index("--config-json")
        seen["cfg_path"] = argv[idx + 1]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reindex.subprocess, "run", _capture)
    reindex.run_index_in_subprocess("/corpus", _fake_cfg())
    import os

    assert not os.path.exists(seen["cfg_path"]), "temp reindex config leaked"


def test_backbone_delta_crosses_the_subprocess_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC-118: the changed-relpaths list is written to a temp JSON the child argv names."""
    import json
    import os

    seen = {}

    def _capture(argv, **k):
        idx = argv.index("--backbone-changed-file")
        seen["delta_path"] = argv[idx + 1]
        seen["payload"] = json.loads(open(argv[idx + 1], encoding="utf-8").read())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reindex.subprocess, "run", _capture)
    reindex.run_index_in_subprocess(
        "/corpus",
        _fake_cfg(),
        backbone_changed_relpaths=["feeds/a/run_1/metadata/e2.metadata.json"],
    )
    assert seen["payload"] == ["feeds/a/run_1/metadata/e2.metadata.json"]
    assert not os.path.exists(seen["delta_path"]), "temp delta file leaked"


def test_no_delta_omits_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """No computed delta (None) must not invent an empty scope in the child."""
    seen = {}

    def _capture(argv, **k):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reindex.subprocess, "run", _capture)
    reindex.run_index_in_subprocess("/corpus", _fake_cfg())
    assert "--backbone-changed-file" not in seen["argv"]
