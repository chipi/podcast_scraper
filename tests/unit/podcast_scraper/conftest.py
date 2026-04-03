"""FAISS + pytest-xdist: avoid native OpenMP/BLAS thread storms that crash workers."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional

import pytest


@pytest.fixture(autouse=True)
def _stub_faiss_if_missing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """When ``faiss-cpu`` is not installed (CI ``.[dev]``), inject a tiny in-memory stand-in.

    Unit tests for ``FaissVectorStore`` must not require the optional ``[ml]`` extra.
    If the real ``faiss`` package exists, it is used unchanged.
    """
    if importlib.util.find_spec("faiss") is not None:
        yield
        return
    fake_path = Path(__file__).with_name("fake_faiss_for_unit_tests.py")
    spec = importlib.util.spec_from_file_location("_podcast_fake_faiss", fake_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "faiss", mod)
    yield


@pytest.fixture(autouse=True)
def _single_thread_blas_for_faiss() -> Iterator[None]:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    previous: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    yield
    for k, val in previous.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
