"""FAISS + pytest-xdist: avoid native OpenMP/BLAS thread storms that crash workers."""

from __future__ import annotations

import os
from typing import Dict, Iterator, Optional

import pytest


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
