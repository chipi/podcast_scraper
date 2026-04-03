"""Minimal in-memory FAISS stand-in for unit tests when ``faiss-cpu`` is not installed (CI .[dev]).

Mimics only the subset of the Python ``faiss`` API used by ``FaissVectorStore``. When the real
``faiss`` package is present, tests use it instead (see ``conftest.py``).
"""

from __future__ import annotations

import pickle
from typing import Any, cast, List

import numpy as np

METRIC_INNER_PRODUCT = 1


def normalize_L2(x: np.ndarray) -> None:
    """In-place L2 row normalization (same contract as ``faiss.normalize_L2``)."""
    if x.size == 0:
        return
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0, 1.0, norms)
    x[:] = (x / norms).astype(np.float32)


def downcast_index(index: Any) -> Any:
    return index


class _IdMap:
    def __init__(self, owner: "IndexIDMap") -> None:
        self._owner = owner

    def at(self, i: int) -> int:
        return int(self._owner._ids[i])


class IndexFlatIP:
    __name__ = "IndexFlatIP"

    def __init__(self, d: int) -> None:
        self.d = int(d)
        self._xb = np.zeros((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._xb.shape[0])

    def reconstruct(self, i: int) -> np.ndarray:
        row = np.asarray(self._xb[int(i)], dtype=np.float32).copy()
        return cast(np.ndarray, row)


class IndexIVFFlat:
    __name__ = "IndexIVFFlat"

    def __init__(self, quantizer: Any, d: int, nlist: int, metric: int) -> None:
        _ = quantizer
        _ = metric
        self.d = int(d)
        self.nlist = int(nlist)
        self.nprobe = 1
        self._xb = np.zeros((0, self.d), dtype=np.float32)

    def train(self, xb: np.ndarray) -> None:
        _ = xb

    @property
    def ntotal(self) -> int:
        return int(self._xb.shape[0])

    def reconstruct(self, i: int) -> np.ndarray:
        row = np.asarray(self._xb[int(i)], dtype=np.float32).copy()
        return cast(np.ndarray, row)


class IndexIVFPQ(IndexIVFFlat):
    __name__ = "IndexIVFPQ"

    def __init__(self, quantizer: Any, d: int, nlist: int, m: int, nbits: int) -> None:
        _ = m
        _ = nbits
        super().__init__(quantizer, d, nlist, METRIC_INNER_PRODUCT)


class IndexIDMap:
    __name__ = "IndexIDMap"

    def __init__(self, inner: Any) -> None:
        self.index = inner
        self._vecs: List[np.ndarray] = []
        self._ids: List[int] = []
        self.id_map = _IdMap(self)

    @property
    def d(self) -> int:
        return int(self.index.d)

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    def _sync_inner(self) -> None:
        inner = self.index
        if not self._vecs:
            inner._xb = np.zeros((0, int(inner.d)), dtype=np.float32)
        else:
            inner._xb = np.stack(self._vecs, axis=0).astype(np.float32)

    def train(self, xb: np.ndarray) -> None:
        if hasattr(self.index, "train"):
            self.index.train(xb)

    def add_with_ids(self, mat: np.ndarray, id_arr: np.ndarray) -> None:
        m = np.asarray(mat, dtype=np.float32)
        ids = np.asarray(id_arr, dtype=np.int64)
        for i in range(m.shape[0]):
            self._vecs.append(m[i].copy())
            self._ids.append(int(ids[i]))
        self._sync_inner()

    def remove_ids(self, arr: np.ndarray) -> None:
        rm = {int(x) for x in np.asarray(arr, dtype=np.int64).flat}
        nv: List[np.ndarray] = []
        ni: List[int] = []
        for v, fid in zip(self._vecs, self._ids):
            if fid not in rm:
                nv.append(v)
                ni.append(fid)
        self._vecs, self._ids = nv, ni
        self._sync_inner()

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if not self._ids:
            return (
                np.zeros((1, k), dtype=np.float32),
                np.full((1, k), -1, dtype=np.int64),
            )
        mat = np.stack(self._vecs, axis=0).astype(np.float32)
        qv = q[0].astype(np.float32)
        scores = mat @ qv
        n = len(scores)
        m = min(k, n)
        order = np.argsort(-scores)[:m]
        out_s = np.zeros(k, dtype=np.float32)
        out_i = np.full(k, -1, dtype=np.int64)
        for t, j in enumerate(order):
            out_s[t] = float(scores[int(j)])
            out_i[t] = int(self._ids[int(j)])
        return out_s.reshape(1, -1), out_i.reshape(1, -1)


def write_index(index: Any, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(index, fh, protocol=4)


def read_index(path: str) -> Any:
    with open(path, "rb") as fh:
        # Test-only FAISS stub: index files are produced by write_index above in unit tests.
        return pickle.load(fh)  # nosec B301
