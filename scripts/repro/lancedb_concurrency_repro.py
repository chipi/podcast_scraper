"""Deterministic repro for the api SIGSEGV under concurrent LanceDB sync queries (#1205).

The api serves search from many threads (the digest route fans out ~8 concurrent hybrid searches).
LanceDB's SYNC query API is not safe for concurrent in-process queries — they race in its single
shared native ``background_loop`` and segfault the process. This harness reproduces exactly that,
with no app code: N threads run concurrent hybrid queries on one connection in a loop.

PLATFORM: the crash is in the **x86_64** native (AVX/SIMD) code path of the lance/pyarrow wheels.
It does NOT reproduce on arm64 (macOS or Linux) nor under x86_64 emulation on an arm64 host (the
emulator SIGILLs on the AVX instructions). Run this on a real x86_64 Linux host — e.g. the
``repro-lancedb-concurrency`` GitHub workflow, which runs on the x86_64 ubuntu runner.

Env knobs:
  THREADS (16)  ROUNDS (30)  QUERIES (64)  USE_LOCK (0/1)

USE_LOCK=1 serializes queries behind a single lock (the temp workaround in LanceDBBackend) and must
never crash. USE_LOCK=0 is the unfixed fan-out and is expected to SIGSEGV (exit 139) on x86_64. The
race is timing-dependent, so raise ROUNDS/THREADS if a given run happens not to crash.
"""

from __future__ import annotations

import os
import platform
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import lancedb
import pyarrow as pa
from lancedb.rerankers import RRFReranker

USE_LOCK = os.environ.get("USE_LOCK") == "1"
THREADS = int(os.environ.get("THREADS", "16"))
ROUNDS = int(os.environ.get("ROUNDS", "30"))
QUERIES = int(os.environ.get("QUERIES", "64"))
DIM = 4

_dir = tempfile.mkdtemp()
_db = lancedb.connect(_dir + "/lance")
_schema = pa.schema(
    [("id", pa.string()), ("text", pa.string()), ("embedding", pa.list_(pa.float32(), DIM))]
)
_tbl = _db.create_table("segments", schema=_schema)
_VEC = [0.1, 0.2, 0.3, 0.4]
for _i in range(400):  # per-doc adds -> many small fragments, like the pipeline indexer
    _tbl.add([{"id": f"s{_i}", "text": f"doc {_i} altman openai scaling", "embedding": _VEC}])
_tbl.create_fts_index("text", replace=True)

_lock = threading.Lock()
_reranker = RRFReranker()


def _query(_ignored: int) -> int:
    if USE_LOCK:
        _lock.acquire()
    try:
        table = _db.open_table("segments")
        req = table.search(query_type="hybrid").vector(_VEC).text("altman").rerank(_reranker)
        return len(list(req.limit(10).to_list()))
    finally:
        if USE_LOCK:
            _lock.release()


def main() -> None:
    print(
        f"platform={platform.machine()} lancedb={lancedb.__version__} pyarrow={pa.__version__} "
        f"USE_LOCK={USE_LOCK} threads={THREADS} rounds={ROUNDS} queries={QUERIES}",
        flush=True,
    )
    for rnd in range(ROUNDS):
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            list(ex.map(_query, range(QUERIES)))
        print(f"round {rnd} ok", flush=True)
    print("=== NO CRASH ===", flush=True)


if __name__ == "__main__":
    main()
