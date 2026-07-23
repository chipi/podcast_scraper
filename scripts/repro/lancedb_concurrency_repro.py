"""Concurrent-LanceDB SIGSEGV guardrail (#1205) — validates the shipped fix.

The api serves search from many threads (the digest route fans out ~8 concurrent
hybrid searches). Before the ``0fe0854b`` fix, LanceDB's in-engine hybrid combine
(``_combine_hybrid_results`` / ``_normalize_scores`` via ``pyarrow.compute``)
segfaulted the api under this concurrent load.

The fix bypassed the native combine by routing every hybrid request through the
Python-side ``search_bm25`` + ``search_vector`` + ``rrf_fuse`` fan-out
(``retrieval.py``). This harness exercises exactly that path under 4×+ concurrent
load — the same shape as the production digest fan-out — and asserts:

- No process crash (no SIGSEGV).
- All requests return.
- Stdout ends with ``=== NO CRASH ===`` (a stable string the pytest wrapper
  greps for).

PLATFORM caveat (from #1205 root-cause investigation, `62c049e5` handover):
the crash is in the **x86_64** native (AVX/SIMD) code path of the lance/pyarrow
wheels. It does NOT reproduce on arm64 (macOS or Linux) nor under x86_64 emulation
on an arm64 host — the emulator SIGILLs on the AVX instructions. So this repro
runs green on any arch, but only earns its keep as a real regression signal on
x86_64. The pytest wrapper skips the assertion on arm64 with a clear reason;
the CI job on the x86_64 ubuntu runner is where the guardrail bites.

Env knobs:
  THREADS (16)  ROUNDS (10)  QUERIES (32)

Exit codes:
  0    — clean; ``=== NO CRASH ===`` printed.
  139  — SIGSEGV (128 + SIGSEGV=11); the #1205 regression fired.
  other — non-crash error; inspect stdout.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Cheap defense: never invoke the LanceDB native reranker path from this harness.
# The whole point is to prove RetrievalLayer.retrieve() (Python-side fan-out) is
# safe. If someone ever wires a hybrid combine back in, this file is a canary.

THREADS = int(os.environ.get("THREADS", "16"))
ROUNDS = int(os.environ.get("ROUNDS", "10"))
QUERIES = int(os.environ.get("QUERIES", "32"))
DIM = 32  # small enough for tests, matches embedding shape used in unit fixtures


def _seed(embed_dim: int, i: int) -> list[float]:
    """Deterministic pseudo-embedding — real embeddings not needed to trigger the crash."""
    return [((i * 7 + k * 11) % 100) / 100.0 for k in range(embed_dim)]


def _build_backend(path: str):
    """Build a small LanceDB backend fixture with enough rows to trigger the concurrency race."""
    from podcast_scraper.search.backend import SegmentDocument
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

    backend = LanceDBBackend(path, embed_dim=DIM)
    # 400 docs is what the original 62c049e5 repro used — per-doc adds produce many small
    # fragments, matching the shape the pipeline indexer produces.
    docs = [
        SegmentDocument(
            id=f"s{i}",
            text=f"doc {i} altman openai scaling laws regulation compute governance",
            show_id="show:s1",
            episode_id="episode:e1",
            start_time=float(i),
            end_time=float(i + 1),
            embedding=_seed(DIM, i),
            publish_date="2026-01-01",
        )
        for i in range(400)
    ]
    backend.upsert_segments(docs)
    # FTS + vector indices — search_bm25 needs the inverted index; search_vector
    # runs brute-force below _MIN_VECTOR_INDEX_ROWS (also fine).
    backend.create_indices()
    return backend


def _query(backend, _ignored: int) -> int:
    """One hybrid query via the shipped Python-side fan-out (RetrievalLayer.retrieve)."""
    from podcast_scraper.search.retrieval import RetrievalLayer

    layer = RetrievalLayer(backend)
    results = layer.retrieve(
        text="altman scaling",
        embedding=_seed(DIM, 0),
        k=10,
        signals="hybrid",
    )
    return len(results)


def main() -> int:
    print(
        f"platform={platform.machine()} python={sys.version.split()[0]} "
        f"threads={THREADS} rounds={ROUNDS} queries={QUERIES}",
        flush=True,
    )

    try:
        import lancedb

        print(f"lancedb={lancedb.__version__}", flush=True)
    except Exception as e:  # pragma: no cover - defensive
        print(f"lancedb import failed: {e!r}", file=sys.stderr, flush=True)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        backend = _build_backend(os.path.join(tmp, "lance"))
        for rnd in range(ROUNDS):
            with ThreadPoolExecutor(max_workers=THREADS) as ex:
                _ = list(ex.map(lambda i: _query(backend, i), range(QUERIES)))
            print(f"round {rnd} ok", flush=True)

    print("=== NO CRASH ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
