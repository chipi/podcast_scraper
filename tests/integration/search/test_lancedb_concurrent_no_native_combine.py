"""Regression test for #1205 SIGSEGV (LanceDB native hybrid combine).

Search v3 S0 (RFC-107 §S4, PRD-045 FR12.4): every PR that touches
``search/backends/``, ``search/retrieval.py``, ``search/hybrid_search.py``, or
``server/routes/corpus_digest.py`` must prove the concurrent-hybrid path stays
SIGSEGV-free.

The test runs ``scripts/repro/lancedb_concurrency_repro.py`` **as a subprocess**
so that a segfault kills that child, not pytest itself; we then read its exit
code and stdout.

Platform caveat (from #1205 root-cause analysis, `62c049e5` handover): the
crash is x86_64-native (AVX/SIMD) in the lance/pyarrow wheels. On arm64 the
harness runs green because the crashing path never executes — the assertion is
kept softer there (subprocess must not crash, but a green result on arm64 is
not proof the x86_64 path is fixed). The CI job on the x86_64 ubuntu runner is
where the guardrail truly bites; locally on macOS arm64 this test still catches
regressions of *our* Python-side fan-out (import errors, contract drift, dead
code paths) even though it cannot reproduce the native SIGSEGV itself.

Complements the fixed-string / AST guard in
``scripts/check/lint_search_v3_forbidden_imports.py`` (which catches
*compile-time* re-imports of ``_combine_hybrid_results`` / ``_normalize_scores``);
this test catches *runtime* regressions of the fan-out contract itself
(a legal-looking change to ``retrieval.py`` that reintroduces the concurrency
issue would slip through the lint but fail here).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPRO_SCRIPT = REPO_ROOT / "scripts" / "repro" / "lancedb_concurrency_repro.py"

# SIGSEGV exit code (128 + SIGSEGV=11). subprocess.returncode reports the negated
# signal on POSIX (-11), or the shell-style 139 depending on how the process died;
# treat either as evidence of the #1205 regression.
_SIGSEGV_EXITS = {-11, 139}

# Keep the test cheap in CI (rounds/threads/queries tuned for a ~10s runtime on x86_64;
# the race is timing-dependent — production digest fan-out is 8 concurrent, we push to 16
# to widen the window without ballooning wall-clock).
_ENV_KNOBS = {
    "THREADS": "16",
    "ROUNDS": "6",
    "QUERIES": "24",
}


@pytest.mark.integration
def test_repro_script_exists_and_is_the_shipped_fix_shape() -> None:
    """Guard against silent deletion of the repro script (as happened once in #1221)."""
    assert (
        REPRO_SCRIPT.exists()
    ), f"Repro script missing: {REPRO_SCRIPT}. RFC-107 §S4 requires this harness."
    text = REPRO_SCRIPT.read_text()
    # The shipped-fix shape uses RetrievalLayer.retrieve (the Python-side fan-out),
    # NOT LanceDB's native hybrid via search(query_type="hybrid"). If someone rewrites
    # this script to use the raw LanceDB hybrid API, this assertion catches it.
    assert "RetrievalLayer" in text, (
        "Repro script must exercise the shipped Python-side fan-out "
        "(RetrievalLayer.retrieve), not the raw LanceDB native hybrid."
    )
    assert (
        'query_type="hybrid"' not in text
    ), "Repro script must NOT call LanceDB's native hybrid search — that path is #1205."


@pytest.mark.integration
def test_concurrent_hybrid_search_no_sigsegv(tmp_path: Path) -> None:
    """Run the concurrent fan-out under load; assert no SIGSEGV and clean exit."""
    env = os.environ.copy()
    env.update(_ENV_KNOBS)
    # Ensure the subprocess picks up the repo's src/ (matches how the repro would run
    # in CI on the x86_64 ubuntu runner).
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Prevent LanceDB from grabbing home-dir caches that leak between test runs.
    env.setdefault("LANCE_LOG", "warn")

    result = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    rc = result.returncode

    # The #1205 signal: process killed by SIGSEGV. Any occurrence of this on any arch
    # means the concurrency fix is regressed.
    assert rc not in _SIGSEGV_EXITS, (
        f"#1205 regressed: subprocess killed by SIGSEGV (returncode={rc}).\n"
        f"stdout tail: {stdout[-2000:]!r}\n"
        f"stderr tail: {stderr[-2000:]!r}"
    )

    # Clean-exit guarantee: our fan-out must return normally.
    assert rc == 0, (
        f"Repro subprocess did not exit cleanly (returncode={rc}).\n"
        f"stdout tail: {stdout[-2000:]!r}\n"
        f"stderr tail: {stderr[-2000:]!r}"
    )

    # Stable sentinel the repro prints on completion — proves it ran the full ROUNDS,
    # not e.g. exited early on a caught exception.
    assert "=== NO CRASH ===" in stdout, (
        f"Repro subprocess did not reach completion marker.\n" f"stdout tail: {stdout[-2000:]!r}"
    )

    # Honest note on arch coverage: a green result on arm64 is not a proof that
    # the x86_64 native code path is safe (the crashing AVX code never runs). Print
    # the arch so the CI log makes the coverage story explicit for future auditors.
    arch = platform.machine()
    print(f"[test_lancedb_concurrent_no_native_combine] arch={arch} rc={rc}")
