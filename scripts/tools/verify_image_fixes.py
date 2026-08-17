"""Verify the #1657 fixes INSIDE the built pipeline image, where production actually runs.

Every check here exists because the same thing passed from source and would not have been
caught. The 14-episode acceptance run executed from a git checkout with the full dev extras;
the production image is a different filesystem, different extras, and no git. A fix that only
works in the first environment is not a fix.

Run inside the container:
    docker run --rm podcast-scraper-stack-pipeline-llm:latest \
        python /app/scripts/tools/verify_image_fixes.py

Exit code is the number of failed checks (0 = all good), so it can gate a deploy.

READ THIS BEFORE DEBUGGING A SILENT RUN. On a host that relays the Docker socket (e.g. a
colima/socat bridge), container **stdout is dropped intermittently** — the container runs, exits
0, and prints nothing. It is not deterministic: the identical command can print once and go
silent the next time. On 2026-08-16 that cost an hour and produced a confident, entirely wrong
diagnosis ("importing torch silently kills the interpreter in the image", complete with a theory
about qemu and AVX512). torch was fine. The output was being lost.

If a run prints nothing, do NOT conclude the process died. Write results to a file inside the
container and copy them out, which is reliable here:

    cid=$(docker create <image> python /tmp/rv.py)
    docker cp verify_image_fixes.py "$cid:/tmp/verify.py"
    docker cp run_verify.py        "$cid:/tmp/rv.py"   # redirects stdout into /tmp/out.txt
    docker start -a "$cid" >/dev/null 2>&1
    docker cp "$cid:/tmp/out.txt" ./result.txt

Note also that bind mounts (-v) and stdin (-i) are unreliable through the same bridge; docker cp
into a created container works.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from typing import Callable, List, Tuple

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    def decorator(fn: Callable[[], str]) -> Callable:
        try:
            detail = fn()
            RESULTS.append((name, True, detail))
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc) or "assertion failed"))
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
            RESULTS.append(
                (name, False, f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}")
            )
        return fn

    return decorator


@check("#30 git_sha is present and non-null")
def _git_sha() -> str:
    """The whole point of #30. From source this passes trivially; here it is the real test."""
    from podcast_scraper.workflow.processing_manifest import git_ground_truth

    truth = git_ground_truth()
    sha = truth.get("git_sha")
    assert sha, (
        "git_sha is None INSIDE THE IMAGE — the build arg did not reach the runtime env. "
        f"PODCAST_GIT_SHA={os.environ.get('PODCAST_GIT_SHA')!r}. Every manifest this image "
        "writes would carry null provenance (ADR-132)."
    )
    assert len(sha) == 7, f"expected a 7-char short SHA, got {sha!r}"
    return f"git_sha={sha} git_dirty={truth.get('git_dirty')}"


@check("#30 git is genuinely absent (so the env var is doing the work)")
def _git_absent() -> str:
    """If git IS installed here, the check above proves nothing about the build arg."""
    git_path = shutil.which("git")
    if git_path:
        return f"NOTE: git IS present at {git_path} — the check above may be passing via the probe"
    return "git binary absent, as expected: the value can only have come from the build arg"


@check("#26 ffmpeg is installed (else every episode fails hard now)")
def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    assert path, (
        "ffmpeg missing from the image. Since #26 this is FATAL, not degraded — every episode "
        "would raise FFmpegUnavailableError. Dockerfile installs it in the runtime stage."
    )
    out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=30)
    return f"{path}: {out.stdout.splitlines()[0] if out.stdout else 'version unknown'}"


@check("#26 the hard-failure path is the code that shipped")
def _ffmpeg_fatal() -> str:
    from podcast_scraper.preprocessing.audio import factory

    assert hasattr(
        factory, "FFmpegUnavailableError"
    ), "FFmpegUnavailableError absent — the image predates #26 and still silently degrades"
    return "FFmpegUnavailableError present"


@check("#8 known_models.yaml is bundled IN THE WHEEL")
def _packaged_registry() -> str:
    """The packaging fix specifically — not merely "the allowlist loads".

    ``_resolve_path`` searches cwd upward for ``config/known_models.yaml`` and then
    ``/app/config/known_models.yaml`` before falling back to the wheel-bundled copy. Both of
    those exist in this image, so a check that only asserts "the allowlist is non-empty" would
    pass even with the packaging fix reverted. ``_bundled_path()`` is asked directly.
    """
    from podcast_scraper.providers import known_models

    bundled = known_models._bundled_path()
    assert bundled is not None and bundled.is_file(), (
        "podcast_scraper/data/known_models.yaml is NOT in the wheel — the packaging fix is "
        "missing. The container path masks this until someone runs the wheel elsewhere."
    )

    governed, by_provider = known_models._load()
    assert governed, "allowlist loaded but EMPTY — model validation would be silently disabled"
    resolved = known_models._resolve_path()
    total = sum(len(v) for v in by_provider.values())
    return (
        f"bundled={bundled} | resolved_from={resolved} | "
        f"{len(governed)} governed providers, {total} model ids"
    )


@check("#27 dedupe threshold is the measured 0.90")
def _dedupe_threshold() -> str:
    from podcast_scraper.gi.chunked_extraction import DEFAULT_LEXICAL_DEDUPE_THRESHOLD

    assert (
        0.86 <= DEFAULT_LEXICAL_DEDUPE_THRESHOLD <= 0.93
    ), f"threshold {DEFAULT_LEXICAL_DEDUPE_THRESHOLD} is outside the measured gap"
    return f"DEFAULT_LEXICAL_DEDUPE_THRESHOLD={DEFAULT_LEXICAL_DEDUPE_THRESHOLD}"


@check("#1657 dedupe RUNS here, whichever tier is available")
def _dedupe_runs() -> str:
    """The original defect: dedup was embedding-only and silently no-opped without the module.

    Reports WHICH tier did the work rather than assuming. The llm image installs
    ``.[llm,search,...]`` and ``[search]`` carries sentence-transformers, so the embedding tier
    is expected to be present here — unlike the macOS dev box, where no ML wheels exist.
    """
    from podcast_scraper.gi.chunked_extraction import dedupe

    pair = [
        "Ryan Greenblatt argues that ML research is less deep than math, so there is less "
        "reliance on individual deep experts.",
        "Ryan Greenblatt says that ML research is less deep than math, so there is less "
        "reliance on individual deep experts.",
    ]
    kept = dedupe(pair, threshold=0.72)
    assert len(kept) == 1, (
        f"a real verb-swap duplicate survived dedup on the production image (kept {len(kept)}) "
        "— dedup is not running here, which is exactly the #1657 defect"
    )

    try:
        import sentence_transformers

        tier = (
            f"sentence-transformers {sentence_transformers.__version__} present "
            "(expected: [search] carries it) — embedding tier available on top of lexical"
        )
    except ImportError:
        tier = "sentence-transformers absent — lexical tier alone did the work, as designed"
    return f"duplicate collapsed 2 -> 1; {tier}"


@check("#19 the placeholder insight is gone from the code")
def _no_placeholder() -> str:
    from podcast_scraper.gi import pipeline

    assert not hasattr(pipeline, "_STUB_INSIGHT_TEXT"), "_STUB_INSIGHT_TEXT still present"
    return "no _STUB_INSIGHT_TEXT in gi.pipeline"


@check("#29 the placeholder repair gate is importable")
def _placeholder_gate() -> str:
    from podcast_scraper.gi.corpus import check_corpus_for_placeholders

    assert callable(check_corpus_for_placeholders)
    return "check_corpus_for_placeholders available (make corpus-placeholder-check)"


@check("#23 the failover pre-flight is importable and wired")
def _preflight() -> str:
    from podcast_scraper.summarization.fallback import (
        log_fallback_chain_preflight,
        preflight_fallback_chain,
    )

    assert callable(preflight_fallback_chain) and callable(log_fallback_chain_preflight)
    import inspect

    from podcast_scraper.workflow import orchestration

    src = inspect.getsource(orchestration)
    assert "log_fallback_chain_preflight" in src, "pre-flight not called from orchestration"
    return "pre-flight present and called at provider creation"


@check("#22 preprocessing ledger reasons are present")
def _ledger_reasons() -> str:
    import inspect

    from podcast_scraper.workflow import episode_processor

    src = inspect.getsource(episode_processor)
    for slug in ("preprocessing_disabled", "media_file_missing", "ffmpeg_unavailable"):
        assert slug in src, f"ledger reason {slug!r} missing"
    return "preprocessing_disabled / media_file_missing / ffmpeg_unavailable all present"


@check("every provider module imports (otel/mistralai version conflict check)")
def _providers_import() -> str:
    """The build logs a real conflict; this asks whether it actually breaks anything.

        ERROR: pip's dependency resolver ...
        mistralai 2.4.5 requires opentelemetry-semantic-conventions<0.61,>=0.60b1,
        but you have opentelemetry-semantic-conventions 0.65b0 which is incompatible.

    ``opentelemetry-bootstrap -a install`` upgrades the otel stack AFTER mistralai is installed,
    so the pin is violated in the finished image. pip prints it and carries on, which means the
    only way to know whether it matters is to import the thing and see.

    A provider that fails to import is not a slow degradation — it is a stage that cannot run.
    """
    import importlib

    modules = [
        "podcast_scraper.providers.openai.openai_provider",
        "podcast_scraper.providers.litellm.litellm_provider",
        "podcast_scraper.providers.deepseek.deepseek_provider",
        "podcast_scraper.providers.anthropic.anthropic_provider",
        "podcast_scraper.providers.gemini.gemini_provider",
        "podcast_scraper.providers.mistral.mistral_provider",
        "podcast_scraper.providers.grok.grok_provider",
        "podcast_scraper.providers.deepgram",
    ]
    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc.__class__.__name__}: {exc}")
    assert not failures, "provider modules failed to import:\n  " + "\n  ".join(failures)
    return f"all {len(modules)} provider modules import cleanly"


@check("the otel semconv pin conflict is recorded, not hidden")
def _otel_conflict() -> str:
    """Report the installed versions so the conflict is visible even when nothing breaks."""
    from importlib import metadata

    def _v(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except Exception:  # noqa: BLE001
            return "(absent)"

    semconv = _v("opentelemetry-semantic-conventions")
    mistral = _v("mistralai")
    return f"mistralai={mistral} opentelemetry-semantic-conventions={semconv} (pin wants <0.61)"


def main() -> int:
    print("=" * 78)
    print("IN-IMAGE VERIFICATION — the environment production actually runs")
    print("=" * 78)
    print(f"python   : {sys.version.split()[0]}")
    print(f"PODCAST_GIT_SHA={os.environ.get('PODCAST_GIT_SHA', '(unset)')}")
    print(f"PODCAST_GIT_BRANCH={os.environ.get('PODCAST_GIT_BRANCH', '(unset)')}")
    print(f"PODCAST_GIT_DIRTY={os.environ.get('PODCAST_GIT_DIRTY', '(unset)')}")
    print()

    failed = 0
    for name, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}")
        for line in str(detail).splitlines():
            print(f"        {line}")
    print()
    print("=" * 78)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} passed, {failed} FAILED")
    print("=" * 78)
    return failed


if __name__ == "__main__":
    sys.exit(main())
