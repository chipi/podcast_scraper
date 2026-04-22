#!/usr/bin/env python3
"""Layer 2 validation for #650: single real Whisper call (~$0.003).

Calls OpenAI Whisper once against a short fixture audio (~21 s) and asserts:

1. ``verbose_json`` response contains a ``.duration`` value (Finding 19).
2. Cost computation matches ``duration_seconds / 60 * $0.006`` within
   floating-point tolerance (Finding 17 wiring through ProviderCallMetrics).
3. The per-stage aggregate ``pipeline_metrics.llm_transcription_cost_usd``
   is populated end-to-end (Finding 20 wiring).

Cost: roughly $0.003 per run. Runtime: a few seconds.

Usage
-----

Requires ``OPENAI_API_KEY`` in the environment (``.env`` is auto-loaded by
``dotenv`` via the provider chain).

    python scripts/validate/validate_whisper_cost_layer2.py

Exit code 0 on success, 1 on any assertion failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env")

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "audio" / "p01_multi_e01.mp3"
WHISPER_PRICE_PER_MINUTE = 0.006  # OpenAI whisper-1


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY not set.", file=sys.stderr)
        return 1
    if not FIXTURE.exists():
        print(f"FAIL: fixture not found: {FIXTURE}", file=sys.stderr)
        return 1

    from podcast_scraper import config
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider
    from podcast_scraper.utils.provider_metrics import ProviderCallMetrics
    from podcast_scraper.workflow.metrics import Metrics

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        transcription_provider="openai",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_transcription_model="whisper-1",
        transcribe_missing=True,
    )
    provider = OpenAIProvider(cfg)
    provider.initialize()
    if provider.client is None:
        # initialize() doesn't always eagerly create the client; force it.
        import openai  # type: ignore[import-not-found]

        provider.client = openai.OpenAI(api_key=cfg.openai_api_key)
    provider._transcription_initialized = True

    pm = Metrics()
    cm = ProviderCallMetrics()

    print(f"Calling OpenAI Whisper on {FIXTURE.name} ...")
    try:
        result, elapsed = provider.transcribe_with_segments(
            str(FIXTURE),
            pipeline_metrics=pm,
            call_metrics=cm,
            # episode_duration_seconds intentionally left None so the code
            # relies on verbose_json.duration (not caller input).
        )
    except Exception as exc:
        print(f"FAIL: Whisper API call raised: {exc}", file=sys.stderr)
        return 1

    text = result.get("text") or ""
    print(f"  text: {text[:80]!r}...")
    print(f"  wall: {elapsed:.2f}s")
    print(f"  call_metrics.estimated_cost: ${cm.estimated_cost}")
    print(
        f"  pipeline.llm_transcription_audio_minutes: " f"{pm.llm_transcription_audio_minutes:.4f}"
    )
    print(f"  pipeline.llm_transcription_cost_usd:    " f"${pm.llm_transcription_cost_usd}")

    # 1. Duration (Finding 19): audio minutes must be non-zero and realistic
    #    for a ~21s fixture. The verbose_json response should populate it.
    assert pm.llm_transcription_audio_minutes > 0.0, (
        "FAIL: llm_transcription_audio_minutes is 0 — verbose_json.duration "
        "may not be propagating."
    )
    assert 0.2 < pm.llm_transcription_audio_minutes < 1.0, (
        f"FAIL: implausible duration "
        f"{pm.llm_transcription_audio_minutes:.4f} min for ~21s fixture."
    )

    # 2. Cost arithmetic (Finding 17): pipeline aggregate must match
    #    duration_min * $0.006 within float tolerance.
    expected_cost = pm.llm_transcription_audio_minutes * WHISPER_PRICE_PER_MINUTE
    actual_cost = pm.llm_transcription_cost_usd
    assert (
        abs(actual_cost - expected_cost) < 1e-6
    ), f"FAIL: pipeline cost ${actual_cost} != expected ${expected_cost}"

    # 3. Per-episode call_metrics mirrors the pipeline (Finding 20).
    assert cm.estimated_cost is not None, (
        "FAIL: call_metrics.estimated_cost not populated — " "set_cost() did not fire."
    )
    assert (
        abs((cm.estimated_cost or 0.0) - expected_cost) < 1e-6
    ), f"FAIL: call_metrics ${cm.estimated_cost} != expected ${expected_cost}"

    print()
    print("PASS: Layer 2 — Whisper verbose_json + cost wiring end-to-end.")
    print(f"      $ per call ≈ ${actual_cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
