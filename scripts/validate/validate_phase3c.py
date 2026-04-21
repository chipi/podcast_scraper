"""Real-episode end-to-end validation for #643 Phase 3C dispatch wiring.

Drives the actual ``_generate_episode_summary`` → ``gi.build_artifact`` →
``kg.build_artifact`` code path with real cloud providers, against real
transcripts (copied from an existing corpus). Verifies the claim that
``extraction_bundled`` drops one LLM call vs staged and ``mega_bundled``
drops two.

Usage::

    .venv/bin/python scripts/validate/validate_phase3c.py

Reads transcripts from ``.test_outputs/_validate_phase3c/transcripts/``
(expects ``short.txt`` and ``medium.txt``). Output JSON +
stdout-rendered table go to ``.test_outputs/_validate_phase3c/results/``.

Exit code: 0 if all gates pass, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Load .env for API keys.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from podcast_scraper import config as config_mod  # noqa: E402
from podcast_scraper.gi.pipeline import build_artifact as gi_build_artifact  # noqa: E402
from podcast_scraper.kg.pipeline import build_artifact as kg_build_artifact  # noqa: E402
from podcast_scraper.summarization.factory import (  # noqa: E402
    create_summarization_provider,
)
from podcast_scraper.workflow import metrics as metrics_mod  # noqa: E402
from podcast_scraper.workflow.metadata_generation import _generate_episode_summary  # noqa: E402

RESULTS_DIR = _REPO_ROOT / ".test_outputs" / "_validate_phase3c" / "results"
TRANSCRIPTS_DIR = _REPO_ROOT / ".test_outputs" / "_validate_phase3c" / "transcripts"

# Pricing (USD per 1M tokens). Keep conservative / in line with provider docs.
PRICING = {
    "anthropic": {"input": 1.00, "output": 5.00},  # claude-haiku-4-5
    "gemini": {"input": 0.10, "output": 0.40},  # gemini-2.5-flash-lite (approx)
    "deepseek": {"input": 0.28, "output": 0.42},  # deepseek-chat
    "openai": {"input": 0.15, "output": 0.60},  # gpt-4o-mini
    "mistral": {"input": 0.20, "output": 0.60},  # mistral-small
    "grok": {"input": 3.00, "output": 15.00},  # grok-4 / grok-3 approx
}


@dataclass
class RunResult:
    transcript: str
    mode: str
    provider: str
    model: str
    call_counts: Dict[str, int] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    elapsed: Dict[str, float] = field(default_factory=dict)
    artifact_counts: Dict[str, int] = field(default_factory=dict)
    prefilled_present: bool = False
    usd_cost: float = 0.0
    errors: List[str] = field(default_factory=list)


def _build_cfg(provider: str, mode: str, output_dir: str) -> config_mod.Config:
    cfg_dict: Dict[str, Any] = {
        "rss_url": "https://example.com/validation.rss",
        "output_dir": output_dir,
        "transcribe_missing": False,
        "auto_speakers": False,
        "generate_summaries": True,
        "generate_gi": True,
        "generate_kg": True,
        "generate_metadata": True,
        "gi_insight_source": "provider",
        "kg_extraction_source": "provider",
        "gi_require_grounding": False,  # skip QA+NLI to isolate dispatch cost
        "summary_provider": provider,
        "llm_pipeline_mode": mode,
        "cloud_llm_structured_min_output_tokens": 4096,
        "gi_max_insights": 12,
        "kg_max_topics": 10,
        "kg_max_entities": 15,
        # Needed by the factory / provider __init__.
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
        "deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "mistral_api_key": os.environ.get("MISTRAL_API_KEY"),
        "grok_api_key": os.environ.get("GROK_API_KEY"),
    }
    if provider == "anthropic":
        cfg_dict["anthropic_summary_model"] = "claude-haiku-4-5"
    elif provider == "gemini":
        cfg_dict["gemini_summary_model"] = "gemini-2.5-flash-lite"
    elif provider == "deepseek":
        cfg_dict["deepseek_summary_model"] = "deepseek-chat"
    elif provider == "openai":
        cfg_dict["openai_summary_model"] = "gpt-4o-mini"
    elif provider == "mistral":
        cfg_dict["mistral_summary_model"] = "mistral-small-latest"
    elif provider == "grok":
        cfg_dict["grok_summary_model"] = "grok-3-mini"
    return config_mod.Config.model_validate(cfg_dict)


def _instrument(provider: Any) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Wrap provider methods to count invocations and track tokens."""
    counts = {
        "summarize": 0,
        "summarize_bundled": 0,
        "summarize_mega_bundled": 0,
        "summarize_extraction_bundled": 0,
        "generate_insights": 0,
        "extract_kg_graph": 0,
        "extract_kg_from_summary_bullets": 0,
        "total": 0,
    }
    tokens = {"input": 0, "output": 0}

    # Also hook the underlying HTTP client so we also capture token totals
    # regardless of which provider-facing method is called. Providers vary:
    # anthropic → client.messages.create; openai-compat → client.chat.completions.create;
    # gemini → client.models.generate_content.
    client = getattr(provider, "client", None)
    if client is not None:
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            orig = client.messages.create

            def wrapped_anth(*a: Any, **kw: Any) -> Any:
                r = orig(*a, **kw)
                u = getattr(r, "usage", None)
                if u is not None:
                    tokens["input"] += int(getattr(u, "input_tokens", 0) or 0)
                    tokens["output"] += int(getattr(u, "output_tokens", 0) or 0)
                return r

            client.messages.create = wrapped_anth
        if hasattr(client, "chat") and hasattr(getattr(client, "chat", None), "completions"):
            orig_oc = client.chat.completions.create

            def wrapped_oc(*a: Any, **kw: Any) -> Any:
                r = orig_oc(*a, **kw)
                u = getattr(r, "usage", None)
                if u is not None:
                    tokens["input"] += int(getattr(u, "prompt_tokens", 0) or 0)
                    tokens["output"] += int(getattr(u, "completion_tokens", 0) or 0)
                return r

            client.chat.completions.create = wrapped_oc
        # Mistral SDK uses chat.complete (no 's') — same usage field shape as OpenAI.
        if hasattr(client, "chat") and hasattr(getattr(client, "chat", None), "complete"):
            orig_mc = client.chat.complete

            def wrapped_mc(*a: Any, **kw: Any) -> Any:
                r = orig_mc(*a, **kw)
                u = getattr(r, "usage", None)
                if u is not None:
                    tokens["input"] += int(getattr(u, "prompt_tokens", 0) or 0)
                    tokens["output"] += int(getattr(u, "completion_tokens", 0) or 0)
                return r

            client.chat.complete = wrapped_mc
        if hasattr(client, "models") and hasattr(
            getattr(client, "models", None), "generate_content"
        ):
            orig_g = client.models.generate_content

            def wrapped_g(*a: Any, **kw: Any) -> Any:
                r = orig_g(*a, **kw)
                u = getattr(r, "usage_metadata", None)
                if u is not None:
                    tokens["input"] += int(getattr(u, "prompt_token_count", 0) or 0)
                    tokens["output"] += int(getattr(u, "candidates_token_count", 0) or 0)
                return r

            client.models.generate_content = wrapped_g

    # Method-level counters (how many times each provider method was called).
    for name in list(counts.keys()):
        if name == "total":
            continue
        orig_m = getattr(provider, name, None)
        if not callable(orig_m):
            continue

        def make_wrapper(orig_fn: Callable[..., Any], key: str) -> Callable[..., Any]:
            def wrapper(*a: Any, **kw: Any) -> Any:
                counts[key] += 1
                counts["total"] += 1
                return orig_fn(*a, **kw)

            return wrapper

        setattr(provider, name, make_wrapper(orig_m, name))

    return counts, tokens


def _count_nodes(payload: Dict[str, Any], node_type: str) -> int:
    return sum(1 for n in payload.get("nodes", []) if n.get("type") == node_type)


def run_one(transcript_path: Path, mode: str, provider_name: str) -> RunResult:
    work_dir = RESULTS_DIR.parent / "runs" / f"{transcript_path.stem}__{mode}__{provider_name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    local_transcript = work_dir / "transcript.txt"
    local_transcript.write_text(transcript_path.read_text(encoding="utf-8"))

    cfg = _build_cfg(provider_name, mode, str(work_dir))
    provider = create_summarization_provider(cfg)
    if hasattr(provider, "initialize"):
        provider.initialize()
    counts, tokens = _instrument(provider)

    model = getattr(provider, "summary_model", "?")
    result = RunResult(
        transcript=transcript_path.name,
        mode=mode,
        provider=provider_name,
        model=str(model),
    )

    pipeline_metrics = metrics_mod.Metrics()

    try:
        # --- Summary stage (dispatch happens here) ---
        t0 = time.time()
        summary_meta, _cm = _generate_episode_summary(
            transcript_file_path="transcript.txt",
            output_dir=str(work_dir),
            cfg=cfg,
            episode_idx=0,
            summary_provider=provider,
            pipeline_metrics=pipeline_metrics,
        )
        result.elapsed["summary"] = time.time() - t0
        result.prefilled_present = bool(
            summary_meta and getattr(summary_meta, "prefilled_extraction", None)
        )

        transcript_text = local_transcript.read_text(encoding="utf-8")

        # --- GI stage ---
        prefilled_insights: Optional[List[Dict[str, Any]]] = None
        if result.prefilled_present and summary_meta is not None:
            pe = summary_meta.prefilled_extraction or {}
            if pe.get("insights"):
                prefilled_insights = pe["insights"]

        t1 = time.time()
        gi_payload = gi_build_artifact(
            "ep_validation",
            transcript_text,
            podcast_id="feed_validation",
            episode_title="Validation Episode",
            cfg=cfg,
            insight_provider=provider,
            summary_provider=provider,
            pipeline_metrics=pipeline_metrics,
            prefilled_insights=prefilled_insights,
        )
        result.elapsed["gi"] = time.time() - t1

        # --- KG stage ---
        prefilled_partial: Optional[Dict[str, Any]] = None
        if result.prefilled_present and summary_meta is not None:
            pe = summary_meta.prefilled_extraction or {}
            if pe.get("topics") or pe.get("entities"):
                prefilled_partial = {
                    "topics": pe.get("topics") or [],
                    "entities": pe.get("entities") or [],
                }

        t2 = time.time()
        kg_payload = kg_build_artifact(
            "ep_validation",
            transcript_text,
            podcast_id="feed_validation",
            episode_title="Validation Episode",
            cfg=cfg,
            kg_extraction_provider=provider,
            pipeline_metrics=pipeline_metrics,
            prefilled_partial=prefilled_partial,
        )
        result.elapsed["kg"] = time.time() - t2

        # Artifact counts
        result.artifact_counts["bullets"] = len(getattr(summary_meta, "bullets", []) or [])
        result.artifact_counts["gi_insights"] = _count_nodes(gi_payload, "Insight")
        result.artifact_counts["kg_topics"] = _count_nodes(kg_payload, "Topic")
        result.artifact_counts["kg_entities"] = _count_nodes(kg_payload, "Entity")

        # Persist artifacts for inspection
        (work_dir / "gi.json").write_text(json.dumps(gi_payload, indent=2))
        (work_dir / "kg.json").write_text(json.dumps(kg_payload, indent=2))
        if summary_meta is not None:
            (work_dir / "summary.json").write_text(
                json.dumps(summary_meta.model_dump(mode="json"), indent=2)
            )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"{type(exc).__name__}: {exc}")

    result.call_counts = counts
    result.tokens = tokens
    pricing = PRICING.get(provider_name, {"input": 0.0, "output": 0.0})
    result.usd_cost = (
        tokens["input"] * pricing["input"] / 1_000_000
        + tokens["output"] * pricing["output"] / 1_000_000
    )
    return result


def _print_table(results: List[RunResult]) -> None:
    hdr = (
        f"{'transcript':<12} {'mode':<22} {'provider':<10} "
        f"{'calls':<6} {'in_tok':<8} {'out_tok':<8} {'cost_$':<8} "
        f"{'bullets':<8} {'ins':<5} {'top':<5} {'ent':<5} "
        f"{'pref':<5} {'t_sum':<7} {'t_gi':<7} {'t_kg':<7} {'err'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        err = "-" if not r.errors else r.errors[0][:30]
        print(
            f"{r.transcript[:11]:<12} "
            f"{r.mode:<22} {r.provider:<10} "
            f"{r.call_counts.get('total', 0):<6} "
            f"{r.tokens.get('input', 0):<8} "
            f"{r.tokens.get('output', 0):<8} "
            f"{r.usd_cost:<8.5f} "
            f"{r.artifact_counts.get('bullets', 0):<8} "
            f"{r.artifact_counts.get('gi_insights', 0):<5} "
            f"{r.artifact_counts.get('kg_topics', 0):<5} "
            f"{r.artifact_counts.get('kg_entities', 0):<5} "
            f"{'Y' if r.prefilled_present else 'N':<5} "
            f"{r.elapsed.get('summary', 0):<7.2f} "
            f"{r.elapsed.get('gi', 0):<7.2f} "
            f"{r.elapsed.get('kg', 0):<7.2f} "
            f"{err}"
        )


def _check_gates(results: List[RunResult]) -> Tuple[bool, List[str]]:
    """Apply per-mode gates from the validation plan."""
    failures: List[str] = []
    # Bucket by (transcript, mode, provider).
    for r in results:
        tag = f"{r.transcript}/{r.mode}/{r.provider}"
        if r.errors:
            failures.append(f"{tag}: run errored: {r.errors[0]}")
            continue
        if r.mode == "staged":
            # Expect summarize + generate_insights + extract_kg_graph = 3 calls.
            if r.call_counts.get("total", 0) < 3:
                failures.append(
                    f"{tag}: staged expected ≥3 LLM calls, got {r.call_counts.get('total', 0)}"
                )
        elif r.mode == "bundled":
            # Expect summarize_bundled + generate_insights + extract_kg_graph = 3 calls.
            if r.call_counts.get("summarize_bundled", 0) != 1:
                failures.append(
                    f"{tag}: bundled expected 1 summarize_bundled call, "
                    f"got {r.call_counts.get('summarize_bundled', 0)}"
                )
        elif r.mode == "extraction_bundled":
            # Expect 1 extraction_bundled + 1 staged summarize = 2 total provider method calls.
            # GIL + KG must NOT hit the provider.
            if r.call_counts.get("summarize_extraction_bundled", 0) != 1:
                failures.append(
                    f"{tag}: extraction_bundled expected 1 extraction call, "
                    f"got {r.call_counts.get('summarize_extraction_bundled', 0)}"
                )
            if r.call_counts.get("generate_insights", 0) != 0:
                failures.append(
                    f"{tag}: extraction_bundled GIL should skip provider, "
                    f"got {r.call_counts.get('generate_insights', 0)} generate_insights calls"
                )
            if (
                r.call_counts.get("extract_kg_graph", 0)
                + r.call_counts.get("extract_kg_from_summary_bullets", 0)
                != 0
            ):
                failures.append(
                    f"{tag}: extraction_bundled KG should skip provider, got "
                    f"{r.call_counts.get('extract_kg_graph', 0)} extract_kg_graph + "
                    f"{r.call_counts.get('extract_kg_from_summary_bullets', 0)} bullet-derived"
                )
            if not r.prefilled_present:
                failures.append(f"{tag}: extraction_bundled missing prefilled_extraction")
        elif r.mode == "mega_bundled":
            # Expect ONE call total: summarize_mega_bundled. GIL + KG skipped.
            if r.call_counts.get("summarize_mega_bundled", 0) != 1:
                failures.append(
                    f"{tag}: mega_bundled expected 1 mega call, "
                    f"got {r.call_counts.get('summarize_mega_bundled', 0)}"
                )
            if r.call_counts.get("generate_insights", 0) != 0:
                failures.append(
                    f"{tag}: mega_bundled GIL should skip provider, "
                    f"got {r.call_counts.get('generate_insights', 0)} generate_insights calls"
                )
            if (
                r.call_counts.get("extract_kg_graph", 0)
                + r.call_counts.get("extract_kg_from_summary_bullets", 0)
                != 0
            ):
                failures.append(
                    f"{tag}: mega_bundled KG should skip provider, got "
                    f"{r.call_counts.get('extract_kg_graph', 0)} extract_kg_graph + "
                    f"{r.call_counts.get('extract_kg_from_summary_bullets', 0)} bullet-derived"
                )
            if not r.prefilled_present:
                failures.append(f"{tag}: mega_bundled missing prefilled_extraction")
            if r.call_counts.get("total", 0) != 1:
                failures.append(
                    f"{tag}: mega_bundled expected total=1, got {r.call_counts.get('total', 0)}"
                )

        # Artifact presence gates.
        if r.artifact_counts.get("gi_insights", 0) < 1:
            failures.append(f"{tag}: GI produced 0 insights")
        if r.artifact_counts.get("kg_topics", 0) < 1:
            failures.append(f"{tag}: KG produced 0 topics")
        if r.artifact_counts.get("kg_entities", 0) < 1:
            failures.append(f"{tag}: KG produced 0 entities")
        if r.artifact_counts.get("bullets", 0) < 1:
            failures.append(f"{tag}: summary produced 0 bullets")

    return (len(failures) == 0, failures)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    transcripts = [
        TRANSCRIPTS_DIR / "short.txt",
        TRANSCRIPTS_DIR / "medium.txt",
    ]
    for t in transcripts:
        if not t.exists():
            print(f"FATAL: missing transcript {t}", file=sys.stderr)
            return 2

    matrix: List[Tuple[Path, str, str]] = []
    for t in transcripts:
        matrix.extend(
            [
                # Gemini baselines (staged vs bundled vs extraction vs mega).
                (t, "staged", "gemini"),
                (t, "bundled", "gemini"),
                (t, "extraction_bundled", "gemini"),
                (t, "mega_bundled", "gemini"),
                # All other providers in mega_bundled — #632 claimed only
                # Anthropic/DeepSeek were tier-capable; #646 real-episode retest
                # showed all 6 work on production traffic.
                (t, "mega_bundled", "anthropic"),
                (t, "mega_bundled", "deepseek"),
                (t, "mega_bundled", "openai"),
                (t, "mega_bundled", "mistral"),
                (t, "mega_bundled", "grok"),
            ]
        )

    results: List[RunResult] = []
    for t, mode, provider_name in matrix:
        print(f"\n[run] {t.name} / {mode} / {provider_name} …", flush=True)
        r = run_one(t, mode, provider_name)
        results.append(r)
        print(
            f"  total_calls={r.call_counts.get('total', 0)} "
            f"tokens={r.tokens.get('input', 0)}/{r.tokens.get('output', 0)} "
            f"cost=${r.usd_cost:.5f} "
            f"ins={r.artifact_counts.get('gi_insights', 0)} "
            f"top={r.artifact_counts.get('kg_topics', 0)} "
            f"ent={r.artifact_counts.get('kg_entities', 0)} "
            f"err={r.errors[0] if r.errors else '-'}"
        )

    print()
    print("=" * 140)
    _print_table(results)
    print("=" * 140)

    passed, failures = _check_gates(results)
    (RESULTS_DIR / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    (RESULTS_DIR / "gates.json").write_text(
        json.dumps({"passed": passed, "failures": failures}, indent=2)
    )

    print()
    if passed:
        print("GATES: PASS")
    else:
        print(f"GATES: FAIL ({len(failures)} failures)")
        for f in failures:
            print(f"  - {f}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
