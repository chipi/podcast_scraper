"""Full pipeline validation: run a provider through all stages and check pass/fail.

Runs summary → GI → KG → bridge for a given provider on the held-out dataset,
then checks pass/fail criteria at each stage. Produces an integration matrix row.

Usage:
    # Single provider:
    python scripts/eval/pipeline_validate.py --provider gemini --model gemini-2.5-flash-lite

    # All cloud providers (sequential):
    python scripts/eval/pipeline_validate.py --all-cloud

    # Local only:
    python scripts/eval/pipeline_validate.py --provider ollama --model qwen3.5:9b

    # ML only:
    python scripts/eval/pipeline_validate.py --provider ml
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATASET = "curated_5feeds_benchmark_v2"
PREPROCESSING = "cleaning_v4"

# Pass/fail criteria per stage
CRITERIA = {
    "summary": {"min_episodes": 5, "min_avg_chars": 500},
    "gi_insights": {"min_per_episode": 8},
    "gi_grounding": {"min_rate": 0.50},
    "gi_topics": {"min_per_episode": 3},
    "kg_topics": {"min_per_episode": 5, "max_label_chars": 50},
    "kg_entities": {"min_per_episode": 2},
    "bridge_topic_merge": {"min_rate": 0.90},
}


# Provider configs: (backend_type, model, prompts_user, extra_params)
def _cloud(backend_type: str, model: str, prompts_prefix: str) -> Dict[str, str]:
    return {
        "type": backend_type,
        "model": model,
        "prompts_user": f"{prompts_prefix}/summarization/long_v1",
        "prompts_system": f"{prompts_prefix}/summarization/system_v1",
    }


# Provider-specific timeout tiers. Reasoning models (Grok) and large local
# models need more time for evidence grounding (60+ LLM calls in GI stage).
TIMEOUT_FAST = 600  # 10 min — Gemini, Mistral, small Ollama
TIMEOUT_NORMAL = 1800  # 30 min — OpenAI, Anthropic, DeepSeek, mid Ollama
TIMEOUT_SLOW = 3600  # 60 min — Grok (reasoning), qwen3.5:35b

# Per-provider timeout override; defaults to TIMEOUT_NORMAL
PROVIDER_TIMEOUTS: Dict[str, int] = {
    "gemini/gemini-2.5-flash-lite": TIMEOUT_FAST,
    "mistral/mistral-small": TIMEOUT_FAST,
    "grok/grok-3-mini": TIMEOUT_SLOW,
    "ollama/qwen3.5:35b": TIMEOUT_SLOW,
    "ollama/llama3.2:3b": TIMEOUT_FAST,
}


CLOUD_PROVIDERS = {
    "openai/gpt-4o-mini": _cloud("openai", "gpt-4o-mini", "openai"),
    "gemini/gemini-2.5-flash-lite": _cloud("gemini", "gemini-2.5-flash-lite", "gemini"),
    "anthropic/claude-haiku-4-5": _cloud("anthropic", "claude-haiku-4-5-20251001", "anthropic"),
    "deepseek/deepseek-chat": _cloud("deepseek", "deepseek-chat", "deepseek"),
    "mistral/mistral-small": _cloud("mistral", "mistral-small-latest", "mistral"),
    "grok/grok-3-mini": _cloud("grok", "grok-3-mini", "grok"),
}


def _ollama(model: str) -> Dict[str, str]:
    return {
        "type": "ollama",
        "model": model,
        "prompts_user": "ollama/summarization/long_v1",
        "prompts_system": "ollama/summarization/system_v1",
    }


# Core 5: one per family + size diversity (ADR-076 decision).
# Dropped: qwen3.5:{27b,35b} (same family, worse or marginal), qwen2.5:7b
# (prev gen), llama3.1:8b (same family), mistral-nemo:12b (worse than 7b),
# mistral-small3.2 (22B for +1%), phi3:mini (4k context, structurally broken).
LOCAL_PROVIDERS = {
    "ollama/qwen3.5:9b": _ollama("qwen3.5:9b"),  # Champion, best local
    "ollama/llama3.1:8b": _ollama("llama3.1:8b"),  # Llama family, full pipeline at 8B
    "ollama/mistral:7b": _ollama("mistral:7b"),  # Best non-Qwen mid-tier
    "ollama/gemma2:9b": _ollama("gemma2:9b"),  # Google arch, different strengths
    "ollama/qwen3.5:35b": _ollama("qwen3.5:35b"),  # "Does bigger help?" reference
}


def _safe_slug(provider_label: str) -> str:
    return provider_label.replace("/", "_").replace(":", "_").replace(".", "_")


def _write_temp_config(
    provider_label: str,
    provider_cfg: Dict[str, str],
    task: str,
) -> Path:
    """Write a temporary experiment config YAML for one task."""
    slug = _safe_slug(provider_label)
    run_id = f"pv_{slug}_{task}"
    cfg: Dict[str, Any] = {
        "id": run_id,
        "task": task,
        "backend": {
            "type": provider_cfg["type"],
            "model": provider_cfg["model"],
        },
        "data": {"dataset_id": DATASET},
        "preprocessing_profile": PREPROCESSING,
        "params": {
            "max_length": 800,
            "min_length": 200,
            "temperature": 0.0,
        },
    }

    if task == "summarization":
        cfg["prompts"] = {
            "user": provider_cfg["prompts_user"],
            "system": provider_cfg.get("prompts_system"),
        }
    elif task == "grounded_insights":
        cfg["prompts"] = {
            "user": provider_cfg["prompts_user"],
            "system": provider_cfg.get("prompts_system"),
        }
        cfg["params"]["gi_insight_source"] = "provider"
        cfg["params"]["gi_max_insights"] = 12
        cfg["params"]["gi_require_grounding"] = True
    elif task == "knowledge_graph":
        cfg["prompts"] = {
            "user": provider_cfg["prompts_user"],
            "system": provider_cfg.get("prompts_system"),
        }
        cfg["params"]["kg_extraction_source"] = "provider"
        cfg["params"]["kg_max_topics"] = 10
        cfg["params"]["kg_max_entities"] = 15

    out_dir = Path("data/eval/configs/_pipeline_validate")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.yaml"

    import yaml

    with out_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return out_path


def _run_experiment(config_path: Path, provider_label: str = "") -> Tuple[bool, str]:
    """Run make experiment-run and return (success, error_message)."""
    cmd = [
        "make",
        "experiment-run",
        f"CONFIG={config_path}",
        "FORCE=1",
    ]
    timeout = PROVIDER_TIMEOUTS.get(provider_label, TIMEOUT_NORMAL)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
        )
        if result.returncode == 0:
            return True, ""
        # Extract error from output
        lines = (result.stdout + result.stderr).strip().split("\n")
        err = next(
            (ln for ln in reversed(lines) if "ERROR" in ln or "Error" in ln),
            lines[-1] if lines else "Unknown error",
        )
        return False, err[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (600s)"
    except Exception as e:
        return False, str(e)[:200]


def _check_summary(run_id: str) -> Dict[str, Any]:
    """Check summary stage pass/fail."""
    preds_path = Path(f"data/eval/runs/{run_id}/predictions.jsonl")
    if not preds_path.exists():
        return {"pass": False, "reason": "no predictions.jsonl"}

    preds = [json.loads(ln) for ln in preds_path.read_text().splitlines() if ln.strip()]
    if len(preds) < CRITERIA["summary"]["min_episodes"]:
        return {
            "pass": False,
            "reason": f"only {len(preds)} episodes (need {CRITERIA['summary']['min_episodes']})",
        }

    avg_chars = sum(len(p.get("output", {}).get("summary_final", "")) for p in preds) / max(
        len(preds), 1
    )
    if avg_chars < CRITERIA["summary"]["min_avg_chars"]:
        return {
            "pass": False,
            "reason": f"avg {avg_chars:.0f} chars (need {CRITERIA['summary']['min_avg_chars']})",
        }

    return {"pass": True, "episodes": len(preds), "avg_chars": round(avg_chars)}


def _check_gi(run_id: str) -> Dict[str, Any]:
    """Check GI stage pass/fail."""
    preds_path = Path(f"data/eval/runs/{run_id}/predictions.jsonl")
    if not preds_path.exists():
        return {"pass": False, "reason": "no predictions.jsonl"}

    preds = [json.loads(ln) for ln in preds_path.read_text().splitlines() if ln.strip()]

    total_insights = 0
    total_quotes = 0
    total_grounded = 0
    total_topics = 0
    ep_count = len(preds)

    for p in preds:
        gil = p.get("output", {}).get("gil", {})
        nodes = gil.get("nodes", [])
        types = Counter(n.get("type") for n in nodes)
        total_insights += types.get("Insight", 0)
        total_quotes += types.get("Quote", 0)
        total_topics += types.get("Topic", 0)
        total_grounded += sum(
            1
            for n in nodes
            if n.get("type") == "Insight" and n.get("properties", {}).get("grounded")
        )

    avg_insights = total_insights / max(ep_count, 1)
    grounding_rate = total_grounded / max(total_insights, 1)
    avg_topics = total_topics / max(ep_count, 1)

    result: Dict[str, Any] = {
        "insights": total_insights,
        "quotes": total_quotes,
        "grounded": total_grounded,
        "grounding_rate": round(grounding_rate, 2),
        "topics": total_topics,
    }

    fails: List[str] = []
    if avg_insights < CRITERIA["gi_insights"]["min_per_episode"]:
        fails.append(
            f"avg insights {avg_insights:.1f} < {CRITERIA['gi_insights']['min_per_episode']}"
        )
    if grounding_rate < CRITERIA["gi_grounding"]["min_rate"]:
        fails.append(f"grounding {grounding_rate:.0%} < {CRITERIA['gi_grounding']['min_rate']:.0%}")
    if avg_topics < CRITERIA["gi_topics"]["min_per_episode"]:
        fails.append(f"avg topics {avg_topics:.1f} < {CRITERIA['gi_topics']['min_per_episode']}")

    result["pass"] = len(fails) == 0
    if fails:
        result["reason"] = "; ".join(fails)
    return result


def _check_kg(run_id: str) -> Dict[str, Any]:
    """Check KG stage pass/fail."""
    preds_path = Path(f"data/eval/runs/{run_id}/predictions.jsonl")
    if not preds_path.exists():
        return {"pass": False, "reason": "no predictions.jsonl"}

    preds = [json.loads(ln) for ln in preds_path.read_text().splitlines() if ln.strip()]

    total_topics = 0
    total_entities = 0
    long_labels = 0
    ep_count = len(preds)

    for p in preds:
        kg = p.get("output", {}).get("kg", {})
        nodes = kg.get("nodes", [])
        for n in nodes:
            if n.get("type") == "Topic":
                total_topics += 1
                label = n.get("properties", {}).get("label", "")
                if len(label) > CRITERIA["kg_topics"]["max_label_chars"]:
                    long_labels += 1
            elif n.get("type") in ("Entity", "Person"):
                total_entities += 1

    avg_topics = total_topics / max(ep_count, 1)
    avg_entities = total_entities / max(ep_count, 1)

    result: Dict[str, Any] = {
        "topics": total_topics,
        "entities": total_entities,
        "long_labels": long_labels,
    }

    fails: List[str] = []
    if avg_topics < CRITERIA["kg_topics"]["min_per_episode"]:
        fails.append(f"avg topics {avg_topics:.1f} < {CRITERIA['kg_topics']['min_per_episode']}")
    if avg_entities < CRITERIA["kg_entities"]["min_per_episode"]:
        fails.append(
            f"avg entities {avg_entities:.1f} < {CRITERIA['kg_entities']['min_per_episode']}"
        )

    result["pass"] = len(fails) == 0
    if fails:
        result["reason"] = "; ".join(fails)
    return result


def _check_bridge(gi_run_id: str, kg_run_id: str) -> Dict[str, Any]:
    """Build bridges and check topic merge rate."""
    gi_path = Path(f"data/eval/runs/{gi_run_id}/predictions.jsonl")
    kg_path = Path(f"data/eval/runs/{kg_run_id}/predictions.jsonl")

    if not gi_path.exists() or not kg_path.exists():
        return {"pass": False, "reason": "missing GI or KG predictions"}

    from podcast_scraper.builders.bridge_builder import build_bridge
    from podcast_scraper.gi.pipeline import _dedupe_topic_node_specs

    gi_by_ep = {}
    for ln in gi_path.read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            gi_by_ep[d["episode_id"]] = d["output"]["gil"]

    kg_by_ep = {}
    for ln in kg_path.read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            kg_by_ep[d["episode_id"]] = d["output"]["kg"]

    total_topics = 0
    merged_topics = 0

    for ep_id in sorted(set(gi_by_ep) & set(kg_by_ep)):
        gi = dict(gi_by_ep[ep_id])
        kg = kg_by_ep[ep_id]

        # Apply KG topic alignment (same as pipeline fix)
        kg_labels = [
            n["properties"]["label"]
            for n in kg.get("nodes", [])
            if n.get("type") == "Topic" and n.get("properties", {}).get("label")
        ]
        if kg_labels:
            new_nodes = [n for n in gi["nodes"] if n.get("type") != "Topic"]
            new_edges = [e for e in gi["edges"] if e.get("type") != "ABOUT"]
            specs = _dedupe_topic_node_specs(kg_labels)
            for tid, label in specs:
                new_nodes.append({"id": tid, "type": "Topic", "properties": {"label": label}})
            insight_ids = [n["id"] for n in new_nodes if n.get("type") == "Insight"]
            for iid in insight_ids:
                for tid, _ in specs:
                    new_edges.append({"type": "ABOUT", "from": iid, "to": tid})
            gi["nodes"] = new_nodes
            gi["edges"] = new_edges

        bridge = build_bridge(ep_id, gi, kg)
        for identity in bridge["identities"]:
            if identity.get("type") == "topic":
                total_topics += 1
                if identity["sources"].get("gi") and identity["sources"].get("kg"):
                    merged_topics += 1

    rate = merged_topics / max(total_topics, 1)
    result: Dict[str, Any] = {
        "total_topics": total_topics,
        "merged": merged_topics,
        "merge_rate": round(rate, 2),
    }

    if rate < CRITERIA["bridge_topic_merge"]["min_rate"]:
        result["pass"] = False
        result["reason"] = (
            f"merge rate {rate:.0%} < {CRITERIA['bridge_topic_merge']['min_rate']:.0%}"
        )
    else:
        result["pass"] = True

    return result


def validate_provider(
    provider_label: str,
    provider_cfg: Dict[str, str],
) -> Dict[str, Any]:
    """Run full pipeline validation for one provider."""
    slug = _safe_slug(provider_label)
    results: Dict[str, Any] = {"provider": provider_label, "stages": {}}
    t0 = time.time()

    # Stage 1: Summary
    print(f"  [{provider_label}] Summary...", end="", flush=True)
    cfg_path = _write_temp_config(provider_label, provider_cfg, "summarization")
    ok, err = _run_experiment(cfg_path, provider_label)
    if ok:
        check = _check_summary(f"pv_{slug}_summarization")
        results["stages"]["summary"] = check
        print(f" {'PASS' if check['pass'] else 'FAIL'}")
    else:
        results["stages"]["summary"] = {"pass": False, "reason": err}
        print(f" FAIL ({err[:60]})")
        results["elapsed"] = round(time.time() - t0, 1)
        return results

    # Stage 2: GI
    print(f"  [{provider_label}] GI...", end="", flush=True)
    cfg_path = _write_temp_config(provider_label, provider_cfg, "grounded_insights")
    ok, err = _run_experiment(cfg_path, provider_label)
    if ok:
        check = _check_gi(f"pv_{slug}_grounded_insights")
        results["stages"]["gi"] = check
        print(f" {'PASS' if check['pass'] else 'FAIL'}")
    else:
        results["stages"]["gi"] = {"pass": False, "reason": err}
        print(f" FAIL ({err[:60]})")

    # Stage 3: KG
    print(f"  [{provider_label}] KG...", end="", flush=True)
    cfg_path = _write_temp_config(provider_label, provider_cfg, "knowledge_graph")
    ok, err = _run_experiment(cfg_path, provider_label)
    if ok:
        check = _check_kg(f"pv_{slug}_knowledge_graph")
        results["stages"]["kg"] = check
        print(f" {'PASS' if check['pass'] else 'FAIL'}")
    else:
        results["stages"]["kg"] = {"pass": False, "reason": err}
        print(f" FAIL ({err[:60]})")

    # Stage 4: Bridge
    gi_run = f"pv_{slug}_grounded_insights"
    kg_run = f"pv_{slug}_knowledge_graph"
    if results["stages"].get("gi", {}).get("pass") and results["stages"].get("kg", {}).get("pass"):
        print(f"  [{provider_label}] Bridge...", end="", flush=True)
        check = _check_bridge(gi_run, kg_run)
        results["stages"]["bridge"] = check
        print(f" {'PASS' if check['pass'] else 'FAIL'}")
    else:
        results["stages"]["bridge"] = {
            "pass": False,
            "reason": "skipped (GI or KG failed)",
        }
        print(f"  [{provider_label}] Bridge... SKIP")

    results["elapsed"] = round(time.time() - t0, 1)
    return results


def print_matrix(all_results: List[Dict[str, Any]]) -> None:
    """Print the integration pass/fail matrix."""
    stages = ["summary", "gi", "kg", "bridge"]
    print()
    print("=" * 90)
    print("PIPELINE VALIDATION MATRIX")
    print("=" * 90)
    header = f"{'Provider':<35s}"
    for s in stages:
        header += f" {s:>10s}"
    header += f" {'Time':>8s}"
    print(header)
    print("-" * 90)

    for r in all_results:
        row = f"{r['provider']:<35s}"
        for s in stages:
            stage = r.get("stages", {}).get(s, {})
            if stage.get("pass"):
                row += f" {'PASS':>10s}"
            elif stage.get("reason", "").startswith("skipped"):
                row += f" {'SKIP':>10s}"
            else:
                row += f" {'FAIL':>10s}"
        row += f" {r.get('elapsed', 0):>7.0f}s"
        print(row)

    print()
    # Detail for failures
    for r in all_results:
        for s in stages:
            stage = r.get("stages", {}).get(s, {})
            if not stage.get("pass") and not stage.get("reason", "").startswith("skipped"):
                print(f"  FAIL {r['provider']}/{s}: {stage.get('reason', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline validation across providers")
    parser.add_argument("--provider", help="Provider type (gemini, openai, ollama, ml, etc.)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--all-cloud", action="store_true", help="Run all 6 cloud providers")
    parser.add_argument("--all-local", action="store_true", help="Run all 11 Ollama models")
    parser.add_argument(
        "--local-fast",
        action="store_true",
        help="Run Ollama models ≤12B only (skip 27b, 35b — ~1hr vs ~4hrs)",
    )
    parser.add_argument("--all", action="store_true", help="Run everything")
    args = parser.parse_args()

    # Models >12B that are slow on MPS — skip in --local-fast mode
    _HEAVY_MODELS = {"ollama/qwen3.5:27b", "ollama/qwen3.5:35b", "ollama/mistral-small3.2"}

    providers_to_run: Dict[str, Dict[str, str]] = {}

    if args.all:
        providers_to_run.update(CLOUD_PROVIDERS)
        providers_to_run.update(LOCAL_PROVIDERS)
    elif args.all_cloud:
        providers_to_run.update(CLOUD_PROVIDERS)
    elif args.local_fast:
        providers_to_run.update(
            {k: v for k, v in LOCAL_PROVIDERS.items() if k not in _HEAVY_MODELS}
        )
    elif args.all_local:
        providers_to_run.update(LOCAL_PROVIDERS)
    elif args.provider and args.model:
        label = f"{args.provider}/{args.model}"
        all_known = {**CLOUD_PROVIDERS, **LOCAL_PROVIDERS}
        if label in all_known:
            providers_to_run[label] = all_known[label]
        else:
            print(f"Unknown provider: {label}")
            print(f"Known: {', '.join(sorted(all_known.keys()))}")
            sys.exit(1)
    elif args.provider == "ml":
        # TODO: ML provider validation (bart-led, hybrid)
        print("ML provider validation not yet implemented")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Pipeline validation: {len(providers_to_run)} provider(s)")
    print(f"Dataset: {DATASET}")
    print()

    all_results = []
    for label, cfg in providers_to_run.items():
        print(f"\n{'='*60}")
        print(f"Validating: {label}")
        print(f"{'='*60}")
        result = validate_provider(label, cfg)
        all_results.append(result)

    print_matrix(all_results)

    # Save results
    out_dir = Path("data/eval/runs/_pipeline_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"validation_{ts}.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
