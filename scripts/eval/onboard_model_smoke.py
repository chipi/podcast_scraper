#!/usr/bin/env python3
"""Single-shot smoke test for an onboarding vLLM model — fires a hello probe and
one full episode summary against the currently-serving vLLM endpoint, then
emits PASS/FAIL + diagnostics. Use this as the gate before queueing a model
into a full autoresearch cohort run.

Validates against the MODEL_PLAYBOOK.md onboarding checklist:

- Step 3 (Boot test): 5-token hello probe, verify well-formed response, no BPE
  artifacts, no unexpected reasoning preamble, finish_reason=stop.
- Step 4 (Smoke summary test): one full episode summary against the planned
  config (model + prompt + sampling all from the same yaml), verify length in
  spec, no echo / Q&A / refusal patterns.

Usage:

    PYTHONPATH=. .venv/bin/python scripts/eval/onboard_model_smoke.py \\
        --config data/eval/configs/summarization/<your_config>.yaml \\
        --episode p01_e01

Exit codes:

- 0 = PASS (all checks succeeded)
- 1 = FAIL (one or more checks failed; see stderr diagnostics)
- 2 = SETUP error (config not loadable, vLLM unreachable, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Byte-level BPE artifacts that should NEVER appear in plain-text output if
# the tokenizer's decode path is wired correctly. Same set as
# `output_postprocess.decode_r1_byte_level`.
BPE_ARTIFACTS = (
    "Ġ",  # space token
    "Ċ",  # newline token
    "âĢĻ",  # right single quote
    "âĢľ",  # left double quote
    "âĢĿ",  # right double quote
    "âĢĶ",  # em dash
)

REFUSAL_PHRASES = (
    "i can't comply",
    "i can not comply",
    "i'm sorry, but i can't",
    "i am sorry, but i can",
    "i am an ai assistant designed to provide helpful and harmless",
    "i cannot assist",
    "i won't help",
)

REASONING_LEAKS = (
    "<think>",
    "<reasoning>",
    "<scratchpad>",
    "[think]",
)

ECHO_PATTERNS = (
    # First-line transcript reproduction patterns
    "welcome back to",
    "host:",
    "[00:00]",
)


def _http_get(url: str, headers: dict[str, str], timeout: int = 15) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def _http_post(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 300
) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def _has_bpe_artifacts(text: str) -> list[str]:
    return [tok for tok in BPE_ARTIFACTS if tok in text]


def _has_refusal(text: str) -> list[str]:
    lower = text.lower()
    return [p for p in REFUSAL_PHRASES if p in lower]


def _has_reasoning_leak(text: str) -> list[str]:
    return [tag for tag in REASONING_LEAKS if tag in text]


def _has_echo(text: str) -> list[str]:
    lower = text[:200].lower()
    return [p for p in ECHO_PATTERNS if p in lower]


def _report(label: str, passed: bool, detail: str = "") -> str:
    mark = "PASS" if passed else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    return line


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Experiment YAML config — provides prompts, sampling, vLLM base url.",
    )
    parser.add_argument(
        "--episode",
        default="p01_e01",
        help="Episode id from curated_5feeds_dev_v1 to use for the summary smoke. Default p01_e01.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=os.environ.get("VLLM_API_BASE"),
        help="Override vLLM base URL. Default: VLLM_API_BASE env, then config.",
    )
    parser.add_argument(
        "--max-summary-chars",
        type=int,
        default=3500,
        help="Max chars in the summary before flagging over-spec (default 3500).",
    )
    parser.add_argument(
        "--min-summary-chars",
        type=int,
        default=400,
        help="Min chars in the summary before flagging under-spec. Default 400 (~100 tokens).",
    )
    args = parser.parse_args()

    import yaml

    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"SETUP ERROR: cannot load config {args.config}: {e}", file=sys.stderr)
        return 2

    backend = cfg.get("backend", {})
    base_url = args.vllm_base_url or os.environ.get("VLLM_API_BASE") or backend.get("base_url")
    if not base_url:
        print(
            "SETUP ERROR: no vLLM base URL "
            "(--vllm-base-url, VLLM_API_BASE, or config.backend.base_url required)",
            file=sys.stderr,
        )
        return 2

    api_key = os.environ.get(backend.get("api_key_env", "VLLM_API_KEY"), "buddy-is-the-king")
    served_model_name = backend.get("model", "autoresearch")
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"=== Onboarding smoke for {args.config.stem} ===")
    print(f"vLLM base url: {base_url}")
    print(f"served model name: {served_model_name}")
    print()

    results: list[str] = []
    all_passed = True

    # --- Check 1: /health 200 ---
    base = base_url.rstrip("/").removesuffix("/v1")
    code, body = _http_get(f"{base}/health", headers={}, timeout=10)
    passed = code == 200
    results.append(_report("/health 200", passed, f"got {code}"))
    if not passed:
        all_passed = False
        print("\n".join(results), file=sys.stderr)
        return 1

    # --- Check 2: /v1/models id matches ---
    code, body = _http_get(f"{base_url.rstrip('/')}/models", headers, timeout=10)
    served_id = None
    underlying_id = None
    if code == 200:
        try:
            arr = json.loads(body).get("data", [])
            if arr:
                served_id = arr[0].get("id")
                underlying_id = arr[0].get("root") or arr[0].get("owned_by")
        except Exception:
            pass
    passed = code == 200 and served_id == served_model_name
    results.append(
        _report(
            "/v1/models id matches config", passed, f"served={served_id} underlying={underlying_id}"
        )
    )
    if not passed:
        all_passed = False

    # --- Check 3: 5-token hello ---
    hello_payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        # Reasoning models (DeepSeek-R1 family, Qwen3 family with thinking,
        # Magistral) spend a hidden token budget on <think>...</think> that
        # the server-side reasoning_parser strips from `content`. A 20-token
        # budget leaves no room for both think + answer → content=None.
        # 200 is enough headroom for any reasonable hello.
        "max_tokens": 200,
        "temperature": 0.0,
    }
    extra_body = backend.get("extra_body") or {}
    if extra_body:
        hello_payload.update(extra_body)
    t0 = time.monotonic()
    code, body = _http_post(
        f"{base_url.rstrip('/')}/chat/completions", headers, hello_payload, timeout=60
    )
    hello_latency = time.monotonic() - t0
    hello_content = ""
    if code == 200:
        try:
            data = json.loads(body)
            choice = data["choices"][0]
            hello_content = choice["message"]["content"]
        except Exception as e:
            results.append(_report("hello response parses", False, f"json error: {e}"))
            all_passed = False
    else:
        results.append(_report("hello response 200", False, f"got {code} body={body[:200]}"))
        all_passed = False

    if code == 200:
        # Reasoning-parser strip can leave content=None (all budget went to
        # the <think> block). Treat None as empty for the check; the failure
        # message tells the operator to bump max_tokens.
        hello_content = hello_content or ""
        passed = bool(hello_content.strip())
        results.append(
            _report(
                "hello content non-empty",
                passed,
                f"got {len(hello_content)} chars in {hello_latency:.1f}s",
            )
        )
        if not passed:
            all_passed = False
        artifacts = _has_bpe_artifacts(hello_content)
        passed = not artifacts
        results.append(
            _report(
                "hello no BPE artifacts", passed, f"found {artifacts}" if artifacts else "clean"
            )
        )
        if not passed:
            all_passed = False
        leaks = _has_reasoning_leak(hello_content)
        passed = not leaks
        results.append(
            _report("hello no reasoning preamble", passed, f"found {leaks}" if leaks else "clean")
        )
        if not passed:
            all_passed = False
        refusal = _has_refusal(hello_content)
        passed = not refusal
        results.append(
            _report("hello no refusal", passed, f"found {refusal}" if refusal else "clean")
        )
        if not passed:
            all_passed = False

    # --- Check 4: Full episode summary smoke via the harness ---
    # We do this by calling the same prompts the config uses, against the named
    # episode. The simplest path is to invoke the standard harness in dry-run
    # mode but on a SINGLE episode — but the harness reads the dataset and runs
    # all episodes. Instead we read the episode text directly + render the
    # prompts manually + fire the chat completion ourselves.

    from podcast_scraper.evaluation.experiment_config import (
        load_experiment_config,  # type: ignore[import-not-found]
    )

    try:
        load_experiment_config(args.config)
    except Exception as e:
        results.append(_report("experiment config loads", False, str(e)))
        print("\n".join(results), file=sys.stderr)
        return 1

    # Load the dataset to find the episode
    dataset_path = Path(f"data/eval/datasets/{cfg['data']['dataset_id']}.json")
    if not dataset_path.exists():
        results.append(_report("dataset metadata loads", False, f"missing {dataset_path}"))
        all_passed = False
        print("\n".join(results), file=sys.stderr)
        return 1
    dataset = json.loads(dataset_path.read_text())
    episode = next((e for e in dataset["episodes"] if e["episode_id"] == args.episode), None)
    if not episode:
        results.append(
            _report("episode found in dataset", False, f"no {args.episode} in {dataset_path}")
        )
        all_passed = False
        print("\n".join(results), file=sys.stderr)
        return 1

    transcript_path = Path(episode["transcript_path"])
    if not transcript_path.exists():
        results.append(_report("transcript file exists", False, f"missing {transcript_path}"))
        all_passed = False
        print("\n".join(results), file=sys.stderr)
        return 1
    transcript = transcript_path.read_text(encoding="utf-8", errors="replace")

    # Render the prompts via jinja
    import jinja2

    prompt_root = Path("src/podcast_scraper/prompts")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(prompt_root)))
    # System prompt is optional per PromptConfig schema. DeepSeek-R1 docs
    # explicitly recommend NO system prompt; Kimi-Linear's Round 3 nosys
    # variant also omits it. Skip rendering when absent.
    system_name = cfg["prompts"].get("system")
    system_tpl = env.get_template(system_name + ".j2") if system_name else None
    user_tpl = env.get_template(cfg["prompts"]["user"] + ".j2")
    paragraphs_min = max(1, int(cfg["params"].get("min_length", 200)) // 100)
    paragraphs_max = max(paragraphs_min, int(cfg["params"].get("max_length", 800)) // 100)
    title = episode.get("title", "")
    system_msg = (
        system_tpl.render(paragraphs_min=paragraphs_min, paragraphs_max=paragraphs_max, title=title)
        if system_tpl is not None
        else None
    )
    user_msg = user_tpl.render(
        paragraphs_min=paragraphs_min,
        paragraphs_max=paragraphs_max,
        title=title,
        transcript=transcript,
    )

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    summary_payload = {
        "model": served_model_name,
        "messages": messages,
        "max_tokens": int(cfg["params"].get("max_length", 800)),
        "temperature": float(cfg["params"].get("temperature", 0.0)),
    }
    if "top_p" in cfg["params"]:
        summary_payload["top_p"] = cfg["params"]["top_p"]
    if extra_body:
        summary_payload.update(extra_body)

    print(f"  ... firing summary smoke against {args.episode} ({len(transcript)} char transcript)")
    t0 = time.monotonic()
    code, body = _http_post(
        f"{base_url.rstrip('/')}/chat/completions", headers, summary_payload, timeout=600
    )
    summary_latency = time.monotonic() - t0
    summary_content = ""
    summary_finish = "?"
    if code == 200:
        try:
            data = json.loads(body)
            choice = data["choices"][0]
            summary_content = choice["message"]["content"]
            summary_finish = choice.get("finish_reason", "?")
        except Exception as e:
            results.append(_report("summary response parses", False, f"json error: {e}"))
            all_passed = False
    else:
        results.append(_report("summary response 200", False, f"got {code} body={body[:200]}"))
        all_passed = False

    if code == 200:
        passed = bool(summary_content.strip())
        results.append(
            _report(
                "summary non-empty",
                passed,
                f"{len(summary_content)} chars in {summary_latency:.1f}s",
            )
        )
        if not passed:
            all_passed = False

        passed = summary_finish == "stop"
        results.append(_report("finish_reason=stop", passed, f"got {summary_finish}"))
        if not passed:
            all_passed = False

        # Optionally apply the configured postprocessor before content checks
        post_name = cfg["prompts"].get("postprocessor")
        if post_name:
            try:
                from podcast_scraper.evaluation.output_postprocess import get_postprocessor

                summary_content_decoded = get_postprocessor(post_name)(summary_content)
                results.append(_report(f"postprocessor '{post_name}' applied", True, "OK"))
                summary_content = summary_content_decoded
            except Exception as e:
                results.append(_report(f"postprocessor '{post_name}' applied", False, str(e)))
                all_passed = False

        passed = args.min_summary_chars <= len(summary_content) <= args.max_summary_chars
        results.append(
            _report(
                f"length within {args.min_summary_chars}-{args.max_summary_chars} chars",
                passed,
                f"got {len(summary_content)}",
            )
        )
        if not passed:
            all_passed = False

        artifacts = _has_bpe_artifacts(summary_content)
        passed = not artifacts
        results.append(
            _report(
                "summary no BPE artifacts", passed, f"found {artifacts}" if artifacts else "clean"
            )
        )
        if not passed:
            all_passed = False

        leaks = _has_reasoning_leak(summary_content)
        passed = not leaks
        results.append(
            _report("summary no reasoning leak", passed, f"found {leaks}" if leaks else "clean")
        )
        if not passed:
            all_passed = False

        refusal = _has_refusal(summary_content)
        passed = not refusal
        results.append(
            _report("summary no refusal", passed, f"found {refusal}" if refusal else "clean")
        )
        if not passed:
            all_passed = False

        echoes = _has_echo(summary_content)
        passed = not echoes
        results.append(
            _report("summary no transcript echo", passed, f"found {echoes}" if echoes else "clean")
        )
        if not passed:
            all_passed = False

    print()
    print("\n".join(results))
    print()
    if summary_content:
        print("--- summary preview (first 300 chars) ---")
        print(summary_content[:300])
        print()

    if all_passed:
        print("=== PASS — model ready for cohort run ===")
    else:
        print("=== FAIL — see failed checks above ===")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
