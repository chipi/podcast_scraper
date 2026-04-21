"""Minimal SummLlama3.2-3B standalone runner for v2 eval framework.

Generates predictions.jsonl compatible with our scoring harness, so we can
score SummLlama standalone against any of the v2 silvers without full
framework integration.

Output: data/eval/runs/<run_id>/ with predictions.jsonl + baseline.json
ready for score_run() to process.
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo on path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str):
    print(f"Loading {model_id} on MPS...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)  # nosec B615
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_id,
        torch_dtype=torch.float16,
        device_map="mps",
    )
    model.eval()
    return tokenizer, model


def generate_summary(
    tokenizer, model, transcript: str, style: str = "paragraph"
) -> tuple[str, float]:
    if style == "paragraph":
        user = (
            "Please write a focused prose summary of the following podcast transcript "
            "in 4-6 paragraphs. Begin the first paragraph with a single sentence naming "
            "the episode's domain and its central argument or premise. Cover all major "
            "discussion segments in the order they appear. Preserve specific terminology "
            "verbatim — do not paraphrase named concepts. Anchor each paragraph in "
            "specific claims, data points, or named entities from the transcript. Ignore "
            "sponsorships, ads, and housekeeping. Do not use quotes or speaker names. Do "
            "not invent information not implied by the transcript.\n\nTranscript:\n\n"
            f"{transcript}"
        )
    else:
        user = (
            "Write a bullet-point summary of the following podcast transcript as 6-8 "
            "single-sentence bullets. Each bullet should cover a distinct major topic "
            "or claim in the order it appears. Preserve specific terminology verbatim — "
            "do not paraphrase named concepts. Anchor each bullet in specific claims, "
            "data points, or named entities from the transcript. Ignore sponsorships, "
            "ads, and housekeeping. Do not use quotes or speaker names. Do not invent "
            "information not implied by the transcript. Output only the bullets, one "
            "per line, each starting with '- '.\n\nTranscript:\n\n"
            f"{transcript}"
        )

    messages = [
        {"role": "system", "content": "You write focused summaries of podcast transcripts."},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    response = tokenizer.decode(
        output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    return response, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DISLab/SummLlama3.2-3B")
    parser.add_argument("--dataset", required=True, help="e.g. curated_5feeds_dev_v1")
    parser.add_argument("--run-id", required=True, help="e.g. summllama32_standalone_dev_v2")
    parser.add_argument("--style", default="paragraph", choices=["paragraph", "bullets"])
    args = parser.parse_args()

    materialized = Path(f"data/eval/materialized/{args.dataset}")
    meta = json.load(open(materialized / "meta.json"))
    episodes = meta["episodes"]
    print(f"Dataset: {args.dataset} — {len(episodes)} episodes")

    run_dir = Path(f"data/eval/runs/{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model)

    preds_path = run_dir / "predictions.jsonl"
    durations = []
    with preds_path.open("w") as f:
        for ep in episodes:
            ep_id = ep["episode_id"]
            transcript_path = materialized / f"{ep_id}.txt"
            transcript = transcript_path.read_text()
            # Truncate to fit prompt + generation in context window (128k but keep fast)
            if len(transcript) > 40000:
                transcript = transcript[:40000]
            print(f"  {ep_id}: {len(transcript)} chars -> ", end="", flush=True)
            summary, elapsed = generate_summary(tokenizer, model, transcript, style=args.style)
            durations.append(elapsed)
            print(f"{len(summary)} chars in {elapsed:.1f}s")
            pred = {
                "episode_id": ep_id,
                "output": {"summary_final": summary},
            }
            f.write(json.dumps(pred) + "\n")

    baseline = {
        "run_id": args.run_id,
        "dataset_id": args.dataset,
        "task": "summarization",
        "backend": {"type": "causal_lm_local", "model": args.model},
        "params": {"model": args.model, "temperature": 0.0, "max_tokens": 600},
        "stats": {
            "num_episodes": len(episodes),
            "total_time_seconds": sum(durations),
            "avg_time_seconds": sum(durations) / len(durations) if durations else 0,
        },
    }
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print(f"\nDone. avg_time_seconds: {baseline['stats']['avg_time_seconds']:.1f}s")
    print(f"Predictions: {preds_path}")


if __name__ == "__main__":
    main()
