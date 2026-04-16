"""Long-T5-XL (pszemraj/long-t5-tglobal-xl-16384-book-summary) standalone runner.

Mirrors run_summllama_v2.py but for a seq2seq model. Feeds full transcript
(up to 16384 tokens), generates paragraph summary directly. Outputs
predictions.jsonl in framework format.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pszemraj/long-t5-tglobal-xl-16384-book-summary")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    print(f"torch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}")
    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # pszemraj/long-t5-tglobal-xl ships both safetensors and pytorch_model.bin. The
    # safetensors variant creates meta tensors that can't be moved to MPS on our
    # transformers version; force .bin format, which loads cleanly.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        use_safetensors=False,
    )
    model = model.to("mps")
    model.eval()

    materialized = Path(f"data/eval/materialized/{args.dataset}")
    meta = json.load(open(materialized / "meta.json"))
    episodes = meta["episodes"]
    print(f"Dataset: {args.dataset} — {len(episodes)} episodes")

    run_dir = Path(f"data/eval/runs/{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / "predictions.jsonl"
    durations = []
    with preds_path.open("w") as f:
        for ep in episodes:
            ep_id = ep["episode_id"]
            transcript = (materialized / f"{ep_id}.txt").read_text()
            # Long-T5-XL has 16k input window; cap to 15k tokens worth of text
            if len(transcript) > 60000:
                transcript = transcript[:60000]
            print(f"  {ep_id}: {len(transcript)} chars -> ", end="", flush=True)
            t0 = time.time()
            inputs = tokenizer(
                transcript,
                return_tensors="pt",
                truncation=True,
                max_length=16384,
            ).to("mps")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=600,
                    min_new_tokens=150,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            elapsed = time.time() - t0
            summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            durations.append(elapsed)
            print(f"{len(summary)} chars in {elapsed:.1f}s")
            pred = {
                "episode_id": ep_id,
                "dataset_id": args.dataset,
                "output": {"summary_final": summary},
            }
            f.write(json.dumps(pred) + "\n")

    baseline = {
        "run_id": args.run_id,
        "dataset_id": args.dataset,
        "task": "summarization",
        "backend": {"type": "seq2seq_local", "model": args.model},
        "params": {"model": args.model, "num_beams": 4, "max_new_tokens": 600},
        "stats": {
            "num_episodes": len(episodes),
            "total_time_seconds": sum(durations),
            "avg_time_seconds": sum(durations) / len(durations) if durations else 0,
        },
    }
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print(f"\nDone. avg_time: {baseline['stats']['avg_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
