"""Materialize QMSum general-query entries into our eval format.

Downloads pszemraj/qmsum-cleaned from HF Hub, filters for general queries (gq),
writes transcripts + meta.json + gold reference predictions.jsonl.

Usage:
    python scripts/eval/materialize_qmsum.py [--split validation] [--max-meetings 35]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download  # nosec B615


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--max-meetings", type=int, default=35)
    parser.add_argument("--dataset-id", default="qmsum_phase21_v1", help="Output dataset ID")
    args = parser.parse_args()

    parquet_path = hf_hub_download(  # nosec B615
        "pszemraj/qmsum-cleaned",
        f"data/{args.split}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)

    gq_rows = []
    for i in range(len(table)):
        pid = table.column("pid")[i].as_py()
        if "-gq-" in pid:
            gq_rows.append(i)

    if args.max_meetings and len(gq_rows) > args.max_meetings:
        gq_rows = gq_rows[: args.max_meetings]

    print(f"Selected {len(gq_rows)} general-query meetings from {args.split} split")

    mat_dir = Path(f"data/eval/materialized/{args.dataset_id}")
    mat_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = Path(f"data/eval/references/gold/{args.dataset_id}_gold_paragraph")
    ref_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    ref_preds = []

    for idx, row_idx in enumerate(gq_rows):
        pid = table.column("pid")[row_idx].as_py()
        raw_input = table.column("input")[row_idx].as_py()
        gold_output = table.column("output")[row_idx].as_py()
        input_tokens = table.column("input_token_count")[row_idx].as_py()

        meeting_id = pid.replace("_0", "").replace("-", "_")

        lines = raw_input.split("\n", 1)
        transcript = lines[1].strip() if len(lines) > 1 else raw_input

        transcript_path = mat_dir / f"{meeting_id}.txt"
        transcript_path.write_text(transcript)

        episodes.append(
            {
                "episode_id": meeting_id,
                "transcript_path": f"{meeting_id}.txt",
                "source": "qmsum",
                "qmsum_pid": pid,
                "input_tokens": input_tokens,
            }
        )

        if gold_output:
            ref_preds.append(
                {
                    "episode_id": meeting_id,
                    "dataset_id": args.dataset_id,
                    "output": {"summary_final": gold_output},
                }
            )

        print(
            f"  {meeting_id}: {len(transcript)} chars transcript, "
            f"{len(gold_output) if gold_output else 0} chars gold ref"
        )

    meta = {
        "dataset_id": args.dataset_id,
        "source": "pszemraj/qmsum-cleaned",
        "split": args.split,
        "filter": "general_query_only",
        "num_episodes": len(episodes),
        "materialized_at": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
    }
    (mat_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    ref_preds_path = ref_dir / "predictions.jsonl"
    with ref_preds_path.open("w") as f:
        for pred in ref_preds:
            f.write(json.dumps(pred) + "\n")

    ref_baseline = {
        "run_id": f"{args.dataset_id}_gold_paragraph",
        "dataset_id": args.dataset_id,
        "task": "summarization",
        "backend": {"type": "human_gold", "source": "qmsum"},
    }
    (ref_dir / "baseline.json").write_text(json.dumps(ref_baseline, indent=2))

    print(f"\nMaterialized: {mat_dir}")
    print(f"Gold refs:    {ref_dir}")
    print(f"Episodes:     {len(episodes)}")
    refs_with_gold = sum(1 for p in ref_preds if p["output"]["summary_final"])
    print(f"With gold:    {refs_with_gold}")


if __name__ == "__main__":
    main()
