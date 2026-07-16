"""Which LOCAL model can judge the value gate, so an all-DGX profile needs no cloud call?

The value gate classifies each insight 0-3 (FILLER / MINOR / USEFUL / CORE) and drops anything below
the floor. Who judges is load-bearing: a model grading its OWN output is far more lenient than an
independent one, so self-grading leaves the filler in and the gate becomes decoration. A pinned
cloud judge fixes that but breaks the one thing the DGX profile exists for -- no cloud call at any
stage. So: can a SECOND LOCAL model do the job?

Everything about the experiment -- the judge field, the silver, the dataset, the scoring -- lives in
the config. Re-running it is one command; changing the field is a config edit.

    PYTHONPATH=. .venv/bin/python scripts/eval/experiment/value_gate_judge_bakeoff.py \
        --config data/eval/configs/value_gate_judge_bakeoff_v1.yaml

Two things this gets right that the first version got wrong:

1. THE INSIGHTS MUST BE PRE-GATE. The first version froze insights from a run whose value gate was
   already ON, judged by anthropic. Re-judging them measured nothing -- the reference judge was
   shown only its own survivors and dutifully dropped ~nothing (9% where it drops 26% on raw
   output). The config now points at an UNGATED run, so the filler is still there to catch.

2. THE SILVER IS A CONSENSUS, NOT ONE OPINION. Two independent cloud judges grade the same insights;
   a label counts only where they AGREE. Both vendors are disjoint from the candidate (qwen), so the
   silver carries no #939 same-vendor bias. Their agreement rate is also the NOISE CEILING of the
   task: where two flagship judges cannot agree whether an insight is CORE or FILLER, the label is
   ambiguous rather than wrong, and no local judge can be held above that bar.

Scored per judge, against the silver consensus:

  drop_rate   share of insights it rejects at the floor
  agreement   raw keep/drop match
  kappa       Cohen's kappa -- agreement CORRECTED FOR CHANCE, and the number that ranks them. A
              judge that keeps everything scores high raw agreement with any permissive baseline
              while deciding nothing; kappa scores that ~0. That is exactly the self-grading failure
              the gate exists to prevent, so it is the failure the metric has to be able to see.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from podcast_scraper.config import Config  # noqa: E402
from podcast_scraper.gi.value_gate import TIER_USEFUL  # noqa: E402
from podcast_scraper.summarization.factory import (  # noqa: E402
    create_summarization_provider,
)
from scripts.eval.experiment.grounding_bakeoff import (  # noqa: E402
    load_frozen_insights,
    subsample,
)


def load_config(path: str) -> Dict[str, Any]:
    """The judge set, the silver, the dataset and the scoring all live in the config.

    Nothing about this experiment is hardcoded here: re-running it, or re-running it with a
    different judge field, is a config edit and one command.
    """
    import yaml

    return yaml.safe_load(pathlib.Path(path).read_text())


def build_judge(provider: str, model: Optional[str]) -> Any:
    update: Dict[str, Any] = {"summary_provider": provider}
    if model:
        update[f"{provider}_summary_model"] = model
    cfg = Config.model_validate({"profile": "experiment_dgx_only", "generate_gi": True, **update})
    p = create_summarization_provider(cfg)
    if hasattr(p, "initialize"):
        p.initialize()
    return p


def kappa(a: List[bool], b: List[bool]) -> float:
    """Cohen's kappa. 0 = chance-level, 1 = perfect. Two lenient judges that both keep almost
    everything score high raw agreement while deciding nothing; kappa exposes that."""
    n = len(a)
    if n == 0:
        return 0.0
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    if exp >= 1.0:
        return 0.0
    return (obs - exp) / (1 - exp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="data/eval/configs/value_gate_judge_bakeoff_v1.yaml",
        help="the experiment config — judges, silver, dataset, scoring",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_id = cfg["id"]
    data = cfg["data"]
    floor = int(cfg.get("params", {}).get("floor", TIER_USEFUL))
    usable = float(cfg.get("scoring", {}).get("usable_kappa", 0.4))

    arms: List[Tuple[str, str, Optional[str]]] = []
    for s_ in cfg["silver"]:
        arms.append((f"SILVER {s_['provider']}", s_["provider"], s_.get("model")))
    b = cfg["baseline"]
    arms.append((b.get("label", "baseline"), b["provider"], b.get("model")))
    for j in cfg["judges"]:
        arms.append((j.get("label", j["model"]), j["provider"], j.get("model")))

    frozen = load_frozen_insights(data["insight_source_run"])
    if not frozen:
        print(f"  no insights in run '{data['insight_source_run']}' — generate it first")
        return 1
    texts: List[str] = []
    for ep in sorted(frozen):
        texts.extend(subsample(frozen[ep], int(data.get("per_episode", 12))))

    print(f"  {run_id}")
    print(
        f"  {len(texts)} insights from '{data['insight_source_run']}' (PRE-gate — filler left in)"
    )
    print(f"  keep if tier >= {floor}. Judges are non-qwen (#939 same-vendor bias).\n")

    keeps: Dict[str, List[bool]] = {}
    results: Dict[str, Dict[str, Any]] = {}

    for label, provider, model in arms:
        try:
            judge = build_judge(provider, model)
            t0 = time.time()
            tiers = judge.classify_insights(texts)
            dt = time.time() - t0
        except Exception as exc:  # noqa: BLE001 — a judge that cannot run is a result
            print(f"  {label:<28} FAILED ({type(exc).__name__}: {exc})", flush=True)
            results[label] = {"failed": f"{type(exc).__name__}: {exc}"}
            continue

        if not tiers or len(tiers) != len(texts):
            print(
                f"  {label:<28} returned {len(tiers or [])} of {len(texts)} tiers — UNUSABLE",
                flush=True,
            )
            results[label] = {"failed": f"returned {len(tiers or [])} of {len(texts)} tiers"}
            continue

        keep = [int(t) >= floor for t in tiers]
        keeps[label] = keep
        results[label] = {
            "drop_rate": round(100 * (1 - sum(keep) / len(keep)), 1),
            "sec": round(dt, 1),
            "tiers": [int(t) for t in tiers],
        }
        print(f"  {label:<28} drops {results[label]['drop_rate']:>5.1f}%   {dt:>6.1f}s", flush=True)

    out = pathlib.Path(f"data/eval/runs/{run_id}")
    out.mkdir(parents=True, exist_ok=True)

    silver_labels = [f"SILVER {s_['provider']}" for s_ in cfg["silver"]]
    got = [keeps.get(lbl) for lbl in silver_labels]
    if any(g is None for g in got):
        print("\n  A silver judge did not run — cannot score. Results saved unscored.")
        (out / "judges.json").write_text(json.dumps({"judges": results}, indent=2))
        return 1

    a, b_ = got[0], got[1]
    idx = [i for i in range(len(texts)) if a[i] == b_[i]]
    ceiling = 100 * len(idx) / len(texts)
    silver = [a[i] for i in idx]
    silver_drop = 100 * (1 - sum(silver) / len(silver)) if silver else 0.0

    print(f"\n  SILVER — the cloud judges agree on {len(idx)}/{len(texts)} insights")
    print(f"    inter-cloud agreement (the task's NOISE CEILING): {ceiling:.1f}%")
    print(f"    silver drop-rate: {silver_drop:.1f}%")
    print("    Where two flagship judges disagree the label is ambiguous, not wrong — excluded.")

    hdr = f"\n  {'judge':<28} {'drop%':>7} {'agree':>8} {'kappa':>7} {'sec':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    ranked = []
    for label, keep in keeps.items():
        sub = [keep[i] for i in idx]
        agree = 100 * sum(1 for x, y in zip(sub, silver) if x == y) / len(sub)
        k = kappa(sub, silver)
        results[label]["silver_agreement"] = round(agree, 1)
        results[label]["silver_kappa"] = round(k, 3)
        ranked.append((k, label))
        print(
            f"  {label:<28} {results[label]['drop_rate']:>6.1f}% {agree:>7.1f}% "
            f"{k:>7.2f} {results[label]['sec']:>6.1f}s"
        )

    ranked.sort(reverse=True)
    print(f"\n  Ranked by kappa (chance-corrected; usable >= {usable}):")
    for k, label in ranked:
        verdict = "USABLE" if k >= usable else ("weak" if k >= 0.2 else "rubber-stamp / noise")
        print(f"    {k:>6.2f}  {label:<28} {verdict}")

    (out / "judges.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": args.config,
                "insight_source_run": data["insight_source_run"],
                "n_insights": len(texts),
                "noise_ceiling_pct": round(ceiling, 1),
                "silver_n": len(idx),
                "silver_drop_rate": round(silver_drop, 1),
                "judges": results,
            },
            indent=2,
        )
    )
    print(f"\n  written: {out/'judges.json'}")
    return 0
