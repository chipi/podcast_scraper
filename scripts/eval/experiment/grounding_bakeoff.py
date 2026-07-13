"""Module-2 bake-off: who should find the quote that backs an insight?

The pipeline does two jobs. Job 1 writes the insight (an LLM). Job 2 finds the sentence in the
transcript that proves it. Today job 2 is done by two local models -- an extractive-QA model that
selects a span, and an NLI model that checks the span supports the claim. The alternative is to let
an LLM do it. Nobody has measured which is better, because the two jobs were always run together and
the eval silently pointed job 2 at whichever LLM was under test.

This script isolates job 2. The insight set is FROZEN -- the same claims, on the same episodes, are
handed to every grounder -- so the only thing that varies is who does the retrieval. Without that,
a model that writes fewer (or blander) insights looks like a better grounder.

It calls the REAL pipeline code (`create_gil_evidence_providers` + `_ground_insights_dispatch`),
not a reimplementation. Reimplementing the wiring is exactly how the eval came to measure a
configuration production does not run.

Three metrics, because coverage alone is gameable -- a grounder that attaches any old sentence to
every insight scores 100%:

  coverage     share of insights that got at least one quote
  fabrication  share of returned quotes that do NOT appear verbatim in the transcript.
               An extractive model CANNOT fabricate: it returns a span it selected out of the
               text. An LLM generates, so it can produce a quote that is nearly-but-not-quite
               right -- and a near-miss fails the offset match and is dropped silently today.
               This is the number that decides whether an LLM is safe here.
  support      does the quote actually back the claim? Judged later, by a model that did not
               produce it (self-grading runs ~6x lenient).

Usage:
    python scripts/eval/experiment/grounding_bakeoff.py --insights ck_h2h_gemini_10ep \
        --grounders transformers,ollama,gemini --out data/eval/runs/bakeoff_gemini_insights
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from podcast_scraper.config import Config  # noqa: E402
from podcast_scraper.gi import pipeline as gi_pipeline  # noqa: E402
from podcast_scraper.gi.deps import create_gil_evidence_providers  # noqa: E402
from podcast_scraper.gi.grounding import resolve_llm_quote_span  # noqa: E402


def load_frozen_insights(run_id: str) -> Dict[str, List[str]]:
    """Pull the insight texts out of a finished run. These become the fixed input for every arm."""
    out: Dict[str, List[str]] = {}
    path = pathlib.Path(f"data/eval/runs/{run_id}/predictions.jsonl")
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        gil = (rec.get("output") or {}).get("gil") or {}
        texts = [
            (n.get("properties") or {}).get("text", "").strip()
            for n in (gil.get("nodes") or [])
            if n.get("type") == "Insight"
        ]
        texts = [t for t in texts if t]
        if texts:
            out[rec["episode_id"]] = texts
    return out


def subsample(texts: List[str], n: int) -> List[str]:
    """Take `n` insights spread EVENLY across the list, not the first n.

    Chunked extraction emits insights in chunk order, so the first n would all come from the start
    of the episode — and a retriever's job gets harder deeper into a transcript. An even stride
    keeps every part of the episode represented. Deterministic, so every arm sees the same claims.
    """
    if n <= 0 or len(texts) <= n:
        return texts
    stride = len(texts) / n
    return [texts[int(i * stride)] for i in range(n)]


def load_transcripts(dataset_id: str) -> Dict[str, str]:
    spec = json.loads(pathlib.Path(f"data/eval/datasets/{dataset_id}.json").read_text())
    out: Dict[str, str] = {}
    for ep in spec["episodes"]:
        p = ep.get("transcript_path")
        if p and pathlib.Path(p).is_file():
            out[ep["episode_id"]] = pathlib.Path(p).read_text()
    return out


def ground_with(
    grounder: str, insights: List[str], transcript: str, base_profile: str
) -> Tuple[List[List[Any]], float]:
    """Run the REAL grounding stage with `grounder` doing job 2."""
    cfg = Config.model_validate(
        {
            "profile": base_profile,
            "generate_gi": True,
            "quote_extraction_provider": grounder,
            "entailment_provider": grounder,
        }
    )
    quote_provider, entail_provider = create_gil_evidence_providers(cfg, summary_provider=None)
    specs: List[Tuple[str, Any]] = [(t, "claim") for t in insights]

    t0 = time.time()
    grounded = gi_pipeline._ground_insights_dispatch(
        cfg=cfg,
        insight_specs=specs,
        transcript=transcript,
        quote_extraction_provider=quote_provider,
        entailment_provider=entail_provider,
        qa_score_min=float(getattr(cfg, "gi_qa_score_min", 0.3)),
        nli_entailment_min=float(getattr(cfg, "gi_nli_entailment_min", 0.75)),
        extract_retries=int(getattr(cfg, "gi_evidence_extract_retries", 0) or 0),
        pipeline_metrics=None,
        prefetched_by_idx=None,
    )
    return grounded, time.time() - t0


def _norm(s: str) -> str:
    """Whitespace- and apostrophe-insensitive, so trivial formatting differences are not 'drift'."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


def score(grounded: List[List[Any]], transcript: str) -> Dict[str, Any]:
    """Coverage, drift and fabrication.

    Two different things get conflated if you only ask "can we locate it":

      verbatim    the quote is an EXACT substring of the transcript. This is what a quote is
                  supposed to be. An extractive model is verbatim by construction -- it returns a
                  span it selected. An LLM has to reproduce the text, and may not.
      drifted     not verbatim, but the pipeline's tolerant matcher still salvaged a span from it
                  (it falls back to matching a sub-phrase). The insight still gets evidence, but
                  the quote we store is NOT the text the model returned.
      fabricated  not verbatim and not salvageable. Dropped silently today, with no counter.

    Reporting only "fabricated" would flatter an LLM, because the sub-phrase fallback quietly
    rescues its near-misses. The drift rate is the honest measure of how much the LLM is
    paraphrasing text it was told to copy.
    """
    n_ins = len(grounded)
    covered = sum(1 for qs in grounded if qs)
    quotes = [q for qs in grounded for q in qs]
    t_norm = _norm(transcript)

    verbatim = drifted = fabricated = 0
    samples: List[str] = []
    for q in quotes:
        text = (getattr(q, "text", "") or "").strip()
        if text and _norm(text) in t_norm:
            verbatim += 1
        elif text and resolve_llm_quote_span(transcript, text) is not None:
            drifted += 1
            if len(samples) < 3:
                samples.append(text[:120])
        else:
            fabricated += 1
            if len(samples) < 3:
                samples.append(f"[UNLOCATABLE] {text[:110]}")

    n_q = len(quotes) or 1
    return {
        "insights": n_ins,
        "covered": covered,
        "coverage_pct": round(100 * covered / n_ins, 1) if n_ins else 0.0,
        "quotes": len(quotes),
        "quotes_per_insight": round(len(quotes) / n_ins, 2) if n_ins else 0.0,
        "verbatim": verbatim,
        "drifted": drifted,
        "fabricated": fabricated,
        "verbatim_pct": round(100 * verbatim / n_q, 1),
        "drift_pct": round(100 * drifted / n_q, 1),
        "fabrication_pct": round(100 * fabricated / n_q, 1),
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insights", required=True, help="finished run_id to freeze insights from")
    ap.add_argument("--dataset", default="prod_v3_10ep_v1")
    ap.add_argument("--grounders", default="transformers,ollama,gemini")
    ap.add_argument("--profile", default="experiment_dgx_only")
    ap.add_argument(
        "--per-episode",
        type=int,
        default=15,
        help="insights per episode, evenly spread (0 = all). Keeps every arm comparable and the "
        "run tractable; the local QA model is slow on a laptop.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frozen = load_frozen_insights(args.insights)
    frozen = {ep: subsample(texts, args.per_episode) for ep, texts in frozen.items()}
    transcripts = load_transcripts(args.dataset)
    episodes = sorted(set(frozen) & set(transcripts))
    total_ins = sum(len(frozen[e]) for e in episodes)
    print(f"  frozen insight set: {args.insights} — {len(episodes)} episodes, {total_ins} insights")
    print("  every grounder sees the IDENTICAL claims. only job 2 varies.\n")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    for grounder in [g.strip() for g in args.grounders.split(",") if g.strip()]:
        agg = {
            "insights": 0,
            "covered": 0,
            "quotes": 0,
            "verbatim": 0,
            "drifted": 0,
            "fabricated": 0,
        }
        secs = 0.0
        per_ep: List[Dict[str, Any]] = []
        samples: List[str] = []
        print(f"  --- {grounder} ---", flush=True)
        for ep in episodes:
            try:
                grounded, dt = ground_with(grounder, frozen[ep], transcripts[ep], args.profile)
            except Exception as exc:  # noqa: BLE001 — a failed arm is a result, not a crash
                print(f"    {ep}: FAILED ({type(exc).__name__}: {exc})", flush=True)
                continue
            s = score(grounded, transcripts[ep])
            secs += dt
            per_ep.append({"episode_id": ep, **s})
            for k in agg:
                agg[k] += s[k]
            samples.extend(s["samples"])
            print(
                f"    {ep}  {s['insights']:>3} ins  cov {s['coverage_pct']:>5.1f}%  "
                f"verbatim {s['verbatim_pct']:>5.1f}%  drift {s['drift_pct']:>5.1f}%  "
                f"fabricated {s['fabrication_pct']:>5.1f}%  {dt:>5.1f}s",
                flush=True,
            )
        if not per_ep:
            print(f"  {grounder:<14} produced nothing", flush=True)
            continue
        n_q = agg["quotes"] or 1
        summary = {
            "grounder": grounder,
            "episodes": len(per_ep),
            "insights": agg["insights"],
            "coverage_pct": round(100 * agg["covered"] / agg["insights"], 1),
            "quotes_per_insight": round(agg["quotes"] / agg["insights"], 2),
            "verbatim_pct": round(100 * agg["verbatim"] / n_q, 1),
            "drift_pct": round(100 * agg["drifted"] / n_q, 1),
            "fabrication_pct": round(100 * agg["fabricated"] / n_q, 1),
            "sec_per_episode": round(secs / len(per_ep), 1),
        }
        results[grounder] = {
            "summary": summary,
            "per_episode": per_ep,
            "drift_samples": samples[:10],
        }
        print(
            f"  = {grounder:<14} coverage {summary['coverage_pct']:>5.1f}%  "
            f"q/ins {summary['quotes_per_insight']:>5.2f}  "
            f"verbatim {summary['verbatim_pct']:>5.1f}%  "
            f"drift {summary['drift_pct']:>5.1f}%  "
            f"fabricated {summary['fabrication_pct']:>5.1f}%  "
            f"{summary['sec_per_episode']:>6.1f}s/ep\n",
            flush=True,
        )

    (outdir / "bakeoff.json").write_text(json.dumps(results, indent=2))

    if results:
        print("\n  MODULE 2 — who should find the quote?\n")
        hdr = (
            f"  {'grounder':<14} {'coverage':>9} {'q/ins':>7} {'verbatim':>9} "
            f"{'drift':>7} {'fabricated':>11} {'s/ep':>8}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for g, r in results.items():
            s = r["summary"]
            print(
                f"  {g:<14} {s['coverage_pct']:>8.1f}% {s['quotes_per_insight']:>7.2f} "
                f"{s['verbatim_pct']:>8.1f}% {s['drift_pct']:>6.1f}% "
                f"{s['fabrication_pct']:>10.1f}% {s['sec_per_episode']:>7.1f}s"
            )
        print(
            "\n  verbatim   = the quote is an exact substring of the transcript (what a quote IS)\n"
            "  drift      = not verbatim, but the tolerant matcher salvaged a sub-phrase.\n"
            "               The insight still gets evidence, but the stored quote is NOT what the\n"
            "               model returned. An extractive model cannot drift; an LLM can.\n"
            "  fabricated = not locatable at all. Dropped silently today, with no counter."
        )
    print(f"\n  written: {outdir/'bakeoff.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
