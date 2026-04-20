"""Cloud transcription model head-to-head: WER × cost × latency (#577 Exp 3).

Transcribes a fixed set of episodes through every cloud transcription model
we support and scores them on WER (vs reference transcripts), wall time, and
cost. Picks the winner(s) per-axis.

Providers tested:
  - OpenAI: whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe
  - Gemini: gemini-2.5-flash-lite (native audio input)
  - Mistral: voxtral-mini-latest

Fixed input: audio re-encoded at 32 kbps / 16 kHz / mono (Exp 1 winner).
Requires references in --reference-dir as <ep>.txt.

Keys (autoresearch convention; falls back to production env var):
  AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY   | OPENAI_API_KEY
  AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY   | GEMINI_API_KEY
  AUTORESEARCH_EXPERIMENT_MISTRAL_API_KEY  | MISTRAL_API_KEY

Usage:
    export AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY=sk-...
    export AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY=...
    export AUTORESEARCH_EXPERIMENT_MISTRAL_API_KEY=...

    python scripts/eval/transcription_cloud_sweep.py \\
        --audio-dir tests/fixtures/audio \\
        --reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \\
        --episodes p01_e03,p02_e03,p03_e03,p04_e03,p05_e03
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".tmp" / "cloud_sweep"
_RESULTS_DIR = _REPO_ROOT / "data" / "eval" / "runs" / "_cloud_sweep"

# Approximate per-minute rates (USD). Experiment logs actual usage where available;
# these are only used as fallback when a provider doesn't return usage info.
_APPROX_PER_MIN_USD: Dict[Tuple[str, str], float] = {
    ("openai", "whisper-1"): 0.006,
    ("openai", "gpt-4o-transcribe"): 0.006,
    ("openai", "gpt-4o-mini-transcribe"): 0.003,
    ("gemini", "gemini-2.5-flash-lite"): 0.0003,  # very rough — audio-tokens * input rate
    ("mistral", "voxtral-mini-latest"): 0.001,  # rough
}

DEFAULT_MATRIX: List[Tuple[str, str]] = [
    ("openai", "whisper-1"),
    ("openai", "gpt-4o-transcribe"),
    ("openai", "gpt-4o-mini-transcribe"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("mistral", "voxtral-mini-latest"),
]


def _resolve_key(provider: str) -> Optional[str]:
    """AUTORESEARCH_EXPERIMENT_<P>_API_KEY → <P>_API_KEY."""
    mapping = {
        "openai": ("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "gemini": ("AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY", "GEMINI_API_KEY"),
        "mistral": ("AUTORESEARCH_EXPERIMENT_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
    }
    ar_env, prod_env = mapping[provider]
    return os.environ.get(ar_env) or os.environ.get(prod_env)


def reencode_32k_mono(src: Path, dst: Path) -> None:
    """Cache-once re-encode at 32 kbps / 16 kHz / mono (Exp 1 winner)."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-b:a",
        "32k",
        "-c:a",
        "libmp3lame",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def probe_duration_s(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def compute_wer(hypothesis: str, reference: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref)][len(hyp)] / len(ref)


def _build_cfg(provider: str, model: str, api_key: str) -> Any:
    """Minimal Config stub for provider instantiation."""
    from podcast_scraper.config import Config

    kwargs: Dict[str, Any] = {
        "rss_url": "https://example.invalid/e2e-placeholder.xml",
        "transcription_provider": provider,
        "language": "en",
    }
    if provider == "openai":
        kwargs["openai_api_key"] = api_key
        kwargs["openai_transcription_model"] = model
    elif provider == "gemini":
        kwargs["gemini_api_key"] = api_key
        kwargs["gemini_transcription_model"] = model
    elif provider == "mistral":
        kwargs["mistral_api_key"] = api_key
        kwargs["mistral_transcription_model"] = model
    return Config(**kwargs)


def _transcribe_one(provider: str, model: str, audio_path: Path, cfg: Any) -> Dict[str, Any]:
    """Instantiate provider and transcribe. Returns dict with text + wall time."""
    t0 = time.time()
    if provider == "openai":
        # OpenAI uses a standalone transcription call via openai client (no provider class wrapper).
        from openai import OpenAI

        client = OpenAI(api_key=cfg.openai_api_key)
        with audio_path.open("rb") as f:
            resp = client.audio.transcriptions.create(model=model, file=f, language="en")
        text = resp.text
    elif provider == "gemini":
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        prov = GeminiProvider(cfg)
        prov.initialize()
        text = prov.transcribe(str(audio_path), language="en")
    elif provider == "mistral":
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        prov = MistralProvider(cfg)
        prov.initialize()
        text = prov.transcribe(str(audio_path), language="en")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    dt = time.time() - t0
    return {"text": text, "transcribe_time": round(dt, 1)}


def run_cell(
    provider: str,
    model: str,
    ep: str,
    audio_path: Path,
    reference: str,
    audio_duration_s: float,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """One (provider, model, ep) cell: transcribe + score."""
    try:
        cfg = _build_cfg(provider, model, api_key)
        print(f"  [{provider}/{model}] {ep}: transcribing...", flush=True)
        t = _transcribe_one(provider, model, audio_path, cfg)
    except Exception as e:
        print(f"  [{provider}/{model}] {ep}: FAIL ({type(e).__name__}: {e})", flush=True)
        return None
    hyp = t["text"]
    wer = compute_wer(hyp, reference) if reference else None
    cost_per_min = _APPROX_PER_MIN_USD.get((provider, model), 0.0)
    cost_est = round(audio_duration_s / 60.0 * cost_per_min, 6)
    wer_str = f"WER={wer:.1%}" if wer is not None else "no ref"
    print(
        f"  [{provider}/{model}] {ep}: {t['transcribe_time']}s, " f"{wer_str}, ~${cost_est:.4f}",
        flush=True,
    )
    return {
        "provider": provider,
        "model": model,
        "episode": ep,
        "audio_duration_s": round(audio_duration_s, 1),
        "transcribe_time_s": t["transcribe_time"],
        "wer": round(wer, 4) if wer is not None else None,
        "chars": len(hyp),
        "estimated_cost_usd": cost_est,
        "cost_basis": "approx_per_min",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud transcription head-to-head (#577 Exp 3)")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--matrix",
        default=",".join(f"{p}:{m}" for p, m in DEFAULT_MATRIX),
        help="Comma-separated provider:model pairs to test",
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, help="Parallel transcription workers (default: 3)"
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ref_dir = Path(args.reference_dir)
    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    matrix = [tuple(m.split(":", 1)) for m in args.matrix.split(",") if m.strip()]

    # Resolve keys up-front; skip providers without a key.
    providers_needed = {p for p, _ in matrix}
    keys: Dict[str, str] = {}
    for p in providers_needed:
        k = _resolve_key(p)
        if k:
            keys[p] = k
        else:
            print(f"  ⚠ No API key for {p} — skipping all {p} rows")
    active_matrix = [(p, m) for p, m in matrix if p in keys]
    if not active_matrix:
        sys.exit("No providers have API keys available. Set AUTORESEARCH_EXPERIMENT_<P>_API_KEY.")

    print(f"Audio dir:        {audio_dir}")
    print(f"Reference dir:    {ref_dir}")
    print(f"Episodes:         {episodes}")
    print(f"Matrix (active):  {active_matrix}")
    print(f"Max workers:      {args.max_workers}")
    print()

    refs: Dict[str, str] = {}
    for ep in episodes:
        p = ref_dir / f"{ep}.txt"
        if p.exists():
            refs[ep] = p.read_text()

    # Pre-encode all episodes at 32k mono once.
    print("Re-encoding episodes at 32 kbps / 16 kHz / mono...", flush=True)
    encoded: Dict[str, Path] = {}
    durations: Dict[str, float] = {}
    for ep in episodes:
        src = audio_dir / f"{ep}.mp3"
        if not src.exists():
            print(f"  {ep}: SKIP (no source)", flush=True)
            continue
        dst = _CACHE_DIR / f"{ep}_32k_16000hz_mono.mp3"
        reencode_32k_mono(src, dst)
        durations[ep] = probe_duration_s(dst)
        encoded[ep] = dst
    print(f"  {len(encoded)} episodes ready.\n", flush=True)

    # Dispatch all (provider, model, ep) cells to the threadpool.
    print("Transcribing across matrix...", flush=True)
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = []
        for provider, model in active_matrix:
            for ep, audio_path in encoded.items():
                futures.append(
                    pool.submit(
                        run_cell,
                        provider,
                        model,
                        ep,
                        audio_path,
                        refs.get(ep, ""),
                        durations[ep],
                        keys[provider],
                    )
                )
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    # Summary
    print()
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    by_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_model[(r["provider"], r["model"])].append(r)
    print(
        f"{'Provider':<10s} {'Model':<26s} {'Avg WER':>8s} "
        f"{'Avg Time':>10s} {'Avg $/ep':>10s} {'Eps':>4s}"
    )
    print("-" * 80)
    total_cost = 0.0
    for (p, m), rows in sorted(by_model.items()):
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        times = [r["transcribe_time_s"] for r in rows]
        costs = [r["estimated_cost_usd"] for r in rows]
        total_cost += sum(costs)
        wer_avg = statistics.mean(wers) if wers else 0.0
        print(
            f"{p:<10s} {m:<26s} {wer_avg:>7.1%}  "
            f"{statistics.mean(times):>8.1f}s  ${statistics.mean(costs):>7.4f}  {len(rows):>4d}"
        )
    print(f"\nTotal run cost (estimate): ${total_cost:.4f}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = _RESULTS_DIR / f"cloud_sweep_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "matrix": active_matrix,
                "episodes": episodes,
                "fixed_input": {"bitrate_kbps": 32, "sample_rate_hz": 16000, "channels": 1},
                "results": results,
                "total_cost_estimate_usd": round(total_cost, 6),
            },
            indent=2,
        )
    )
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
