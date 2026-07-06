#!/usr/bin/env python3
"""Speaker-detection naming accuracy bake-off (#997).

Measures how well each ``speaker_detector_provider`` backend recovers the
speakers' names from EPISODE METADATA (title + description) on the v3
fixture bed. v3 has 25 episodes with explicit ground-truth annotations
under ``tests/fixtures/ground-truth/v3/ground_truth/*.json`` and v2 RSS feeds carry
matching ``<description>`` text for 22 of those episodes.

This is NOT the same task as diarization (audio → time-aligned speakers
— covered in ``diarization_dgx_vs_cloud_v1.py``). This task is:

    (episode_title, episode_description, known_hosts) → list[speakers]

Backends tested by default:

- ``spacy``  — NER on the title+description text (local, free, baseline)
- ``gemini`` — Gemini 2.5 Flash-Lite via the existing speaker-detector
  protocol (cloud, ~$0.001 / call)
- ``ollama`` — Ollama on DGX (local-on-DGX, free; needs DGX up with the
  configured ollama_speaker_model)

Scoring axes (per episode, per backend):

- ``exact_match_guest``: detected list contains the ground-truth
  ``primary_guest_canonical_name`` exactly.
- ``soft_match_guest``: any token of the ground-truth guest name appears
  in the detected list (handles "Liam Verbeek" reported as "Liam").
- ``host_recovered``: detected hosts set contains the ground-truth host
  ``host_canonical_name`` (exact or token-overlap).
- ``hallucinated_names``: detected names that don't match ground-truth
  host or guest — surfaces backend hallucination patterns.

Output: ``data/eval/runs/speaker_detection_naming/<run_id>/metrics.json``
plus a per-backend per-episode result line. Designed to be re-runnable —
pass ``--backends spacy gemini`` to skip Ollama when DGX is down.

Usage::

    # Quick local-only smoke (spaCy only)
    python scripts/eval/score/speaker_detection_naming_v1.py --backends spacy

    # spaCy + Gemini (cloud, needs GEMINI_API_KEY in env)
    python scripts/eval/score/speaker_detection_naming_v1.py --backends spacy gemini

    # Full 3-way (needs DGX up + Ollama daemon + OLLAMA_API_BASE)
    python scripts/eval/score/speaker_detection_naming_v1.py --backends spacy gemini ollama
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Set, Tuple

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("speaker_detection_naming")


def _load_v3_episodes() -> List[Dict[str, Any]]:
    """Load v3 manifest + ground_truth + v2 RSS description per episode."""
    manifest_path = _REPO_ROOT / "tests/fixtures/ground-truth/v3/manifest.json"
    decoder = json.JSONDecoder()
    raw = manifest_path.read_text()
    manifest, _ = decoder.raw_decode(raw)

    v2_desc: Dict[str, Dict[str, str]] = {}
    for fp in glob.glob(str(_REPO_ROOT / "tests/fixtures/rss/*.xml")):
        xml = pathlib.Path(fp).read_text(errors="replace")
        for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
            guid_m = re.search(r"<guid[^>]*>([^<]+)</guid>", item)
            if not guid_m:
                continue
            gid = guid_m.group(1).strip()
            title_m = re.search(r"<title>([^<]+)</title>", item)
            desc_m = re.search(r"<description>([^<]+)</description>", item)
            v2_desc[gid] = {
                "title": title_m.group(1).strip() if title_m else "",
                "description": desc_m.group(1).strip() if desc_m else "",
            }

    out: List[Dict[str, Any]] = []
    for ep in manifest["episodes"]:
        eid = ep["episode_id"]
        gt_path = _REPO_ROOT / ep["ground_truth_path"]
        gt = json.loads(gt_path.read_text())
        v2 = None
        for gid, meta in v2_desc.items():
            if gid.startswith(eid):
                v2 = meta
                break
        if v2 is None or not v2.get("description"):
            continue
        out.append(
            {
                "episode_id": eid,
                "title": v2["title"] or ep.get("title", ""),
                "description": v2["description"],
                "host_canonical_name": gt.get("host_canonical_name", ""),
                "primary_guest_canonical_name": gt.get("primary_guest_canonical_name", ""),
            }
        )
    return out


def _tokenize_name(name: str) -> Set[str]:
    return {tok for tok in re.split(r"\s+", name.lower()) if tok}


def _name_in_list(target: str, detected: List[str]) -> Tuple[bool, bool]:
    if not target:
        return False, False
    target_tokens = _tokenize_name(target)
    exact = any(d.strip().lower() == target.strip().lower() for d in detected)
    if exact:
        return True, True
    detected_tokens: Set[str] = set()
    for d in detected:
        detected_tokens.update(_tokenize_name(d))
    soft = bool(target_tokens & detected_tokens)
    return False, soft


def _build_spacy_detector():
    from podcast_scraper import config as cfg_module
    from podcast_scraper.speaker_detectors.factory import create_speaker_detector

    cfg = cfg_module.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "speaker_detector_provider": "spacy",
            "auto_speakers": True,
        }
    )
    det = create_speaker_detector(cfg)
    det.initialize()
    return det


def _build_gemini_detector():
    from podcast_scraper import config as cfg_module
    from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set; skip gemini backend.")
    cfg = cfg_module.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "speaker_detector_provider": "gemini",
            "gemini_api_key": api_key,
            "auto_speakers": True,
        }
    )
    det = GeminiProvider(cfg)
    det.initialize()
    return det


def _build_ollama_detector():
    from podcast_scraper import config as cfg_module
    from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
    cfg = cfg_module.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "speaker_detector_provider": "ollama",
            "ollama_api_base": api_base,
            "auto_speakers": True,
        }
    )
    det = OllamaProvider(cfg)
    det.initialize()
    return det


_BACKEND_BUILDERS = {
    "spacy": _build_spacy_detector,
    "gemini": _build_gemini_detector,
    "ollama": _build_ollama_detector,
}


def _run_backend(backend: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        det = _BACKEND_BUILDERS[backend]()
    except Exception as exc:  # noqa: BLE001
        logger.warning("skip backend=%s: %s", backend, exc)
        return {"backend": backend, "skipped": True, "skip_reason": str(exc)}

    import time as _time

    per_ep: List[Dict[str, Any]] = []
    hallucinated_counter: Dict[str, int] = {}
    t_backend_start = _time.time()
    for idx, ep in enumerate(episodes, start=1):
        t_ep_start = _time.time()
        host = ep["host_canonical_name"]
        guest = ep["primary_guest_canonical_name"]
        try:
            detected, hosts, ok, used_defaults = det.detect_speakers(
                ep["title"], ep["description"], {host} if host else set()
            )
        except Exception as exc:  # noqa: BLE001
            per_ep.append(
                {
                    "episode_id": ep["episode_id"],
                    "error": str(exc),
                    "exact_guest": False,
                    "soft_guest": False,
                    "host_recovered": False,
                    "detected": [],
                }
            )
            continue

        detected_list = [str(d) for d in detected]
        exact_guest, soft_guest = _name_in_list(guest, detected_list)
        host_exact, host_soft = _name_in_list(host, list(hosts) + detected_list)
        host_recovered = host_exact or host_soft

        # "Faithful" credit: a detected name that appears as a substring in
        # the RSS title or description is correct EXTRACTION even if it
        # doesn't match v3 transcript-truth (v2/v3 alias-drift is an
        # intentional v3 failure mode — ``alias_invention``). Separates
        # "model extracted what was in input" from "model hallucinated".
        metadata_text = (ep["title"] + " " + ep["description"]).lower()
        faithful_extractions: List[str] = []
        hallucinated_here: List[str] = []
        ground_tokens = _tokenize_name(host) | _tokenize_name(guest)
        for d in detected_list:
            d_tokens = _tokenize_name(d)
            in_metadata = d.lower() in metadata_text
            matches_truth = bool(d_tokens & ground_tokens)
            if in_metadata or matches_truth:
                faithful_extractions.append(d)
            else:
                hallucinated_here.append(d)
                hallucinated_counter[d] = hallucinated_counter.get(d, 0) + 1

        # ``faithful_guest_credit``: did the backend extract ANY non-host
        # name that's in the input metadata? Best single signal for
        # "backend can read the input."
        faithful_non_host = [
            d for d in faithful_extractions if not (_tokenize_name(d) & _tokenize_name(host))
        ]
        faithful_guest_credit = bool(faithful_non_host)

        per_ep.append(
            {
                "episode_id": ep["episode_id"],
                "exact_guest": exact_guest,
                "soft_guest": soft_guest,
                "faithful_guest_credit": faithful_guest_credit,
                "host_recovered": host_recovered,
                "detected": detected_list,
                "faithful": faithful_extractions,
                "hallucinated": hallucinated_here,
                "used_defaults": used_defaults,
                "ok": ok,
            }
        )
        # Per-episode progress so a long-running backend (Ollama on
        # DGX with the model loading cold) is visible while it runs.
        dt = _time.time() - t_ep_start
        elapsed_total = _time.time() - t_backend_start
        eta_remaining = (elapsed_total / idx) * (len(episodes) - idx)
        print(
            f"  [{backend}] {idx:>2}/{len(episodes)} {ep['episode_id']:>8} "
            f"dt={dt:.1f}s elapsed={elapsed_total:.0f}s eta_remaining={eta_remaining:.0f}s "
            f"| exact_guest={int(exact_guest)} faithful={int(faithful_guest_credit)} "
            f"| detected={detected_list}",
            flush=True,
        )

    n = len(per_ep)
    if n == 0:
        return {"backend": backend, "skipped": False, "episodes_scored": 0}
    exact_acc = sum(1 for r in per_ep if r.get("exact_guest")) / n
    soft_acc = sum(1 for r in per_ep if r.get("soft_guest")) / n
    faithful_acc = sum(1 for r in per_ep if r.get("faithful_guest_credit")) / n
    host_acc = sum(1 for r in per_ep if r.get("host_recovered")) / n
    hall_total = sum(len(r.get("hallucinated", [])) for r in per_ep)

    return {
        "backend": backend,
        "skipped": False,
        "episodes_scored": n,
        "exact_guest_accuracy": round(exact_acc, 3),
        "soft_guest_accuracy": round(soft_acc, 3),
        "faithful_guest_extraction": round(faithful_acc, 3),
        "host_recovery_accuracy": round(host_acc, 3),
        "hallucinated_total": hall_total,
        "hallucinated_top": sorted(hallucinated_counter.items(), key=lambda kv: -kv[1])[:10],
        "per_episode": per_ep,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["spacy", "gemini", "ollama"],
        choices=list(_BACKEND_BUILDERS),
    )
    ap.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data/eval/runs/speaker_detection_naming"),
    )
    ap.add_argument(
        "--run-id",
        default="v1_initial",
    )
    args = ap.parse_args()

    episodes = _load_v3_episodes()
    print(f"# Loaded {len(episodes)} episodes from v3 manifest + v2 RSS descriptions")
    print()
    out_root = pathlib.Path(args.output_dir) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "episodes_total": len(episodes),
        "backends": {},
    }
    for backend in args.backends:
        print(f"## Running backend: {backend}")
        r = _run_backend(backend, episodes)
        results["backends"][backend] = r
        if r.get("skipped"):
            print(f"  skipped: {r.get('skip_reason')}")
        else:
            print(f"  episodes_scored:           {r['episodes_scored']}")
            print(
                f"  exact_guest_accuracy:      {r['exact_guest_accuracy']:.1%}  "
                "(matches v3 transcript-truth exactly)"
            )
            print(
                f"  soft_guest_accuracy:       {r['soft_guest_accuracy']:.1%}  "
                "(token overlap with v3 transcript-truth)"
            )
            print(
                f"  faithful_guest_extraction: {r['faithful_guest_extraction']:.1%}  "
                "(extracted a non-host name from the input metadata)"
            )
            print(f"  host_recovery_accuracy:    {r['host_recovery_accuracy']:.1%}")
            print(
                f"  hallucinated_total:        {r['hallucinated_total']}  "
                "(names not in metadata AND not in ground truth)"
            )
            if r["hallucinated_top"]:
                print(f"  top hallucinations:        {r['hallucinated_top'][:5]}")
        print()

    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"# Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
