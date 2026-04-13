#!/usr/bin/env python3
"""Diagnose Hugging Face hub reachability vs Transformers download (same path as preload).

Run from repo root::

    .venv/bin/python3 scripts/cache/hf_hub_smoke_test.py
    .venv/bin/python3 scripts/cache/hf_hub_smoke_test.py --model google/flan-t5-base

Steps:
1. Print relevant environment (HF_HUB_CACHE, proxy vars).
2. Raw HTTPS GET to huggingface.co (TLS + routing; no HF client).
3. Optional ``huggingface_hub`` API probe (repo metadata).
4. ``AutoTokenizer.from_pretrained`` + one encode (same stack as ``preload_ml_models.py``).

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.cache import get_transformers_cache_dir
from podcast_scraper.config_constants import get_pinned_revision_for_model
from podcast_scraper.providers.ml.summarizer import DEFAULT_SUMMARY_MODELS


def _print_env() -> None:
    keys = (
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    )
    print("=== Environment (subset) ===")
    for k in keys:
        v = os.environ.get(k)
        print(f"  {k}={v!r}" if v else f"  {k}=(unset)")
    print()


def _https_probe(url: str, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Return (ok, message) for a simple GET."""
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "podcast_scraper-hf_hub_smoke_test/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            _ = resp.read(512)
        return True, f"HTTP {code} from {url}"
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code} from {url}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"URLError from {url}: {e.reason!r}"
    except Exception as e:
        return False, f"{type(e).__name__} from {url}: {e}"


def _hub_api_probe(repo_id: str) -> tuple[bool, str]:
    try:
        from huggingface_hub import model_info
    except ImportError:
        return True, "huggingface_hub not installed (skip API probe)"
    try:
        info = model_info(repo_id, timeout=30.0)
        sha = getattr(info, "sha", None) or "unknown"
        return True, f"model_info ok, sha={sha!r}"
    except Exception as e:
        return False, f"model_info failed: {type(e).__name__}: {e}"


def _from_pretrained_probe(
    repo_id: str,
    *,
    cache_dir: Path,
    revision: str | None,
) -> tuple[bool, str]:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        return False, f"transformers not importable: {e}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_str = str(cache_dir.resolve())
    os.environ["HF_HUB_CACHE"] = cache_str
    kw: dict = {"cache_dir": cache_str, "local_files_only": False}
    if revision:
        kw["revision"] = revision
    t0 = time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, **kw)
        _ = tok.encode("smoke test", add_special_tokens=True)
        elapsed = time.perf_counter() - t0
        return True, f"AutoTokenizer.from_pretrained ok in {elapsed:.1f}s"
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return False, f"after {elapsed:.1f}s: {type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="HF hub smoke test (network + Transformers)")
    parser.add_argument(
        "--model",
        default="facebook/bart-base",
        help="Hugging Face repo id (default: same resolved target as bart-small preload)",
    )
    parser.add_argument(
        "--cache",
        default="",
        help="Override HF cache dir (default: get_transformers_cache_dir())",
    )
    parser.add_argument(
        "--skip-https",
        action="store_true",
        help="Skip raw HTTPS probe",
    )
    parser.add_argument(
        "--skip-hub-api",
        action="store_true",
        help="Skip huggingface_hub model_info probe",
    )
    args = parser.parse_args()
    alias = args.model
    resolved = DEFAULT_SUMMARY_MODELS.get(alias, alias)
    revision = get_pinned_revision_for_model(resolved)
    cache = Path(args.cache) if args.cache.strip() else get_transformers_cache_dir()

    print("=== Hugging Face smoke test ===")
    print(f"  alias/input:   {alias}")
    print(f"  resolved repo: {resolved}")
    print(f"  pinned rev:    {revision!r}")
    print(f"  cache dir:     {cache}")
    print()

    _print_env()

    any_fail = False

    if not args.skip_https:
        print("=== 1) Raw HTTPS (no Hugging Face Python client) ===")
        for url in (
            "https://huggingface.co/",
            f"https://huggingface.co/{resolved}/raw/main/config.json",
        ):
            ok, msg = _https_probe(url)
            if not ok:
                any_fail = True
            print(f"  {'OK ' if ok else 'FAIL'} {msg}")
        print()

    if not args.skip_hub_api:
        print("=== 2) huggingface_hub.model_info ===")
        ok, msg = _hub_api_probe(resolved)
        if not ok:
            any_fail = True
        print(f"  {'OK ' if ok else 'FAIL'} {msg}")
        print()

    print("=== 3) transformers.AutoTokenizer.from_pretrained (preload-equivalent) ===")
    ok, msg = _from_pretrained_probe(resolved, cache_dir=cache, revision=revision)
    if not ok:
        any_fail = True
    print(f"  {'OK ' if ok else 'FAIL'} {msg}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
