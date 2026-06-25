#!/usr/bin/env python3
"""Capture the project's data + setup state as a JSON snapshot.

Runs zero-cost: no tests, no eval, no model loads. Just inspects what
files / configs / registries are present and emits a JSON document the
weekly baseline workflow appends into ``data/baselines/``.

Output shape:

    {
      "captured_at": "2026-06-25T04:00:00+00:00",
      "git": {"commit": "...", "branch": "..."},
      "profiles": {"count": 13, "names": ["airgapped", ...]},
      "model_registry": {"preset_count": 9, "preset_hash": "..."},
      "datasets": {"<dataset_id>": {"episodes": int, "feeds": int}, ...},
      "recent_runs": [{"run_id": "...", "completed_at": "..."}, ...],
      "dgx_vllm": {"reachable": bool, "model": str|null, ...}
    }

Usage::

    .venv/bin/python scripts/baselines/capture_setup_probe.py \\
        [--output path.json] [--dgx-host host:port]

Reads the DGX host from ``--dgx-host`` or the ``DGX_VLLM_ENDPOINT``
env var; falls back to "not probed" when neither is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _capture_git() -> Dict[str, str]:
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _git(["describe", "--tags", "--always", "--dirty"]),
    }


def _capture_profiles() -> Dict[str, Any]:
    pdir = _REPO_ROOT / "config" / "profiles"
    if not pdir.is_dir():
        return {"count": 0, "names": []}
    names = sorted(p.stem for p in pdir.glob("*.yaml"))
    return {"count": len(names), "names": names}


def _capture_model_registry() -> Dict[str, Any]:
    """Hash the model_registry preset module so we can detect silent
    drift between snapshots. Doesn't unpack contents."""
    p = _REPO_ROOT / "src" / "podcast_scraper" / "providers" / "ml" / "model_registry.py"
    if not p.is_file():
        return {"file_present": False}
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    # Cheap preset count via text scan — exact value isn't critical,
    # the hash is the drift signal.
    text = p.read_text(encoding="utf-8")
    preset_count = text.count("ProfilePreset(")
    return {"file_present": True, "preset_hash": h, "preset_count_hint": preset_count}


def _capture_datasets() -> Dict[str, Any]:
    ddir = _REPO_ROOT / "data" / "eval" / "datasets"
    if not ddir.is_dir():
        return {}
    out: Dict[str, Any] = {}
    for p in sorted(ddir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        episodes = data.get("episodes") or data.get("episode_ids") or []
        out[p.stem] = {
            "episodes": len(episodes) if isinstance(episodes, list) else 0,
            "feeds": (
                len({e.get("feed_id") for e in episodes if isinstance(e, dict)})
                if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict)
                else 0
            ),
        }
    return out


def _capture_recent_runs(limit: int = 5) -> List[Dict[str, Any]]:
    rdir = _REPO_ROOT / "data" / "eval" / "runs"
    if not rdir.is_dir():
        return []
    by_mtime = sorted(
        (p for p in rdir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    out: List[Dict[str, Any]] = []
    for run_dir in by_mtime:
        fp = run_dir / "fingerprint.json"
        captured_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        out.append(
            {
                "run_id": run_dir.name,
                "completed_at": captured_at,
                "fingerprint_present": fp.is_file(),
            }
        )
    return out


def _capture_dgx(endpoint: str | None) -> Dict[str, Any]:
    if not endpoint:
        return {"reachable": False, "reason": "no_endpoint"}
    url = endpoint.rstrip("/") + "/v1/models"
    try:
        # Short timeout so a missing tailnet doesn't stall the workflow.
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.load(resp)
        models = [m.get("id") for m in payload.get("data") or [] if isinstance(m, dict)]
        return {
            "reachable": True,
            "endpoint": endpoint,
            "models": models,
        }
    except (urllib.error.URLError, socket.timeout, json.JSONDecodeError) as exc:
        return {"reachable": False, "endpoint": endpoint, "reason": str(exc)[:120]}


def capture() -> Dict[str, Any]:
    return {
        "captured_at": _now_iso(),
        "git": _capture_git(),
        "profiles": _capture_profiles(),
        "model_registry": _capture_model_registry(),
        "datasets": _capture_datasets(),
        "recent_runs": _capture_recent_runs(),
        "dgx_vllm": _capture_dgx(
            os.environ.get("DGX_VLLM_ENDPOINT") or None,
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="Write JSON here (default: stdout)")
    p.add_argument(
        "--dgx-host",
        default=None,
        help="DGX vLLM endpoint, e.g. http://host:8003 (overrides env var)",
    )
    args = p.parse_args()
    if args.dgx_host:
        os.environ["DGX_VLLM_ENDPOINT"] = args.dgx_host
    snap = capture()
    blob = json.dumps(snap, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
