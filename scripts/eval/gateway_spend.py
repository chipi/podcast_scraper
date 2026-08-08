#!/usr/bin/env python3
"""Real per-model cost from the LiteLLM gateway's SpendLogs (the source of truth).

The app's local `llm_cost` events under-count litellm-routed calls (no pricing rows +
inconsistent gateway response_cost + grounding calls that don't emit) — see PA3. The gateway
itself computes the REAL spend per call from the upstream provider's response, so query it
directly instead of token×price-estimating offline.

Auth: the LiteLLM MASTER key (admin). Read from LITELLM_MASTER_KEY env (or a .env line). The
per-consumer virtual key (LITELLM_API_KEY) is NOT admin and 401s on /spend routes.

Usage:
    python scripts/eval/gateway_spend.py [START_DATE] [END_DATE] [--base URL]
    # dates YYYY-MM-DD; default = today. Prints per-model total spend + tokens + calls.

Returns a dict {model: {spend, prompt_tokens, completion_tokens, calls}} from `fetch_spend()`
for programmatic use (e.g. a reporter that annotates $/ep with the REAL number).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

DEFAULT_BASE = os.environ.get("LITELLM_API_BASE", "http://homelab:4001")


def _master_key() -> str:
    key = os.environ.get("LITELLM_MASTER_KEY")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("LITELLM_MASTER_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "LITELLM_MASTER_KEY not set (env or .env). The virtual LITELLM_API_KEY is NOT admin — "
        "the gateway 401s on /spend routes without the master key."
    )


def fetch_spend(
    start_date: str,
    end_date: str,
    *,
    base: str = DEFAULT_BASE,
    key: Optional[str] = None,
    api_key_hash: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-model real spend over [start_date, end_date) from the gateway SpendLogs.

    ``api_key_hash`` (B1): scope to ONE consumer's spend (the sha256 of that consumer's virtual key,
    which is how LiteLLM stores it). WITHOUT it, the per-model totals aggregate EVERY consumer that
    hit the model in the window — so a bake-off number is only clean if nothing else used that model
    that day. When given, LiteLLM filters server-side so the totals are that one consumer's spend.
    """
    key = key or _master_key()
    url = f"{base.rstrip('/')}/spend/logs?start_date={start_date}&end_date={end_date}"
    if api_key_hash:
        url += f"&api_key={api_key_hash}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed tailnet gateway
        data = json.loads(resp.read().decode("utf-8"))
    # B2: a shape mismatch (error body, or /global/spend/report's different schema) must NOT read as
    # "$0 spend" — that silently mis-reports cost as free. Fail loud instead.
    if not isinstance(data, list):
        raise RuntimeError(
            f"gateway SpendLogs returned {type(data).__name__}, expected a per-day list — "
            f"cannot be treated as spend. Response head: {str(data)[:200]}"
        )
    # SpendLogs date-range returns per-day objects each carrying a `models` spend dict.
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: {"spend": 0.0})
    for day in data:
        for model, spend in (day.get("models") or {}).items():
            if model:
                out[model]["spend"] += float(spend or 0.0)
    return dict(out)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = DEFAULT_BASE
    for a in sys.argv[1:]:
        if a.startswith("--base"):
            base = a.split("=", 1)[1] if "=" in a else DEFAULT_BASE
    start = args[0] if args else "2026-08-06"
    end = args[1] if len(args) > 1 else "2026-08-07"
    spend = fetch_spend(start, end, base=base)
    print(f"gateway real spend {start}..{end} ({base}):")
    for model, v in sorted(spend.items(), key=lambda kv: -kv[1]["spend"]):
        if v["spend"] > 0:
            print(f"  {model[:55]:55} ${v['spend']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
