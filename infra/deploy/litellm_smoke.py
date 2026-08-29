"""Gateway smoke: does the deployed LiteLLM serve what we shipped, end to end?

ONE implementation, two callers — deploy-litellm.sh (fatal post-deploy gate) and
scripts/ops/prod_ops_health.sh (daily between-deploys check, #1876). It was extracted from the
deploy script the day both were needed, precisely so the two can never drift: a health check that
runs different code than the deploy gate would eventually pass where the gate fails, and that
disagreement is worse than either bug alone.

Two checks, both hard failures:

1. CONFIG IS LIVE — every alias in the shipped config.yaml appears in the served /v1/models.
   Catches the stale-inode class (2026-08-29: a bind-mounted single file pinned the old inode and
   a green deploy served a two-week-old config), a config the gateway rejected, or a partial load.
2. E2E THROUGH THE ROUTE — a 1-token completion through each podcast-* chat alias. Catches an
   expired/revoked upstream key, a dead route, or a bad model id — the failure modes that
   otherwise surface mid-corpus as silent fail-open degradation. Costs a fraction of a cent.
   Scoped to podcast-* because those are the pipeline's contract; homelab-*/groq-* are covered by
   check 1 only (groq-whisper is audio_transcription — it has no chat endpoint to smoke).

Env contract (all required):
  SMOKE_MASTER_KEY   gateway master key (never printed)
  SMOKE_CONFIG       path to the shipped infra/litellm/config.yaml
  SMOKE_BASE         gateway base URL, e.g. http://127.0.0.1:4001

Exit 0 = both checks pass. Exit 1 = any failure, with one actionable line per failure on stderr.
Runs on the box's system python3 (has PyYAML); stdlib otherwise — no pip installs on prod.
"""

import json
import os
import sys
import urllib.request

import yaml


def main() -> int:
    base = os.environ["SMOKE_BASE"].rstrip("/")
    hdr = {
        "Authorization": "Bearer " + os.environ["SMOKE_MASTER_KEY"],
        "Content-Type": "application/json",
    }

    with open(os.environ["SMOKE_CONFIG"], encoding="utf-8") as fh:
        shipped = [m["model_name"] for m in yaml.safe_load(fh)["model_list"]]

    req = urllib.request.Request(base + "/v1/models", headers=hdr)
    served = {m["id"] for m in json.load(urllib.request.urlopen(req, timeout=15))["data"]}

    missing = [a for a in shipped if a not in served]
    if missing:
        print(
            f"SMOKE FAIL: shipped aliases not served (stale container or rejected config): "
            f"{missing}",
            file=sys.stderr,
        )
        return 1
    print(f"smoke 1/2 OK: all {len(shipped)} shipped aliases served")

    failures = []
    for alias in (a for a in shipped if a.startswith("podcast-")):
        body = json.dumps(
            {
                "model": alias,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Say OK"}],
            }
        ).encode()
        try:
            r = urllib.request.Request(base + "/v1/chat/completions", data=body, headers=hdr)
            resp = json.load(urllib.request.urlopen(r, timeout=90))
            resp["choices"][0]["message"]["content"]
            tokens = resp.get("usage", {}).get("total_tokens")
            print(f"smoke 2/2: {alias} -> completion OK ({tokens} tok)")
        except Exception as exc:  # noqa: BLE001 — every failure mode here means the same: not e2e
            failures.append(f"{alias}: {type(exc).__name__}: {exc}")
    if failures:
        print(
            "SMOKE FAIL: alias(es) not working end-to-end (expired upstream key? dead route?):",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("smoke 2/2 OK: every podcast-* alias completes end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
