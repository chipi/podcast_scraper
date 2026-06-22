# Homelab compose drift sync — apply instructions (2026-06-22)

Per operator-question #6 on PR #1039: push the Phase 2c autoresearch
defaults (`--max-num-seqs=4`, `--enforce-eager`) into the homelab
compose source so it matches what
`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` documents.

Cross-repo change per `feedback_cross_repo_apply`: the fix lives in
`/Users/markodragoljevic/Projects/agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`,
not in this repo. Apply via the steps below from the homelab repo.

## What changed in the homelab repo

Two new flags added to the `vllm-autoresearch` service `command:`
list, just after `--max-model-len=32768`:

```yaml
      - --max-num-seqs=4
      - --enforce-eager
```

Each with a comment explaining the rationale and linking back to
`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` § "Compose defaults updated
(2026-06-17, levers A+C applied)".

## Apply instructions

From the agentic-ai-homelab repo:

```bash
cd ~/Projects/agentic-ai-homelab

# 1. Review the diff
git diff infra/vllm/autoresearch/docker-compose.yml

# 2. Commit on a feature branch
git checkout -b feat/autoresearch-compose-phase2c-flags
git add infra/vllm/autoresearch/docker-compose.yml
git commit -m "feat(autoresearch): apply Phase 2c flags (--max-num-seqs=4 --enforce-eager)

Documented in podcast_scraper/autoresearch/PER_MODEL_OPTIMAL_PARAMS.md
§ Compose defaults updated (2026-06-17). KV cache peak measured at 2.5%
even at max-num-seqs=64 → drop to 4 frees 20-40 GiB. --enforce-eager
saves 2-4 min per cold boot at ~10-15% steady-state perf cost,
acceptable for the eval-loop swap cadence.

Closes the drift surfaced as op-Q #6 on
chipi/podcast_scraper#1039 (RFC-097 v2 foundation PR)."

# 3. Push + PR via gh
git push -u origin feat/autoresearch-compose-phase2c-flags
gh pr create --title "feat(autoresearch): Phase 2c compose flags (--max-num-seqs=4 --enforce-eager)" \
  --body "$(cat <<'EOFBODY'
## Summary

Apply the Phase 2c flags that landed in podcast_scraper's
\`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md\` (2026-06-17) to the
homelab autoresearch compose, closing the drift surfaced as
operator-question #6 on chipi/podcast_scraper#1039.

- \`--max-num-seqs=4\` (down from vLLM default 256 / current 128 on live DGX). KV cache peak observed at 2.5% across the chunk-7 9-candidate sweep; max-num-seqs at 64 reserved 20-40 GiB unused. Single-stream eval workload doesn't need concurrent slots.
- \`--enforce-eager\` (added). Skips CUDA graph capture, saving 2-4 min per cold boot at ~10-15% steady-state perf cost. Acceptable for the autoresearch swap cadence (15-40 min between model swaps during a sweep); not for prod where graph capture amortizes.

## Test plan

- [ ] \`make dgx-deploy\` from laptop redeploys without error
- [ ] \`gpu-mode-swap.sh research\` brings the slot up healthy
- [ ] \`curl http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1/models\` returns 200
- [ ] First model swap during next sweep boots faster than baseline (graph-capture skip visible in vLLM startup log: "Cudagraph is disabled under eager mode")
EOFBODY
)"
```

## Drift to flag separately (operator decision)

Beyond the Phase 2c flags, the LIVE DGX compose has diverged from the
homelab compose source on:

| Field | Homelab source | Live DGX (verified 2026-06-22) |
| --- | --- | --- |
| Model id | `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` | `Qwen/Qwen3.6-35B-A3B` |
| `--gpu-memory-utilization` | 0.75 | 0.60 |
| `--max-num-seqs` | (not set; vLLM default) | 128 |
| `--api-key` | `${VLLM_API_KEY:-buddy-is-the-king}` | not set |

These suggest the LIVE compose was hand-edited at some point and the
homelab source wasn't re-synced. The Phase 2c PR above doesn't touch
those fields — they're a separate "which one is canonical" decision.

Recommendation: a follow-up `infra/vllm/autoresearch/decisions/`
ADR-style note documenting which divergence is intentional (if any),
or a `make dgx-deploy` to re-apply the homelab source verbatim and
let the divergences land as a single audit-able diff.

## Verification after deploy

```bash
ssh dgx-llm-1 'docker inspect vllm-autoresearch --format "{{.Config.Cmd}}"'
# Expected: ...--max-num-seqs=4 --enforce-eager...

ssh dgx-llm-1 'docker logs vllm-autoresearch 2>&1 | grep "Cudagraph"'
# Expected: "Cudagraph is disabled under eager mode"
```
