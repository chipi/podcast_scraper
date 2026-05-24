# RFC-089: DGX Spark — tailnet-integrated AI workhorse for dev, eval, and pre-prod

> **Status:** Draft. Companion to [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) (prod hosting) and [RFC-081](RFC-081-pre-prod-environment-and-control-plane.md) (pre-prod control plane). Introduces a third infra axis: an operator-owned, always-on GPU node that joins the existing tailnet and serves as the LLM/embedding backend for non-prod workloads.

## Abstract

The operator has acquired an **NVIDIA DGX Spark Founders Edition** (GB10 Grace Blackwell superchip, 128 GB unified memory, ~1 PFLOP FP16). The machine lives at home, always-on, on residential power + ISP. This RFC defines how it integrates into the existing podcast_scraper flows — laptop, GitHub Actions, prod VPS, drill VPS — via Tailscale, and how it changes the cost / quality / latency math for LLM-heavy work.

The proposal is **explicitly scoped to non-prod**. DGX augments the system as a tailnet-resident LLM + embedding backend for dev iteration, autoresearch eval, and pre-prod LLM validation. It does **not** sit in the prod request path: prod stays on cloud providers (Gemini, OpenAI). This keeps the prod blast-radius story (RFC-082) intact while unlocking meaningful capacity gains for everything upstream of prod.

Three tiers of integration ship in sequence:

- **Tier 1 (immediate)**: replace laptop-hosted Ollama with DGX-hosted Ollama; offload embeddings; point autoresearch at DGX.
- **Tier 2 (strategic)**: new `tailnet_dgx` provider in the podcast_scraper provider abstraction; new `local_dgx_*` profiles; expand AI comparison guide with 70B-class local models.
- **Tier 3 (heavier infra)**: GHA self-hosted runner with workflow allow-list; pre-prod (per RFC-082 follow-up) uses DGX-hosted LLMs by default.

Explicit non-goal: DGX as a prod backend or prod failover target.

## Problem Statement

Today's LLM topology has three weaknesses for non-prod work:

1. **Local Ollama runs on the operator's laptop.** This couples LLM workload with editor / dev work; one heavy autoresearch run starves the IDE. Laptop hardware is also the floor on what models can run locally — 32B-class is the practical ceiling.
2. **Autoresearch evaluation is bottlenecked.** The `autoresearch/` eval matrix wants to compare many providers / models at scale. Running 70B-class locally requires DGX-class hardware; the alternative (paid API calls for every cell in the eval matrix) is expensive enough that the matrix gets pruned by cost rather than by what we want to learn.
3. **Pre-prod (RFC-082 deferred work) has no realistic local-LLM backend.** When pre-prod ships (separate v2.7 ticket #800), validating the `airgapped_*` profile end-to-end means either running a small model on the drill VPS (slow, no GPU) or running it on the laptop (not a real pre-prod). DGX gives pre-prod a real LLM backend.

A fourth, latent issue: **the AI comparison guide currently lists local models that the operator cannot practically run**, so the comparison is theoretical. DGX makes the local column real.

## Goals

1. **Move laptop-resident Ollama to DGX**, accessed via tailnet, with zero code changes to laptop-side consumers (same Ollama HTTP API, only hostname differs).
2. **Point autoresearch eval at DGX** so 70B-class local models are routinely in the matrix.
3. **Add a `tailnet_dgx` provider** to the podcast_scraper provider abstraction, with `local_dgx_balanced` and `local_dgx_full` profiles that mirror cloud profiles but route LLM stages to DGX.
4. **Provide a viable LLM backend for pre-prod** (when it ships per #800) — pre-prod validates the LLM-local path end-to-end without paying cloud costs.
5. **Establish DGX as a GHA self-hosted runner** for explicitly allow-listed workflows (nightly autoresearch, ML CI, heavy stack-test variants) — never for build/deploy/release workflows.
6. **Expand the AI comparison guide** with real measurements from DGX-hosted 70B-class models: quality, cost (electricity + amortized capex), latency.

## Non-Goals

1. **DGX is not in the prod request path.** Prod stays on Gemini + OpenAI per RFC-082. DGX downtime cannot affect prod.
2. **DGX is not a backup or failover target.** Backup-corpus lives in `chipi/podcast_scraper-backup`; failover lives on the DR drill row. DGX has no role in either.
3. **No public ingress to DGX.** DGX is reachable only via tailnet, same model as prod (RFC-082).
4. **No multi-user serving.** Single operator; no rate limiting or per-tenant quotas needed.
5. **No automated wake-on-LAN / sleep / power management.** DGX is always-on by operator commitment; this RFC does not add automation around that.
6. **No DGX-hosted secrets / state.** API keys stay in their existing locations (laptop password manager, VPS `.env`, GHA secrets). DGX accesses none of them.

## Use Cases (delta vs. existing topology)

| Use case | Today | With DGX | Delta |
| --- | --- | --- | --- |
| Operator runs `ollama` locally during dev | Ollama on laptop, 32B ceiling | Ollama on DGX, 70B+ ceiling, laptop free for IDE | Capacity + ergonomics |
| Autoresearch eval matrix sweep | Cloud APIs for every cell ($$), or trimmed matrix to fit laptop | DGX-hosted models join the matrix without API spend | Cost + breadth |
| Cluster-topics / embedding-heavy corpus rebuild | sentence-transformers on laptop CPU (~minutes for 100 episodes) | DGX-hosted embedding endpoint (~seconds) | Latency |
| Pre-prod validates `airgapped_thin` profile | Either drill VPS (slow, no GPU) or laptop (not real pre-prod) | Pre-prod calls DGX-hosted LLM = realistic local path | Realism |
| Nightly ML CI (Whisper model verification, sentence-transformers warmup) | `ubuntu-latest` GHA runners (cold every time) | DGX self-hosted runner with persistent model cache | Speed + GHA minute savings |
| AI comparison guide measurements | Cloud columns measured; local column theoretical | Both columns measured | Honesty |

## Design

### Hardware envelope

DGX Spark Founders Edition (as confirmed by operator):

| Spec | Value | Implication |
| --- | --- | --- |
| Compute | GB10 Grace Blackwell, ~1 PFLOP FP16 | Comfortably serves 70B-class at usable latency |
| Memory | 128 GB unified (CPU + GPU shared) | 70B FP16 fits with room to spare; 200B+ quantized possible |
| Storage | 4 TB NVMe (typical FE config) | ~30 model checkpoints with comfortable margin |
| Network | 10 GbE, residential ISP upstream | Tailnet links via residential upload — bandwidth-aware design needed for any large transfers |
| Power | Always-on, residential | Treat availability as best-effort, ~99%; cloud fallback mandatory for all consumers |

Models the operator will host at minimum:

- **Llama 3.3 70B Instruct** (general-purpose; comparison anchor against Gemini)
- **Qwen 2.5 72B Instruct** (strong on summarization, multilingual; KG benchmark)
- **Gemma 2 27B Instruct** (lighter; latency benchmark)
- **Whisper Large v3** (transcription benchmark vs OpenAI Whisper API)

Additional models added based on autoresearch findings.

### Tailnet integration

DGX joins the existing tailnet (`tail6d0ed4.ts.net`) via the operator's Tailscale account, **not** via an auth-key (auth-keys are reserved for ephemeral GHA + cloud-init use per RFC-082). Operator manually authenticates DGX once; tag set: `tag:dgx-llm-host`.

ACL changes in `tailscale/policy.hujson`:

```jsonc
{
  "tagOwners": {
    // existing tags...
    "tag:dgx-llm-host": ["autogroup:admin"],
  },
  "acls": [
    // existing rules...
    // Operator laptop + GHA deployer (for self-hosted runner) reach DGX Ollama (11434) and embedding shim (8001)
    {
      "action": "accept",
      "src":    ["autogroup:admin", "tag:gha-deployer"],
      "dst":    ["tag:dgx-llm-host:11434", "tag:dgx-llm-host:8001"]
    },
    // Drill VPS reaches DGX for pre-prod LLM validation; prod VPS does NOT
    {
      "action": "accept",
      "src":    ["tag:drill-app"],
      "dst":    ["tag:dgx-llm-host:11434", "tag:dgx-llm-host:8001"]
    },
    // Explicitly: no rule allowing tag:prod-app → tag:dgx-llm-host
  ],
}
```

DGX MagicDNS host (typical): `dgx-llm-1.tail6d0ed4.ts.net`. Suffix-drift handling: same `scripts/ops/resolve_*_tailnet_host.sh` pattern as prod / drill — new resolver `scripts/ops/resolve_dgx_tailnet_host.sh`.

### Server software

Ollama is the choice for v1. Rationale:

- **API compatibility** — podcast_scraper's existing `Ollama` provider works unchanged.
- **Model management** — `ollama pull llama3.3` is operator-friendly; no quantization gymnastics.
- **Concurrency** — Ollama 0.3.0+ handles modest concurrent requests adequately for autoresearch eval workloads.

vLLM is a tier-2 addition if throughput becomes a bottleneck (it almost certainly will for big eval sweeps). vLLM offers tensor parallelism and ~5–10× higher tokens/sec for large models. Decision deferred — first ship Ollama; measure; add vLLM behind a different port + provider variant if measurements justify it.

Embedding endpoint: small FastAPI shim wrapping the same `sentence-transformers/all-MiniLM-L6-v2` model that's used today (so corpora produced via DGX-hosted embedding are bit-identical to laptop-hosted, modulo CPU vs GPU determinism). Endpoint at `:8001`, HTTP POST `/embed` accepting batches.

### Provider abstraction — `tailnet_dgx`

Add a new provider in `src/podcast_scraper/providers/`:

```python
class TailnetDgxProvider(LLMProvider):
    """Routes LLM stages to an Ollama instance on the tailnet.

    Configuration:
      tailnet_dgx:
        host: dgx-llm-1.tail6d0ed4.ts.net
        port: 11434
        models:
          summarization: llama3.3:70b-instruct
          gi: qwen2.5:72b-instruct
          kg: qwen2.5:72b-instruct
          speaker_detection: gemma2:27b-instruct
        health_check_path: /api/tags
        fallback_provider: gemini    # if DGX unreachable
        request_timeout_sec: 120
    """
```

The provider exposes the same methods as cloud providers (`summarize`, `extract_gi`, `extract_kg`, etc.) and **mandatorily** delegates to `fallback_provider` if a health check fails before the call. No code path may hard-require DGX availability.

New profiles in `config/acceptance/`:

- **`local_dgx_balanced.yaml`** — Whisper local on DGX, summarization/GI/KG on DGX (Llama 3.3 70B), embedding on DGX. Fallback to `cloud_balanced` per-stage if DGX unreachable.
- **`local_dgx_full.yaml`** — same but no cloud fallback (used only for "what does pure local look like" measurements).

### Use case rollout (the three tiers)

**Tier 1 — Immediate (days)**

1. DGX joins tailnet, Ollama installed, 4 models pulled, `/api/tags` health probe responds.
2. Operator's existing local-Ollama config flips `host: localhost` → `host: dgx-llm-1.tail6d0ed4.ts.net`. Same models the laptop had, served from DGX. Single config change; no code.
3. Embedding shim deployed; existing CLI tools (`cluster-topics`, `build-validation-index`) gain an `--embedding-endpoint http://...` flag that defaults to local but accepts the DGX endpoint.
4. Autoresearch eval configs at `data/eval/configs/` add DGX-hosted models as new provider rows. Re-run baseline sweeps; compare to cloud baselines.

**Tier 2 — Strategic (weeks)**

1. `TailnetDgxProvider` implemented + unit tests + integration test (mocked Ollama).
2. `local_dgx_balanced` + `local_dgx_full` profiles land.
3. AI comparison guide gets a new section: "DGX-hosted local models vs cloud providers — quality, cost, latency." Tables backed by autoresearch data.
4. Per-episode cost comparison documented: Gemini $/episode vs DGX $/episode (electricity + amortized capex / expected lifetime episodes).

**Tier 3 — Heavier infra (weeks-months)**

1. DGX registers as GitHub Actions self-hosted runner under label `dgx-spark`.
2. Workflow allow-list policy: only workflows with `runs-on: [self-hosted, dgx-spark]` AND in the explicit allow-list at `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md` (autoresearch eval workflows, ML-CI workflows). Build/deploy/release/security workflows **never** run on self-hosted.
3. Pre-prod (per #800) configures `tailnet_dgx` as the default LLM backend when `PODCAST_ENV=pre-prod`.

### Availability + fallback contract

- DGX availability target: **99% best-effort** (residential power + ISP). No SLA.
- Every consumer of DGX (provider, autoresearch, CLI, pre-prod) **must** implement one of:
  - Cloud fallback (`tailnet_dgx → gemini` per-stage), OR
  - Hard-fail with operator-visible error (acceptable for autoresearch eval runs; not acceptable for any pipeline run).
- Health check: HTTP `GET http://dgx-llm-1.tail6d0ed4.ts.net:11434/api/tags` with 5s timeout. Healthy if responds 200 + at least one model listed.
- Self-hosted runner availability: if DGX is unreachable when a workflow dispatches, the workflow falls back to `ubuntu-latest` (slower but functional) via a `runs-on:` strategy matrix. Implementation note: GHA self-hosted runner fallback isn't built-in; needs explicit `runs-on: ${{ <determined-by-prior-step> }}` pattern.

### Operator runbook additions

New section in PROD_RUNBOOK or a separate `docs/guides/DGX_RUNBOOK.md`:

- DGX bring-up checklist (Ollama install, model pulls, embedding shim, tailnet join, ACL update)
- Day-2 ops (model updates, OS updates, log inspection, GPU utilization check via `nvidia-smi` over SSH)
- Troubleshooting (DGX unreachable from laptop / GHA / drill VPS — same suffix-drift / ACL debugging playbook as prod)
- Model catalog (which models are pulled, when, and the autoresearch findings that justified each)
- Cost ledger (electricity tracking — useful for the comparison guide)

### Security

- DGX joins tailnet with operator account credentials, not auth-key. Tailscale ACL is the only network-layer auth.
- No app-layer auth on Ollama / embedding endpoints. Tailnet is the trust boundary, same model as prod's tailnet-only API.
- DGX hosts no secrets, no API keys, no production data. Models are public weights. Embedding shim is stateless.
- DGX is at operator's residence. Physical access = root access — no different from any other operator-owned machine on the tailnet.
- For self-hosted GHA runner (Tier 3): explicit workflow allow-list at `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md`. Public-repo PRs from external contributors **must not** trigger self-hosted runs. GHA's default behavior already requires `runs-on: self-hosted` to be in the workflow file itself, and external PRs run from the base ref's workflow file (which the contributor can't change without review) — this is sufficient if the policy is followed.

### Costs

| Line | Today | With DGX (Tier 1+2 shipped) | Delta |
| --- | --- | --- | --- |
| Capex | $0 | One-time DGX Spark purchase (~$3-4k retail) | already paid; treat as sunk |
| Operator laptop | unchanged | freed from GPU work | qualitative |
| Cloud LLM spend (dev / autoresearch) | ~$10-30/month estimated (Gemini, OpenAI for eval matrix) | ~$0-5/month (most eval runs DGX) | $5-25/mo saved |
| Cloud LLM spend (prod) | unchanged | unchanged (prod doesn't touch DGX) | $0 |
| GHA minutes (heavy ML jobs) | a few hundred / month | DGX self-hosted = ~0 GHA minutes for those jobs | small saving (operator is below GHA free-tier ceiling anyway) |
| Electricity | unchanged | DGX always-on at ~150-300W → ~$15-30/month at $0.20/kWh | -$15-30/mo |
| Net monthly | baseline | -$10 to +$10 (depends on eval intensity) | Roughly neutral; the win is qualitative (capacity, ergonomics, breadth) |

The financial argument is not "save money." The argument is "for the same monthly run-rate, get 70B-class local models in the eval matrix + freed-up laptop + faster embeddings + a realistic pre-prod LLM."

## Phased Rollout

| Phase | Scope | Acceptance |
| --- | --- | --- |
| **P0 — DGX bring-up** | Hardware on tailnet, Ollama installed, 4 models pulled, embedding shim deployed, ACL updated, `resolve_dgx_tailnet_host.sh` added. | Operator can `curl http://dgx-llm-1.<tailnet>:11434/api/tags` from laptop and get a populated model list. ACL refuses prod VPS. |
| **P1 — Tier 1 shipped** | Laptop's Ollama config flipped to DGX; autoresearch sample run completed against DGX; embedding shim wired into one CLI tool. | One end-to-end autoresearch eval run produces results identical (within determinism noise) to cloud baseline. |
| **P2 — Tier 2 shipped** | `TailnetDgxProvider` lands; `local_dgx_*` profiles ship; AI comparison guide updated with measurements. | A pipeline run with `--profile local_dgx_balanced` produces an episode end-to-end identical (within determinism) to `cloud_balanced` for the LLM-driven stages. Comparison-guide PR cites real numbers. |
| **P3 — Tier 3 shipped** | GHA self-hosted runner registered; allow-list policy in place; nightly autoresearch runs on DGX; pre-prod (per #800) uses DGX. | Autoresearch nightly workflow runs on `[self-hosted, dgx-spark]` for ≥7 consecutive nights without operator intervention. |

Each phase ends with an explicit checkpoint. Stop-and-ship after any phase if priorities shift; nothing later builds load-bearing assumptions on later phases.

## Integration with v2.7 issues (optional touchpoints)

This RFC is **independent of every v2.7 infra issue filed** (#796–#806). Nothing in those issues requires DGX to ship; nothing in this RFC requires those issues to ship. The two tracks can land in any order.

There are three soft touchpoints where DGX-work *enhances* an existing issue when both happen to be available — but in each case the issue ships first as a complete unit, and DGX adds a quality upgrade later if/when its phase lands:

| Issue | Soft touchpoint with RFC-089 |
| --- | --- |
| **#800 parkable pre-prod** | Pre-prod ships with cloud LLM (or laptop Ollama) as the v1 backend. When RFC-089 P3 lands, pre-prod gains a tailnet-resident LLM backend (DGX-hosted) that makes the `airgapped_*` path realistic + free to validate. No code change to #800's deliverables — only a config flip in pre-prod's `.env`. |
| **#803 deploy observability + MCP** | The MCP server in #803 D3 ships with 7 prod-state tools. When RFC-089 P0+ is up, an optional 8th tool `dgx_health()` can surface DGX availability to the same agent surface. Not in the #803 acceptance criteria; additive. |
| **#804 LLM cost monitoring** | #804's daily cost rollup + soft caps cover cloud LLM spend regardless of DGX. RFC-089's AI comparison guide cites measurements from #804's metrics for the "DGX vs cloud" cost section. Bidirectional reuse, no dependency direction. |

Hard rule for the next agent picking up any of these: **do not write code or docs that assume DGX exists.** Treat DGX as "if available, route to it; if not, the v1 path is the contract." This protects both tracks from rotting if one slips.

| Option | Pro | Con | Why rejected |
| --- | --- | --- | --- |
| **A. Keep Ollama on laptop, no DGX** | Status quo, no setup | 32B ceiling; laptop contention; AI comparison guide stays theoretical | This is the problem the RFC is solving |
| **B. Use cloud GPU for non-prod (RunPod / Modal / Lambda)** | Pay-per-use; no capex | Per-hour cost adds up fast for autoresearch; no model persistence; cold starts | Operator already owns DGX |
| **C. DGX in prod request path** | Cost savings on cloud LLM spend; smaller cloud bill | Residential SPOF; prod blast radius depends on home power + ISP; conflicts with RFC-082 isolation goals | Non-goal; documented above |
| **D. DGX as backup/snapshot target** | Local storage; cheap | Duplicates `chipi/podcast_scraper-backup`; introduces home dependency on backup path | Non-goal; backups stay in their own repo |
| **E. vLLM instead of Ollama at v1** | ~5-10× throughput on large models | Operator already runs Ollama locally; model management more complex | Defer; add vLLM behind separate port if measurements justify |
| **F. Public ingress on DGX (Cloudflare Tunnel + OAuth)** | Operator can reach from anywhere | Adds attack surface; not needed (tailnet client on every device the operator uses) | Operator's laptop and phone are already on the tailnet |

## Decisions made

### Foundational (locked at draft)

1. **Always-on, no power management.** Operator commits to leaving DGX powered. RFC adds no wake/sleep automation.
2. **Tailnet-only, no public ingress.** Same trust model as prod.
3. **Ollama at v1**, vLLM deferred to a measurement-driven decision; explicit trigger pinned under "Locked-in answers" below.
4. **DGX excluded from prod path by ACL.** `tag:prod-app` → `tag:dgx-llm-host` has no rule; explicit non-rule.
5. **Cloud fallback mandatory** for every consumer. No hard-required-DGX code paths.
6. **GHA self-hosted runner is opt-in via explicit allow-list**, never default.
7. **Model SHAs pinned in autoresearch configs** for reproducibility — not floating Ollama tags.
8. **No model cache backup.** Models are redownloadable; document the re-pull procedure in `DGX_RUNBOOK.md`.

### Locked-in answers to the pre-draft open questions

1. **Self-hosted runner safety on a public repo.** Three mandatory layers before P3 ships:
   - Runner runs in **ephemeral mode** (workspace + runner reset between every job) — defeats persistent-state poisoning.
   - Repo setting **"Require approval for all outside collaborators"** stays enabled — defeats unreviewed-fork-PR workflow triggers.
   - **Explicit workflow allow-list** at `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md`. Pre-commit hook or CI check refuses any new workflow that adds `runs-on: [self-hosted, dgx-spark]` while not in the allow-list. Build / deploy / release / security workflows never on self-hosted.
2. **Embedding GPU vs CPU determinism — known quirk, accept the gap.** sentence-transformers on GPU produces vectors that are **not bit-identical** to CPU but **functionally equivalent** (top-K query results within ~1 rank shuffle). FAISS index files built on DGX are not byte-comparable to laptop-CPU-built indexes; tests assert functional equivalence (top-K membership), not byte equality. Documented in `DGX_RUNBOOK.md` so future contributors don't hunt this.
3. **Ollama → vLLM switch trigger.** Stay on Ollama until **a recurring autoresearch matrix takes >2h end-to-end on Ollama** for in-scope eval models. At that point: spike vLLM on the same matrix, measure, decide on numbers. Until that trigger fires, vLLM is off the roadmap.
4. **DGX as graph-analysis backend — deferred to v2.8.** Reopens when BOTH conditions are met: corpus exceeds ~1000 episodes (today ~100) AND graph algorithms on the api container become CPU-bound. New RFC at that point; no DGX-graph work in 2.7.
5. **Model pulls: directly from public registry to DGX, pre-staged off the critical path.** Never relay model files through other tailnet hosts. Documented in `DGX_RUNBOOK.md` as a "do it overnight or during a meeting" task. If a model is needed urgently and not pre-staged, fall back to cloud for that stage.

## References

- [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) — prod hosting; this RFC explicitly excludes DGX from the prod path defined there
- [RFC-083](RFC-083-prod-failover-orchestration-and-cutover.md) — failover; this RFC explicitly does not use DGX as a failover target
- [RFC-081](RFC-081-pre-prod-environment-and-control-plane.md) — pre-prod control plane; this RFC provides the LLM backend for pre-prod once #800 ships
- [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md) — operational baseline that `DGX_RUNBOOK.md` will mirror
- Related v2.7 issues:
  - #800 — parkable pre-prod (will consume DGX as LLM backend per Tier 3)
  - #803 — deploy observability + MCP (could surface DGX health as an MCP tool)
  - #804 — LLM cost monitoring (will measure DGX vs cloud cost in the comparison guide)
  - #806 — public DNS fallback RFC (independent; both RFCs are about expanding the operator's reachability story but along different axes)
- DGX Spark spec: NVIDIA product page (consult before final review)
- Ollama: <https://ollama.com>
- vLLM: <https://github.com/vllm-project/vllm>
