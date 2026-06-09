# ADR-096: DGX Spark in prod — primary-with-fallback contract

- **Status**: Accepted
- **Date**: 2026-05-23
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md), [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- **Related ADRs**: [ADR-093](ADR-093-canonical-stack-contract-and-environment-adapters.md), [ADR-097](ADR-097-self-hosted-gha-runner-policy.md)

> **Revision note (2026-05-23, same day as creation):** the original draft of this ADR excluded DGX from the prod path entirely on residential-SPOF grounds. Pivoted before any code shipped against it: cost arbitrage on prod Whisper transcription is meaningful (~90% of per-episode cost), and the mandatory-fallback pattern already required for non-prod consumers handles residential-SPOF cleanly. The new direction is: **DGX is allowed in prod with a per-stage cloud fallback contract.**

## Context

The operator's NVIDIA DGX Spark Founders Edition joins the tailnet as `tag:dgx-llm-host` per [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md). It can host Whisper Large v3, Llama 3.3 70B, Qwen 2.5 72B, and similar models at ~$0 marginal cost.

The economic case for prod usage is asymmetric:

| Stage | Cloud cost per 1h podcast | DGX cost | Worth routing? |
| --- | --- | --- | --- |
| Whisper transcription | ~$0.36 (OpenAI) | ~$0 marginal | **Yes — dominates per-episode cost** |
| Gemini summarization | ~$0.01 | ~$0 | Marginal; Gemini wins on reasoning |
| Gemini GI extraction | ~$0.02 | ~$0 | Marginal; same |
| Gemini KG extraction | ~$0.02 | ~$0 | Marginal; same |
| Embeddings (sentence-transformers) | $0 (already local CPU) | $0 (faster on GPU) | Yes for speed, not cost |

The architectural concern is residential SPOF: DGX runs on home power + ISP, no SLA. Naively routing prod to DGX would mean home-power-blip = prod pipeline failure.

The solution that makes this safe is the same pattern RFC-089 already mandates for non-prod consumers: **every DGX consumer must have automatic cloud fallback.** The pattern was originally framed as "non-prod-only safety"; this ADR realizes the same pattern works equally well for prod.

## Decision

DGX may serve prod traffic, **only under the primary-with-fallback contract**. Specifically:

### 1. Tailscale ACL permits prod → DGX

The ACL allows `tag:prod-app` → `tag:dgx-llm-host` on the LLM (`11434`) and embedding (`8001`) ports. This is a real architectural commitment, not a coincidence:

```jsonc
{
  "action": "accept",
  "src":    ["tag:prod-app"],
  "dst":    ["tag:dgx-llm-host:11434", "tag:dgx-llm-host:8001"]
}
```

### 2. Every prod LLM stage that targets DGX MUST specify a cloud fallback

In every prod profile that uses DGX, the affected stage configuration is required to include `fallback_provider`:

```yaml
transcription:
  primary:  tailnet_dgx_whisper      # DGX-hosted Whisper Large v3
  fallback: openai_whisper           # cloud fallback (existing prod provider)
```

A pull request that adds a DGX provider to a prod profile without `fallback` set fails CI validation. There is no opt-out.

### 3. Failover is automatic, atomic per batch, and observable

When a DGX call fails (timeout, connection refused, health check pre-flight fail), the provider abstraction:

1. Logs the failure with structured context (`stage`, `model`, `failure_reason`, `episode_id`).
2. Retries on DGX once with 5s backoff (defends against transient blips without flapping; budget intentionally tiny).
3. If retry fails: delegates the batch to the configured `fallback` provider transparently. Pipeline continues without operator intervention.
4. Emits a Sentry breadcrumb tagged `dgx_fallback_active=true` so the operator sees how often fallback fires.

The batch boundary matters: failover is per-API-call, not per-pipeline-run. A 10-episode pipeline that hits DGX for 7 episodes and falls back to cloud for 3 is a normal partial-degradation event, not a failure.

### 4. Monitoring surfaces fallback rate

The deploy observability work (#803) surfaces a Grafana panel **"DGX fallback rate (last 24h / 7d)"** computed from the breadcrumbs in #3. Thresholds:

- <1% — normal
- 1-10% — surface as a Sentry weekly digest item
- >10% sustained over 7 days — operator review; either fix DGX or move the stage back to cloud-primary

### 5. ACL rule reversal does not require a new ADR

If the operator chooses to **disable** prod-DGX traffic (for any reason — DGX down for maintenance, cost re-analysis, debugging), the ACL rule in §1 can be deleted by editing `tailscale/policy.hujson` directly. Prod will fall back to cloud per the per-stage configuration. No ADR revision needed for that direction; the contract is "DGX is permitted, not required."

The reverse — adding NEW DGX-served stages to prod (e.g., summarization, GI, KG) — is a per-stage decision. Each stage that moves primary-to-DGX gets:

- A specific cost-arbitrage justification (per the asymmetry table in §Context).
- A pre-prod validation period of ≥4 weeks running DGX-primary with measured fallback rate.
- A PR with operator review (no automation; this is a deliberate scope expansion).

## Rationale

- **Fallback IS the safety mechanism.** ADR-093's environment-adapter discipline is about controlled differences between environments. Primary-with-fallback is the same discipline applied to LLM stages: prod's *guarantee* is "cloud is always available"; DGX is a *cost optimization* layered on top. Residential-SPOF is a non-event because cloud is one health-check away.
- **Whisper is asymmetric.** No other prod stage offers comparable cost reduction. Starting with Whisper isolates the change and the learning. If Whisper-on-DGX works for 3-6 months at <1% fallback rate, the pattern can extend; if not, no other stage was put at risk.
- **Cost discipline is operator-friendly.** Even at hobby scale, the per-episode cost reduction (~90% savings on Whisper) is meaningful enough to be felt in monthly LLM spend. The architecture should support obvious wins, not block them on infrastructure orthodoxy.
- **No new ADR needed to disable.** This is important. The previous draft of this ADR required a full new ADR to *enable* prod-DGX. The pivoted version is symmetric — the operator can disable a stage by editing the ACL or the profile YAML. That symmetry matches how a single-operator hobby project actually evolves.

## Consequences

- **Positive**: ~90% per-episode prod cost reduction when DGX is healthy. Fallback ensures prod never breaks because of DGX. ACL is the one-line kill switch.
- **Positive**: Operator can disable a problematic DGX stage without governance overhead — edit profile YAML or ACL, redeploy.
- **Negative**: A new failure mode exists ("prod ran on fallback for the last 24h, did anyone notice?"). Mitigated by the monitoring contract in §4.
- **Negative**: Slight prod pipeline latency variance — DGX is fast when up, fallback adds the failed-call timeout (~5s × retry = ~10s) before cloud-primary kicks in. Negligible per pipeline run.
- **Neutral**: Pre-prod still uses DGX as documented in RFC-089. The non-prod use case is unaffected; this ADR adds prod as an additional consumer.

## Implementation Notes

1. **Provider abstraction must be the only enforcement point.** Pipeline code calling `provider.transcribe(audio)` should not need to know about DGX vs cloud. The provider class (`tailnet_dgx_whisper` etc.) handles health check + retry + fallback internally.
2. **Profile YAML schema validates `fallback` presence.** A profile config validator (likely a Pydantic model) refuses to load a profile that names DGX as a primary without `fallback` set.
3. **CI test exists for the contract.** A unit test deliberately knocks out the DGX endpoint for a `cloud_with_dgx_primary` profile (or whatever the prod profile is named) and verifies the pipeline still completes via cloud fallback.

## References

- [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) — full DGX integration design; this ADR supersedes that RFC's §"DGX excluded from prod path" framing
- [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) — prod hosting + blast-radius contract; primary-with-fallback is the layer above
- [ADR-093](ADR-093-canonical-stack-contract-and-environment-adapters.md) — environment-adapter discipline this ADR builds on
- [ADR-097](ADR-097-self-hosted-gha-runner-policy.md) — sister ADR on the GHA-runner side of RFC-089
- #803 (deploy observability) — implements the fallback-rate Grafana panel + Sentry breadcrumb
- Tailscale ACL: `tailscale/policy.hujson`
