# ADR-082: GitOps App Deploy via stack-test and GitHub Actions

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)

## Context & Problem Statement

The always-on host must run **the same** GHCR image tags and compose overlays as validated in CI.
Operators should not SSH for routine releases; **`main`** should remain the source of truth for
what is intended to run after gates pass.

## Decision

**Application** promotion to the production VPS follows this **contract**:

1. **`main`** is the branch that reflects production intent after review.
2. **`Stack test`** on **`main`** must succeed before treating that SHA as a release candidate for the
   Docker compose stack.
3. **Image publish** to GHCR (tags including **`:sha-<short>`** for **`api`**, **`viewer`**, and
   **`pipeline-llm`**) is tied to the validated **`main`** pipeline.
4. **`deploy-prod.yml`** performs Tailscale SSH to **`deploy@`** and runs **`infra/deploy/deploy.sh`**,
   wiring **`PODCAST_RELEASE=sha-<short>`** for the chosen SHA.

**Automation target (RFC-082 Decision 6):** add **`workflow_run`** on successful **`Stack test`**
so deploy starts automatically after publish for the same commit family. Until that trigger is merged,
**`deploy-prod.yml`** may remain **`workflow_dispatch`** only; operators then **`gh workflow run
deploy-prod.yml`** after the same gates (see [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md)).

**Infrastructure** (`infra/**`) is **not** auto-applied on merge: PR CI shows **`tofu plan`**;
**`infra-apply.yml`** stays **manual** **`workflow_dispatch`** with operator approval so a bad merge
cannot immediately mutate Hetzner or shared Tailscale resources.

## Rationale

- **App vs infra blast radius** — frequent image rollouts are low-touch and gate on stack-test;
  rare cloud changes stay human-gated (RFC-082 Decision 6).
- **Single promotion line** — stack-test ties the running digest to integration validation on **`main`**.

## Alternatives Considered

1. **Deploy on every green `main` push without stack-test** — Rejected; weakens the compose-level
   gate that catches failures Python-only CI can miss.
2. **Auto `tofu apply` on merge for `infra/**`** — Rejected; risks destroy/typo blast radius; plan in
   PR + manual apply is the control.
3. **Kubernetes + Flux/ArgoCD** — Explicit non-goal in RFC-082 for this footprint.

## Consequences

- **Positive**: Routine releases do not require ad-hoc `docker pull` on the laptop; deploy logs live
  in GitHub Actions once **`workflow_run`** is enabled.
- **Negative**: If stack-test is red, **`main`** must not be treated as deploy-ready; with manual
   **`deploy-prod`**, operators must not deploy before confirming stack-test green.
- **Neutral**: Drill uses **`drill-deploy.yml`** with the same image contract, different SSH secret
  and environment.

## Implementation Notes

- **Workflows**: `.github/workflows/deploy-prod.yml`, stack-test workflow(s), image publish jobs on
  **`main`**
- **Operator docs**: [PROD_RUNBOOK.md](../guides/PROD_RUNBOOK.md)

## References

- [RFC-082 — Decision 6 — GitOps loop](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
