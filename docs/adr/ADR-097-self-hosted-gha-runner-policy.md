# ADR-097: GitHub Actions self-hosted runner policy on public repos

- **Status**: Accepted
- **Date**: 2026-05-23
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md)
- **Related ADRs**: [ADR-096](ADR-096-dgx-spark-non-prod-scope.md) (companion — DGX non-prod scope)

## Context

[RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) Tier 3 introduces an NVIDIA DGX Spark as a GitHub Actions self-hosted runner for autoresearch eval and ML-CI workloads. The repository (`chipi/podcast_scraper`) is **public**.

GitHub's own guidance is unambiguous: "We recommend that you only use self-hosted runners with private repositories." The threat model on a public repo with a self-hosted runner includes:

- **Fork-PR workflow execution**: a malicious contributor opens a PR that modifies a workflow file to run code on the operator's hardware.
- **Persistent runner state pollution**: one job's residue (cached dependencies, environment variables, filesystem changes) affects the next job.
- **Token/secret exposure**: a compromised job could exfiltrate runner-scoped tokens or environment secrets.
- **Long-running malicious processes**: a job that doesn't terminate properly leaves resource-consuming processes on the runner.

The operator wants the throughput + cost benefits of self-hosted (GPU-class compute for autoresearch, no GHA-minute spend on heavy ML CI) without taking on those threats.

## Decision

Self-hosted runners on this repo are governed by a **three-layer mandatory policy**. All three layers must be in place before any workflow may target `runs-on: [self-hosted, dgx-spark]`. Removing any layer requires this ADR's explicit revision.

### Layer 1 — Ephemeral runner mode

The DGX runner is registered as **ephemeral** (the `--ephemeral` flag to `config.sh` or its container equivalent). After every job, the runner instance is destroyed and a fresh one is registered. Implications:

- Workspace files, environment variables, caches do not persist between jobs.
- A malicious job cannot poison the next job's environment.
- A job that fails to clean up after itself cannot affect subsequent jobs.

The cost: every job pays a ~10-30s registration overhead. For autoresearch eval matrices (minutes to hours of compute), this is negligible.

### Layer 2 — Outside-collaborator approval gate

The repository setting **"Require approval for all outside collaborators"** stays enabled (verified by repo settings audit checklist; see Implementation Notes below). This means:

- A fork-PR from a first-time external contributor does **not** trigger workflows automatically.
- The operator must explicitly approve workflow execution after reviewing the PR diff.
- Subsequent contributions from the same contributor may or may not require approval based on contribution history (per GitHub's policy semantics).

This setting addresses the most common attack: opportunistic fork-PRs scanning public repos for self-hosted runners to abuse.

### Layer 3 — Explicit workflow allow-list

A workflow may use `runs-on: [self-hosted, dgx-spark]` **only if** that workflow filename is listed in `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md`. Enforcement:

- Pre-commit hook OR CI check (TBD during P3 implementation): refuses commits / PRs that add `runs-on: self-hosted` (or `[self-hosted, ...]`) to a workflow file not in the allow-list.
- The allow-list itself is owned by repository admins; PRs that modify the allow-list require explicit operator review.
- **Hard exclusion**: workflows that handle deployment, release, security review, secrets management, or any external-system credentials may **never** appear in the allow-list. The allow-list is restricted to **autoresearch eval workflows** and **ML-CI workflows** (model verification, sentence-transformer warmup, heavy stack-test variants).

## Consequences

- **Positive**: The three layers together close the documented attack surfaces. Public-repo + self-hosted is operationally viable. Operator gets the throughput + cost wins of self-hosted without unbounded threat exposure.
- **Positive**: The allow-list is mechanically enforceable; future contributors cannot quietly add self-hosted runs to new workflows.
- **Negative**: Each layer has a small operational cost — ephemeral overhead (~30s/job), approval friction for first-time contributors, allow-list maintenance.
- **Negative**: GHA's runner-token compromise threat is not fully eliminated; it is reduced to "an approved workflow with allow-list privileges goes rogue." Mitigation is the operator's review discipline on PRs that modify allow-listed workflow files.

## Implementation Notes

The pre-commit hook / CI check should:

1. Parse every `.github/workflows/*.yml` for any job declaring `runs-on:` that includes `self-hosted` or any custom label.
2. Read the workflow file basenames from `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md` (one filename per line, comment lines starting with `#` ignored).
3. Fail the check if any workflow targets self-hosted while not listed.
4. Run in CI on every PR (so a PR adding `runs-on: self-hosted` to a non-allow-listed workflow fails the check, surfacing the policy gap before merge).

The allow-list file format:

```markdown
# Allow-listed workflows that may target [self-hosted, dgx-spark]
# Per ADR-097. Adding to this list requires operator review.

autoresearch-eval-nightly.yml
ml-ci-warmup.yml
```

The repository-settings audit (per Layer 2) should be a periodic operator check — there is no GitHub API to verify "Require approval for all outside collaborators" is set, so it must be a documented checklist item (see PROD_RUNBOOK § Credential rotation or similar standing audit).

## Reversal criteria

This ADR is revisitable only under one of these conditions:

1. The repository becomes **private** (the original GitHub guidance no longer applies; the layers can be relaxed).
2. A **better mechanism** ships in GitHub Actions itself (e.g., per-job runner sandboxing, native allow-list enforcement) — at which point the layers should be re-evaluated rather than blindly retained.
3. A documented threat model change (e.g., the operator runs the runner in a way that breaks one of the layer assumptions) — at which point the corresponding layer must be redesigned, not removed.

## References

- [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) §"Security" + §"Locked-in answers" #1
- GitHub guidance on self-hosted runners: <https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security>
- `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md` (to be created during RFC-089 P3 implementation)
