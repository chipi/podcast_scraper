# ADR-093: Canonical Stack Contract Versus Environment Adapters

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- **Related**: [GitHub #762](https://github.com/chipi/podcast_scraper/issues/762)

## Context & Problem Statement

The product stack is exercised from several surfaces — GitHub Codespaces pre-prod, production and
DR-drill VPS hosts, ephemeral Compose in CI (**stack-test**), and orchestrated drill workflows.
When **topology** (Compose merges, corpus paths on disk, how health is probed, or behavioral
gates) silently diverges across those surfaces, operators see **“green in CI, wrong on a host”**
failures despite legitimate differences in **transport** (SSH over Tailscale, Codespace lifecycle,
runner-local Docker, HTTPS to a drill hostname).

DR-drill orchestration surfaced the need for a **repeatable discipline**: deploy is not credible
until checks match the rigor used on hosts, without pretending every playbook runs identical
steps end to end.

## Decision

1. **Stack contract** (what stays the same wherever the product runs):
   - Compose layering and **service identities** (**`api`**, **`viewer`**, **`pipeline-llm`**, …)
     as defined per surface but **without ad hoc one-off overlays** unless documented at the hub
     (see **`compose/`** and **`infra/deploy/deploy.sh`** for VPS parity).
   - Corpus **logical** contract — host corpus root bind-mounted consistently to **`/app/output`**
     in long-running containers, with codespace versus VPS differences recorded in **one mapping**
     (RFC-082, **`compose/docker-compose.prod.yml`**, runbooks).
   - **Health semantics** aligned with GH-745: validate **`/api/health`** from **inside** the
     **`api`** container on port **8000** when host publish differs; reject “loopback :8000 on the
     host” as sufficient proof for Compose layouts that do not publish the API on the host.
2. **Environment adapters** (what may differ deliberately):
   - Network path (**Tailscale** SSH **`deploy@`**, **Codespaces** **`postStart`**, **GHA runner**
     Docker Compose, **HTTPS MagicDNS** for drill Playwright).
   - Credential names (**`PROD_*`**, **`DRILL_*`**, **`TS_AUTHKEY`**, deploy keys).
   - Typed confirmation strings on high-blast workflows.
   - Optional image **`sha-<short>`** overrides for incidents.
   - **Explicit non-goal**: we do **not** require one literal SSH invocation style everywhere.
3. **Behavioral gate**: wherever we claim “deployed stack is correct”, prefer the **same contract**
    as **`tests/stack-test`** (for example **`stack-viewer.spec.ts`**). Ingress differences (HTTPS,
    TLS pinning flags) change **transport only**, not the semantic assertions performed on the stack.
4. **Steady-state vs recovery playbooks**: the **routine** story is **preflight → deploy or bring-up →
    health smoke → behavioral gate when required**. **Corpus restore from tarball** is **not**
    part of that routine path for Codespaces or day-to-day prod; it belongs to **DR drill
    choreography** and discrete **manual restore** workflows (**`prod-restore-corpus`**, drill
    restore), or runbook sections explicitly labeled disaster recovery — not the default headline
    sequence for steady operations.
5. **Implementation coupling**: mirrored automation (**e.g.** prod versus drill corpus restore steps)
    must share **`scripts/ops/`** (or thin **`Makefile`** wrappers) so mechanically identical
    procedures cannot silently diverge. **#762** landed shared VPS restore scripts and workflow
    upload paths; future contract edits must update [STACK_CONTRACT.md](../guides/STACK_CONTRACT.md)
    and [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md)
    together.

## Rationale

- Separates **reusable truth** about the stack from **replaceable wires**, matching how the repo
  actually deploys (**`infra/deploy/deploy.sh`**, Compose README, **`stack-test`**).
- Preserves DR-drill rigor (**smoke + Playwright**) as the **template for seriousness**, without
  copying every drill-only phase (simulate restore, destroy) into normal prod workflows.
- Reduces onboarding load: operators learn **contract + adapter table** rather than unrelated
  per-surface folklore.

## Alternatives Considered

1. **One reusable workflow YAML for every host** — Rejected as primary approach: diminishing
   returns versus shared shell and documentation; YAML parameter explosion for small deltas.
2. **Restore as a universal playbook step after every deploy** — Rejected as misleading and
   high-blast-radius for steady state (conflicts with **#762** clarification).
3. **RFC-only prose, no ADR** — Rejected because this is an ongoing **boundary** affecting many
   files; an **Accepted** ADR gives a durable pointer for reviewers.

## Consequences

- **Positive**: Clear criterion for edits (“contract change” vs “adapter change”); easier reviews;
  shared Playwright contract across CI and drill.
- **Negative**: Requires keeping an **audit table** (or equivalent hub doc) and indices in sync as
  new surfaces appear.
- **Neutral**: Does not by itself mandate a single reusable deploy workflow YAML; prod and drill
  deploy stay parallel callers of **`deploy.sh`**.

## Implementation Notes

- **Hub**: [STACK_CONTRACT.md](../guides/STACK_CONTRACT.md) — audit table and steady vs recovery
  playbooks; linked from **`infra/README.md`**, runbooks, **`docs/ci/WORKFLOWS.md`**, and
  **`compose/README.md`**.
- **VPS deploy script**: **`infra/deploy/deploy.sh`** is the canonical deploy path for prod and
  drill app deploy over SSH.
- **VPS corpus restore**: **`scripts/ops/restore_corpus_from_tarball_host.sh`** and
  **`scripts/ops/resolve_latest_snapshot_prod_tag.sh`** and runner
  **`scripts/ops/corpus_snapshot/download_and_verify_snapshot.sh`** — shared by **`prod-restore-corpus.yml`**
  and **`drill-restore-corpus.yml`** (runner uploads script + tarball over Tailscale SSH).
- **CI gate**: **`.github/workflows/stack-test.yml`** + **`tests/stack-test/`**; drill uses
  **`drill-stack-playwright.yml`** with HTTPS adapter only.
- **Corpus manifest and version-aware restore:** [ADR-092](ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md),
  [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).

## References

- [RFC-082: Production hosting](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [ADR-082: GitOps app deploy via stack-test and GHA](ADR-082-gitops-app-deploy-via-stack-test-and-gha.md)
- [ADR-084: Full-stack Docker Compose topology](ADR-084-full-stack-docker-compose-topology.md)
- [ADR-085: Ephemeral stack-test integration gate](ADR-085-ephemeral-stack-test-integration-gate.md)
- [Issue #762](https://github.com/chipi/podcast_scraper/issues/762)
