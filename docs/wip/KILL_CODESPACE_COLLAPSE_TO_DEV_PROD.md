# Kill codespace + collapse env vocabulary to dev + prod

**Status**: PLAN — awaiting go.
**Trigger**: 2026-06-12. The legacy "canonical stack contract" (ADR-093) covers
four envs: `dev`, `codespace`, `pre-prod`, `prod`. The codespace was the
operator's pre-prod hop — build locally → push to codespace → `stack-test`
there → push to prod. Operator has retired that hop; the codespace branch
no longer runs.

**Goal**: drop `codespace` as a deploy target, collapse the env vocabulary to
two stages — `dev` (operator's laptop) and `prod` (Hetzner VPS) — and stop
treating `pre-prod` as its own thing. The "pre-prod-shape validation" the
codespace gave us is now covered by `make stack-test` running locally on
the laptop before deploy. The corpus + model backend selection that used to
need its own env is now a `profile:` argument the runtime resolves through
the registry (PR #977).

**What this is NOT**: this is not about killing the always-on prod VPS or
the stack-test discipline. Both stay. The shrink is purely in env
vocabulary and the codespace machinery that fanned out from the 4-env
model.

---

## Why the 4-env model existed

ADR-093 framed the original contract: same image, four env wrappers.

- `dev` — laptop, fast iteration, broken state OK.
- `codespace` — GitHub Codespace, "pre-prod" hop. Operator paid for a
  long-lived codespace, deployed prod-shaped Docker stack there, ran
  stack-test against it before promoting to the VPS.
- `pre-prod` — same shape as codespace but with a different env-var set
  (mostly distinct ports, distinct Sentry DSNs).
- `prod` — Hetzner VPS, always-on Tailscale ingress.

The codespace existed because:
- (a) **Stack-test discipline** — full Docker stack validation needed a
      reproducible Linux box, and the operator didn't want to maintain
      that locally.
- (b) **Corpus access** — eval and validation used a corpus that lived in
      codespace storage; laptop didn't have it.

Both conditions changed:
- (a) `make stack-test` now runs cleanly on the laptop via Docker
      Desktop / Colima. The codespace's "reproducible Linux" value is
      gone.
- (b) Corpora live in `data/eval/sources/`, `.test_outputs/manual/`,
      and remote backup (RFC-084's manifest + version-aware restore).
      Operator pulls what they need locally; no need for a long-lived
      remote workspace.

## Why "pre-prod" stops being its own thing

The naming was inherited: codespace+vps both speak the same Docker-stack
language; "pre-prod" was the configuration profile that ran on the
codespace box. With the codespace gone, "pre-prod" is just "prod-shape
running on the laptop or in CI" — which is exactly what `make
stack-test` is. Folding the name removes one piece of confusing
vocabulary; the discipline survives intact.

---

## Proposed model (after)

| Surface | Where it lives | What it's for |
| --- | --- | --- |
| Code | Single repo (this one) | Source of truth |
| Local dev + `make stack-test` | Laptop | Fast iteration + full Docker-stack validation before deploy |
| Production | Hetzner VPS | Real podcast feeds, operator |
| Eval corpora | `data/eval/sources/`, `.test_outputs/manual/`, remote backup | Runtime arg (`--config`, `profile:`) — not an env |
| Model backends | Cloud APIs / DGX (separate `agentic-ai-homelab` repo) / Ollama / vLLM | Wired via `profile:` + registry — runtime arg, not env |

Concrete vocab shift:
- "codespace" — **removed** (deploy target dies).
- "pre-prod" — **removed** as a separate env. Where it currently means "the
  stack-test config", say `stack-test` directly. Where it currently means
  "the codespace box", remove.
- "dev" — keeps meaning the laptop.
- "prod" — keeps meaning the VPS.

---

## Blast radius — 35 files

### Class A: Infrastructure (pure delete)

- `.devcontainer/devcontainer.json`
- `.devcontainer/Dockerfile`
- `.devcontainer/README.md`
- `.devcontainer/start.sh`
- `.github/workflows/deploy-codespace.yml` (98 lines)
- Makefile targets / helpers:
  - `deploy-codespace:` (line 857)
  - `_resolve-codespace-name:` (the helper)
  - Any other phony-list mention of `deploy-codespace`

### Class B: Workflows (remove codespace refs/branches)

- `.github/workflows/backup-corpus.yml` — currently backs up codespace
  corpus alongside prod. Drop the codespace branch.
- `.github/workflows/backup-corpus-prod.yml` — likely just a doc comment.
- `.github/workflows/post-deploy-smoke.yml` — likely conditional on
  deploy target. Drop the codespace path.

### Class C: Compose (clean refs)

- `compose/docker-compose.prod.yml` — likely just comment lines.
- `compose/docker-compose.stack.yml` — could be load-bearing; needs
  inspection.
- `compose/docker-compose.stack-test.yml` — likely the new "pre-prod
  shape" — keep as-is, possibly rename to make purpose clearer.
- `compose/grafana-agent.yaml` — probably env-label refs.
- `compose/README.md` — narrative rewrite.

### Class D: Source code (real logic)

- `src/podcast_scraper/server/pipeline_docker_factory.py` — needs
  inspection. Could be runtime branching on env, in which case the
  codespace branch dies.
- `src/podcast_scraper/server/scheduler.py` — same.

### Class E: Tests

- `tests/integration/server/test_pipeline_docker_factory.py` — drop
  codespace-specific cases.
- `tests/integration/server/test_preprod_compose_contract.py` — rename
  to `test_stack_test_compose_contract.py` (or similar) since "pre-prod"
  goes away. Test still valuable — it's the stack-shape contract.
- `tests/unit/scripts/test_corpus_snapshot_integration.py` — drop
  codespace pieces of the integration test.

### Class F: ADRs / RFCs (architectural rewrites)

- `docs/adr/ADR-083-tailscale-private-ingress-always-on-vps.md` —
  references codespace; check if load-bearing.
- `docs/adr/ADR-093-canonical-stack-contract-and-environment-adapters.md`
  — **the central one**. Either rewrite to "2-stage contract" or
  supersede with a short successor ADR.
- `docs/rfc/RFC-081-pre-prod-environment-and-control-plane.md` — likely
  obsolete; supersede.
- `docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md` — rewrite or
  supersede.
- `docs/rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md`
  — references codespace as one backup source.
- `docs/rfc/RFC-087-vps-public-edge-multi-compose.md` — references the
  multi-env compose layout.
- `docs/rfc/index.md` — index entries.

### Class G: Guides / docs (reference rewrites)

- `docs/guides/PROD_RUNBOOK.md` — codespace mentioned in the deploy flow.
- `docs/guides/STACK_CONTRACT.md` — references the 4-env contract.
- `docs/guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md` — codespace as
  backup source.
- `docs/guides/DR_DRILL_RUNBOOK.md` — codespace step in DR drill.
- `docs/guides/SERVER_GUIDE.md` — env mentions.
- `docs/ci/WORKFLOWS.md` — workflow descriptions.

### Class H: Root / top-level

- `README.md` (line 49: "dev / codespace / pre-prod / prod with one
  health surface and one `stack-test` discipline") — collapse to
  "dev / prod with one health surface and one `stack-test`
  discipline".
- `CONTRIBUTING.md` (line 401 checklist item: "If you touched
  `.devcontainer/*`, `compose/docker-compose.prod.yml`,
  `deploy-codespace.yml`, or `backup-corpus.yml`: a real codespace
  boot + operator workflow ran end-to-end") — rewrite as "If you
  touched `compose/docker-compose.prod.yml` or `backup-corpus.yml`:
  `make stack-test` + a prod deploy dry-run ran end-to-end".

### Class I: WIP / historical (probably leave-as-is)

- `docs/wip/MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md` — analysis snapshot;
  the codespace mention is part of that point-in-time picture.
- `docs/releases/RELEASE_v2.6.0.md` — release notes are historical.
- Decision: leave both. They're dated.

---

## Phased execution

Three PRs, each independently reviewable.

### PR 1 — Infrastructure delete + root refs

Scope: Classes A, H. Net: -200 lines of dead infrastructure, plus README
and CONTRIBUTING brought in line.

Files:
- Delete `.devcontainer/` (4 files)
- Delete `.github/workflows/deploy-codespace.yml`
- Remove Makefile `deploy-codespace:` target + helpers
- Edit README.md line 49 (env list)
- Edit CONTRIBUTING.md line 401 (checklist)

Risk: low. The deploy-codespace workflow hasn't fired in N days (per
operator); removing it is reversible via git history.

Why first: smallest, most reversible, highest signal that the cleanup
is real. Subsequent PRs reference this one when justifying ADR
rewrites.

### PR 2 — Source / tests / compose / workflow cleanup

Scope: Classes B, C, D, E. Net: rename `pre-prod` → `stack-test` in
test names, drop codespace-conditional branches, simplify
`pipeline_docker_factory.py`.

Files: 3 workflow files, 5 compose / compose-README files, 2 source
files, 3 test files.

Risk: medium. Source code changes need test coverage; compose changes
need a real `make stack-test` validation before merge.

Why second: depends on PR 1 having landed so the codespace deploy
workflow is already dead.

### PR 3 — ADRs / RFCs / guides rewrite

Scope: Classes F, G. Net: collapse the 4-env contract to 2 stages in
all architectural and operator-facing docs.

Files: 6 ADR/RFC files, 6 guides.

Approach options for each ADR/RFC:
- **Rewrite in place** with a "2026-06-12 amendment" block — cheap, but
  the original 4-env framing gets buried.
- **Supersede with a successor doc** — cleaner reader experience, more
  work to write.

Lean toward in-place amendments for the secondary RFCs (082, 084,
087) and a clean successor ADR for ADR-093 (the central one). Concrete
choice can wait for PR 3's scoping.

Risk: low (docs only). Won't break runtime.

Why third: it's the rewrite scope. Useful to do AFTER infra + code
cleanup so the rewrite reflects what's actually true, not what's
about to be true.

---

## Acceptance per PR

Each PR is its own gate:

**PR 1**: `make ci-fast` green; no test names contain `codespace`;
`.devcontainer/` and `deploy-codespace.yml` absent.

**PR 2**: `make ci-fast` green; `make stack-test` green on laptop;
no source/test reference to `codespace` survives outside historical
WIP docs.

**PR 3**: `make docs` (mkdocs strict) green; ADR-093 either rewritten
or superseded; all guides updated.

---

## What this does NOT touch

- Always-on Hetzner VPS hosting (RFC-082) — stays exactly as-is.
- `make stack-test` discipline — stays. This is the work the codespace
  used to do, now running locally.
- Corpus backup / restore (RFC-084) — stays. The codespace as a
  backup source goes away; the manifest + version-aware restore
  mechanism stays.
- Tailscale ingress (ADR-083) — stays.
- The autoresearch programme — orthogonal.

---

## Open questions

1. **Is "pre-prod" alive in any form?** I'm reading the operator's intent
   as "no". If pre-prod is still meaningful (e.g. a staging VPS, a docker
   profile, anything), the rewrite needs to keep it. Check before PR 3.
2. **Compose simplification scope** — `docker-compose.stack-test.yml`
   currently exists alongside `docker-compose.stack.yml` and
   `docker-compose.prod.yml`. With pre-prod gone, is the stack vs
   stack-test distinction still meaningful, or should they collapse?
3. **Backup workflow split** — `backup-corpus.yml` vs
   `backup-corpus-prod.yml`. The former was "codespace + prod"; with
   codespace dead, can they merge into one?
4. **ADR/RFC supersede policy** — for the documents most affected
   (ADR-093, RFC-081, RFC-082), is the operator preference to (a) edit
   in place with amendments or (b) write successor docs that mark the
   originals superseded?
5. **Renaming `pre-prod` tests** — drop them entirely (the contract is
   now just the prod-shape contract), or rename (`test_stack_test_compose_contract.py`)?
6. **Codespace mentions in CI workflow descriptions** (`docs/ci/WORKFLOWS.md`)
   — likely just delete the codespace sections.
