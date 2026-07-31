# ADR-141: Operator = two planes (control + public); keep the split, fix the naming

- **Status**: Accepted
- **Date**: 2026-08-01
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8), advisor review (Fable 5)
- **Related**: [ADR-116](ADR-116-privilege-split-public-control-api.md) (the governing
  privilege-split decision), [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md)
  (shared edge), RFC-108 / #1320 (public operator surface), `docs/security/THREAT_MODEL.md`

## Context

The prod VPS runs the operator as **two planes**, which repeatedly reads as "a confusing
duplicate." It is not a duplicate — it is the privilege split ADR-116 prescribes — but the
**naming hides that**, and this ADR records the corrected mental model + the decision to keep
the split and fix only the names/docs.

The same `api`/`viewer` image runs in two modes (like a CMS: an **authoring/control** env and
a **read-only consumption** env), linked only by the **corpus volume**, not by the network:

| Plane | Compose project | Deploy workflow (file) | Contains | Exposure |
| --- | --- | --- | --- | --- |
| **Operator CONTROL** (authoring) | `-p compose` | `deploy-prod.yml` ⚠ misnamed | full `api` **+ docker.sock + provider keys**, `viewer`, **`pipeline`** (owns/writes the corpus) | **tailnet-only** |
| **Operator PUBLIC** (consumption) | `-p operator` | `deploy-operator.yml` | app-only `api` (`PODCAST_SERVE_OPERATOR_PUBLIC=1`, **no sock, no keys**, curated read-only routes), `viewer`; corpus **read-only** | **public** (`operator.closelistening.app`) |
| **Player PUBLIC** | `-p player` | `deploy-player.yml` | app-only `api`, `learning-app`; corpus **read-only** | **public** (`closelistening.app`) |

The two operator planes talk to each other **only through the shared `corpus_data` volume**:
the control plane's pipeline **writes** it; both public surfaces **read** it (`:ro`).

## Decision

**Keep the two-plane split. Do NOT consolidate. Fix the naming + document it.**

1. **Keep the split** — it is the security wall (network + volume isolation between the open
   internet and host-root/provider-keys), not RBAC. ADR-116 already made this decision
   (Option B target, today's read-only public plane = its sanctioned Option-C fallback). A
   consolidation was drafted and **rejected**: co-locating the keyed pipeline into the public
   compose project re-merges the secret blast radius the split exists to prevent (T-01/T-08),
   and it contradicted ADR-116 without reconciliation. (Advisor review, 2026-08-01.)
2. **Fix the naming** (the actual problem): `deploy-prod.yml` deploys the *control plane*, not
   "prod generically." Renamed the workflow **display names** so the Actions UI reads clearly:
   - `deploy-prod.yml` → **"Deploy operator — CONTROL plane (tailnet, privileged)"** + a loud
     header comment. **The file name stays `deploy-prod.yml`** — it is load-bearing in
     `scripts/ops/*`, `infra/cloud-init/prod.user-data`, `infra/terraform/outputs.tf`, the
     `prod-ssh-key` action, and ~40 docs/ADRs/RFCs/an incident report (history must not be
     rewritten). A full file rename is a separate, larger change and is out of scope here.
   - `deploy-operator.yml` → **"Deploy operator — PUBLIC surface"**;
     `deploy-player.yml` → **"Deploy player — PUBLIC surface"**.
3. **Document it** — PROD_RUNBOOK gains "The two operator planes + deploy map" (the picture +
   a which-workflow-deploys-what table + the coupling rules).

## Deploy map (what goes together, what's independent)

- **One build** (`stack-test` publishes one `sha-X` api/viewer image). **The three serving
  planes should roll to the SAME sha** ("one engine") — they drifted once (operator/player/
  control on three different shas); after a merge, deploy all three.
- **Coupled by data, not lifecycle:** the control plane **owns + writes** `corpus_data`; the
  public planes **read** it. They deploy independently but the control plane must exist to
  own the volume. ⚠ **`corpus_data` is project-prefixed (`compose_corpus_data`) and mounted
  `external` by the public planes — never `down -v` `-p compose`, and renaming that project
  would orphan the live corpus.**
- **Independent:** each plane is its own compose project; the **pipeline** runs on the control
  plane on-demand via `reprocess-prod.yml`; the **config layer** (caddy/alloy,
  `deploy-config.yml`) deploys separately.

## Consequences

**Positive:** the Actions UI + runbook now say what each job actually deploys; the security
split is preserved + explained; future readers won't mistake the two planes for a bug.

**Negative:** a lingering mismatch remains — the *file* `deploy-prod.yml` still reads "prod"
while its display name says "control plane." Documented as deliberate (history + tooling);
revisit if a full file rename is ever scoped.

**Neutral:** no runtime change — display-name + docs only. Player unchanged.

## Alternatives considered

- **Consolidate to one public operator surface** (draft, rejected): re-merges keys into the
  public project (T-01/T-08 regression) and contradicts ADR-116. See Decision §1.
- **Full file rename** `deploy-prod.yml` → `deploy-operator-control.yml`: correct end-state,
  but touches functional scripts + cloud-init + terraform + would rewrite historical
  ADRs/RFCs/an incident. Deferred as its own scoped change.
