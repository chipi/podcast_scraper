# DR drill runbook (Hetzner + GitHub Actions)

This guide is **drill-only**: Hetzner workspace **`drill`**, Tailscale **`tag:dr-drill`**, and the
GitHub Actions workflows that target **`deploy@`** on the drill VPS. **Production** VPS steps,
shared Tailscale credential setup, and full disaster-recovery narrative stay in
[PROD_RUNBOOK.md](PROD_RUNBOOK.md) (see **GitHub Actions SSH to prod**, **GitHub Actions deploy to DR drill**,
and **Disaster recovery**).

**Full-system context (Tailscale + OpenTofu + GHA + Compose):** [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md).

**IaC layout and OpenTofu workspace rules:** [`infra/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/infra/README.md) (section **DR drill workspace**).

**Workflow index (CI file names and triggers):** [WORKFLOWS.md](../ci/WORKFLOWS.md) (OpenTofu / DR drill paragraph).

**Timed exercise prerequisites:** [RFC-082 DR drill prerequisite checklist](../wip/RFC-082_DR_DRILL_PREREQ_CHECKLIST.md).

---

## When to use which doc

| Need | Doc |
| --- | --- |
| Prod **`deploy@`**, prod Tailscale, prod backup/restore | [PROD_RUNBOOK.md](PROD_RUNBOOK.md) |
| Drill **`deploy@`**, drill workflows, full-cycle orchestrator, drill destroy | This runbook + [`infra/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/infra/README.md) |
| OpenTofu state encryption, **`terraform.tfstate.enc.drill`** | [`infra/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/infra/README.md) |
| Whole-system narrative (diagrams, prod + drill) | [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md) |

Keeping drill material here avoids duplicating long prod sections and keeps PROD_RUNBOOK shorter for
day-to-day prod operators.

---

## Repository secrets and variables (drill)

| Name | Kind | Used by |
| --- | --- | --- |
| `HCLOUD_TOKEN_DRILL` | secret | `drill-infra-plan`, `drill-infra-apply`, `drill-infra-destroy`, orchestrator |
| `TS_API_KEY` | secret | Same infra workflows (Tailscale provider) |
| `TFSTATE_AGE_KEY` | secret | sops decrypt/encrypt for drill state |
| `OPERATOR_SSH_PUBLIC_KEY` | secret | Same as prod infra (public half only; log masking) |
| `TS_AUTHKEY` | secret | `deploy-drill`, `drill-e2e`, `drill-restore-corpus`, orchestrator app jobs |
| `DRILL_DEPLOY_SSH_PRIVATE_KEY` | secret | Same app jobs (PEM for **`deploy@`** on drill) |
| `BACKUP_REPO_TOKEN` | secret (optional) | `drill-restore-corpus` when `chipi/podcast_scraper-backup` is private |
| `TAILNET_NAME` | variable | Infra `TF_VAR_tailscale_tailnet` |
| `DRILL_TAILNET_FQDN` | variable | Resolver input (e.g. `dr-podcast.tail-xxxx.ts.net`) |

ACL note: **`tag:gha-deployer` → `tag:dr-drill:22`** must exist for CI SSH; prod **`infra-apply`** is
the usual path to land `tailscale/policy.hujson` changes.

---

## Typed confirms (copy exactly)

| String | Workflow | Meaning |
| --- | --- | --- |
| `DRILL_FULL_CYCLE` or `DRILL_EXERCISE` | `dr-drill-exercise.yml` | Gates full orchestrator (infra + app + always destroy) |
| `APPLY` | `drill-infra-apply.yml` (manual only) | OpenTofu apply workspace **`drill`** |
| `DRILL_DESTROY` | `drill-infra-destroy.yml` (manual only) | **`tofu destroy`** + **Hetzner `hcloud` sweep** + **Tailscale device delete** (`tag:dr-drill` or drill `tailnet_hostname`); standalone uses git state — orchestrator uses apply artifact (same run) |
| `DRILL_RESTORE` | `drill-restore-corpus.yml` (manual only) | Overwrite drill **`corpus/`** from backup **`snapshot.tgz`** |
| `DRILL_SMOKE` | `drill-e2e.yml` (manual or orchestrator) | Read-only **`/api/health`** via tailnet SSH |
| `DRILL_STACK_PLAYWRIGHT` | `drill-stack-playwright.yml` (manual; orchestrator uses `skip_confirm`) | Run **`tests/stack-test/stack-viewer.spec.ts`** against drill HTTPS |

The orchestrator calls **`drill-infra-apply`** and **`drill-infra-destroy`** with internal
**`skip_confirm: true`** after the single **`DRILL_FULL_CYCLE`** / **`DRILL_EXERCISE`** gate; standalone
runs still require **`APPLY`** / **`DRILL_DESTROY`** as above.

---

## Orchestrated full cycle (`dr-drill-exercise.yml`)

Ordered jobs:

1. **`drill-infra-plan`** — `tofu fmt -check`, **`tofu validate`**, **`tofu plan`** (no apply).
2. **`drill-infra-apply`** — **`tofu apply`** for workspace **`drill`**; uploads **`terraform-state-after-apply-drill`**.
3. **`drill-tfstate-bridge`** — caller job: re-downloads that artifact and uploads **`drill-tfstate-for-teardown`** so **`drill-infra-destroy`** (a reusable workflow) can read state without a git commit.
4. **`deploy-drill`** — git pull, compose pull/up, host **`/api/health`** smoke.
5. **`drill-restore-corpus`** — download **`snapshot.tgz`**, extract **`corpus/`**, recreate **api** + **viewer**.
6. **`drill-e2e`** — tailnet SSH **`curl`**:8080 **`/api/health`** (quick smoke).
7. **`drill-stack-playwright`** — **`tests/stack-test/stack-viewer.spec.ts`** over **HTTPS** against the live drill host (browser + API + corpus).
8. **`finalize`** — runs **`if: always()`** so the next step still runs when a middle job failed.
9. **`drill-infra-destroy`** — downloads **`drill-tfstate-for-teardown`** (or apply artifact as fallback), then **`tofu destroy`**, **Hetzner API sweep**, **Tailscale API** removal of drill devices (always after finalize).

Each job that uses GitHub **Environment `drill`** may require a separate approval if your org
configured reviewers on that environment.

**Dispatch example:**

```bash
gh workflow run dr-drill-exercise.yml -R chipi/podcast_scraper \
  -f confirm=DRILL_FULL_CYCLE
```

Optional inputs: **`backup_tag`**, **`backup_repo`**, **`override_image_sha`** (see workflow file).

---

## App-only path (no infra create/destroy)

Use individual workflows when you already have a drill VM and only want the app pipeline:

- **`deploy-drill.yml`**
- **`drill-restore-corpus.yml`** (confirm **`DRILL_RESTORE`**)
- **`drill-e2e.yml`** (confirm **`DRILL_SMOKE`**)
- **`drill-stack-playwright.yml`** (confirm **`DRILL_STACK_PLAYWRIGHT`** — runs **`stack-viewer.spec.ts`** against drill HTTPS)

Do **not** use **`dr-drill-exercise.yml`** for that case: it always ends in **`drill-infra-destroy`**.

---

## Drill host `.env` checklist (after first boot or restore)

On **`/srv/podcast-scraper/.env`** (modes **`600`**), typical values include:

- **`PODCAST_ENV=dr-drill`** (or another non-**`prod`** label for Sentry/Grafana filters).
- **`PODCAST_CORPUS_HOST_PATH=/srv/podcast-scraper/corpus`** (bind mount source on host).
- **`PODCAST_DOCKER_PROJECT_DIR=/srv/podcast-scraper`** (compose bind mount for api).
- **`PODCAST_DEFAULT_CORPUS_PATH=/app/output`** — pre-fills the viewer corpus path (container view of
  the same bind mount as the api). See compose **`PODCAST_DEFAULT_CORPUS_PATH`** and nginx templates
  under **`docker/viewer/`**.

Grafana Cloud: if **`GRAFANA_CLOUD_*`** URLs are placeholders (for example **`invalid.invalid`**), the
Grafana Agent will not ship metrics; that is expected for isolated drills unless you point drill at
real Grafana Cloud endpoints.

---

## Manual infra destroy (without orchestrator)

```bash
gh workflow run drill-infra-destroy.yml -R chipi/podcast_scraper -f confirm=DRILL_DESTROY
```

The job runs **`tofu destroy`**, **Hetzner `hcloud` sweep**, and **Tailscale `DELETE /device`** for nodes with **`tag:dr-drill`** or hostname equal to **`tailnet_hostname`** in **`terraform.drill.ci.tfvars`**.

When you run **`dr-drill-exercise`**, **`drill-infra-destroy`** downloads **`drill-tfstate-for-teardown`** from the **`drill-tfstate-bridge`** job (same workflow run), so destroy matches what apply created **without committing** `terraform.tfstate.enc.drill` to git. A **standalone** destroy (only this workflow) still uses the encrypted file from the **git** checkout; commit that file after standalone apply if you need destroy/plan to track that state.

Download the uploaded **`terraform-state-after-destroy-drill`** artifact when you need to commit an
updated **`infra/terraform/terraform.tfstate.enc.drill`** (same pattern as **`drill-infra-apply`**).

---

## Prod corpus restore (separate workflow)

**`prod-restore-corpus.yml`** is documented in [PROD_RUNBOOK.md](PROD_RUNBOOK.md) (corpus backups and
cross-references). It uses **`PROD_SSH_PRIVATE_KEY`**, Environment **`prod`**, and confirm **`PROD_RESTORE`**.
It is **not** part of the drill orchestrator.

---

## See also

- [PROD_OPERATOR_CHEAT_SHEET.md](PROD_OPERATOR_CHEAT_SHEET.md) — short prod-oriented reminders.
- [RFC-082 — Always-on hosting](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) — design context.
