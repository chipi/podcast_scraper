# Stack contract and environment adapters

**Audience:** Operators and workflow authors who need one place to compare how the product
stack runs on Codespaces, production VPS, DR drill VPS, CI stack-test, and the orchestrated
drill exercise — without rereading every workflow file.

**Law:** [ADR-093](../adr/ADR-093-canonical-stack-contract-and-environment-adapters.md).
**Hosting narrative:** [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) (Design
section [Stack contract vs environment adapters](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md#stack-contract-vs-adapters)).
**Corpus backup/restore mechanics (manifest, Make vs Actions):**
[CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).

---

## Steady-state playbook (routine)

Use this sequence for **normal** bring-up and deploy validation. It does **not** include
restoring corpus from a backup tarball — that is **DR exercise** or **manual high-blast**
restore only (see [Recovery-only](#recovery-only-not-steady-state) below).

1. **Preflight** — secrets, tailnet variables, and typed confirms (where the workflow requires them).
2. **Deploy or bring-up** — pull GHCR images (or build in CI), `compose up` / `deploy.sh`.
3. **Health smoke** — `/api/health` with the **authoritative probe** for the surface (see audit table);
   adapter/ingress probes may follow but do not replace the contract check (GH-745).
4. **Behavioral gate** (when claiming “deployed correctly”) — `tests/stack-test` Playwright on CI;
   drill HTTPS Playwright after smoke in the orchestrated exercise.

**Transport** (Tailscale SSH, Codespace `postStart`, runner Docker, HTTPS MagicDNS) is an
**adapter** only; it must not change compose topology, corpus semantics, or health meaning.

---

## Audit table

| Surface | Compose files | Env / corpus path | Bring-up entrypoint | Authoritative health probe | Adapter / ingress probes | Behavioral gate | Restore from `snapshot.tgz` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Codespaces (pre-prod)** | `stack.yml` + `prod.yml` (no `vps-prod`) | `.devcontainer/start.sh` exports `PODCAST_DOCKER_PROJECT_DIR`, `PODCAST_CORPUS_HOST_PATH` → `.codespace_corpus` | `.devcontainer/start.sh`; optional `deploy-codespace.yml` | In-container `api` `curl` `http://127.0.0.1:8000/api/health` for parity with VPS | Operator / forwarded viewer port; optional `post-deploy-smoke.yml` codespace SSH `:8090` | `stack-test.yml` on main before publish; optional `post-deploy-smoke.yml` after codespace deploy | Manual / Make paths in [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) — **not** post-deploy routine |
| **Prod VPS** | `stack.yml` + `prod.yml` + `vps-prod.yml` | `/srv/podcast-scraper/.env`; corpus `/srv/podcast-scraper/corpus` | `infra/deploy/deploy.sh` via `deploy-prod.yml` | In-container `api` `curl` `http://127.0.0.1:8000/api/health` in `deploy.sh` and `restore_corpus_from_tarball_host.sh` (GH-745) | `deploy-prod.yml` HTTPS tailnet probe after deploy | `stack-test.yml` on main before GHCR publish (no prod Playwright gate in GHA today) | `prod-restore-corpus.yml` — manual **`PROD_RESTORE`** only |
| **DR drill VPS** | Same triple as prod | Same host layout as prod | Same `deploy.sh` via `drill-deploy.yml` | Same in-container `api` `:8000` probe in `deploy.sh` and shared restore host script | `drill-deploy.yml` and `drill-e2e.yml` SSH to host viewer **8080**; `drill-stack-playwright.yml` HTTPS MagicDNS | `drill-stack-playwright.yml` HTTPS + `stack-viewer.spec.ts` after `drill-e2e` | `drill-restore-corpus.yml` in **orchestrator** + manual **`DRILL_RESTORE`** |
| **CI stack-test** | `stack.yml` + `stack-test.yml` | Ephemeral compose project; seeded corpus via `make stack-test-seed` | `make stack-test-build` / `up` / `seed` / `playwright` in `stack-test.yml` | Runner wait on published viewer `:8090` `/api/health` before Playwright | N/A | Full `tests/stack-test` on runner | Not used |
| **Drill orchestrator** | (inherits drill VPS row after apply) | N/A — chains workflows | `drill-exercise.yml`: plan → apply → deploy → **restore** → e2e → Playwright → destroy | Per-step authoritative probes on the drill VPS row | Per-step adapter probes on the drill VPS row | `drill-stack-playwright` after `drill-e2e` | **`drill-restore-corpus`** as simulated recovery — **not** steady prod |

Shared host restore: runner resolves tag via `resolve_latest_snapshot_prod_tag.sh` (ADR-092
newest-compatible `snapshot-prod-*`), **`download_and_verify_snapshot.sh`** on the runner,
uploads tarball + `scripts/ops/restore_corpus_from_tarball_host.sh`, then one SSH invoke. Details:
[CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).

---

## Recovery-only (not steady-state)

- **DR full cycle** — `drill-exercise.yml` includes corpus restore to prove recovery on a
  throwaway drill host; always ends in destroy.
- **Prod manual restore** — `prod-restore-corpus.yml`; environment **prod**, confirm
  **`PROD_RESTORE`**.
- **Drill manual restore** — `drill-restore-corpus.yml`; confirm **`DRILL_RESTORE`** when not
  called from the orchestrator.

Do not document these as steps in daily prod or Codespace bring-up.

---

## Related docs

| Doc | Role |
| --- | --- |
| [PROD_RUNBOOK.md](PROD_RUNBOOK.md) | Prod operator commands |
| [DR_DRILL_RUNBOOK.md](DR_DRILL_RUNBOOK.md) | Drill workflows and confirms |
| [`infra/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/infra/README.md) | OpenTofu + deploy script layout |
| [WORKFLOWS.md](../ci/WORKFLOWS.md) | CI workflow index |
| [`compose/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/compose/README.md) | Compose file purposes |
| [CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) | Manifest, tag selection, Make vs Actions restore |

When you change compose overlays, corpus bind paths, health probes, or shared restore scripts,
update this table and [ADR-093](../adr/ADR-093-canonical-stack-contract-and-environment-adapters.md)
implementation notes in the same change.
