# Operator handoff audit — gap notes (#805)

**Date:** 2026-06-21 · **Method:** static fresh-eyes audit of the bootstrap path
(not a live drill — see caveat). **Auditor:** agent walk of `PROD_RUNBOOK.md §First-time
bootstrap`, `DR_DRILL_RUNBOOK.md`, `infra/deploy/deploy.sh`, cross-checked against the
actual scripts and the (closed) prerequisites issue #714.

**Premise under test (from #805):** *can a second person deploy/recover prod using only
what's in this repo?* Each gap below is something that breaks that premise.

> **Caveat — this is ~70% of #805.** A static audit catches repo-self-containment and
> drift gaps. It does **not** catch live-only failures: token expiry at call time,
> account/billing state, cloud-init timing, network reachability, Tailscale DNS latency.
> Those need the live VPS drill (AC #1), which stays operator-owned.

This file is transient (per #805 AC #4): delete once all gaps below are closed.

---

## Confirmed gaps (verified against the actual files)

### G1 — Prerequisites checklist is not in the repo, and the pointer is stale  ★ bus-factor critical
`PROD_RUNBOOK.md:28-32` says *"For the prerequisites checklist … see [#714]. All commands
below assume those are done."* But:
- **#714 is CLOSED** and lives outside the repo → fails the "only what's in this repo" test.
- **#714 is stale.** It documents the Tailscale **OAuth** model (`TS_OAUTH_CLIENT_ID` +
  `TS_OAUTH_CLIENT_SECRET`, Premium+). The current runbook (`:72-77`) uses the **Free-plan**
  `TS_AUTHKEY` + `TS_API_KEY` workaround. A fresh operator following #714 sets up the wrong creds.
- #714 also omits the runtime accounts entirely (6 LLM providers, Sentry org + 3 projects,
  Grafana Cloud stack, backup-repo PAT) — those are "out of scope" in #714 but a fresh operator
  still needs them.

**Fix:** add a self-contained `docs/guides/BOOTSTRAP_PREREQUISITES.md` (accounts + credentials,
corrected to the Free-plan Tailscale model, including runtime provider/Sentry/Grafana accounts).
Repoint `PROD_RUNBOOK.md:28` to it; demote #714 to a historical reference.

### G2 — Bootstrap sections are topical, not linearly ordered
A fresh operator reading `§First-time bootstrap` top-to-bottom hits **"GitHub Actions SSH to
prod"** (`:91`) and **"deploy to DR drill"** (`:138`) *before* **"First `tofu apply`"** (`:184`)
— but you cannot SSH to a VPS that doesn't exist yet. There's no "do these in this order" map.

**Fix:** add a numbered **"Bootstrap sequence at a glance"** ordered list at the top of
`§First-time bootstrap` (prereqs → tofu apply → wait for cloud-init → set `PROD_TAILNET_FQDN` →
CI SSH key → stage `.env` secrets → first deploy → smoke), each linking its section.

### G3 — Secret retrieval hardcodes 1Password (`op read`)
`PROD_RUNBOOK.md:188,190` use `op read 'op://Personal/…'`. The operator does not use 1Password,
and a fresh operator may not either — the command fails with no fallback. (Same `op read`
pattern is copied into #714's smoke test.)

**Fix:** make retrieval password-manager-agnostic — placeholder `<your Hetzner API token>` /
`<your Tailscale API key>` with a note "(from your secrets store; e.g. `op read …`, `pass`, env)".

### G4 — No cloud-init readiness/wait guidance between `tofu apply` and first SSH
**Verified.** `§First tofu apply` (`:184-209`) ends at the GHA-variable step. Nothing tells the
operator that the VPS needs minutes of cloud-init before SSH/Tailscale work, or how to poll
readiness. The only timing hint is "~30–45 min wall" at the very top (`:57`).

Actual readiness mechanism (confirmed in `infra/cloud-init/prod.user-data`): cloud-init runs
`tailscale up` (`:187`), clones the repo to `/srv/podcast-scraper` (`:194-196`), enables
`podcast-scraper.service` (`:241`) — but the service's `ExecStartPre=/usr/bin/test -f .env`
(`:97`) means the stack does **not** come up until `.env` exists, which `deploy-prod.yml` renders
at first deploy. cloud-init prints a `final_message` (`:246`) after `$UPTIME` seconds.

**Fix:** add a short readiness note after first apply — expected cloud-init duration + how to
check (Tailscale machine online in admin, an `ssh deploy@… 'echo ok'` attempt). Make explicit
that the stack stays down until the **first `deploy-prod.yml`** stages `.env` — that's by design,
not a failure.

### G7 — `#844` removed the `.bootstrap-needs-env` sentinel, but stale operator instructions remain  ★ correctness
**Verified at code level.** `#841`/`#844` changed first-boot: `.env` is now workflow-staged
(`deploy-prod.yml` renders it from GH Secrets) and the systemd unit gates on
`ExecStartPre=/usr/bin/test -f /srv/podcast-scraper/.env` (`prod.user-data:97`). The
`.bootstrap-needs-env` sentinel is gone — `write_files` no longer creates it (only the history
comment at `prod.user-data:40-45` remains). **But four operator-facing references still describe
the old flow:**
- `infra/cloud-init/prod.user-data:246-249` — `final_message` tells the operator to *"stage
  `.env`, then `sudo rm /srv/podcast-scraper/.bootstrap-needs-env && sudo systemctl restart`"*.
  The `rm` targets a file that is never created; the "manually stage `.env`" step is superseded by
  workflow staging (#841). **This is the message a fresh operator sees on the console first.**
- `infra/cloud-init/prod.user-data:240` — comment "Enable systemd unit (will block on
  `.bootstrap-needs-env`)" → the unit blocks on `test -f .env`, not the sentinel.
- `infra/cloud-init/prod.user-data:190-193` — comment "write_files may already have created
  `.bootstrap-needs-env`" → it no longer does.
- `PROD_RUNBOOK.md:158` (drill section) — repeats *"stage `.env`, then `sudo rm
  /srv/podcast-scraper/.bootstrap-needs-env` so Docker Compose can start"*.

**Fix:** rewrite the `final_message` to the #841/#844 flow ("stage the 15 PROD_* secrets in repo
settings, then run `deploy-prod.yml` — it renders `.env` and brings the stack up"); drop the
sentinel `rm` from `PROD_RUNBOOK.md:158`; correct the two stale history comments (`:190`, `:240`)
so they don't re-mislead. This is a cross-repo-consistency correctness fix, not just docs.

### G5 — `deploy.sh` exit code 4 is undocumented in its own header
`infra/deploy/deploy.sh:9-13` documents exit codes 0–3. The script also exits **4** on
`DEPLOY_GIT_SHA`/`DEPLOY_IMAGE_SHA` validation failures (`:65,:72,:82`). An operator reading the
header to interpret a failed deploy has no entry for 4.

**Fix:** add exit 4 to the header comment block. (In-script, trivial.)

---

## Minor / optional (low ROI — note, don't necessarily fix this round)

### G6 — `deploy.sh` defensive behaviors not reflected in the runbook narrative
`deploy.sh` auto-creates `.env` and injects `PODCAST_DOCKER_PROJECT_DIR` /
`PODCAST_CORPUS_HOST_PATH` if missing (`:26-40`), and auto-repairs
`/usr/local/sbin/podcast-tailscale-serve.sh` from the repo (`:98-107`, `:131-146`, needs a
sudoers `install` rule). Harmless belt-and-suspenders, but a fresh operator debugging `.env`
or MagicDNS wouldn't know the deploy script self-heals these. Optional one-line note.

---

## Audit false-positives (claimed by the first-pass scan, but actually fine — do NOT "fix")

- **`.sops.yaml` edit path** — already given explicitly at `PROD_RUNBOOK.md:68`
  ("Copy the public key … into `infra/.sops.yaml`").
- **`./tofu` wrapper location** — runbook already does `cd infra && ./tofu init` (`:187-194`);
  it's unambiguously run from `infra/`. (Could add one line on *what* the wrapper does, but not a gap.)
- **Per-secret "feature off if missing" mapping** — already documented in the
  `§Environment variable reference` tables ("Notes if missing" column, `:1253+`). The
  `§Stage .env` table (`:255`) could cross-link it, but the information exists.

---

## Disposition for #805

- **AC #2 (notes file with itemized gaps):** this file. ✅
- **AC #3 (gaps fixed via runbook PRs):** G1–G5 fixed in the same PR (see scope doc).
- **AC #1 (live drill completed):** operator-owned. Either run later, or de-scope explicitly
  with a one-liner in the issue (the AC permits "explicitly de-scoped"). **Decision pending.**
- **AC #4 (delete this file once gaps close):** after G1–G5 merge + AC #1 resolved.
- **AC #5 (cadence: repeat every 6 months / after major infra change):** add to the runbook's
  maintenance notes.

## Deferred — DR-drill validation of the cloud-init change

The only functional file this PR touches is `infra/cloud-init/prod.user-data` (G7: `final_message`
rewrite + 2 comments — cosmetic, no behavior change). Validate it via the DR drill **after this
batch ships and `main` is stable** (operator decision, 2026-06-21 — not now):

- **Cheap guard:** `drill-infra-plan` against the branch — `tofu validate`/`plan` renders the
  `prod.user-data` templatefile and catches any breakage; **provisions no infra**.
- **Full end-to-end (optional):** `drill-exercise` (`DRILL_FULL_CYCLE`) — boots a real drill VPS
  from this cloud-init, deploys, always destroys (~€0.01, ~15-20 min).

**Caveat (do not oversell):** the drill validates cloud-init **boot**, not the runbook prose.
G1–G4 + the G7 runbook text are human-followability fixes — only the AC #1 human walk validates
those. A green drill ≠ "the hardened runbook is followable." Local `mkdocs --strict` + `markdownlint`
+ `shellcheck` already passed for this PR; template render is the only unvalidated piece, and it's
cosmetic.

Source-of-truth scope: `docs/wip/V27-OBSERVABILITY-SCOPE-803-805-426_2026-06-21.md`.
