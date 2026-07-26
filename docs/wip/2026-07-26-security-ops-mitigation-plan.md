# Mitigation Plan — Production Security & Ops (2026-07-26)

Companion to [`2026-07-26-production-security-ops-review.md`](2026-07-26-production-security-ops-review.md).
Turns the Fable 5 review findings into a phased, owner-tagged plan. Owners:
**[me]** = in-repo change I make; **[op]** = operator action (CF dashboard,
GH var, SSH, registrar) I can't do from the repo; **[me+op]** = I prep, you flip.

---

## Phase 0 — Verification (done tonight, read-only)

| Item | Result | Evidence |
|---|---|---|
| **P0-1 origin-lock applied?** | ✅ **RESOLVED** — `:443` firewall locked to **22 CF ranges** (`103.21.244.0/22`…); `:80` world-open as designed (ACME) | decrypted `terraform.tfstate.enc`, `hcloud_firewall` port-443 `source_ips` = CF ranges |
| P2-1 operator secrets at rest? | ✅ confirmed real — `OPERATOR_SECRETS_VIA_FILES` absent (player+pipeline = `1`) | `gh variable list` |
| P2-2 `.com` in admin emails? | ✅ confirmed real — `APP_ADMIN_EMAILS = …gmail.com,info@closelistening.com` | `gh variable list` |
| P1-3 blast radius today | ✅ nil — `OPERATOR_ALLOWED_EMAILS` = 1 email, `OPERATOR_SIGNUP_MODE=allowlist` | `gh variable list` |

**Still needs live access I don't have (→ [op]):**
- CF managed-WAF + rate-limit rules configured? (needs CF dashboard/token)
- fail2ban `caddy-access` jail actually loaded on the *running* box? (needs SSH)
- Full inbound firewall rule set — I checked only `:443/:80/:22`; confirm no
  privileged port is open (default-deny Hetzner fw makes this likely, unconfirmed).

**Net effect:** the single P0 is closed. Because the origin is CF-locked, every
"no origin rate-limit / no WAF" finding drops one severity band — an attacker
can't reach the origin bypassing Cloudflare. They remain real for *authenticated*
abuse and for defense-in-depth, but they're no longer the fire.

---

## Phase 1 — P1 quick wins (in-repo, ~half day, all [me])

Sequence these first; all are additive + low-risk.

1. **P1-1 — viewer origin rate-limiting + fix the RFC divergence.**
   Add a `limit_req` zone to `docker/viewer/default.conf.template` mirroring the
   player (`lp_api` 20r/s, `lp_auth` 2r/s on `/api/app/auth/`, `429`) + real-IP
   recovery. Correct RFC-108:42,53 to match reality (or point the claim at CF).
   → needs a viewer image rebuild + `deploy-operator` (the ~80-min build cycle).
2. **P1-3 — refuse `signup_mode=open` under `operator_public`.**
   Startup assertion in `create_app` (`app.py`): if `PODCAST_SERVE_OPERATOR_PUBLIC`
   and signup mode is `open` → refuse to boot. Closes the "one env flip opens the
   whole corpus" footgun. (Deeper fix — gate operator-read on an allowlist-derived
   role instead of the `?grant=creator` hint — is larger; Phase 2.)
3. **P1-4 — operator-appdata backup.** Clone `backup-player-appdata-prod.yml` →
   `operator-appdata` path + daily cron. Pure workflow add.
4. **P1-5 — corpus backup schedule.** Add a cron (start weekly) to
   `backup-corpus-prod.yml` — decouple from the #723 cutover. One line + confirm
   the restore drill has targeted **prod** (DR drill currently blocked on the
   segfault image publish — tracked, Phase 3).

**Batching:** items 2–4 are workflow/code-only → one branch, one deploy-less merge.
Item 1 needs the image-rebuild cycle → fold into the next operator deploy.

---

## Phase 2 — hardening (mix [me]/[op])

5. **P1-2 — doorman creds.** [op verify] Confirm `PLAYER_PREVIEW_PASS` (and the
   `guest` password) are high-entropy random, not dictionary words — the cost-14
   hashes are public (public repo). If weak → [me] rotate both vhosts + drop the
   shared `guest` account.
6. **P2-1 — operator secrets to tmpfs.** [me+op] Set `OPERATOR_SECRETS_VIA_FILES=1`
   (machinery already in `deploy-operator.yml:248-279`) + redeploy. Removes the
   OAuth/session secret from disk-at-rest.
7. **CF WAF + rate-limit rules.** [op] Managed WAF ruleset + rate-limit on
   `/preview` and `/api/app/auth/` on both zones — the actual T-05 L7 control.
8. **A-9 — security headers.** [me] Add `X-Frame-Options: DENY`,
   `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`
   to the `(hardened)` Caddy snippet. (CSP is a separate, larger audit.)
9. **P2-2 — `info@closelistening.com`.** [op] Confirm you own `.com` or drop it
   from `APP_ADMIN_EMAILS`.
10. **P2-3 — HSTS ramp.** [me] `max-age=31536000; includeSubDomains; preload`
    once confident (two live vhosts now).
11. **P2-4 — narrow `set_real_ip_from`.** [me] From `0.0.0.0/0` to the
    Caddy/compose bridge CIDR in the player nginx.
12. **P2-5 — external uptime alerting.** [me+op] Synthetic probe on `/` for both
    domains → ntfy/PagerDuty; closes the "silent 3am container crash" gap.
13. **P2-6 — fail2ban live verify.** [op/SSH] `fail2ban-client status caddy-access`
    on the box; confirm loaded, not just in cloud-init.

---

## Phase 3 — maturity (B-items, later)

- Digest-pin GHCR images (T-09); pin Caddy apt version.
- Per-env LLM keys + provider billing caps (T-08 D5).
- Secret rotation cadence + automation (OAuth client, session, gate cookie).
- Gate operator-read on allowlist-derived role, not the request hint (deeper P1-3).
- DR drill unblock (needs green `stack-test` → complete image publish after the
  segfault fix) + a documented, rehearsed <5-min per-surface rollback.
- CORS allowlist for future non-same-origin clients.
- orrery → own VPS (real multi-tenant isolation) when budget allows.

---

## Recommended order of operations

1. **Tonight/now:** Phase 0 done. Origin-lock closed the P0.
2. **Next session (in-repo, no deploy):** Phase 1 items 2–4 (assertion + 2 backup
   workflows) on one branch.
3. **Next operator deploy:** Phase 1 item 1 (viewer rate-limit, needs rebuild) +
   Phase 2 item 6 (`OPERATOR_SECRETS_VIA_FILES=1`) + item 8 (security headers).
4. **[op] out-of-band:** CF WAF/rate-limit rules (item 7), doorman-password check
   (item 5), `.com` decision (item 9), fail2ban live check (item 13).
5. **Track:** Phase 3.

Nothing here is acted on yet — this is the plan to review.

---

## Addendum (2026-07-26) — backup alignment + DR appdata (in PR #1334)

Operator direction on P1-4/P1-5 refined the backups and DR:

**Backup alignment — all three backups now consistent.**
- All of `backup-player-appdata-prod.yml`, `backup-operator-appdata-prod.yml`,
  `backup-corpus-prod.yml` run on `environment: prod-backup` (a **new, UNGATED**
  environment — `protection_rules: []`) and a **daily** schedule.
- `prod-backup` is a **provenance label**, not a gate: the literal `prod` environment
  has `required_reviewers: [chipi]` and is shared with the deploy workflows (which must
  stay gated), so a scheduled job on `prod` would pend forever. An ungated environment
  gives the "which env did this backup come from" label + future multi-env clarity
  without blocking the schedule.
- **Cascade check (operator asked): none.** The corpus backup references only repo-level
  secrets (`BACKUP_REPO_TOKEN`, `PROD_SSH_PRIVATE_KEY`, `TS_OAUTH_*`) + repo-level vars —
  the *identical* set the always-ungated player backup uses. **Zero overlap** with the
  prod-env-scoped secrets (`PROD_OPENAI/HF/GEMINI_*`, `PROD_SENTRY_DSN_*`, `PROD_GRAFANA_*`,
  `INFRA_STATE_COMMIT_TOKEN` — those belong to deploys/infra, still on `environment: prod`).
  Moving the corpus backup off `prod` loses no secret/var access. No other workflow keys on
  its prod deployment (only a comment in `reprocess-prod.yml`). Verified via `gh api
  .../environments/prod/secrets` + `grep secrets\\.` on the workflows.

**DR gap closed — appdata restore + validate (was corpus-only).**
- Both DR paths restored + validated only the **corpus**; user data (playback/notes/
  favorites + operator prefs/role grants — not regenerable) was never rehearsed.
- `verify-backup-restore.yml` (Sun compose smoke): now also downloads the latest
  `player-appdata-prod-*` + `operator-appdata-prod-*`, verifies each tarball, extracts, and
  asserts the dir landed.
- `drill-restore-corpus.yml` (real Hetzner, via `drill-exercise.yml` Wed): after the corpus
  restore, ships both appdata tarballs to the drill VPS, extracts under
  `/srv/podcast-scraper`, and asserts the dirs are present on the box.
- Both are **tolerant** of a not-yet-created backup (warn+skip — the operator-appdata
  backup is brand new) and **strict** once a release exists.

## Addendum 2 (2026-07-26) — D-item progress

- **P2-1 (operator secrets → tmpfs) — in-repo gap CLOSED in PR #1334.** The deploy was
  already fully wired for ADR-115 Option A (drops the 3 runtime secrets from `.env.operator`,
  stages them to `/dev/shm/operator-secrets/`, passes the flag to the script) — but the compose
  overlay `docker-compose.operator-secrets.yml` that `deploy-operator.sh` joins was **missing**,
  so flipping the var today would have failed the deploy on the `-f` join. Created the overlay
  (mirrors the player's, operator-scoped tmpfs) + a contract guard. **Activation (post-merge):**
  set `OPERATOR_SECRETS_VIA_FILES=1` **after** this merges (so the overlay is on the box), then
  redeploy operator — do NOT flip before merge.
- **P2-2 (`.com` admin email) — DONE.** Operator confirmed it was a typo for `.app`;
  `APP_ADMIN_EMAILS` corrected to `marko.dragoljevic@gmail.com,info@closelistening.app` (GH var,
  effective next deploy). No `closelistening.com` remains in any variable.
- **P2-5 (external uptime) — DONE (operator-side).** Operator confirmed external uptime
  monitoring is already covered. Dismissed.
- **P1-2 (doorman password) — dismissed** by operator (not a concern).
